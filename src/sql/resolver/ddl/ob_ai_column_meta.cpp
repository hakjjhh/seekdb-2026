/**
 * Copyright (c) 2021 OceanBase
 * SPDX-License-Identifier: Apache-2.0
 */

#define USING_LOG_PREFIX SQL_RESV

#include <string.h>
#include "sql/resolver/ddl/ob_ai_column_meta.h"
#include "lib/allocator/ob_malloc.h"
#include "lib/oblog/ob_log_module.h"
#include "share/ob_errno.h"
#include <pthread.h>
#include <unistd.h>
#include "share/ob_server_struct.h"
#include "lib/oblog/ob_log.h"
#include "lib/string/ob_sql_string.h"
#include "common/mysqlclient/ob_mysql_proxy.h"
#include "common/mysqlclient/ob_isql_client.h"
#include "common/mysqlclient/ob_mysql_result.h"
#include <stdio.h>
#include <stdlib.h>
#include <curl/curl.h>
#include "lib/allocator/page_arena.h"
#include "common/object/ob_obj_type.h"

namespace oceanbase
{
namespace sql
{

using namespace common;

namespace
{

bool node_name_equal(const ParseNode *node, const char *name)
{
  bool bret = false;
  if (NULL != node && NULL != node->str_value_ && NULL != name) {
    const int64_t len = static_cast<int64_t>(node->str_len_);
    bret = (len > 0) && (len == static_cast<int64_t>(strlen(name))) &&
           (0 == strncasecmp(node->str_value_, name, static_cast<size_t>(len)));
  }
  return bret;
}

// the metadata lives until the observer exits, the small amount of memory is not
// released on purpose, it keeps the registry lock free during queries
int dup_string(const ObString &src, ObString &dst)
{
  int ret = OB_SUCCESS;
  char *buf = NULL;
  if (src.empty()) {
    dst.reset();
  } else if (OB_ISNULL(buf = static_cast<char *>(
                         ob_malloc(src.length() + 1, "AiColumnMeta")))) {
    ret = OB_ALLOCATE_MEMORY_FAILED;
  } else {
    MEMCPY(buf, src.ptr(), src.length());
    buf[src.length()] = '\0';
    dst.assign_ptr(buf, src.length());
  }
  return ret;
}

int64_t func_param_count(const ParseNode *func)
{
  int64_t cnt = 0;
  if (NULL != func && func->num_child_ > 1 && NULL != func->children_[1]) {
    const ParseNode *params = func->children_[1];
    cnt = (T_EXPR_LIST == params->type_) ? params->num_child_ : 1;
  }
  return cnt;
}

const ParseNode *func_param(const ParseNode *func, const int64_t idx)
{
  const ParseNode *param = NULL;
  if (NULL != func && func->num_child_ > 1 && NULL != func->children_[1] && idx >= 0) {
    const ParseNode *params = func->children_[1];
    if (T_EXPR_LIST == params->type_) {
      if (idx < params->num_child_) {
        param = params->children_[idx];
      }
    } else if (0 == idx) {
      param = params;
    }
  }
  return param;
}

const ParseNode *find_ai_func(const ParseNode *node)
{
  const ParseNode *found = NULL;
  if (NULL == node) {
    // do nothing
  } else if (T_FUN_SYS == node->type_ && node->num_child_ > 0) {
    if (node_name_equal(node->children_[0], "AI_COMPLETE") ||
        node_name_equal(node->children_[0], "AI_EMBED")) {
      found = node;
    }
  }
  for (int64_t i = 0; NULL != node && NULL == found && i < node->num_child_; ++i) {
    found = find_ai_func(node->children_[i]);
  }
  return found;
}

void collect_ai_dependency(const ParseNode *node,
                           const ObIArray<ObString> &ai_column_names,
                           bool &depends)
{
  if (NULL == node || depends) {
    // do nothing
  } else {
    if (T_COLUMN_REF == node->type_ && node->num_child_ >= 3 && NULL != node->children_[2]) {
      const ParseNode *column_node = node->children_[2];
      if (NULL != column_node->str_value_) {
        const ObString column_name(static_cast<int32_t>(column_node->str_len_),
                                   column_node->str_value_);
        for (int64_t i = 0; !depends && i < ai_column_names.count(); ++i) {
          if (ai_column_names.at(i).case_compare_equal(column_name)) {
            depends = true;
          }
        }
      }
    }
    for (int64_t i = 0; !depends && i < node->num_child_; ++i) {
      collect_ai_dependency(node->children_[i], ai_column_names, depends);
    }
  }
}

// the dimension declared by "vector(n)", -1 when the type is not a vector
int64_t get_vector_dimension(const ParseNode *data_type)
{
  int64_t dim = -1;
  if (NULL != data_type && T_COLLECTION == data_type->type_ &&
      data_type->num_child_ == 1 && NULL != data_type->children_[0] &&
      T_INT == data_type->children_[0]->type_ &&
      1 == data_type->int32_values_[0]) {
    dim = data_type->children_[0]->value_;
  }
  return dim;
}


// copy a value into the given allocator, the copy is NUL terminated
int ai_copy_text(const ObString &src, ObIAllocator &allocator, ObString &dst)
{
  int ret = OB_SUCCESS;
  char *buf = NULL;
  dst.reset();
  if (src.empty() || NULL == src.ptr()) {
    // an empty value needs no memory
  } else if (OB_ISNULL(buf = static_cast<char *>(allocator.alloc(src.length() + 1)))) {
    ret = OB_ALLOCATE_MEMORY_FAILED;
    LOG_WARN("failed to allocate the value of the ai column", K(ret));
  } else {
    MEMCPY(buf, src.ptr(), src.length());
    buf[src.length()] = '\0';
    dst.assign_ptr(buf, static_cast<int32_t>(src.length()));
  }
  return ret;
}

// walk the prompt expression of AI_COMPLETE and collect the literals and the
// referenced columns in the order they appear
int collect_prompt_parts(const ParseNode *node,
                         ObIAllocator &allocator,
                         ObAiColumnMeta &meta)
{
  int ret = OB_SUCCESS;
  if (NULL == node) {
    // nothing to collect
  } else if (T_VARCHAR == node->type_ || T_CHAR == node->type_) {
    if (NULL != node->str_value_ && node->str_len_ > 0) {
      ObAiPromptPart part;
      const ObString text(static_cast<int32_t>(node->str_len_), node->str_value_);
      part.is_column_ = false;
      if (OB_FAIL(ai_copy_text(text, allocator, part.text_))) {
        LOG_WARN("failed to copy the prompt literal of the ai column", K(ret));
      } else if (OB_FAIL(meta.prompt_parts_.push_back(part))) {
        LOG_WARN("failed to push back the prompt literal", K(ret));
      }
    }
  } else if (T_COLUMN_REF == node->type_ && node->num_child_ >= 3 &&
             NULL != node->children_[2] && NULL != node->children_[2]->str_value_ &&
             node->children_[2]->str_len_ > 0) {
    ObAiPromptPart part;
    const ParseNode *name_node = node->children_[2];
    const ObString name(static_cast<int32_t>(name_node->str_len_), name_node->str_value_);
    part.is_column_ = true;
    if (OB_FAIL(ai_copy_text(name, allocator, part.text_))) {
      LOG_WARN("failed to copy the prompt column of the ai column", K(ret));
    } else if (OB_FAIL(meta.prompt_parts_.push_back(part))) {
      LOG_WARN("failed to push back the prompt column", K(ret));
    } else if (OB_FAIL(meta.dep_columns_.push_back(part.text_))) {
      LOG_WARN("failed to push back the dependency column", K(ret));
    }
  } else {
    for (int64_t i = 0; OB_SUCC(ret) && i < node->num_child_; ++i) {
      if (OB_FAIL(collect_prompt_parts(node->children_[i], allocator, meta))) {
        LOG_WARN("failed to collect the prompt part of the ai column", K(ret));
      }
    }
  }
  return ret;
}

}  // namespace

ObAiColumnRegistry &ObAiColumnRegistry::get_instance()
{
  static ObAiColumnRegistry instance;
  return instance;
}

int ObAiColumnRegistry::add_meta(const ObAiColumnMeta &meta)
{
  int ret = OB_SUCCESS;
  ObAiColumnMeta stored;
  lib::ObMutexGuard guard(mutex_);
  if (OB_FAIL(dup_string(meta.db_name_, stored.db_name_))) {
  } else if (OB_FAIL(dup_string(meta.table_name_, stored.table_name_))) {
  } else if (OB_FAIL(dup_string(meta.column_name_, stored.column_name_))) {
  } else if (OB_FAIL(dup_string(meta.normalized_expr_, stored.normalized_expr_))) {
  } else if (OB_FAIL(dup_string(meta.state_column_name_, stored.state_column_name_))) {
  } else if (OB_FAIL(dup_string(meta.request_column_name_, stored.request_column_name_))) {
  } else if (OB_FAIL(dup_string(meta.result_spec_column_name_, stored.result_spec_column_name_))) {
  } else {
    stored.max_retries_ = meta.max_retries_;
    stored.dim_ = meta.dim_;
    stored.is_json_ = meta.is_json_;
    stored.is_embed_ = meta.is_embed_;
    stored.result_type_ = meta.result_type_;
    stored.pending_ = false;
    if (OB_SUCC(ret) && OB_FAIL(dup_string(meta.model_name_, stored.model_name_))) {
      LOG_WARN("failed to copy the model name", K(ret));
    }
    for (int64_t i = 0; OB_SUCC(ret) && i < meta.dep_columns_.count(); ++i) {
      ObString copied;
      if (OB_FAIL(dup_string(meta.dep_columns_.at(i), copied))) {
        LOG_WARN("failed to copy the dependency column", K(ret));
      } else if (OB_FAIL(stored.dep_columns_.push_back(copied))) {
        LOG_WARN("failed to push back the dependency column", K(ret));
      }
    }
    for (int64_t i = 0; OB_SUCC(ret) && i < meta.pk_columns_.count(); ++i) {
      ObString copied;
      if (OB_FAIL(dup_string(meta.pk_columns_.at(i), copied))) {
        LOG_WARN("failed to copy the primary key column", K(ret));
      } else if (OB_FAIL(stored.pk_columns_.push_back(copied))) {
        LOG_WARN("failed to push back the primary key column", K(ret));
      }
    }
    for (int64_t i = 0; OB_SUCC(ret) && i < meta.prompt_parts_.count(); ++i) {
      ObAiPromptPart part;
      part.is_column_ = meta.prompt_parts_.at(i).is_column_;
      if (OB_FAIL(dup_string(meta.prompt_parts_.at(i).text_, part.text_))) {
        LOG_WARN("failed to copy the prompt part", K(ret));
      } else if (OB_FAIL(stored.prompt_parts_.push_back(part))) {
        LOG_WARN("failed to push back the prompt part", K(ret));
      }
    }
    // the same (db, table, column) triple is registered only once
    for (int64_t i = 0; OB_SUCC(ret) && i < metas_.count(); ++i) {
      const ObAiColumnMeta &old_meta = metas_.at(i);
      if (old_meta.db_name_.case_compare_equal(stored.db_name_) &&
          old_meta.table_name_.case_compare_equal(stored.table_name_) &&
          old_meta.column_name_.case_compare_equal(stored.column_name_)) {
        ret = OB_ENTRY_EXIST;
      }
    }
    if (OB_ENTRY_EXIST == ret) {
      ret = OB_SUCCESS;
    } else if (OB_FAIL(ret)) {
    } else if (OB_FAIL(metas_.push_back(stored))) {
      LOG_WARN("failed to push back ai column meta", K(ret));
    }
  }
  return ret;
}

int ObAiColumnRegistry::set_pending(const ObString &db_name,
                                    const ObString &table_name,
                                    const ObString &column_name,
                                    const bool pending)
{
  int ret = OB_SUCCESS;
  lib::ObMutexGuard guard(mutex_);
  bool found = false;
  for (int64_t i = 0; !found && i < metas_.count(); ++i) {
    ObAiColumnMeta &meta = metas_.at(i);
    if (meta.table_name_.case_compare_equal(table_name) &&
        meta.column_name_.case_compare_equal(column_name) &&
        (db_name.empty() || meta.db_name_.case_compare_equal(db_name))) {
      meta.pending_ = pending;
      found = true;
    }
  }
  if (!found) {
    ret = OB_ENTRY_NOT_EXIST;
  }
  return ret;
}

int ObAiColumnRegistry::get_pending(const ObString &db_name,
                                    const ObString &table_name,
                                    const ObString &column_name,
                                    bool &pending)
{
  int ret = OB_SUCCESS;
  lib::ObMutexGuard guard(mutex_);
  bool found = false;
  pending = false;
  for (int64_t i = 0; !found && i < metas_.count(); ++i) {
    const ObAiColumnMeta &meta = metas_.at(i);
    if (meta.table_name_.case_compare_equal(table_name) &&
        meta.column_name_.case_compare_equal(column_name) &&
        (db_name.empty() || meta.db_name_.case_compare_equal(db_name))) {
      pending = meta.pending_;
      found = true;
    }
  }
  if (!found) {
    ret = OB_ENTRY_NOT_EXIST;
  }
  return ret;
}

int ObAiColumnRegistry::get_metas(ObIArray<ObAiColumnMeta> &metas)
{
  int ret = OB_SUCCESS;
  lib::ObMutexGuard guard(mutex_);
  if (OB_FAIL(metas.assign(metas_))) {
    LOG_WARN("failed to assign ai column metas", K(ret));
  }
  return ret;
}

int ObAiColumnRegistry::remove_table(const ObString &db_name, const ObString &table_name)
{
  int ret = OB_SUCCESS;
  lib::ObMutexGuard guard(mutex_);
  for (int64_t i = metas_.count() - 1; OB_SUCC(ret) && i >= 0; --i) {
    if (metas_.at(i).table_name_.case_compare_equal(table_name) &&
        metas_.at(i).db_name_.case_compare_equal(db_name)) {
      if (OB_FAIL(metas_.remove(i))) {
        LOG_WARN("failed to remove ai column meta", K(ret));
      }
    }
  }
  return ret;
}

int64_t ObAiColumnRegistry::count()
{
  lib::ObMutexGuard guard(mutex_);
  return metas_.count();
}

void ObAiColumnRegistry::reset()
{
  lib::ObMutexGuard guard(mutex_);
  metas_.reset();
}

const ParseNode *ai_column_find_attr(const ParseNode *column_def)
{
  const ParseNode *attr = NULL;
  if (NULL != column_def && T_COLUMN_DEFINITION == column_def->type_ &&
      column_def->num_child_ >= 3 && NULL != column_def->children_[2]) {
    const ParseNode *attrs = column_def->children_[2];
    for (int64_t i = 0; NULL == attr && i < attrs->num_child_; ++i) {
      if (NULL != attrs->children_[i] && T_CONSTR_AI_COLUMN == attrs->children_[i]->type_) {
        attr = attrs->children_[i];
      }
    }
  }
  return attr;
}

int ai_column_parse_options(const ParseNode *options,
                            int64_t &max_retries,
                            ObString &sync_mode)
{
  int ret = OB_SUCCESS;
  max_retries = 0;
  sync_mode.reset();
  if (NULL == options) {
    // no WITH clause, the default options are used
  } else if (T_AI_COLUMN_OPTIONS != options->type_) {
    ret = OB_ERR_UNEXPECTED;
    LOG_WARN("unexpected ai column options node", K(ret), "node type", options->type_);
  } else {
    for (int64_t i = 0; OB_SUCC(ret) && i < options->num_child_; ++i) {
      const ParseNode *option = options->children_[i];
      if (NULL == option || 2 != option->num_child_ ||
          NULL == option->children_[0] || NULL == option->children_[1]) {
        ret = OB_ERR_UNEXPECTED;
        LOG_WARN("unexpected ai column option node", K(ret));
      } else if (node_name_equal(option->children_[0], "max_retries")) {
        if (T_INT != option->children_[1]->type_) {
          ret = OB_INVALID_ARGUMENT;
          LOG_WARN("max_retries of an ai column must be an integer", K(ret));
        } else {
          max_retries = option->children_[1]->value_;
        }
      } else if (node_name_equal(option->children_[0], "sync_mode")) {
        if (NULL == option->children_[1]->str_value_) {
          ret = OB_INVALID_ARGUMENT;
          LOG_WARN("sync_mode of an ai column must be a string", K(ret));
        } else {
          sync_mode.assign_ptr(option->children_[1]->str_value_,
                               static_cast<int32_t>(option->children_[1]->str_len_));
        }
      } else {
        // unknown options are ignored
      }
    }
  }
  if (OB_SUCC(ret) && !sync_mode.empty() &&
      !sync_mode.case_compare_equal(ObString::make_string("async"))) {
    ret = OB_NOT_SUPPORTED;
  }
  return ret;
}

int ai_column_check_expr(const ParseNode *expr,
                         const ParseNode *data_type,
                         const ObIArray<ObString> &ai_column_names,
                         const bool is_json,
                         ObIAllocator &allocator,
                         ObString &normalized_expr)
{
  int ret = OB_SUCCESS;
  const ParseNode *func = NULL;
  bool depends_on_ai_column = false;
  normalized_expr.reset();
  if (OB_ISNULL(func = find_ai_func(expr))) {
    ret = OB_NOT_SUPPORTED;
  } else {
    collect_ai_dependency(func, ai_column_names, depends_on_ai_column);
    if (depends_on_ai_column) {
      ret = OB_NOT_SUPPORTED;
      }
  }
  // AI_EMBED(model, text, dim) declares the dimension explicitly, it must match
  // the declared dimension of the vector column
  if (OB_SUCC(ret) && node_name_equal(func->children_[0], "AI_EMBED") &&
      func_param_count(func) > 2) {
    const ParseNode *dim_node = func_param(func, 2);
    const int64_t expected_dim = get_vector_dimension(data_type);
    int64_t declared_dim = 0;
    if (NULL != dim_node && T_INT == dim_node->type_) {
      declared_dim = dim_node->value_;
    }
    if (expected_dim > 0 && declared_dim != expected_dim) {
      ret = OB_ERR_INVALID_VECTOR_DIM;
      LOG_USER_ERROR(OB_ERR_INVALID_VECTOR_DIM,
                     static_cast<uint32_t>(expected_dim),
                     static_cast<uint32_t>(declared_dim));
      LOG_WARN("inconsistent dimension of the ai column", K(ret),
               K(expected_dim), K(declared_dim));
    }
  }
  if (OB_SUCC(ret)) {
    char buf[512];
    char *copy = NULL;
    const int64_t len = snprintf(buf, sizeof(buf), "AI COLUMN(%s)%s",
                                 is_json ? "json" : "scalar",
                                 is_json ? " | Return exactly one valid JSON value" : "");
    if (OB_UNLIKELY(len <= 0 || len >= static_cast<int64_t>(sizeof(buf)))) {
      ret = OB_ERR_UNEXPECTED;
    } else if (OB_ISNULL(copy = static_cast<char *>(allocator.alloc(len)))) {
      ret = OB_ALLOCATE_MEMORY_FAILED;
    } else {
      MEMCPY(copy, buf, len);
      normalized_expr.assign_ptr(copy, static_cast<int32_t>(len));
    }
  }
  return ret;
}

int ai_column_gen_hidden_name(ObIAllocator &allocator,
                              const ObString &column_name,
                              const char *prefix,
                              ObString &hidden_name)
{
  int ret = OB_SUCCESS;
  hidden_name.reset();
  if (NULL == prefix) {
    ret = OB_INVALID_ARGUMENT;
  } else {
    const int64_t prefix_len = static_cast<int64_t>(strlen(prefix));
    const int64_t total_len = prefix_len + column_name.length();
    char *buf = NULL;
    if (OB_ISNULL(buf = static_cast<char *>(allocator.alloc(total_len)))) {
      ret = OB_ALLOCATE_MEMORY_FAILED;
    } else {
      MEMCPY(buf, prefix, prefix_len);
      if (!column_name.empty()) {
        MEMCPY(buf + prefix_len, column_name.ptr(), column_name.length());
      }
      hidden_name.assign_ptr(buf, static_cast<int32_t>(total_len));
    }
  }
  return ret;
}

int ai_column_describe_expr(const ParseNode *expr,
                            const int32_t result_type,
                            ObIAllocator &allocator,
                            ObAiColumnMeta &meta)
{
  int ret = OB_SUCCESS;
  const ParseNode *func = NULL;
  meta.result_type_ = result_type;
  meta.prompt_parts_.reset();
  meta.dep_columns_.reset();
  meta.model_name_.reset();
  meta.dim_ = 0;
  meta.is_embed_ = false;
  if (NULL == expr) {
    ret = OB_INVALID_ARGUMENT;
    LOG_WARN("the ai column expression is null", K(ret));
  } else if (OB_ISNULL(func = find_ai_func(expr))) {
    ret = OB_NOT_SUPPORTED;
    LOG_WARN("the ai column expression has no ai function", K(ret));
  } else if (node_name_equal(func->children_[0], "AI_EMBED")) {
    meta.is_embed_ = true;
  }
  // parameter 1 is the name of the AI model
  if (OB_SUCC(ret)) {
    const ParseNode *model_node = func_param(func, 0);
    if (NULL != model_node && NULL != model_node->str_value_) {
      const ObString model_name(static_cast<int32_t>(model_node->str_len_),
                                model_node->str_value_);
      if (OB_FAIL(ai_copy_text(model_name, allocator, meta.model_name_))) {
        LOG_WARN("failed to copy the model name of the ai column", K(ret));
      }
    }
  }
  // parameter 3 of AI_EMBED is the dimension of the vector
  if (OB_SUCC(ret) && meta.is_embed_ && func_param_count(func) > 2) {
    const ParseNode *dim_node = func_param(func, 2);
    if (NULL != dim_node && T_INT == dim_node->type_) {
      meta.dim_ = dim_node->value_;
    }
  }
  // parameter 2 of AI_COMPLETE is the prompt expression
  if (OB_SUCC(ret) && !meta.is_embed_) {
    const ParseNode *prompt_node = func_param(func, 1);
    if (NULL != prompt_node) {
      if (OB_FAIL(collect_prompt_parts(prompt_node, allocator, meta))) {
        LOG_WARN("failed to collect the prompt parts of the ai column", K(ret));
      }
    }
  }
  return ret;
}

bool ai_column_is_meta_table(const ParseNode &table_node)
{
  bool bret = false;
  if (T_RELATION_FACTOR == table_node.type_ && table_node.num_child_ >= 2 &&
      NULL != table_node.children_[1] && NULL != table_node.children_[1]->str_value_) {
    const ParseNode *table_name_node = table_node.children_[1];
    const ObString table_name(static_cast<int32_t>(table_name_node->str_len_),
                              table_name_node->str_value_);
    bret = table_name.case_compare_equal(ObString::make_string("__all_ai_column"));
  }
  return bret;
}

namespace
{

// ---------------------------------------------------------------------------
// background worker of the AI columns
//
// In every iteration the worker takes a snapshot of the registered AI columns,
// groups them by table, reads the pending rows of the user table through the
// inner sql proxy, computes the results by calling the AI endpoint over http
// and publishes them with one update statement per row, so that all AI columns
// of the same row become visible atomically.
// ---------------------------------------------------------------------------

const int64_t AI_WORKER_INTERVAL_US = 200 * 1000;
const int64_t AI_WORKER_BATCH_SIZE = 64;
const int64_t AI_WORKER_HTTP_TIMEOUT_SEC = 20;
const int64_t AI_STATE_DONE = 0;
const int64_t AI_STATE_PENDING = 100;
const int64_t AI_STATE_RETRY_BASE = 300;
const int64_t AI_STATE_GIVEUP_BASE = 400;

enum AiResultKind
{
  AI_RESULT_TEXT = 0,
  AI_RESULT_INT,
  AI_RESULT_DOUBLE,
  AI_RESULT_JSON,
  AI_RESULT_VECTOR
};

// the url and the access key of one AI model endpoint
struct AiEndpoint
{
  ObString model_name_;
  ObString url_;
  ObString access_key_;
  AiEndpoint() : model_name_(), url_(), access_key_() {}
};

// the pending state of one AI column of one row: the state value that was read
// and the values of the columns the AI expression depends on
struct AiColumnTask
{
  bool found_;
  int64_t state_;
  ObSEArray<ObString, 8> dep_values_;
  ObSEArray<int8_t, 8> dep_is_null_;
  AiColumnTask() : found_(false), state_(0), dep_values_(), dep_is_null_() {}
};

// one row of the user table together with the pending state of every AI column
// of that table
struct AiRowTask
{
  ObSEArray<ObString, 4> pk_values_;
  ObSEArray<AiColumnTask, 4> columns_;
  AiRowTask() : pk_values_(), columns_() {}
};

void ai_worker_sleep()
{
  usleep(static_cast<unsigned int>(AI_WORKER_INTERVAL_US));
}

// `name`
int ai_append_ident(ObSqlString &sql, const ObString &name)
{
  int ret = OB_SUCCESS;
  if (OB_FAIL(sql.append("`"))) {
  } else if (OB_FAIL(sql.append(name))) {
  } else {
    ret = sql.append("`");
  }
  return ret;
}

// 'value', a quote inside the value is doubled and a backslash is escaped
int ai_append_literal(ObSqlString &sql, const ObString &value)
{
  int ret = OB_SUCCESS;
  const char *ptr = value.ptr();
  if (OB_FAIL(sql.append("'"))) {
  } else {
    for (int64_t i = 0; OB_SUCC(ret) && NULL != ptr && i < value.length(); ++i) {
      if ('\'' == ptr[i]) {
        ret = sql.append("''");
      } else if ('\\' == ptr[i]) {
        ret = sql.append("\\\\");
      } else {
        ret = sql.append(ptr + i, 1);
      }
    }
    if (OB_SUCC(ret)) {
      ret = sql.append("'");
    }
  }
  return ret;
}

int64_t ai_to_int64(const ObString &value)
{
  int64_t number = 0;
  char buf[32];
  if (!value.empty() && NULL != value.ptr() &&
      value.length() < static_cast<int64_t>(sizeof(buf))) {
    MEMCPY(buf, value.ptr(), value.length());
    buf[value.length()] = '\0';
    number = strtoll(buf, NULL, 10);
  }
  return number;
}

ObString ai_trim(const ObString &value)
{
  ObString trimmed;
  const char *ptr = value.ptr();
  int64_t begin = 0;
  int64_t end = (NULL == ptr) ? 0 : value.length();
  while (begin < end && (' ' == ptr[begin] || '\t' == ptr[begin] ||
                         '\n' == ptr[begin] || '\r' == ptr[begin])) {
    ++begin;
  }
  while (begin < end && (' ' == ptr[end - 1] || '\t' == ptr[end - 1] ||
                         '\n' == ptr[end - 1] || '\r' == ptr[end - 1])) {
    --end;
  }
  if (end > begin) {
    trimmed.assign_ptr(ptr + begin, static_cast<int32_t>(end - begin));
  }
  return trimmed;
}

bool ai_is_numeric_literal(const ObString &value, const bool allow_double)
{
  bool bret = !value.empty() && NULL != value.ptr();
  int64_t digit_cnt = 0;
  int64_t dot_cnt = 0;
  int64_t i = 0;
  if (bret && ('+' == value.ptr()[0] || '-' == value.ptr()[0])) {
    i = 1;
  }
  for (; bret && i < value.length(); ++i) {
    const char c = value.ptr()[i];
    if (c >= '0' && c <= '9') {
      ++digit_cnt;
    } else if (allow_double && '.' == c && 0 == dot_cnt) {
      ++dot_cnt;
    } else {
      bret = false;
    }
  }
  return bret && digit_cnt > 0;
}

int ai_append_json_escaped(ObSqlString &out, const ObString &value)
{
  int ret = OB_SUCCESS;
  const char *ptr = value.ptr();
  for (int64_t i = 0; OB_SUCC(ret) && NULL != ptr && i < value.length(); ++i) {
    const char c = ptr[i];
    if ('"' == c) {
      ret = out.append("\\\"");
    } else if ('\\' == c) {
      ret = out.append("\\\\");
    } else if ('\n' == c) {
      ret = out.append("\\n");
    } else if ('\r' == c) {
      ret = out.append("\\r");
    } else if ('\t' == c) {
      ret = out.append("\\t");
    } else if (static_cast<unsigned char>(c) < 0x20) {
      ret = out.append_fmt("\\u%04x",
                           static_cast<unsigned int>(static_cast<unsigned char>(c)));
    } else {
      ret = out.append(ptr + i, 1);
    }
  }
  return ret;
}

void ai_skip_json_space(const char *ptr, const int64_t len, int64_t &pos)
{
  while (pos < len && NULL != ptr && (' ' == ptr[pos] || '\t' == ptr[pos] ||
                                      '\n' == ptr[pos] || '\r' == ptr[pos])) {
    ++pos;
  }
}

// find the string value of "key" in a json text, the value is unescaped and
// copied into the allocator
int ai_find_json_string(const ObString &body, const char *key, ObIAllocator &allocator,
                        ObString &value, bool &found)
{
  int ret = OB_SUCCESS;
  const char *ptr = body.ptr();
  const int64_t len = body.length();
  const int64_t key_len = (NULL == key) ? 0 : static_cast<int64_t>(strlen(key));
  value.reset();
  found = false;
  if (NULL == key || key_len <= 0 || NULL == ptr || len <= 0) {
    ret = OB_INVALID_ARGUMENT;
  } else {
    for (int64_t i = 0; OB_SUCC(ret) && !found && i + key_len + 2 <= len; ++i) {
      if ('"' != ptr[i] || '"' != ptr[i + key_len + 1] ||
          0 != strncmp(ptr + i + 1, key, static_cast<size_t>(key_len))) {
        continue;
      }
      int64_t pos = i + key_len + 2;
      ai_skip_json_space(ptr, len, pos);
      if (pos >= len || ':' != ptr[pos]) {
        continue;
      }
      ++pos;
      ai_skip_json_space(ptr, len, pos);
      if (pos >= len || '"' != ptr[pos]) {
        continue;
      }
      ++pos;
      char *buf = static_cast<char *>(allocator.alloc(len - pos + 1));
      if (OB_ISNULL(buf)) {
        ret = OB_ALLOCATE_MEMORY_FAILED;
        LOG_WARN("failed to allocate the json value", K(ret));
      } else {
        int64_t out = 0;
        bool closed = false;
        while (OB_SUCC(ret) && !closed && pos < len) {
          const char c = ptr[pos];
          if ('"' == c) {
            closed = true;
            ++pos;
          } else if ('\\' == c && pos + 1 < len) {
            const char escaped = ptr[pos + 1];
            if ('u' == escaped) {
              // \uXXXX is copied verbatim, the worker does not decode it
              if (pos + 6 <= len) {
                MEMCPY(buf + out, ptr + pos, 6);
                out += 6;
                pos += 6;
              } else {
                ret = OB_ERR_UNEXPECTED;
              }
            } else {
              if ('n' == escaped) {
                buf[out++] = '\n';
              } else if ('t' == escaped) {
                buf[out++] = '\t';
              } else if ('r' == escaped) {
                buf[out++] = '\r';
              } else if ('b' == escaped) {
                buf[out++] = '\b';
              } else if ('f' == escaped) {
                buf[out++] = '\f';
              } else {
                buf[out++] = escaped;
              }
              pos += 2;
            }
          } else {
            buf[out++] = c;
            ++pos;
          }
        }
        if (OB_SUCC(ret)) {
          if (!closed) {
            ret = OB_ERR_UNEXPECTED;
            LOG_WARN("the json string of the ai answer is not closed", K(ret));
          } else {
            buf[out] = '\0';
            value.assign_ptr(buf, static_cast<int32_t>(out));
            found = true;
          }
        }
      }
    }
  }
  return ret;
}

// find the numeric array value of "key" in a json text, the array is rebuilt
// without any space
int ai_find_json_array(const ObString &body, const char *key, ObIAllocator &allocator,
                       ObString &value, bool &found)
{
  int ret = OB_SUCCESS;
  const char *ptr = body.ptr();
  const int64_t len = body.length();
  const int64_t key_len = (NULL == key) ? 0 : static_cast<int64_t>(strlen(key));
  value.reset();
  found = false;
  if (NULL == key || key_len <= 0 || NULL == ptr || len <= 0) {
    ret = OB_INVALID_ARGUMENT;
  } else {
    for (int64_t i = 0; OB_SUCC(ret) && !found && i + key_len + 2 <= len; ++i) {
      if ('"' != ptr[i] || '"' != ptr[i + key_len + 1] ||
          0 != strncmp(ptr + i + 1, key, static_cast<size_t>(key_len))) {
        continue;
      }
      int64_t pos = i + key_len + 2;
      ai_skip_json_space(ptr, len, pos);
      if (pos >= len || ':' != ptr[pos]) {
        continue;
      }
      ++pos;
      ai_skip_json_space(ptr, len, pos);
      if (pos >= len || '[' != ptr[pos]) {
        continue;
      }
      const int64_t begin = pos;
      int64_t depth = 0;
      bool closed = false;
      while (!closed && pos < len) {
        if ('[' == ptr[pos]) {
          ++depth;
        } else if (']' == ptr[pos]) {
          --depth;
          if (0 == depth) {
            closed = true;
          }
        }
        ++pos;
      }
      if (!closed) {
        ret = OB_ERR_UNEXPECTED;
        LOG_WARN("the json array of the ai answer is not closed", K(ret));
      } else {
        char *buf = static_cast<char *>(allocator.alloc(pos - begin + 1));
        if (OB_ISNULL(buf)) {
          ret = OB_ALLOCATE_MEMORY_FAILED;
        } else {
          int64_t out = 0;
          for (int64_t k = begin; k < pos; ++k) {
            const char c = ptr[k];
            if (' ' != c && '\t' != c && '\n' != c && '\r' != c) {
              buf[out++] = c;
            }
          }
          buf[out] = '\0';
          value.assign_ptr(buf, static_cast<int32_t>(out));
          found = true;
        }
      }
    }
  }
  return ret;
}

size_t ai_http_write_cb(void *contents, size_t size, size_t nmemb, void *userp)
{
  size_t consumed = 0;
  const size_t total = size * nmemb;
  ObSqlString *buffer = static_cast<ObSqlString *>(userp);
  if (NULL != buffer && NULL != contents && total > 0) {
    if (OB_SUCCESS == buffer->append(static_cast<const char *>(contents),
                                     static_cast<int64_t>(total))) {
      consumed = total;
    }
  }
  return consumed;
}

int ai_http_post(const ObString &url,
                 const ObString &access_key,
                 const ObString &body,
                 ObSqlString &response)
{
  int ret = OB_SUCCESS;
  CURL *curl = NULL;
  struct curl_slist *headers = NULL;
  ObSqlString bearer;
  long http_code = 0;
  response.reuse();
  if (url.empty() || NULL == url.ptr() || body.empty()) {
    ret = OB_INVALID_ARGUMENT;
    LOG_WARN("invalid argument of the ai http request", K(ret));
  } else if (OB_ISNULL(curl = curl_easy_init())) {
    ret = OB_CURL_ERROR;
    LOG_WARN("failed to initialize the curl handle of the ai worker", K(ret));
  } else if (OB_ISNULL(headers = curl_slist_append(headers, "Content-Type: application/json"))) {
    ret = OB_CURL_ERROR;
    LOG_WARN("failed to append the content type header", K(ret));
  } else if (!access_key.empty() &&
             OB_FAIL(bearer.append_fmt("Authorization: Bearer %.*s",
                                       static_cast<int>(access_key.length()),
                                       access_key.ptr()))) {
    LOG_WARN("failed to build the authorization header", K(ret));
  } else if (!bearer.empty() &&
             OB_ISNULL(headers = curl_slist_append(headers, bearer.ptr()))) {
    ret = OB_CURL_ERROR;
    LOG_WARN("failed to append the authorization header", K(ret));
  } else {
    char *url_ptr = const_cast<char *>(url.ptr());
    char *body_ptr = const_cast<char *>(body.ptr());
    if (CURLE_OK != curl_easy_setopt(curl, CURLOPT_URL, url_ptr) ||
        CURLE_OK != curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, ai_http_write_cb) ||
        CURLE_OK != curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response) ||
        CURLE_OK != curl_easy_setopt(curl, CURLOPT_NOSIGNAL, 1L) ||
        CURLE_OK != curl_easy_setopt(curl, CURLOPT_TIMEOUT,
                                     static_cast<long>(AI_WORKER_HTTP_TIMEOUT_SEC)) ||
        CURLE_OK != curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers) ||
        CURLE_OK != curl_easy_setopt(curl, CURLOPT_POST, 1L) ||
        CURLE_OK != curl_easy_setopt(curl, CURLOPT_POSTFIELDS, body_ptr) ||
        CURLE_OK != curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE,
                                     static_cast<long>(body.length()))) {
      ret = OB_CURL_ERROR;
      LOG_WARN("failed to set the option of the ai http request", K(ret));
    } else if (CURLE_OK != curl_easy_perform(curl)) {
      ret = OB_CURL_ERROR;
      LOG_WARN("failed to perform the ai http request", K(ret), K(url));
    } else if (CURLE_OK != curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code)) {
      ret = OB_CURL_ERROR;
      LOG_WARN("failed to get the response code of the ai http request", K(ret));
    } else if (http_code < 200 || http_code >= 300) {
      ret = OB_CURL_ERROR;
      LOG_WARN("the ai endpoint returned an error", K(ret), K(http_code), K(url));
    }
  }
  if (NULL != headers) {
    curl_slist_free_all(headers);
    headers = NULL;
  }
  if (NULL != curl) {
    curl_easy_cleanup(curl);
    curl = NULL;
  }
  return ret;
}

AiResultKind ai_result_kind(const ObAiColumnMeta &meta)
{
  AiResultKind kind = AI_RESULT_TEXT;
  const int32_t type = meta.result_type_;
  if (meta.is_embed_) {
    kind = AI_RESULT_VECTOR;
  } else if (meta.is_json_) {
    kind = AI_RESULT_JSON;
  } else if ((type >= static_cast<int32_t>(ObTinyIntType) &&
              type <= static_cast<int32_t>(ObIntType)) ||
             (type >= static_cast<int32_t>(ObUTinyIntType) &&
              type <= static_cast<int32_t>(ObUInt64Type))) {
    kind = AI_RESULT_INT;
  } else if (static_cast<int32_t>(ObFloatType) == type ||
             static_cast<int32_t>(ObDoubleType) == type ||
             static_cast<int32_t>(ObNumberType) == type ||
             static_cast<int32_t>(ObUFloatType) == type ||
             static_cast<int32_t>(ObUDoubleType) == type ||
             static_cast<int32_t>(ObUNumberType) == type) {
    kind = AI_RESULT_DOUBLE;
  }
  return kind;
}

const char *ai_system_prompt(const AiResultKind kind)
{
  const char *prompt = "";
  if (AI_RESULT_INT == kind) {
    prompt = "Return exactly one base-10 integer value";
  } else if (AI_RESULT_JSON == kind) {
    prompt = "Return exactly one valid JSON value";
  }
  return prompt;
}

int ai_build_user_prompt(const ObAiColumnMeta &meta,
                         const ObIArray<ObString> &dep_values,
                         ObSqlString &prompt)
{
  int ret = OB_SUCCESS;
  prompt.reuse();
  for (int64_t i = 0; OB_SUCC(ret) && i < meta.prompt_parts_.count(); ++i) {
    const ObAiPromptPart &part = meta.prompt_parts_.at(i);
    if (!part.is_column_) {
      ret = prompt.append(part.text_);
    } else {
      int64_t idx = -1;
      for (int64_t k = 0; -1 == idx && k < meta.dep_columns_.count(); ++k) {
        if (meta.dep_columns_.at(k).case_compare_equal(part.text_)) {
          idx = k;
        }
      }
      if (-1 != idx && idx < dep_values.count()) {
        ret = prompt.append(dep_values.at(idx));
      }
    }
  }
  return ret;
}

int ai_build_request(const ObAiColumnMeta &meta,
                     const AiResultKind kind,
                     const ObString &user_prompt,
                     ObSqlString &request)
{
  int ret = OB_SUCCESS;
  request.reuse();
  if (OB_FAIL(request.append("{\"model\":\""))) {
  } else if (OB_FAIL(ai_append_json_escaped(request, meta.model_name_))) {
  } else if (AI_RESULT_VECTOR == kind) {
    if (OB_FAIL(request.append("\",\"input\":\""))) {
    } else if (OB_FAIL(ai_append_json_escaped(request, user_prompt))) {
    } else {
      ret = request.append("\"}");
    }
  } else if (OB_FAIL(request.append("\",\"messages\":[{\"role\":\"system\",\"content\":\""))) {
  } else if (OB_FAIL(ai_append_json_escaped(request,
                                            ObString::make_string(ai_system_prompt(kind))))) {
  } else if (OB_FAIL(request.append("\"},{\"role\":\"user\",\"content\":\""))) {
  } else if (OB_FAIL(ai_append_json_escaped(request, user_prompt))) {
  } else {
    ret = request.append("\"}]}");
  }
  return ret;
}

// call the AI endpoint and build the sql literal of the result
int ai_compute_value(ObIAllocator &allocator,
                     const ObAiColumnMeta &meta,
                     const AiEndpoint &endpoint,
                     const AiColumnTask &cell,
                     ObSqlString &value_literal)
{
  int ret = OB_SUCCESS;
  const AiResultKind kind = ai_result_kind(meta);
  ObSqlString user_prompt;
  ObSqlString request;
  ObSqlString response;
  value_literal.reuse();
  if (OB_FAIL(ai_build_user_prompt(meta, cell.dep_values_, user_prompt))) {
    LOG_WARN("failed to build the user prompt of the ai column", K(ret));
  } else if (OB_FAIL(ai_build_request(meta, kind, user_prompt.string(), request))) {
    LOG_WARN("failed to build the request of the ai column", K(ret));
  } else if (OB_FAIL(ai_http_post(endpoint.url_, endpoint.access_key_,
                                  request.string(), response))) {
    LOG_WARN("failed to call the ai endpoint", K(ret), K(meta.column_name_));
  } else if (AI_RESULT_VECTOR == kind) {
    ObString array;
    bool found = false;
    if (OB_FAIL(ai_find_json_array(response.string(), "embedding",
                                   allocator, array, found))) {
      LOG_WARN("failed to parse the embedding of the ai answer", K(ret));
    } else if (!found || array.empty()) {
      ret = OB_ERR_UNEXPECTED;
      LOG_WARN("the ai answer has no embedding", K(ret));
    } else {
      ret = value_literal.append(array);
    }
  } else {
    ObString answer;
    bool found = false;
    if (OB_FAIL(ai_find_json_string(response.string(), "content",
                                    allocator, answer, found))) {
      LOG_WARN("failed to parse the content of the ai answer", K(ret));
    } else if (!found) {
      ret = OB_ERR_UNEXPECTED;
      LOG_WARN("the ai answer has no content", K(ret));
    } else if (AI_RESULT_INT == kind || AI_RESULT_DOUBLE == kind) {
      const ObString trimmed = ai_trim(answer);
      if (!ai_is_numeric_literal(trimmed, AI_RESULT_DOUBLE == kind)) {
        ret = OB_ERR_UNEXPECTED;
        LOG_WARN("the ai answer is not a number", K(ret), K(answer),
                 K(meta.column_name_));
      } else {
        ret = value_literal.append(trimmed);
      }
    } else {
      ret = ai_append_literal(value_literal, answer);
    }
  }
  return ret;
}

// the url and the access key of a model are kept in the system table
int ai_lookup_endpoint(common::ObISQLClient *proxy,
                       ObIAllocator &allocator,
                       const ObString &model_name,
                       AiEndpoint &endpoint)
{
  int ret = OB_SUCCESS;
  ObSqlString sql;
  endpoint.model_name_ = model_name;
  endpoint.url_.reset();
  endpoint.access_key_.reset();
  if (OB_ISNULL(proxy)) {
    ret = OB_INVALID_ARGUMENT;
  } else if (OB_FAIL(sql.append("select url, access_key from "
                                "oceanbase.__all_ai_model_endpoint where ai_model_name = "))) {
    LOG_WARN("failed to build the endpoint query of the ai model", K(ret));
  } else if (OB_FAIL(ai_append_literal(sql, model_name))) {
  } else if (OB_FAIL(sql.append(" limit 1"))) {
  } else {
    common::ObISQLClient::ReadResult res;
    if (OB_FAIL(proxy->read(res, sql.ptr()))) {
      LOG_WARN("failed to query the endpoint of the ai model", K(ret), K(model_name));
    } else {
      common::sqlclient::ObMySQLResult *result = res.get_result();
      if (NULL == result) {
        ret = OB_ENTRY_NOT_EXIST;
      } else {
        ret = result->next();
        if (OB_ITER_END == ret) {
          ret = OB_ENTRY_NOT_EXIST;
        } else if (OB_FAIL(ret)) {
          LOG_WARN("failed to fetch the endpoint of the ai model", K(ret));
        } else {
          ObString url;
          ObString access_key;
          if (OB_FAIL(result->get_varchar("url", url))) {
            LOG_WARN("failed to read the url of the ai model", K(ret));
          } else if (OB_FAIL(ai_copy_text(url, allocator, endpoint.url_))) {
          } else {
            const int access_ret = result->get_varchar("access_key", access_key);
            if (OB_SUCCESS == access_ret) {
              ret = ai_copy_text(access_key, allocator, endpoint.access_key_);
            } else if (OB_ERR_NULL_VALUE == access_ret) {
              endpoint.access_key_.reset();
            } else {
              ret = access_ret;
              LOG_WARN("failed to read the access key of the ai model", K(ret));
            }
          }
        }
      }
    }
  }
  return ret;
}

int ai_find_endpoint(common::ObISQLClient *proxy,
                     ObIAllocator &allocator,
                     const ObString &model_name,
                     ObIArray<AiEndpoint> &endpoints,
                     const AiEndpoint *&endpoint)
{
  int ret = OB_SUCCESS;
  endpoint = NULL;
  for (int64_t i = 0; NULL == endpoint && i < endpoints.count(); ++i) {
    if (endpoints.at(i).model_name_.case_compare_equal(model_name)) {
      endpoint = &endpoints.at(i);
    }
  }
  if (NULL == endpoint) {
    AiEndpoint lookup;
    if (OB_FAIL(ai_lookup_endpoint(proxy, allocator, model_name, lookup))) {
      LOG_WARN("failed to look up the endpoint of the ai model", K(ret), K(model_name));
    } else if (OB_FAIL(endpoints.push_back(lookup))) {
      LOG_WARN("failed to cache the endpoint of the ai model", K(ret));
    } else {
      endpoint = &endpoints.at(endpoints.count() - 1);
    }
  }
  return ret;
}

int ai_build_pending_query(const ObAiColumnMeta &meta, ObSqlString &sql)
{
  int ret = OB_SUCCESS;
  sql.reuse();
  if (OB_FAIL(sql.append("select "))) {
  }
  for (int64_t i = 0; OB_SUCC(ret) && i < meta.pk_columns_.count(); ++i) {
    if (i > 0) {
      ret = sql.append(", ");
    }
    if (OB_SUCC(ret)) {
      ret = sql.append("cast(");
    }
    if (OB_SUCC(ret)) {
      ret = ai_append_ident(sql, meta.pk_columns_.at(i));
    }
    if (OB_SUCC(ret)) {
      ret = sql.append_fmt(" as char) as pk%d", static_cast<int>(i));
    }
  }
  if (OB_SUCC(ret)) {
    ret = sql.append(", cast(");
  }
  if (OB_SUCC(ret)) {
    ret = ai_append_ident(sql, meta.state_column_name_);
  }
  if (OB_SUCC(ret)) {
    ret = sql.append(" as char) as st, cast(");
  }
  if (OB_SUCC(ret)) {
    ret = ai_append_ident(sql, meta.request_column_name_);
  }
  if (OB_SUCC(ret)) {
    ret = sql.append(" as char) as rq");
  }
  for (int64_t i = 0; OB_SUCC(ret) && i < meta.dep_columns_.count(); ++i) {
    if (OB_FAIL(sql.append(", cast("))) {
    } else if (OB_FAIL(ai_append_ident(sql, meta.dep_columns_.at(i)))) {
    } else {
      ret = sql.append_fmt(" as char) as d%d", static_cast<int>(i));
    }
  }
  if (OB_SUCC(ret)) {
    ret = sql.append(" from ");
  }
  if (OB_SUCC(ret)) {
    ret = ai_append_ident(sql, meta.db_name_);
  }
  if (OB_SUCC(ret)) {
    ret = sql.append(".");
  }
  if (OB_SUCC(ret)) {
    ret = ai_append_ident(sql, meta.table_name_);
  }
  if (OB_SUCC(ret)) {
    ret = sql.append(" where ");
  }
  if (OB_SUCC(ret)) {
    ret = ai_append_ident(sql, meta.state_column_name_);
  }
  if (OB_SUCC(ret)) {
    ret = sql.append_fmt(" >= %ld and ", static_cast<long>(AI_STATE_PENDING));
  }
  if (OB_SUCC(ret)) {
    ret = ai_append_ident(sql, meta.state_column_name_);
  }
  if (OB_SUCC(ret)) {
    ret = sql.append(" limit ");
  }
  if (OB_SUCC(ret)) {
    ret = sql.append_fmt("%ld", static_cast<long>(AI_WORKER_BATCH_SIZE));
  }
  return ret;
}

int ai_read_text(ObIAllocator &allocator,
                 common::sqlclient::ObMySQLResult *result,
                 const char *alias,
                 ObString &value,
                 bool &is_null)
{
  int ret = OB_SUCCESS;
  ObString raw;
  value.reset();
  is_null = false;
  const int read_ret = result->get_varchar(alias, raw);
  if (OB_SUCCESS == read_ret) {
    ret = ai_copy_text(raw, allocator, value);
  } else if (OB_ERR_NULL_VALUE == read_ret) {
    is_null = true;
  } else {
    ret = read_ret;
    LOG_WARN("failed to read the value of the ai column", K(ret), K(alias));
  }
  return ret;
}

// compare two primary key values exactly, two rows whose keys differ only in the
// case are different rows
bool ai_pk_equal(const ObIArray<ObString> &lhs, const ObIArray<ObString> &rhs)
{
  bool bret = (lhs.count() == rhs.count()) && lhs.count() > 0;
  for (int64_t i = 0; bret && i < lhs.count(); ++i) {
    bret = (0 == lhs.at(i).compare(rhs.at(i)));
  }
  return bret;
}

// read the pending rows of one AI column and remember their state and the
// values of the columns the AI expression depends on, the rows are merged by
// the primary key so that all AI columns of one row are published together
int ai_collect_pending_rows(common::ObISQLClient *proxy,
                            ObIAllocator &allocator,
                            const ObAiColumnMeta &meta,
                            const int64_t column_idx,
                            const int64_t column_cnt,
                            ObIArray<AiRowTask> &tasks)
{
  int ret = OB_SUCCESS;
  ObSqlString sql;
  char alias[16];
  if (meta.pk_columns_.count() == 0) {
    LOG_DEBUG("the table of the ai column has no primary key", K(meta.table_name_));
  } else if (OB_FAIL(ai_build_pending_query(meta, sql))) {
    LOG_WARN("failed to build the pending query of the ai column", K(ret));
  } else {
    common::ObISQLClient::ReadResult res;
    const int read_ret = proxy->read(res, sql.ptr());
    if (OB_FAIL(read_ret)) {
      ret = read_ret;
      LOG_WARN("failed to query the pending rows of the ai column", K(ret),
               K(meta.table_name_));
    } else {
      common::sqlclient::ObMySQLResult *result = res.get_result();
      if (NULL == result) {
        // no pending row
      } else {
        bool done = false;
        while (OB_SUCC(ret) && !done) {
          const int next_ret = result->next();
          if (OB_ITER_END == next_ret) {
            done = true;
          } else if (OB_SUCCESS != next_ret) {
            ret = next_ret;
            LOG_WARN("failed to fetch the pending row of the ai column", K(ret));
          } else {
            ObSEArray<ObString, 4> pk_values;
            ObSEArray<ObString, 8> dep_values;
            ObSEArray<int8_t, 8> dep_is_null;
            ObString state_text;
            int64_t state = 0;
            bool is_null = false;
            for (int64_t i = 0; OB_SUCC(ret) && i < meta.pk_columns_.count(); ++i) {
              ObString value;
              snprintf(alias, sizeof(alias), "pk%d", static_cast<int>(i));
              if (OB_FAIL(ai_read_text(allocator, result, alias, value, is_null))) {
              } else if (is_null) {
                ret = OB_ERR_UNEXPECTED;
                LOG_WARN("the primary key value of the ai column is null", K(ret));
              } else if (OB_FAIL(pk_values.push_back(value))) {
                LOG_WARN("failed to push back the primary key value", K(ret));
              }
            }
            if (OB_FAIL(ret)) {
            } else if (OB_FAIL(ai_read_text(allocator, result, "st", state_text, is_null))) {
            } else if (is_null) {
              ret = OB_ERR_UNEXPECTED;
              LOG_WARN("the state of the ai column is null", K(ret));
            } else {
              state = ai_to_int64(state_text);
            }
            for (int64_t i = 0; OB_SUCC(ret) && i < meta.dep_columns_.count(); ++i) {
              ObString value;
              snprintf(alias, sizeof(alias), "d%d", static_cast<int>(i));
              if (OB_FAIL(ai_read_text(allocator, result, alias, value, is_null))) {
              } else if (OB_FAIL(dep_values.push_back(value))) {
                LOG_WARN("failed to push back the dependency value", K(ret));
              } else if (OB_FAIL(dep_is_null.push_back(is_null ? 1 : 0))) {
                LOG_WARN("failed to push back the dependency flag", K(ret));
              }
            }
            if (OB_SUCC(ret) && state <= 0) {
              // the state was cleared by a concurrent update, drop the row
              continue;
            }
            if (OB_SUCC(ret)) {
              int64_t task_idx = -1;
              for (int64_t i = 0; -1 == task_idx && i < tasks.count(); ++i) {
                if (ai_pk_equal(tasks.at(i).pk_values_, pk_values)) {
                  task_idx = i;
                }
              }
              if (-1 == task_idx) {
                AiRowTask task;
                if (OB_FAIL(task.pk_values_.assign(pk_values))) {
                  LOG_WARN("failed to copy the primary key of the ai column", K(ret));
                } else {
                  for (int64_t c = 0; OB_SUCC(ret) && c < column_cnt; ++c) {
                    AiColumnTask empty;
                    if (OB_FAIL(task.columns_.push_back(empty))) {
                      LOG_WARN("failed to init the ai column task", K(ret));
                    }
                  }
                }
                if (OB_SUCC(ret) && OB_FAIL(tasks.push_back(task))) {
                  LOG_WARN("failed to push back the ai column task", K(ret));
                } else if (OB_SUCC(ret)) {
                  task_idx = tasks.count() - 1;
                }
              }
              if (OB_SUCC(ret)) {
                AiColumnTask &cell = tasks.at(task_idx).columns_.at(column_idx);
                cell.found_ = true;
                cell.state_ = state;
                if (OB_FAIL(cell.dep_values_.assign(dep_values))) {
                  LOG_WARN("failed to copy the dependency values", K(ret));
                } else if (OB_FAIL(cell.dep_is_null_.assign(dep_is_null))) {
                  LOG_WARN("failed to copy the dependency flags", K(ret));
                }
              }
            }
          }
        }
      }
    }
  }
  return ret;
}

int ai_append_condition(ObSqlString &cond, int64_t &cond_cnt)
{
  int ret = OB_SUCCESS;
  if (cond_cnt > 0) {
    ret = cond.append(" and ");
  }
  if (OB_SUCC(ret)) {
    ++cond_cnt;
  }
  return ret;
}

int ai_append_ident_condition(ObSqlString &cond, int64_t &cond_cnt,
                              const ObString &column, const ObString &value,
                              const bool is_null)
{
  int ret = OB_SUCCESS;
  if (OB_FAIL(ai_append_condition(cond, cond_cnt))) {
  } else if (OB_FAIL(ai_append_ident(cond, column))) {
  } else if (is_null) {
    ret = cond.append(" is null");
  } else if (OB_FAIL(cond.append(" = "))) {
  } else {
    ret = ai_append_literal(cond, value);
  }
  return ret;
}

int ai_append_number_condition(ObSqlString &cond, int64_t &cond_cnt,
                               const ObString &column, const int64_t value)
{
  int ret = OB_SUCCESS;
  if (OB_FAIL(ai_append_condition(cond, cond_cnt))) {
  } else if (OB_FAIL(ai_append_ident(cond, column))) {
  } else {
    ret = cond.append_fmt(" = %ld", static_cast<long>(value));
  }
  return ret;
}

int ai_append_assignment(ObSqlString &set_clause, int64_t &item_cnt,
                         const ObString &column, const ObString &literal)
{
  int ret = OB_SUCCESS;
  if (item_cnt > 0) {
    ret = set_clause.append(", ");
  }
  if (OB_SUCC(ret)) {
    ret = ai_append_ident(set_clause, column);
  }
  if (OB_SUCC(ret)) {
    ret = set_clause.append(" = ");
  }
  if (OB_SUCC(ret)) {
    ret = set_clause.append(literal);
  }
  if (OB_SUCC(ret)) {
    ++item_cnt;
  }
  return ret;
}

int ai_append_number_assignment(ObSqlString &set_clause, int64_t &item_cnt,
                                const ObString &column, const int64_t value)
{
  int ret = OB_SUCCESS;
  if (item_cnt > 0) {
    ret = set_clause.append(", ");
  }
  if (OB_SUCC(ret)) {
    ret = ai_append_ident(set_clause, column);
  }
  if (OB_SUCC(ret)) {
    ret = set_clause.append_fmt(" = %ld", static_cast<long>(value));
  }
  if (OB_SUCC(ret)) {
    ++item_cnt;
  }
  return ret;
}

// publish the computed values of all AI columns of one row with one statement,
// every published column is guarded by the state and the dependency values that
// were read, so that a result computed from stale values is dropped
int ai_publish_row(common::ObISQLClient *proxy,
                   ObIAllocator &allocator,
                   const ObIArray<ObAiColumnMeta> &metas,
                   const ObIArray<int64_t> &columns,
                   ObIArray<AiEndpoint> &endpoints,
                   const AiRowTask &task,
                   ObIArray<int8_t> &cell_unfinished)
{
  int ret = OB_SUCCESS;
  const ObAiColumnMeta *first_meta = NULL;
  ObSqlString set_clause;
  ObSqlString cond_clause;
  int64_t set_cnt = 0;
  int64_t cond_cnt = 0;
  // the state that the publish statement writes to each cell, -1 means the cell
  // is not touched by the statement
  ObSEArray<int64_t, 8> next_state;
  if (columns.count() <= 0) {
    ret = OB_INVALID_ARGUMENT;
  } else {
    first_meta = &metas.at(columns.at(0));
    for (int64_t c = 0; OB_SUCC(ret) && c < columns.count(); ++c) {
      if (OB_FAIL(next_state.push_back(-1))) {
        LOG_WARN("failed to init the next state of the ai column", K(ret));
      }
    }
    for (int64_t p = 0; OB_SUCC(ret) && p < task.pk_values_.count(); ++p) {
      if (OB_FAIL(ai_append_ident_condition(cond_clause, cond_cnt,
                                            first_meta->pk_columns_.at(p),
                                            task.pk_values_.at(p), false))) {
        LOG_WARN("failed to append the primary key condition", K(ret));
      }
    }
  }
  for (int64_t c = 0; OB_SUCC(ret) && c < columns.count(); ++c) {
    const ObAiColumnMeta &meta = metas.at(columns.at(c));
    const AiColumnTask &cell = task.columns_.at(c);
    // the cell is unfinished unless it is published by the statement below
    if (c < cell_unfinished.count()) {
      cell_unfinished.at(c) = (cell.found_ ? 1 : 0);
    }
    if (!cell.found_ || cell.state_ >= AI_STATE_GIVEUP_BASE) {
      // the row is not pending, or the ai column has given up on it
      continue;
    }
    const AiEndpoint *endpoint = NULL;
    const int endpoint_ret = ai_find_endpoint(proxy, allocator, meta.model_name_,
                                              endpoints, endpoint);
    if (OB_SUCCESS != endpoint_ret || NULL == endpoint) {
      // the endpoint of the model is not configured yet, keep the row pending so
      // that the result is computed once the endpoint becomes available
      LOG_DEBUG("the endpoint of the ai model is not ready", K(endpoint_ret),
                K(meta.model_name_));
    } else {
      ObSqlString value_literal;
      bool publish_value = false;
      int64_t publish_state = AI_STATE_DONE;
      const int compute_ret = ai_compute_value(allocator, meta, *endpoint,
                                               cell, value_literal);
      if (OB_SUCCESS == compute_ret) {
        publish_value = true;
      } else {
        const int64_t retry_count = (cell.state_ < 100) ? 0 : (cell.state_ % 100);
        if (retry_count + 1 > meta.max_retries_) {
          publish_state = AI_STATE_GIVEUP_BASE + retry_count + 1;
        } else {
          publish_state = AI_STATE_RETRY_BASE + retry_count + 1;
        }
        LOG_WARN("failed to compute the value of the ai column", K(compute_ret),
                 K(meta.column_name_), K(cell.state_));
      }
      if (OB_SUCC(ret) && publish_value &&
          OB_FAIL(ai_append_assignment(set_clause, set_cnt,
                                       meta.column_name_, value_literal.string()))) {
        LOG_WARN("failed to append the value of the ai column", K(ret));
      } else if (OB_SUCC(ret) &&
                 OB_FAIL(ai_append_number_assignment(set_clause, set_cnt,
                                                     meta.state_column_name_,
                                                     publish_state))) {
        LOG_WARN("failed to append the state of the ai column", K(ret));
      } else if (OB_SUCC(ret) &&
                 OB_FAIL(ai_append_number_condition(cond_clause, cond_cnt,
                                                    meta.state_column_name_,
                                                    cell.state_))) {
        LOG_WARN("failed to append the state condition of the ai column", K(ret));
      } else if (OB_SUCC(ret) && c < next_state.count()) {
        next_state.at(c) = publish_state;
      }
      for (int64_t d = 0; OB_SUCC(ret) && d < meta.dep_columns_.count(); ++d) {
        if (d >= cell.dep_values_.count() || d >= cell.dep_is_null_.count()) {
          ret = OB_ERR_UNEXPECTED;
          LOG_WARN("the dependency value of the ai column is missing", K(ret));
        } else if (OB_FAIL(ai_append_ident_condition(cond_clause, cond_cnt,
                                                     meta.dep_columns_.at(d),
                                                     cell.dep_values_.at(d),
                                                     0 != cell.dep_is_null_.at(d)))) {
          LOG_WARN("failed to append the dependency condition", K(ret));
        }
      }
    }
  }
  if (OB_SUCC(ret) && set_cnt > 0) {
    ObSqlString sql;
    int64_t affected = 0;
    // the cells are published by the update statement. A cell whose published
    // state still needs the worker (a retry, or a column that gave up its
    // retries) stays unfinished, so that the strict read of the next statement
    // reports it instead of reading a stale NULL value.
    for (int64_t c = 0; c < cell_unfinished.count() && c < next_state.count(); ++c) {
      if (next_state.at(c) >= 0) {
        cell_unfinished.at(c) = (AI_STATE_DONE == next_state.at(c)) ? 0 : 1;
      }
    }
    if (OB_FAIL(sql.append("update "))) {
    } else if (OB_FAIL(ai_append_ident(sql, first_meta->db_name_))) {
    } else if (OB_FAIL(sql.append("."))) {
    } else if (OB_FAIL(ai_append_ident(sql, first_meta->table_name_))) {
    } else if (OB_FAIL(sql.append(" set "))) {
    } else if (OB_FAIL(sql.append(set_clause.string()))) {
    } else if (OB_FAIL(sql.append(" where "))) {
    } else if (OB_FAIL(sql.append(cond_clause.string()))) {
    } else if (OB_FAIL(proxy->write(sql.ptr(), affected))) {
      LOG_WARN("failed to publish the result of the ai column", K(ret));
      for (int64_t c = 0; c < cell_unfinished.count(); ++c) {
        cell_unfinished.at(c) = 1;
      }
    } else if (0 == affected) {
      // the row was changed concurrently, the next round recomputes the values
      LOG_DEBUG("the result of the ai column is stale", K(affected));
      for (int64_t c = 0; c < cell_unfinished.count(); ++c) {
        cell_unfinished.at(c) = 1;
      }
    }
  }
  return ret;
}

int ai_process_table(common::ObISQLClient *proxy,
                     ObIAllocator &allocator,
                     const ObIArray<ObAiColumnMeta> &metas,
                     const ObIArray<int64_t> &columns)
{
  int ret = OB_SUCCESS;
  ObSEArray<AiRowTask, 8> tasks;
  ObSEArray<AiEndpoint, 4> endpoints;
  if (columns.count() <= 0) {
    ret = OB_INVALID_ARGUMENT;
  }
  for (int64_t c = 0; OB_SUCC(ret) && c < columns.count(); ++c) {
    const ObAiColumnMeta &meta = metas.at(columns.at(c));
    if (meta.model_name_.empty()) {
      LOG_DEBUG("the ai column has no model name", K(meta.column_name_));
    } else if (OB_FAIL(ai_collect_pending_rows(proxy, allocator, meta,
                                               c, columns.count(), tasks))) {
      LOG_WARN("failed to collect the pending rows of the ai column", K(ret));
    }
  }
  ObSEArray<int8_t, 64> unfinished;
  ObSEArray<int8_t, 8> cell_unfinished;
  for (int64_t t = 0; OB_SUCC(ret) && t < tasks.count(); ++t) {
    cell_unfinished.reuse();
    for (int64_t c = 0; OB_SUCC(ret) && c < columns.count(); ++c) {
      if (OB_FAIL(cell_unfinished.push_back(0))) {
        LOG_WARN("failed to init the unfinished flag", K(ret));
      }
    }
    if (OB_SUCC(ret) && OB_FAIL(ai_publish_row(proxy, allocator, metas, columns,
                                               endpoints, tasks.at(t),
                                               cell_unfinished))) {
      LOG_WARN("failed to publish the results of the ai columns of the row", K(ret));
    }
    for (int64_t c = 0; OB_SUCC(ret) && c < columns.count(); ++c) {
      if (OB_FAIL(unfinished.push_back(cell_unfinished.at(c)))) {
        LOG_WARN("failed to push back the unfinished flag", K(ret));
      }
    }
  }
  // remember which AI columns still have unfinished rows, the strict read mode
  // of the next statements reads the flags instead of querying the table. The
  // flags are refreshed after the publish so that a strict read of the next
  // statement never observes an unfinished marker of an already published row,
  // and a row that gave up its retries is still reported as unfinished.
  for (int64_t c = 0; OB_SUCC(ret) && c < columns.count(); ++c) {
    const ObAiColumnMeta &meta = metas.at(columns.at(c));
    bool pending = false;
    for (int64_t t = 0; !pending && t < tasks.count(); ++t) {
      const int64_t idx = t * columns.count() + c;
      pending = idx < unfinished.count() && 0 != unfinished.at(idx);
    }
    const int flag_ret = ObAiColumnRegistry::get_instance().set_pending(
        meta.db_name_, meta.table_name_, meta.column_name_, pending);
    if (OB_SUCCESS != flag_ret) {
      LOG_WARN("failed to set the pending flag of the ai column", K(flag_ret),
               K(meta.column_name_));
    }
  }
  return ret;
}

int ai_worker_run_once(ObIAllocator &allocator)
{
  int ret = OB_SUCCESS;
  common::ObISQLClient *proxy = GCTX.sql_proxy_;
  ObSEArray<ObAiColumnMeta, 16> metas;
  if (OB_ISNULL(proxy)) {
    // the server is not ready yet
  } else if (OB_FAIL(ObAiColumnRegistry::get_instance().get_metas(metas))) {
    LOG_WARN("failed to get the metas of the ai columns", K(ret));
  } else if (0 == metas.count()) {
    // no ai column exists yet
  } else {
    for (int64_t i = 0; OB_SUCC(ret) && i < metas.count(); ++i) {
      const ObAiColumnMeta &meta = metas.at(i);
      bool handled = false;
      for (int64_t k = 0; !handled && k < i; ++k) {
        if (metas.at(k).db_name_.case_compare_equal(meta.db_name_) &&
            metas.at(k).table_name_.case_compare_equal(meta.table_name_)) {
          handled = true;
        }
      }
      if (handled) {
        continue;
      }
      ObSEArray<int64_t, 8> columns;
      for (int64_t k = 0; OB_SUCC(ret) && k < metas.count(); ++k) {
        if (metas.at(k).db_name_.case_compare_equal(meta.db_name_) &&
            metas.at(k).table_name_.case_compare_equal(meta.table_name_)) {
          if (OB_FAIL(columns.push_back(k))) {
            LOG_WARN("failed to push back the ai column index", K(ret));
          }
        }
      }
      if (OB_SUCC(ret)) {
        const int table_ret = ai_process_table(proxy, allocator, metas, columns);
        if (OB_SUCCESS != table_ret) {
          LOG_WARN("failed to process the ai columns of the table", K(table_ret),
                   K(meta.table_name_));
        }
      }
    }
  }
  return ret;
}

void *ai_column_worker_thread(void *arg)
{
  UNUSED(arg);
  LOG_INFO("the ai column worker thread is started");
  while (true) {
    ObArenaAllocator allocator("AiColWorker");
    const int ret = ai_worker_run_once(allocator);
    if (OB_SUCCESS != ret) {
      LOG_WARN("failed to run one round of the ai column worker", K(ret));
    }
    ai_worker_sleep();
  }
  return NULL;
}

}  // namespace

int ai_column_worker_start_once()
{
  int ret = OB_SUCCESS;
  static bool started = false;
  if (!started) {
    started = true;
    // curl is initialized by the thread that starts the worker, before any other
    // thread posts a request
    const int curl_init_ret = static_cast<int>(curl_global_init(CURL_GLOBAL_DEFAULT));
    if (0 != curl_init_ret) {
      LOG_WARN("failed to initialize curl for the ai column worker", K(curl_init_ret));
    }
    pthread_t tid;
    const int rc = pthread_create(&tid, NULL, ai_column_worker_thread, NULL);
    if (0 != rc) {
      ret = OB_ERR_UNEXPECTED;
      LOG_WARN("failed to create the ai column worker thread", K(ret), K(rc));
    } else {
      pthread_detach(tid);
      LOG_INFO("the ai column worker thread is started");
    }
  }
  return ret;
}

namespace {

const char *TASK_TABLE_NAME = "__all_ai_column_task";

// find the position of the task table name in the sql text, the returned
// position is the beginning of the whole table name, the prefix such as
// "oceanbase." is skipped by the caller
int64_t find_task_table(const ObString &sql, const int64_t start_pos)
{
  int64_t pos = -1;
  const int64_t name_len = static_cast<int64_t>(strlen(TASK_TABLE_NAME));
  for (int64_t i = start_pos; -1 == pos && i + name_len <= sql.length(); ++i) {
    if (0 == strncasecmp(sql.ptr() + i, TASK_TABLE_NAME, static_cast<size_t>(name_len))) {
      pos = i;
    }
  }
  return pos;
}

// the beginning of the table reference, the database name before the table name
// is skipped too
int64_t find_table_ref_begin(const ObString &sql, const int64_t name_pos)
{
  int64_t begin = name_pos;
  const char *dbs[] = {"oceanbase.", "def.", "sys."};
  for (int64_t k = 0; k < 3; ++k) {
    const int64_t db_len = static_cast<int64_t>(strlen(dbs[k]));
    if (name_pos >= db_len &&
        0 == strncasecmp(sql.ptr() + name_pos - db_len, dbs[k], static_cast<size_t>(db_len))) {
      begin = name_pos - db_len;
      break;
    }
  }
  return begin;
}

int append_escaped(ObSqlString &sql, const ObString &value)
{
  int ret = OB_SUCCESS;
  if (OB_FAIL(sql.append_fmt("'%.*s'", value.length(), value.ptr()))) {
    LOG_WARN("failed to append the escaped string", K(ret));
  }
  return ret;
}

// build the synthesized query of __all_ai_column_task, one union all branch per
// table that has AI columns
int build_task_table_query(const ObIArray<ObAiColumnMeta> &metas,
                           ObIAllocator &allocator,
                           ObSqlString &sql)
{
  int ret = OB_SUCCESS;
  int64_t branch_cnt = 0;
  ObSEArray<const ObAiColumnMeta *, 8> emitted_tables;
  for (int64_t i = 0; OB_SUCC(ret) && i < metas.count(); ++i) {
    const ObAiColumnMeta &meta = metas.at(i);
    bool emitted = false;
    for (int64_t k = 0; !emitted && k < emitted_tables.count(); ++k) {
      if (emitted_tables.at(k)->table_name_.case_compare_equal(meta.table_name_) &&
          emitted_tables.at(k)->db_name_.case_compare_equal(meta.db_name_)) {
        emitted = true;
      }
    }
    if (emitted) {
      continue;
    } else if (OB_FAIL(emitted_tables.push_back(&meta))) {
      LOG_WARN("failed to push back the table name", K(ret));
    } else if (0 == branch_cnt && OB_FAIL(sql.append("("))) {
      LOG_WARN("failed to append the query", K(ret));
    } else if (branch_cnt > 0 && OB_FAIL(sql.append(" union all "))) {
      LOG_WARN("failed to append the query", K(ret));
    }
    // collect the AI columns of this table
    ObSEArray<const ObAiColumnMeta *, 8> columns;
    for (int64_t k = 0; OB_SUCC(ret) && k < metas.count(); ++k) {
      const ObAiColumnMeta &other = metas.at(k);
      if (other.table_name_.case_compare_equal(meta.table_name_) &&
          other.db_name_.case_compare_equal(meta.db_name_)) {
        if (OB_FAIL(columns.push_back(&other))) {
          LOG_WARN("failed to push back the ai column", K(ret));
        }
      }
    }
    // the state columns of all AI columns of the table, a row is pending when
    // any of them is positive
    ObSqlString pending;
    for (int64_t k = 0; OB_SUCC(ret) && k < columns.count(); ++k) {
      if (OB_FAIL(pending.append_fmt("%.*s %.*s >= %ld and %.*s < %ld",
                                     0 == k ? 0 : 4,
                                     0 == k ? "" : " or ",
                                     columns.at(k)->state_column_name_.length(),
                                     columns.at(k)->state_column_name_.ptr(),
                                     static_cast<long>(AI_STATE_PENDING),
                                     columns.at(k)->state_column_name_.length(),
                                     columns.at(k)->state_column_name_.ptr(),
                                     static_cast<long>(AI_STATE_GIVEUP_BASE)))) {
        LOG_WARN("failed to append the pending condition", K(ret));
      }
    }
    for (int64_t k = 0; OB_SUCC(ret) && k < columns.count(); ++k) {
      const ObAiColumnMeta &column = *columns.at(k);
      if (k > 0 && OB_FAIL(sql.append(" union all "))) {
        LOG_WARN("failed to append the query", K(ret));
      } else if (OB_FAIL(sql.append_fmt(
                   "select (select t.table_id from oceanbase.__all_table t "
                   "join oceanbase.__all_database d on d.database_id = t.database_id "
                   "where d.database_name = '%.*s' and t.table_name = '%.*s' limit 1) "
                   "as table_id, concat(",
                   column.db_name_.length(),
                   column.db_name_.ptr(),
                   column.table_name_.length(),
                   column.table_name_.ptr()))) {
        LOG_WARN("failed to append the query", K(ret));
      } else {
        // the request id of a row is derived from its primary key, so that all
        // the AI columns of one row share the same request id
        for (int64_t p = 0; OB_SUCC(ret) && p < column.pk_columns_.count(); ++p) {
          if (p > 0 && OB_FAIL(sql.append(", '|', "))) {
            LOG_WARN("failed to append the query", K(ret));
          } else if (OB_FAIL(sql.append_fmt("cast(`%.*s` as char)",
                                            column.pk_columns_.at(p).length(),
                                            column.pk_columns_.at(p).ptr()))) {
            LOG_WARN("failed to append the query", K(ret));
          }
        }
        if (OB_SUCC(ret) && 0 == column.pk_columns_.count()
            && OB_FAIL(sql.append("''"))) {
          LOG_WARN("failed to append the query", K(ret));
        } else if (OB_FAIL(sql.append_fmt(
                     ") as request_id, "
                     "cast(`%.*s` / 100 as signed) as state, "
                     "cast(`%.*s` %% 100 as signed) as retry_count, concat(",
                     column.state_column_name_.length(),
                     column.state_column_name_.ptr(),
                     column.state_column_name_.length(),
                     column.state_column_name_.ptr()))) {
          LOG_WARN("failed to append the query", K(ret));
        } else {
          // "`pk1` = " , cast(pk1 as char), ...
          for (int64_t p = 0; OB_SUCC(ret) && p < column.pk_columns_.count(); ++p) {
            if (p > 0 && OB_FAIL(sql.append(", "))) {
              LOG_WARN("failed to append the query", K(ret));
            } else if (OB_FAIL(sql.append_fmt("'%s`%s` = ', cast(%s as char)",
                                              0 == p ? "" : " and ",
                                              column.pk_columns_.at(p).ptr(),
                                              column.pk_columns_.at(p).ptr()))) {
              LOG_WARN("failed to append the query", K(ret));
            }
          }
          if (OB_SUCC(ret) && 0 == column.pk_columns_.count()
              && OB_FAIL(sql.append("''"))) {
            LOG_WARN("failed to append the query", K(ret));
          } else if (OB_FAIL(sql.append_fmt(
                       ") as rowkey_predicate, "
                       "cast('2000-01-01 00:00:00.000000' as datetime(6)) "
                       "as eligible_ts from %.*s.%.*s "
                       "where %.*s >= %ld and %.*s < %ld",
                       column.db_name_.length(), column.db_name_.ptr(),
                       column.table_name_.length(), column.table_name_.ptr(),
                       column.state_column_name_.length(),
                       column.state_column_name_.ptr(),
                       static_cast<long>(AI_STATE_PENDING),
                       column.state_column_name_.length(),
                       column.state_column_name_.ptr(),
                       static_cast<long>(AI_STATE_GIVEUP_BASE)))) {
            LOG_WARN("failed to append the query", K(ret));
          }
        }
      }
    }
    if (OB_SUCC(ret)) {
      branch_cnt += columns.count();
    }
  }
  if (OB_SUCC(ret)) {
    if (0 == branch_cnt) {
      ret = OB_ENTRY_NOT_EXIST;
    } else if (OB_FAIL(sql.append(")"))) {
      LOG_WARN("failed to append the query", K(ret));
    }
  }
  UNUSED(allocator);
  return ret;
}

}  // namespace

namespace {

bool is_ai_space(const char c)
{
  return ' ' == c || '\t' == c || '\n' == c || '\r' == c;
}

bool is_ai_word_char(const char c)
{
  return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z')
      || (c >= '0' && c <= '9') || '_' == c;
}

void read_word_bound(const common::ObString &sql, int64_t pos, int64_t &begin, int64_t &end)
{
  begin = pos;
  end = pos;
  while (end < sql.length() && is_ai_word_char(sql.ptr()[end])) {
    ++end;
  }
}

bool is_clause_keyword(const common::ObString &sql, const int64_t begin, const int64_t end)
{
  static const char *KEYWORDS[] = {"join", "on", "where", "group", "order", "limit",
                                   "having", "union", "left", "right", "inner",
                                   "outer", "full", "cross", "natural", "use",
                                   "force", "ignore", "partition", "for", "using",
                                   "set", "values", "select", "straight_join", "as"};
  bool found = false;
  const int64_t len = end - begin;
  for (int64_t i = 0; !found && i < static_cast<int64_t>(sizeof(KEYWORDS) / sizeof(KEYWORDS[0])); ++i) {
    if (len == static_cast<int64_t>(strlen(KEYWORDS[i]))
        && 0 == strncasecmp(sql.ptr() + begin, KEYWORDS[i], static_cast<size_t>(len))) {
      found = true;
    }
  }
  return found;
}

// reads the optional alias that follows a table reference, returns an empty
// range when the table reference has no alias
void read_table_alias(const common::ObString &sql, const int64_t pos,
                      int64_t &alias_begin, int64_t &alias_end)
{
  alias_begin = pos;
  alias_end = pos;
  int64_t scan = pos;
  while (scan < sql.length() && is_ai_space(sql.ptr()[scan])) {
    ++scan;
  }
  int64_t begin = scan;
  int64_t end = scan;
  read_word_bound(sql, scan, begin, end);
  if (end > begin && 2 == end - begin
      && 0 == strncasecmp(sql.ptr() + begin, "as", 2)) {
    int64_t next = end;
    while (next < sql.length() && is_ai_space(sql.ptr()[next])) {
      ++next;
    }
    begin = next;
    end = next;
    read_word_bound(sql, next, begin, end);
    if (end > begin) {
      alias_begin = begin;
      alias_end = end;
    }
  } else if (end > begin && !is_clause_keyword(sql, begin, end)) {
    alias_begin = begin;
    alias_end = end;
  }
}

}  // namespace

int ai_column_rewrite_task_table(const common::ObString &sql,
                                 ObIAllocator &allocator,
                                 ObString &rewritten,
                                 bool &rewritten_flag)
{
  int ret = OB_SUCCESS;
  rewritten.reset();
  rewritten_flag = false;
  int64_t pos = find_task_table(sql, 0);
  if (-1 == pos) {
    // the statement does not reference the task table
  } else {
    ObSEArray<ObAiColumnMeta, 16> metas;
    ObSqlString task_query;
    ObSqlString out;
    if (OB_FAIL(ObAiColumnRegistry::get_instance().get_metas(metas))) {
      LOG_WARN("failed to get the ai column metas", K(ret));
    } else if (0 == metas.count()) {
      // no AI column at all, the task table is empty
      ret = OB_ENTRY_NOT_EXIST;
    } else if (OB_FAIL(build_task_table_query(metas, allocator, task_query))) {
      LOG_WARN("failed to build the task table query", K(ret));
    } else {
      int64_t copied = 0;
      while (OB_SUCC(ret) && copied < sql.length()) {
        const int64_t next = find_task_table(sql, copied);
        if (-1 == next) {
          if (OB_FAIL(out.append(sql.ptr() + copied, sql.length() - copied))) {
            LOG_WARN("failed to append the sql", K(ret));
          }
          copied = sql.length();
        } else {
          const int64_t begin = find_table_ref_begin(sql, next);
          if (begin > copied &&
              OB_FAIL(out.append(sql.ptr() + copied, begin - copied))) {
            LOG_WARN("failed to append the sql", K(ret));
          } else if (OB_SUCC(ret) &&
                     OB_FAIL(out.append(task_query.string().ptr(),
                                        task_query.string().length()))) {
            LOG_WARN("failed to append the task table query", K(ret));
          } else if (OB_SUCC(ret)) {
            int64_t alias_begin = 0;
            int64_t alias_end = 0;
            read_table_alias(sql, next + static_cast<int64_t>(strlen(TASK_TABLE_NAME)),
                             alias_begin, alias_end);
            if (alias_end > alias_begin) {
              if (OB_FAIL(out.append(" as "))) {
                LOG_WARN("failed to append the alias", K(ret));
              } else if (OB_FAIL(out.append(sql.ptr() + alias_begin, alias_end - alias_begin))) {
                LOG_WARN("failed to append the alias", K(ret));
              } else {
                copied = alias_end;
              }
            } else if (OB_FAIL(out.append(" as __all_ai_column_task"))) {
              LOG_WARN("failed to append the alias", K(ret));
            } else {
              copied = next + static_cast<int64_t>(strlen(TASK_TABLE_NAME));
            }
          }
        }
      }
      if (OB_SUCC(ret)) {
        char *buf = NULL;
        if (OB_ISNULL(buf = static_cast<char *>(allocator.alloc(out.length())))) {
          ret = OB_ALLOCATE_MEMORY_FAILED;
        } else {
          MEMCPY(buf, out.ptr(), out.length());
          rewritten.assign_ptr(buf, static_cast<int32_t>(out.length()));
          rewritten_flag = true;
        }
      }
    }
    if (OB_ENTRY_NOT_EXIST == ret) {
      // no AI column, keep the statement unchanged, the task table is surely empty
      ret = OB_SUCCESS;
    }
  }
  return ret;
}

namespace {

void collect_column_names(const ParseNode *node, ObIArray<ObString> &names)
{
  if (NULL == node) {
    // do nothing
  } else {
    if (T_COLUMN_REF == node->type_ && node->num_child_ >= 3 &&
        NULL != node->children_[2] && NULL != node->children_[2]->str_value_) {
      const ParseNode *name_node = node->children_[2];
      ObString name(static_cast<int32_t>(name_node->str_len_), name_node->str_value_);
      if (OB_SUCCESS != names.push_back(name)) {
        LOG_WARN_RET(OB_ERR_UNEXPECTED, "failed to push back the column name");
      }
    } else if (T_IDENT == node->type_ && NULL != node->str_value_) {
      ObString name(static_cast<int32_t>(node->str_len_), node->str_value_);
      if (OB_SUCCESS != names.push_back(name)) {
        LOG_WARN_RET(OB_ERR_UNEXPECTED, "failed to push back the column name");
      }
    }
    for (int64_t i = 0; i < node->num_child_; ++i) {
      collect_column_names(node->children_[i], names);
    }
  }
}

}  // namespace

bool is_primary_key_node(const ParseNode *node)
{
  return NULL != node
      && (T_PRIMARY_KEY == node->type_ || T_CONSTR_PRIMARY_KEY == node->type_);
}

bool has_primary_key_attribute(const ParseNode *attributes)
{
  bool found = is_primary_key_node(attributes);
  if (!found && NULL != attributes && T_COLUMN_ATTRIBUTES == attributes->type_) {
    for (int64_t i = 0; !found && i < attributes->num_child_; ++i) {
      found = is_primary_key_node(attributes->children_[i]);
    }
  }
  return found;
}

void ai_column_collect_pk_columns(const ParseNode *elements, ObIArray<ObString> &names)
{
  if (NULL != elements) {
    for (int64_t i = 0; i < elements->num_child_; ++i) {
      const ParseNode *element = elements->children_[i];
      if (NULL != element && T_PRIMARY_KEY == element->type_) {
        // primary key (c1, c2)
        collect_column_names(element, names);
      } else if (NULL != element && T_COLUMN_DEFINITION == element->type_
                 && element->num_child_ > 2
                 && has_primary_key_attribute(element->children_[2])) {
        // c1 bigint primary key
        ObSEArray<ObString, 4> ref_names;
        collect_column_names(element->children_[0], ref_names);
        if (ref_names.count() > 0) {
          names.push_back(ref_names.at(ref_names.count() - 1));
        }
      }
    }
  }
}

namespace ai_column_rw {

bool is_ident_char(const char c)
{
  return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
         (c >= '0' && c <= '9') || '_' == c || '$' == c;
}

void skip_space(const ObString &sql, int64_t &pos)
{
  while (pos < sql.length() && isspace(static_cast<unsigned char>(sql.ptr()[pos]))) {
    ++pos;
  }
}

bool read_word(const ObString &sql, int64_t &pos, const char *word)
{
  bool bret = false;
  skip_space(sql, pos);
  const int64_t len = static_cast<int64_t>(strlen(word));
  if (pos + len <= sql.length() &&
      0 == strncasecmp(sql.ptr() + pos, word, static_cast<size_t>(len))) {
    const int64_t end = pos + len;
    if (end >= sql.length() || !is_ident_char(sql.ptr()[end])) {
      pos = end;
      bret = true;
    }
  }
  return bret;
}

bool read_ident(const ObString &sql, int64_t &pos, ObString &ident)
{
  bool bret = false;
  ident.reset();
  skip_space(sql, pos);
  if (pos < sql.length() && '`' == sql.ptr()[pos]) {
    const int64_t begin = ++pos;
    while (pos < sql.length() && '`' != sql.ptr()[pos]) {
      ++pos;
    }
    if (pos < sql.length()) {
      ident.assign_ptr(sql.ptr() + begin, static_cast<int32_t>(pos - begin));
      ++pos;
      bret = !ident.empty();
    }
  } else if (pos < sql.length() && is_ident_char(sql.ptr()[pos])) {
    const int64_t begin = pos;
    while (pos < sql.length() && is_ident_char(sql.ptr()[pos])) {
      ++pos;
    }
    ident.assign_ptr(sql.ptr() + begin, static_cast<int32_t>(pos - begin));
    bret = true;
  }
  return bret;
}

bool read_table_ref(const ObString &sql, int64_t &pos, ObString &db, ObString &table)
{
  bool bret = false;
  ObString first;
  db.reset();
  table.reset();
  if (read_ident(sql, pos, first)) {
    int64_t p = pos;
    skip_space(sql, p);
    if (p < sql.length() && '.' == sql.ptr()[p]) {
      ++p;
      if (read_ident(sql, p, table)) {
        db = first;
        pos = p;
        bret = true;
      }
    } else {
      table = first;
      bret = true;
    }
  }
  return bret;
}

bool mentions(const ObString &text, const ObString &name)
{
  bool bret = false;
  if (!name.empty() && name.length() <= text.length()) {
    for (int64_t i = 0; !bret && i + name.length() <= text.length(); ++i) {
      if (0 == strncasecmp(text.ptr() + i, name.ptr(), static_cast<size_t>(name.length()))) {
        const bool left_ok = (0 == i) || !is_ident_char(text.ptr()[i - 1]);
        const int64_t end = i + name.length();
        const bool right_ok = (end >= text.length()) || !is_ident_char(text.ptr()[end]);
        bret = left_ok && right_ok;
      }
    }
  }
  return bret;
}

int copy_quoted(const ObString &text, int64_t &pos, ObSqlString &out)
{
  int ret = OB_SUCCESS;
  const char quote = text.ptr()[pos];
  const int64_t begin = pos++;
  while (pos < text.length() && text.ptr()[pos] != quote) {
    if ('\\' == text.ptr()[pos] && pos + 1 < text.length()) {
      pos += 2;
    } else {
      ++pos;
    }
  }
  if (pos < text.length()) {
    ++pos;
  }
  if (OB_FAIL(out.append(text.ptr() + begin, pos - begin))) {
    LOG_WARN("failed to append the quoted text", K(ret));
  }
  return ret;
}

int append_quoted_ident(ObSqlString &out, const ObString &name)
{
  int ret = OB_SUCCESS;
  if (OB_FAIL(out.append_fmt("`%.*s`", name.length(), name.ptr()))) {
    LOG_WARN("failed to append the identifier", K(ret));
  }
  return ret;
}

int rewrite_status_pseudo(const ObString &pred, const ObString &state_col, ObSqlString &out)
{
  int ret = OB_SUCCESS;
  int64_t pos = 0;
  while (OB_SUCC(ret) && pos < pred.length()) {
    const char c = pred.ptr()[pos];
    if ('`' == c || '\'' == c || '"' == c) {
      if (OB_FAIL(copy_quoted(pred, pos, out))) {
        LOG_WARN("failed to copy the quoted text", K(ret));
      }
    } else if (is_ident_char(c)) {
      const int64_t begin = pos;
      while (pos < pred.length() && is_ident_char(pred.ptr()[pos])) {
        ++pos;
      }
      const int64_t len = pos - begin;
      bool replaced = false;
      if (6 == len && 0 == strncasecmp(pred.ptr() + begin, "status", 6)) {
        int64_t p = pos;
        skip_space(pred, p);
        if (p < pred.length() && '=' == pred.ptr()[p]) {
          ++p;
          skip_space(pred, p);
          if (p < pred.length() && '\'' == pred.ptr()[p]) {
            const int64_t vbegin = ++p;
            while (p < pred.length() && '\'' != pred.ptr()[p]) {
              ++p;
            }
            const int64_t vlen = p - vbegin;
            if (p < pred.length()) {
              ++p;
            }
            replaced = true;
            if (6 == vlen && 0 == strncasecmp(pred.ptr() + vbegin, "FAILED", 6)) {
              ret = out.append_fmt("(`%.*s` / 100 >= 4)", state_col.length(), state_col.ptr());
            } else if (5 == vlen && 0 == strncasecmp(pred.ptr() + vbegin, "READY", 5)) {
              ret = out.append_fmt("(`%.*s` = 0)", state_col.length(), state_col.ptr());
            } else {
              ret = out.append_fmt("(`%.*s` > 0 and `%.*s` / 100 < 4)",
                                   state_col.length(), state_col.ptr(),
                                   state_col.length(), state_col.ptr());
            }
            if (OB_FAIL(ret)) {
              LOG_WARN("failed to append the status condition", K(ret));
            }
            pos = p;
          }
        }
      }
      if (OB_SUCC(ret) && !replaced && OB_FAIL(out.append(pred.ptr() + begin, len))) {
        LOG_WARN("failed to append the text", K(ret));
      }
    } else {
      if (OB_FAIL(out.append(pred.ptr() + pos, 1))) {
        LOG_WARN("failed to append the text", K(ret));
      }
      ++pos;
    }
  }
  return ret;
}

const ObAiColumnMeta *find_meta(const ObIArray<ObAiColumnMeta> &metas,
                                const ObString &db,
                                const ObString &table,
                                const ObString &column)
{
  const ObAiColumnMeta *found = NULL;
  for (int64_t i = 0; NULL == found && i < metas.count(); ++i) {
    const ObAiColumnMeta &meta = metas.at(i);
    if (meta.table_name_.case_compare_equal(table) &&
        meta.column_name_.case_compare_equal(column) &&
        (db.empty() || meta.db_name_.case_compare_equal(db))) {
      found = &meta;
    }
  }
  return found;
}

}  // namespace ai_column_rw

int ai_column_rewrite_retry_stmt(const ObString &sql,
                                 ObIAllocator &allocator,
                                 ObString &rewritten,
                                 bool &rewritten_flag)
{
  int ret = OB_SUCCESS;
  using namespace ai_column_rw;
  rewritten.reset();
  rewritten_flag = false;
  int64_t pos = 0;
  skip_space(sql, pos);
  ObString db;
  ObString table;
  ObString column;
  if (!read_word(sql, pos, "alter")) {
    // not an alter statement
  } else if (!read_word(sql, pos, "table")) {
    // not an alter table statement
  } else if (!read_table_ref(sql, pos, db, table)) {
    // not a table reference
  } else if (!read_word(sql, pos, "retry")) {
    // not a retry statement
  } else if (!read_word(sql, pos, "ai") || !read_word(sql, pos, "column")) {
    ret = OB_NOT_SUPPORTED;
    LOG_WARN("the alter table retry statement is not supported", K(ret));
  } else if (!read_ident(sql, pos, column)) {
    ret = OB_NOT_SUPPORTED;
    LOG_WARN("failed to parse the retried ai column", K(ret));
  } else {
    int64_t p = pos;
    skip_space(sql, p);
    ObString pred;
    if (p < sql.length() && ',' == sql.ptr()[p]) {
      // RETRY can not be combined with another alter table action
      ret = OB_NOT_SUPPORTED;
    } else if (read_word(sql, p, "where")) {
      pred.assign_ptr(sql.ptr() + p, static_cast<int32_t>(sql.length() - p));
    } else if (p < sql.length()) {
      ret = OB_NOT_SUPPORTED;
    }
    ObSEArray<ObAiColumnMeta, 16> metas;
    const ObAiColumnMeta *meta = NULL;
    if (OB_FAIL(ret)) {
      LOG_WARN("invalid alter table retry ai column statement", K(ret));
    } else if (OB_FAIL(ObAiColumnRegistry::get_instance().get_metas(metas))) {
      LOG_WARN("failed to get the ai column metas", K(ret));
    } else if (NULL == (meta = find_meta(metas, db, table, column))) {
      ret = OB_NOT_SUPPORTED;
      LOG_WARN("retrying a column that is not an ai column", K(ret), K(column));
    } else {
      ObSqlString out;
      if (OB_FAIL(out.append_fmt("update "))) {
      } else if (OB_FAIL(append_quoted_ident(out, db))) {
      } else if (OB_FAIL(out.append("."))) {
      } else if (OB_FAIL(append_quoted_ident(out, table))) {
      } else if (OB_FAIL(out.append(" set "))) {
      } else if (OB_FAIL(append_quoted_ident(out, meta->column_name_))) {
      } else if (OB_FAIL(out.append(" = NULL, "))) {
      } else if (OB_FAIL(append_quoted_ident(out, meta->state_column_name_))) {
      } else if (OB_FAIL(out.append_fmt(" = %ld, ",
                                        static_cast<long>(AI_STATE_PENDING)))) {
      } else if (OB_FAIL(append_quoted_ident(out, meta->request_column_name_))) {
      } else if (OB_FAIL(out.append(" = 0"))) {
        LOG_WARN("failed to build the retry statement", K(ret));
      }
      if (OB_SUCC(ret) && !pred.empty()) {
        ObSqlString new_pred;
        if (OB_FAIL(rewrite_status_pseudo(pred, meta->state_column_name_, new_pred))) {
          LOG_WARN("failed to rewrite the status pseudo column", K(ret));
        } else if (OB_FAIL(out.append(" where "))) {
        } else if (OB_FAIL(out.append(new_pred.ptr(), new_pred.length()))) {
          LOG_WARN("failed to append the predicate", K(ret));
        }
      }
      if (OB_SUCC(ret)) {
        char *buf = NULL;
        if (OB_ISNULL(buf = static_cast<char *>(allocator.alloc(out.length())))) {
          ret = OB_ALLOCATE_MEMORY_FAILED;
        } else {
          MEMCPY(buf, out.ptr(), out.length());
          rewritten.assign_ptr(buf, static_cast<int32_t>(out.length()));
          rewritten_flag = true;
          (void)ObAiColumnRegistry::get_instance().set_pending(
              meta->db_name_, meta->table_name_, meta->column_name_, true);
        }
      }
    }
  }
  return ret;
}

int ai_column_rewrite_update_stmt(const ObString &sql,
                                  ObIAllocator &allocator,
                                  ObString &rewritten,
                                  bool &rewritten_flag)
{
  int ret = OB_SUCCESS;
  using namespace ai_column_rw;
  rewritten.reset();
  rewritten_flag = false;
  int64_t pos = 0;
  skip_space(sql, pos);
  ObString db;
  ObString table;
  if (!read_word(sql, pos, "update")) {
    // not an update statement
  } else if (!read_table_ref(sql, pos, db, table)) {
    // not a table reference
  } else if (!read_word(sql, pos, "set")) {
    // not a simple update statement
  } else {
    // find the where keyword that ends the set list
    int64_t where_pos = -1;
    int64_t depth = 0;
    int64_t p = pos;
    while (p < sql.length() && -1 == where_pos) {
      const char c = sql.ptr()[p];
      if ('`' == c || '\'' == c || '"' == c) {
        ObSqlString ignored;
        if (OB_FAIL(copy_quoted(sql, p, ignored))) {
          LOG_WARN("failed to skip the quoted text", K(ret));
        }
      } else if ('(' == c) {
        ++depth;
        ++p;
      } else if (')' == c) {
        --depth;
        ++p;
      } else if (0 == depth && is_ident_char(c)) {
        const int64_t begin = p;
        while (p < sql.length() && is_ident_char(sql.ptr()[p])) {
          ++p;
        }
        if (5 == p - begin && 0 == strncasecmp(sql.ptr() + begin, "where", 5)) {
          where_pos = begin;
        }
      } else {
        ++p;
      }
    }
    // collect the assigned column names
    const int64_t set_end = (-1 == where_pos) ? sql.length() : where_pos;
    ObSEArray<ObString, 16> assigned;
    depth = 0;
    p = pos;
    while (OB_SUCC(ret) && p < set_end) {
      const char c = sql.ptr()[p];
      if ('`' == c || '\'' == c || '"' == c) {
        ObSqlString ignored;
        if (OB_FAIL(copy_quoted(sql, p, ignored))) {
          LOG_WARN("failed to skip the quoted text", K(ret));
        }
      } else if ('(' == c) {
        ++depth;
        ++p;
      } else if (')' == c) {
        --depth;
        ++p;
      } else if (0 == depth && is_ident_char(c)) {
        const int64_t begin = p;
        while (p < sql.length() && is_ident_char(sql.ptr()[p])) {
          ++p;
        }
        int64_t q = p;
        skip_space(sql, q);
        if (q < sql.length() && '=' == sql.ptr()[q] &&
            (q + 1 >= sql.length() || '=' != sql.ptr()[q + 1])) {
          ObString name(static_cast<int32_t>(p - begin), sql.ptr() + begin);
          if (OB_FAIL(assigned.push_back(name))) {
            LOG_WARN("failed to push back the assigned column", K(ret));
          }
          p = q + 1;
        }
      } else {
        ++p;
      }
    }
    ObSEArray<ObAiColumnMeta, 16> metas;
    if (OB_FAIL(ret)) {
      LOG_WARN("failed to parse the update statement", K(ret));
    } else if (OB_FAIL(ObAiColumnRegistry::get_instance().get_metas(metas))) {
      LOG_WARN("failed to get the ai column metas", K(ret));
    } else {
      ObSqlString reset;
      bool needs_reset = false;
      for (int64_t i = 0; OB_SUCC(ret) && i < metas.count(); ++i) {
        const ObAiColumnMeta &meta = metas.at(i);
        if (!meta.table_name_.case_compare_equal(table) ||
            (!db.empty() && !meta.db_name_.case_compare_equal(db))) {
          continue;
        }
        bool affected = false;
        bool touched_by_self = false;
        for (int64_t k = 0; k < assigned.count(); ++k) {
          const ObString &name = assigned.at(k);
          if (name.case_compare_equal(meta.column_name_)) {
            touched_by_self = true;
          }
          if (name.length() >= 5 && 0 == strncasecmp(name.ptr(), "__ai_", 5)) {
            touched_by_self = true;
          }
          for (int64_t d = 0; !affected && d < meta.dep_columns_.count(); ++d) {
            if (name.case_compare_equal(meta.dep_columns_.at(d))) {
              affected = true;
            }
          }
          for (int64_t d = 0; !affected && d < meta.pk_columns_.count(); ++d) {
            if (name.case_compare_equal(meta.pk_columns_.at(d))) {
              affected = true;
            }
          }
        }
        if (!affected || touched_by_self) {
          // the updated columns do not affect this AI column
        } else {
          needs_reset = true;
          (void)ObAiColumnRegistry::get_instance().set_pending(
              meta.db_name_, meta.table_name_, meta.column_name_, true);
          if (OB_FAIL(reset.append(", "))) {
          } else if (OB_FAIL(append_quoted_ident(reset, meta.column_name_))) {
          } else if (OB_FAIL(reset.append(" = NULL, "))) {
          } else if (OB_FAIL(append_quoted_ident(reset, meta.state_column_name_))) {
          } else if (OB_FAIL(reset.append_fmt(" = %ld, ",
                                              static_cast<long>(AI_STATE_PENDING)))) {
          } else if (OB_FAIL(append_quoted_ident(reset, meta.request_column_name_))) {
          } else if (OB_FAIL(reset.append(" = 0"))) {
            LOG_WARN("failed to build the reset clause", K(ret));
          }
        }
      }
      if (OB_SUCC(ret) && needs_reset) {
        ObSqlString out;
        if (OB_FAIL(out.append(sql.ptr(), set_end))) {
          LOG_WARN("failed to append the statement", K(ret));
        } else if (OB_FAIL(out.append(reset.ptr(), reset.length()))) {
          LOG_WARN("failed to append the reset clause", K(ret));
        } else if (OB_FAIL(out.append(" "))) {
          LOG_WARN("failed to append the space", K(ret));
        } else if (OB_FAIL(out.append(sql.ptr() + set_end, sql.length() - set_end))) {
          LOG_WARN("failed to append the predicate", K(ret));
        } else {
          char *buf = NULL;
          if (OB_ISNULL(buf = static_cast<char *>(allocator.alloc(out.length())))) {
            ret = OB_ALLOCATE_MEMORY_FAILED;
          } else {
            MEMCPY(buf, out.ptr(), out.length());
            rewritten.assign_ptr(buf, static_cast<int32_t>(out.length()));
            rewritten_flag = true;
          }
        }
      }
    }
  }
  return ret;
}

int ai_column_check_strict_read(const ObString &sql, const bool strict_read)
{
  int ret = OB_SUCCESS;
  using namespace ai_column_rw;
  if (!strict_read) {
    // the strict read mode is off
  } else {
    int64_t pos = 0;
    skip_space(sql, pos);
    ObSEArray<ObAiColumnMeta, 16> metas;
    if (read_word(sql, pos, "alter")) {
      // the retry statement is always allowed
    } else if (OB_FAIL(ObAiColumnRegistry::get_instance().get_metas(metas))) {
      LOG_WARN("failed to get the ai column metas", K(ret));
    } else {
      for (int64_t i = 0; OB_SUCC(ret) && i < metas.count(); ++i) {
        const ObAiColumnMeta &meta = metas.at(i);
        bool pending = false;
        if (!mentions(sql, meta.column_name_) || !mentions(sql, meta.table_name_)) {
          // the statement does not read this AI column
        } else if (OB_FAIL(ObAiColumnRegistry::get_instance().get_pending(
                     meta.db_name_, meta.table_name_, meta.column_name_, pending))) {
          LOG_WARN("failed to get the pending flag of the ai column", K(ret),
                   K(meta.column_name_));
        } else if (pending) {
          for (int64_t k = 0; k < metas.count(); ++k) {
          }
          ret = OB_ERR_AI_COLUMN_UNFINISHED_TASK;
          LOG_USER_ERROR(OB_ERR_AI_COLUMN_UNFINISHED_TASK, 0L, 0L);
        }
      }
    }
  }
  return ret;
}

}  // namespace sql
}  // namespace oceanbase
