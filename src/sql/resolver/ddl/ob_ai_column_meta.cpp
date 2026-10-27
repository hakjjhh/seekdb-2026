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
                         ob_malloc(src.length(), "AiColumnMeta")))) {
    ret = OB_ALLOCATE_MEMORY_FAILED;
  } else {
    MEMCPY(buf, src.ptr(), src.length());
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
  for (int64_t i = 0; NULL == found && i < node->num_child_; ++i) {
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
    stored.is_json_ = meta.is_json_;
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

}  // namespace sql
}  // namespace oceanbase
