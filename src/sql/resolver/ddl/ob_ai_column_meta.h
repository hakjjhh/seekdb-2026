/**
 * Copyright (c) 2021 OceanBase
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef OCEANBASE_SQL_RESOLVER_DDL_OB_AI_COLUMN_META_H_
#define OCEANBASE_SQL_RESOLVER_DDL_OB_AI_COLUMN_META_H_

#include "lib/lock/ob_mutex.h"
#include "lib/string/ob_string.h"
#include "lib/container/ob_se_array.h"
#include "lib/allocator/ob_allocator.h"
#include "lib/utility/ob_print_utils.h"
#include "lib/utility/ob_macro_utils.h"
#include "sql/parser/parse_node.h"

namespace oceanbase
{
namespace sql
{

// one part of the prompt template of an AI column, a part is either a string
// literal or a column of the table
struct ObAiPromptPart
{
  bool is_column_;
  common::ObString text_;
  ObAiPromptPart() : is_column_(false), text_() {}
};

// metadata of one AI column, kept in memory for the lifetime of the observer
struct ObAiColumnMeta
{
  common::ObString db_name_;
  common::ObString table_name_;
  common::ObString column_name_;
  common::ObString normalized_expr_;
  common::ObString state_column_name_;
  common::ObString request_column_name_;
  common::ObString result_spec_column_name_;
  common::ObString model_name_;
  int64_t max_retries_;
  int64_t dim_;
  bool is_json_;
  bool is_embed_;
  // ObObjType of the result column of the AI column
  int32_t result_type_;
  // whether the AI column has unfinished rows, maintained by the worker thread
  // and by the statements that invalidate the AI column
  bool pending_;
  // the prompt template of AI_COMPLETE, and the columns the AI expression
  // depends on, and the primary key columns of the table
  common::ObSEArray<ObAiPromptPart, 8> prompt_parts_;
  common::ObSEArray<common::ObString, 8> dep_columns_;
  common::ObSEArray<common::ObString, 4> pk_columns_;

  ObAiColumnMeta()
    : db_name_(), table_name_(), column_name_(), normalized_expr_(),
      state_column_name_(), request_column_name_(), result_spec_column_name_(),
      model_name_(), max_retries_(0), dim_(0), is_json_(false), is_embed_(false),
      result_type_(0), pending_(false),
      prompt_parts_(), dep_columns_(), pk_columns_()
  {}
  TO_STRING_KV(K_(db_name), K_(table_name), K_(column_name), K_(max_retries), K_(is_json),
               K_(state_column_name), K_(request_column_name), K_(result_spec_column_name));
};

// process wide registry of the AI columns, the rows of the system view
// __all_ai_column are generated from it
class ObAiColumnRegistry
{
public:
  static ObAiColumnRegistry &get_instance();
  int add_meta(const ObAiColumnMeta &meta);
  int get_metas(common::ObIArray<ObAiColumnMeta> &metas);
  int remove_table(const common::ObString &db_name, const common::ObString &table_name);
  int set_pending(const common::ObString &db_name,
                  const common::ObString &table_name,
                  const common::ObString &column_name,
                  const bool pending);
  int get_pending(const common::ObString &db_name,
                  const common::ObString &table_name,
                  const common::ObString &column_name,
                  bool &pending);
  int64_t count();
  void reset();
private:
  ObAiColumnRegistry() : mutex_(), metas_() {}
  ~ObAiColumnRegistry() {}
  DISALLOW_COPY_AND_ASSIGN(ObAiColumnRegistry);
  lib::ObMutex mutex_;
  common::ObSEArray<ObAiColumnMeta, 16> metas_;
};

// find the T_CONSTR_AI_COLUMN attribute of a column definition, NULL if the
// column is a normal column
const ParseNode *ai_column_find_attr(const ParseNode *column_def);

// parse "WITH (sync_mode = 'async', max_retries = 2)"
int ai_column_parse_options(const ParseNode *options,
                            int64_t &max_retries,
                            common::ObString &sync_mode);

// validate the AI expression of an AI column and build the normalized expression
int ai_column_check_expr(const ParseNode *expr,
                         const ParseNode *data_type,
                         const common::ObIArray<common::ObString> &ai_column_names,
                         const bool is_json,
                         common::ObIAllocator &allocator,
                         common::ObString &normalized_expr);

// describe the AI expression of an AI column: the model name, the dimension
// of the embedding and the prompt template of the expression
int ai_column_describe_expr(const ParseNode *expr,
                            const int32_t result_type,
                            common::ObIAllocator &allocator,
                            ObAiColumnMeta &meta);

// __ai_state_<column>, __ai_request_<column>, __ai_result_spec_<column>
int ai_column_gen_hidden_name(common::ObIAllocator &allocator,
                              const common::ObString &column_name,
                              const char *prefix,
                              common::ObString &hidden_name);

// collect the primary key column names from the table element list
void ai_column_collect_pk_columns(const ParseNode *elements,
                                  common::ObIArray<common::ObString> &names);

// rewrite the references of oceanbase.__all_ai_column_task with the
// synthesized query over the tables that have AI columns
int ai_column_rewrite_task_table(const common::ObString &sql,
                                 common::ObIAllocator &allocator,
                                 common::ObString &rewritten,
                                 bool &rewritten_flag);

// start the background worker of the AI columns, at most once per process
int ai_column_worker_start_once();


// rewrite "alter table t retry ai column c where ..." into an update statement
int ai_column_rewrite_retry_stmt(const common::ObString &sql,
                                 common::ObIAllocator &allocator,
                                 common::ObString &rewritten,
                                 bool &rewritten_flag);

// rewrite "update t set c = ..." so that the AI columns that depend on the
// updated columns are invalidated
int ai_column_rewrite_update_stmt(const common::ObString &sql,
                                  common::ObIAllocator &allocator,
                                  common::ObString &rewritten,
                                  bool &rewritten_flag);

// the strict read mode, returns OB_ERR_AI_COLUMN_UNFINISHED_TASK when the
// statement reads an AI column that still has unfinished tasks
int ai_column_check_strict_read(const common::ObString &sql, const bool strict_read);

// whether the table factor refers to the system view __all_ai_column
bool ai_column_is_meta_table(const ParseNode &table_node);

}  // namespace sql
}  // namespace oceanbase

#endif  // OCEANBASE_SQL_RESOLVER_DDL_OB_AI_COLUMN_META_H_
