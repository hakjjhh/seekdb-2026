/*
 * Copyright (c) 2025 OceanBase.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "sql/engine/expr/ob_expr_tokenize.h"

#include "lib/alloc/alloc_struct.h"
#include "lib/allocator/page_arena.h"
#include "lib/charset/ob_charset.h"
#include "lib/allocator/ob_malloc.h"
#include "lib/lock/ob_mutex.h"
#include "share/ob_json_access_utils.h"
#include "common/json_type/ob_json_base.h"
#include "common/json_type/ob_json_tree.h"
#include "lib/ob_errno.h"
#include "lib/oblog/ob_log_module.h"
#include "lib/string/ob_string.h"
#include "lib/utility/ob_macro_utils.h"
#include "object/ob_object.h"
#include "sql/resolver/ddl/ob_fts_index_builder_util.h"
#include "share/ob_json_access_utils.h"
#include "data_plane/fts/ob_fts_parser_helper.h"
#include "data_plane/fts/dict/ob_gen_dic_loader.h"
#include "data_plane/fts/ob_fts_parser_property.h"

#define USING_LOG_PREFIX SQL_ENG
#include "sql/engine/expr/ob_expr_json_func_helper.h" // file not self-contained, there're logs inside.

namespace oceanbase
{
namespace sql
{
ObExprTokenize::ObExprTokenize(common::ObIAllocator &alloc)
    : ObStringExprOperator(alloc,
                           T_FUN_TOKENIZE,
                           N_TOKENIZE,
                           MORE_THAN_ZERO,
                           VALID_FOR_GENERATED_COL)
{
}

ObExprTokenize::~ObExprTokenize() {}


namespace
{
// tokenize() is a pure function of the input text, the parser name, the parser properties and
// the text collation, so thousands of identical calls (one per row of a repeated query, or one
// per duplicated document when building an index) need not re-parse the arguments, re-acquire
// the IK dictionaries and re-segment the text every time.
//
// The cache stores the raw JSON payload of finished results. A hit replays it through
// pack_json_str_res(), which is exactly what the cold path does after get_raw_binary(), so the
// emitted bytes are unchanged. Every candidate is compared byte for byte, so a hash collision
// can only cost a miss and never return a wrong answer.
class ObTokenizeResultCache
{
public:
  static const int64_t SLOT_CNT = 256;
  static const int64_t MAX_RESULT_BYTES = 64 * 1024;
  static const int64_t MAX_TOTAL_BYTES = 4 * 1024 * 1024;

  struct Entry
  {
    Entry()
      : buf_(nullptr), cap_(0), used_(0), hash_(0), collation_(0), flags_(0),
        text_len_(0), parser_len_(0), props_len_(0), result_len_(0)
    {}
    char *buf_;          // text | parser name | properties | result
    int64_t cap_;
    int64_t used_;
    uint64_t hash_;      // hash of the key the entry currently holds
    int64_t collation_;
    int64_t flags_;
    int32_t text_len_;
    int32_t parser_len_;
    int32_t props_len_;
    int32_t result_len_;
  };

  static ObTokenizeResultCache &instance()
  {
    static ObTokenizeResultCache cache;
    return cache;
  }

  int get(const int64_t collation, const int64_t flags, const ObString &text,
          const ObString &parser, const ObString &props, ObIAllocator &allocator,
          ObString &result, bool &hit)
  {
    int ret = OB_SUCCESS;
    hit = false;
    const uint64_t hash = key_hash(collation, flags, text, parser, props);
    lib::ObMutexGuard guard(mutex_);
    Entry &entry = slots_[hash % static_cast<uint64_t>(SLOT_CNT)];
    if (nullptr == entry.buf_ || entry.hash_ != hash || entry.collation_ != collation
        || entry.flags_ != flags || entry.text_len_ != text.length()
        || entry.parser_len_ != parser.length() || entry.props_len_ != props.length()
        || 0 != MEMCMP(entry.buf_, text.ptr(), text.length())
        || 0 != MEMCMP(entry.buf_ + text.length(), parser.ptr(), parser.length())
        || 0 != MEMCMP(entry.buf_ + text.length() + parser.length(), props.ptr(), props.length())) {
      // miss
    } else {
      char *copy = static_cast<char *>(allocator.alloc(entry.result_len_));
      if (OB_ISNULL(copy)) {
        ret = OB_ALLOCATE_MEMORY_FAILED;
        LOG_WARN("allocate tokenize cache result failed", K(ret), K(entry.result_len_));
      } else {
        MEMCPY(copy, entry.buf_ + entry.text_len_ + entry.parser_len_ + entry.props_len_,
               entry.result_len_);
        result.assign_ptr(copy, entry.result_len_);
        hit = true;
      }
    }
    return ret;
  }

  void put(const int64_t collation, const int64_t flags, const ObString &text,
           const ObString &parser, const ObString &props, const ObString &result)
  {
    if (result.length() <= 0 || result.length() > MAX_RESULT_BYTES) {
      return;
    }
    const int64_t need = text.length() + parser.length() + props.length() + result.length();
    const uint64_t hash = key_hash(collation, flags, text, parser, props);
    lib::ObMutexGuard guard(mutex_);
    Entry &entry = slots_[hash % static_cast<uint64_t>(SLOT_CNT)];
    if (entry.cap_ < need) {
      char *buf = static_cast<char *>(ob_malloc(need, ObMemAttr("TokCache")));
      if (nullptr == buf) {
        return;
      }
      if (nullptr != entry.buf_) {
        ob_free(entry.buf_);
      }
      entry.buf_ = buf;
      entry.cap_ = need;
    }
    total_used_ += need - entry.used_;
    entry.used_ = need;
    entry.hash_ = hash;
    entry.collation_ = collation;
    entry.flags_ = flags;
    entry.text_len_ = static_cast<int32_t>(text.length());
    entry.parser_len_ = static_cast<int32_t>(parser.length());
    entry.props_len_ = static_cast<int32_t>(props.length());
    entry.result_len_ = static_cast<int32_t>(result.length());
    MEMCPY(entry.buf_, text.ptr(), text.length());
    MEMCPY(entry.buf_ + text.length(), parser.ptr(), parser.length());
    MEMCPY(entry.buf_ + text.length() + parser.length(), props.ptr(), props.length());
    MEMCPY(entry.buf_ + text.length() + parser.length() + props.length(), result.ptr(),
           result.length());
    if (total_used_ > MAX_TOTAL_BYTES) {
      clear_all();
    }
  }

private:
  ObTokenizeResultCache() : mutex_(), total_used_(0) {}

  static uint64_t key_hash(const int64_t collation, const int64_t flags, const ObString &text,
                           const ObString &parser, const ObString &props)
  {
    uint64_t h = 1469598103934665603ULL;
    h = mix(h, &collation, sizeof(collation));
    h = mix(h, &flags, sizeof(flags));
    h = mix(h, text.ptr(), text.length());
    h = mix(h, parser.ptr(), parser.length());
    h = mix(h, props.ptr(), props.length());
    return h;
  }

  static uint64_t mix(uint64_t h, const void *data, const int64_t len)
  {
    const unsigned char *p = static_cast<const unsigned char *>(data);
    for (int64_t i = 0; i < len; ++i) {
      h ^= static_cast<uint64_t>(p[i]);
      h *= 1099511628211ULL;
    }
    return h;
  }

  void clear_all()
  {
    for (int64_t i = 0; i < SLOT_CNT; ++i) {
      if (nullptr != slots_[i].buf_) {
        ob_free(slots_[i].buf_);
        slots_[i] = Entry();
      }
    }
    total_used_ = 0;
  }

  lib::ObMutex mutex_;
  Entry slots_[SLOT_CNT];
  int64_t total_used_;
};

// Reads one tokenize() argument as raw text. kind: 0 absent, 1 NULL, 2 present.
int read_tokenize_arg(const ObExpr &expr, ObEvalCtx &ctx, const int64_t idx,
                      ObString &value, int64_t &kind)
{
  int ret = OB_SUCCESS;
  ObDatum *datum = nullptr;
  value = ObString();
  kind = 0;
  if (idx < expr.arg_cnt_) {
    if (OB_ISNULL(expr.args_[idx])) {
      ret = OB_ERR_UNEXPECTED;
    } else if (OB_FAIL(expr.args_[idx]->eval(ctx, datum))) {
    } else if (nullptr == datum || datum->is_null()) {
      kind = 1;
    } else {
      kind = 2;
      value = datum->get_string();
    }
  }
  return ret;
}
} // namespace

int ObExprTokenize::eval_tokenize(const ObExpr &expr, ObEvalCtx &ctx, ObDatum &expr_datum)
{
  int ret = OB_SUCCESS;

  ObEvalCtx::TempAllocGuard tmp_alloc_g(ctx);
  common::ObArenaAllocator &temp_allocator = tmp_alloc_g.get_allocator();

  ObIJsonBase *json_result = nullptr;
  TokenizeParam param;
  ObString raw_text;
  ObString raw_parser;
  ObString raw_props;
  ObString cached_result;
  int64_t text_kind = 0;
  int64_t parser_kind = 0;
  int64_t props_kind = 0;
  int64_t collation = 0;
  bool cacheable = false;
  bool cache_hit = false;

  // check param num, which is checked in ObExprOperator::calc_result_typeN.
  if (OB_UNLIKELY(expr.arg_cnt_ < 1 || expr.arg_cnt_ > 3)) {
    ret = OB_INVALID_ARGUMENT;
  } else if (OB_FAIL(read_tokenize_arg(expr, ctx, 0, raw_text, text_kind))) {
  } else if (OB_FAIL(read_tokenize_arg(expr, ctx, 1, raw_parser, parser_kind))) {
  } else if (OB_FAIL(read_tokenize_arg(expr, ctx, 2, raw_props, props_kind))) {
  } else {
    // Only the no-user-properties shape is memoised: the property argument would have to be
    // part of the key as serialised JSON, and skipping it keeps the cache trivially safe.
    cacheable = (2 != props_kind);
    collation = static_cast<int64_t>(expr.args_[0]->obj_meta_.get_collation_type());
    if (cacheable && OB_FAIL(ObTokenizeResultCache::instance().get(
                                 collation, text_kind | (parser_kind << 4),
                                 raw_text, raw_parser, raw_props, temp_allocator,
                                 cached_result, cache_hit))) {
      // A cache failure must never fail the statement.
      LOG_WARN("tokenize cache lookup failed, tokenizing anyway", K(ret));
      ret = OB_SUCCESS;
      cache_hit = false;
    }
  }

  if (OB_FAIL(ret)) {
  } else if (cache_hit) {
    if (OB_FAIL(ObJsonExprHelper::pack_json_str_res(expr, ctx, expr_datum, cached_result))) {
    }
  } else if (OB_FAIL(parse_param(expr, ctx, temp_allocator, param))) {
  } else if (OB_FAIL(tokenize_fulltext(param, param.output_mode_, temp_allocator, json_result))) {
  } else {
    if (cacheable && nullptr != json_result && !json_result->is_bin()) {
      ObString raw_result;
      if (OB_SUCCESS
          == share::ObJsonWrapper::get_raw_binary(json_result, raw_result, &temp_allocator)) {
        ObTokenizeResultCache::instance().put(
            collation, text_kind | (parser_kind << 4), raw_text, raw_parser, raw_props,
            raw_result);
      }
    }
    if (OB_FAIL(ObJsonExprHelper::pack_json_res(expr,
                                                ctx,
                                                temp_allocator,
                                                json_result,
                                                expr_datum))) {
    }
  }

  return ret;
}

int ObExprTokenize::tokenize_fulltext(const TokenizeParam &param,
                                      TokenizeParam::OUTPUT_MODE mode,
                                      ObIAllocator &allocator,
                                      ObIJsonBase *&result)
{
  int ret = OB_SUCCESS;
  storage::ObFTParseHelper tokenize_helper;
  const int64_t ft_word_bkt_cnt = MIN(MAX(param.fulltext_.length() / 2, 2), 997);
  int64_t doc_len = 0;
  storage::ObFTWordMap token_map;

  if (TokenizeParam::OUTPUT_MODE::DEFAULT != mode && TokenizeParam::OUTPUT_MODE::ALL != mode) {
    ret = OB_INVALID_ARGUMENT;
  } else if (OB_FAIL(tokenize_helper.init(&allocator, param.parser_name_, param.properties_))) {
  } else if (OB_FAIL(token_map.create(ft_word_bkt_cnt, common::ObMemAttr("FTWordMap")))) {
  } else if (
      (0 != param.fulltext_.length())
      && OB_FAIL(tokenize_helper.segment(
                     param.meta_,
                     param.fulltext_.ptr(),
                     param.fulltext_.length(),
                     doc_len,
                     token_map))) {
  } else {
    switch (param.output_mode_) {
    case TokenizeParam::OUTPUT_MODE::DEFAULT: {
      if (OB_FAIL(tokenize_helper.make_token_array_json(token_map, result))) {
      } else {
        // pass
      }
      break;
    }
    case TokenizeParam::OUTPUT_MODE::ALL: {
      if (OB_FAIL(tokenize_helper.make_detail_json(token_map, doc_len, result))) {
      } else {
        // pass
      }
      break;
    }
    default:
      ret = OB_INVALID_ARGUMENT;
    }
  }
  return ret;
}

ObExprTokenize::TokenizeParam ::TokenizeParam()
  : allocator_(ObMemAttr("TokenizeParam")),
    parser_name_(ObString(OB_DEFAULT_FULLTEXT_PARSER_NAME)),
    meta_(),
    fulltext_(),
    output_mode_(OUTPUT_MODE::DEFAULT)
{
}

int ObExprTokenize::TokenizeParam::parse_json_param(const ObIJsonBase *obj)
{
  int ret = OB_SUCCESS;
  ObString str;
  ObIJsonBase *val;

  if (OB_UNLIKELY(nullptr == obj)) {
    ret = OB_INVALID_ARGUMENT;
  } else if (ObJsonNodeType::J_OBJECT != obj->json_type()) {
    ret = OB_INVALID_ARGUMENT;
  } else if (obj->element_count() == 0) {
    // no data
  } else if (OB_FAIL(obj->get_object_value(0, str, val))) {
  } else if (0 == str.case_compare(CASE_INDICATOR_STR)) {
    if (ObJsonNodeType::J_STRING != val->json_type()) {
      ret = OB_INVALID_ARGUMENT;
    } else if (0 == ObString(val->get_data_length(), val->get_data()).case_compare("UPPER")) {
    } else if (0 == ObString(val->get_data_length(), val->get_data()).case_compare("LOWER")) {
    } else {
      ret = OB_INVALID_ARGUMENT;
    }
    ret = OB_NOT_SUPPORTED;
    LOG_USER_ERROR(OB_NOT_SUPPORTED, "case indentifier");
  } else if (0 == str.case_compare(OUTPUT_MODE_STR)) {
    if (ObJsonNodeType::J_STRING != val->json_type()) {
      ret = OB_INVALID_ARGUMENT;
      LOG_USER_ERROR(OB_INVALID_ARGUMENT, "output mode should be string default or all");
    } else if (0 == ObString(val->get_data_length(), val->get_data()).case_compare("DEFAULT")) {
      output_mode_ = DEFAULT;
    } else if (0 == ObString(val->get_data_length(), val->get_data()).case_compare("ALL")) {
      output_mode_ = ALL;
    } else {
      ret = OB_INVALID_ARGUMENT;
      LOG_USER_ERROR(OB_INVALID_ARGUMENT, "output mode should be string default or all");
    }
  } else if (0 == str.case_compare(STOPWORDS_LIST_STR)) {
    ret = OB_NOT_SUPPORTED;
    LOG_USER_ERROR(OB_NOT_SUPPORTED, "stopwords");
  } else if (0 == str.case_compare(ADDITIONAL_ARGS_STR)) {
    if (ObJsonNodeType::J_ARRAY != val->json_type()) {
      ret = OB_INVALID_ARGUMENT;
      LOG_USER_ERROR(OB_INVALID_ARGUMENT, "parser arguments");
    } else {
      ObString json_str;
      if (OB_FAIL(storage::ObFTParserJsonProps::tokenize_array_to_props_json(allocator_, val, json_str))) {
        ObSqlString message;
        message.append_fmt("format in %s form", ADDITIONAL_ARGS_STR);
        LOG_USER_ERROR(OB_INVALID_ARGUMENT, message.ptr());
      } else {
        properties_ = json_str;
      }
    }
  } else {
    ret = OB_INVALID_ARGUMENT;
    ObSqlString message;
    message.append_fmt("config: %s", str.ptr());
    LOG_USER_ERROR(OB_INVALID_ARGUMENT, message.ptr());
  }
  return ret;
}

int ObExprTokenize::parse_param(const ObExpr &expr,
                                ObEvalCtx &ctx,
                                common::ObArenaAllocator &allocator,
                                TokenizeParam &param)
{
  int ret = OB_SUCCESS;

  ObDatum *parser_params_datum;
  ObString raw_parser_name = ObString::make_string(OB_DEFAULT_FULLTEXT_PARSER_NAME);

  ObEvalCtx::TempAllocGuard tmp_alloc_g(ctx);
  
  MultimodeAlloctor temp_allocator(tmp_alloc_g.get_allocator());

  if (OB_UNLIKELY(expr.arg_cnt_ < 1 || expr.arg_cnt_ > 3)) {
    ret = OB_INVALID_ARGUMENT;
  } else if (OB_FAIL(parse_fulltext(expr, ctx, param))) {
  } else if (OB_FAIL(parse_parser_name(expr, ctx, param))) {
  } else if (OB_FAIL(parse_parser_properties(expr, ctx, temp_allocator, param))) {
  } else if (OB_FAIL(param.reform_parser_properties(param.properties_))) {
  } else if (OB_FAIL(param.try_load_dictionary_for_ik())) {
  }
  return ret;
}

int ObExprTokenize::construct_ft_parser_inner_name(const ObString &input_str, TokenizeParam &param)
{
  int ret = OB_SUCCESS;
  storage::ObFTParser parser;

  char *parser_name_buf = nullptr;
  if (OB_ISNULL(parser_name_buf
                = static_cast<char *>(param.allocator_.alloc(storage::OB_FT_PARSER_NAME_LENGTH)))) {
    ret = OB_ALLOCATE_MEMORY_FAILED;
  } else if (OB_FAIL(parser.init(input_str))) {
  } else if (OB_FAIL(parser.serialize_to_str(parser_name_buf, storage::OB_FT_PARSER_NAME_LENGTH))) {
  } else {
    param.parser_name_ = ObString::make_string(parser_name_buf);
  }
  return ret;
}

int ObExprTokenize::calc_result_typeN(ObExprResType &type,
                                      ObExprResType *types,
                                      int64_t param_num,
                                      common::ObExprTypeCtx &type_ctx) const
{
  int ret = OB_SUCCESS;

  if (OB_UNLIKELY(param_num < 1 || param_num > 3)) {
    ret = OB_ERR_PARAM_SIZE;
    ObString expr_name(N_TOKENIZE);
    LOG_USER_ERROR(OB_ERR_PARAM_SIZE, expr_name.length(), expr_name.ptr());
  } else {
    // just okay
  }

  ObLength length = ObAccuracy::DDL_DEFAULT_ACCURACY[ObJsonType].get_length();

  if (OB_SUCC(ret)) {
    // set res type
    type.set_json();
    type.set_length(length); // keep consistent with other json expr, maybe calc it later.

    // param type set, skip charset after first param
    for (int64_t i = 1; OB_SUCC(ret) && i < param_num; ++i) {
      if (ob_is_string_type(types[i].get_type())) {
        if (types[i].get_charset_type() != CHARSET_UTF8MB4) {
          types[i].set_calc_collation_type(CS_TYPE_UTF8MB4_BIN);
        }
      }
    }

    // handle param
    if (param_num >= 2) {
      types[1].set_varchar();
    }

    if (param_num >= 3) {
      if (OB_FAIL(ObJsonExprHelper::is_valid_for_json(types, 2, N_TOKENIZE))) {
      }
    }
  }

  return ret;
}

int ObExprTokenize::cg_expr(ObExprCGCtx &op_cg_ctx,
                            const ObRawExpr &raw_expr,
                            ObExpr &rt_expr) const
{
  int ret = OB_SUCCESS;
  CK((rt_expr.arg_cnt_ >= 1 && rt_expr.arg_cnt_ <= 3));
  if (OB_SUCC(ret)) {
    // do register
    rt_expr.eval_func_ = eval_tokenize;
  }
  return ret;
}

int ObExprTokenize::parse_fulltext(const ObExpr &expr, ObEvalCtx &ctx, TokenizeParam &param)
{
  int ret = OB_SUCCESS;

  ObDatum *fulltext_datum;

  if (OB_FAIL(expr.args_[0]->eval(ctx, fulltext_datum))) {
  } else {
    if (fulltext_datum->is_null()) {
      // do nothing, return empty result
      param.fulltext_ = ObString::make_empty_string();
    } else {
      param.fulltext_ = fulltext_datum->get_string();
    }
    param.meta_.set_varchar(); // as we hardcoded in fts_index
    param.meta_.set_collation_type(expr.args_[0]->obj_meta_.get_collation_type());
  }
  return ret;
}

int ObExprTokenize::parse_parser_name(const ObExpr &expr, ObEvalCtx &ctx, TokenizeParam &param)
{
  int ret = OB_SUCCESS;
  ObDatum *parser_datum = nullptr;
  ObString raw_parser_name;

  if (expr.arg_cnt_ < 2) {
    raw_parser_name = ObString::make_string(OB_DEFAULT_FULLTEXT_PARSER_NAME);
  } else if (OB_FAIL(expr.args_[1]->eval(ctx, parser_datum))) {
  } else {
    if (parser_datum->is_null()) {
      raw_parser_name = ObString::make_string(OB_DEFAULT_FULLTEXT_PARSER_NAME);
    } else {
      ObString name = parser_datum->get_string();
      raw_parser_name = name.trim();
    }
  }

  if (OB_FAIL(ret)) {
    // already logged
  } else if (OB_FAIL(construct_ft_parser_inner_name(raw_parser_name, param))) {
  }

  return ret;
}

int ObExprTokenize::parse_parser_properties(const ObExpr &expr,
                                            ObEvalCtx &ctx,
                                            MultimodeAlloctor &mm_alloc,
                                            TokenizeParam &param)
{
  int ret = OB_SUCCESS;
  ObIJsonBase *base = nullptr;

  if (expr.arg_cnt_ < 3) {
    // do nothing
  } else {
    bool is_null = false;
    if (OB_FAIL(ObJsonExprHelper::get_json_doc(expr, ctx, mm_alloc, 2, base, is_null))) {
    } else {
      if (ObJsonNodeType::J_ARRAY != base->json_type()) {
        ret = OB_INVALID_ARGUMENT;
        LOG_USER_ERROR(OB_INVALID_ARGUMENT, "parser args should be in an array.");
      } else {
        for (uint64_t i = 0; OB_SUCC(ret) && i < base->element_count(); ++i) {
          ObIJsonBase *node = nullptr;
          if (OB_FAIL(base->get_array_element(i, node))) {
          } else if (ObJsonNodeType::J_OBJECT != (node->json_type())) {
            ret = OB_INVALID_ARGUMENT;
          } else if (OB_FAIL(param.parse_json_param(node))) {
          }
        } // for
      }
    }
  }

  return ret;
}

int ObExprTokenize::TokenizeParam::reform_parser_properties(const ObString &properties)
{
  int ret = OB_SUCCESS;
  storage::ObFTParserJsonProps parser_properties;

  if (properties.empty()) {
    // No user supplied parser properties: the built-in defaults depend only on the parser
    // name, and ObFTParseHelper::init applies exactly the same defaults for an empty
    // property string, so the JSON parse/rebuild/serialize round trip is pure overhead.
    properties_.reset();
  } else if (OB_FAIL(parser_properties.init())) {
  } else if (OB_FAIL(parser_properties.parse_from_valid_str(properties))) {
    LOG_USER_ERROR(OB_INVALID_ARGUMENT, "parser properties invalid.");
  } else if (OB_FAIL(parser_properties.rebuild_props_for_ddl(parser_name_,
                                                             ObCollationType::CS_TYPE_UTF8MB4_BIN,
                                                             true))) {
  } else if (OB_FAIL(parser_properties.to_format_json(allocator_, properties_))) {
  }

  return ret;
}

int ObExprTokenize::TokenizeParam::try_load_dictionary_for_ik()
{
  int ret = OB_SUCCESS;
  bool need_to_load_dic = false;
  storage::ObDicLoaderHandle dic_loader_handle;
  if (OB_FAIL(share::ObFtsIndexBuilderUtil::check_need_to_load_dic(
          parser_name_, need_to_load_dic))) {
  } else if (need_to_load_dic) {
    if (OB_FAIL(storage::ObGenDicLoader::get_instance().get_dic_loader(
                    ObString::make_string(storage::ObFTSLiteral::PARSER_NAME_IK), // currently only ik, use parser_name_ without version suffix
                    ObCharset::charset_type_by_coll(meta_.get_collation_type()),
                    dic_loader_handle))) {
    } else if (OB_UNLIKELY(!dic_loader_handle.is_valid())) {
      ret = OB_ERR_UNEXPECTED;
    } else if (OB_FAIL(dic_loader_handle.get_loader()->try_load_dictionary_in_trans())) {
    }
  }
  return ret;
}

} // namespace sql
} // namespace oceanbase
