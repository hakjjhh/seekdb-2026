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

#define USING_LOG_PREFIX STORAGE

#include "ob_text_daat_iter.h"

namespace oceanbase
{
namespace storage
{
int ObTextDaaTIter::init(const ObTextDaaTParam &param)
{
  int ret = OB_SUCCESS;
  if (OB_ISNULL(param.base_param_) || OB_ISNULL(param.dim_iters_) || OB_ISNULL(param.allocator_)
      || OB_ISNULL(param.relevance_collector_)) {
    ret = OB_INVALID_ARGUMENT;
  } else if (OB_FAIL(ObSRDaaTIterImpl::init(*param.base_param_, *param.dim_iters_,
                                            *param.allocator_, *param.relevance_collector_))) {
  } else if (param.bm25_param_est_ctx_.is_valid()
             && OB_FAIL(bm25_param_estimator_.init(param.bm25_param_est_ctx_))) {
  } else {
    mode_flag_ = param.mode_flag_;
    function_lookup_mode_ = param.function_lookup_mode_;
  }
  return ret;
}

void ObTextDaaTIter::reset()
{
  for (int64_t i = 0; i < SHORT_UNION_MAX_DIMS; ++i) {
    presence_batches_[i].reset();
  }
  bm25_param_estimator_.reset();
  // this class can know the type of dim_iters_, but ObSRDaaTIterImpl maybe not
  if (OB_NOT_NULL(dim_iters_)) {
    for (int64_t i = 0; i < dim_iters_->count(); ++i) {
      static_cast<ObTextRetrievalDaaTTokenIter *>(dim_iters_->at(i))->reset();
    }
  }
  ObSRDaaTIterImpl::reset();
}

void ObTextDaaTIter::reuse(const bool switch_tablet)
{
  for (int64_t i = 0; i < SHORT_UNION_MAX_DIMS; ++i) {
    presence_batches_[i].reset();
  }
  if ((OB_NOT_NULL(dim_iters_) && 0 == dim_iters_->count())) {
    // do nothing
  } else {
    bm25_param_estimator_.reuse(switch_tablet);
  }
  if (OB_NOT_NULL(dim_iters_)) {
    for (int64_t i = 0; i < dim_iters_->count(); ++i) {
      static_cast<ObTextRetrievalDaaTTokenIter *>(dim_iters_->at(i))->reuse();
    }
  }
  ObSRDaaTIterImpl::reuse(switch_tablet);
}

int ObTextDaaTIter::pre_process()
{
  int ret = OB_SUCCESS;
  if (dim_iters_->count() == 0) {
    ret = OB_ITER_END;
  } else if (iter_param_->need_project_relevance() && bm25_param_estimator_.is_inited()) {
    if (OB_FAIL(bm25_param_estimator_.do_estimation(*iter_param_->eval_ctx_))) {
    }
  }
  return ret;
}

int ObTextDaaTIter::get_next_rows(const int64_t capacity, int64_t &count)
{
  const ObDatumMeta *meta = is_inited_ ? &iter_param_->id_proj_expr_->datum_meta_ : nullptr;
  // Small natural-language unions whose result carries no relevance value can
  // merge the children's ID batches directly. Scored queries keep the generic
  // path so BM25 accumulation order (and therefore the exact float result) and
  // the on-demand filter/threshold behaviour are untouched.
  if (nullptr != meta
      && mode_flag_ == sql::ObMatchAgainstMode::NATURAL_LANGUAGE_MODE
      && !function_lookup_mode_
      && dim_iters_->count() > 0
      && dim_iters_->count() <= SHORT_UNION_MAX_DIMS
      && nullptr == iter_param_->relevance_expr_
      && iter_param_->eval_ctx_->is_vectorized()
      && iter_param_->id_proj_expr_->is_batch_result()
      && (meta->type_ == common::ObUInt64Type
          || (meta->type_ == common::ObVarcharType
              && meta->cs_type_ == common::CS_TYPE_BINARY))) {
    return get_presence_union_rows(capacity, count);
  }
  return ObSRDaaTIterImpl::get_next_rows(capacity, count);
}

int ObTextDaaTIter::get_presence_union_rows(const int64_t capacity, int64_t &count)
{
  int ret = OB_SUCCESS;
  count = 0;
  const int64_t dim_count = dim_iters_->count();
  const int64_t real_capacity = MIN(capacity, iter_param_->max_batch_size_);
  const common::ObLimitParam &limit = *iter_param_->limit_param_;
  const bool integer_id = iter_param_->id_proj_expr_->datum_meta_.type_ == common::ObUInt64Type;
  int64_t equal_heads[SHORT_UNION_MAX_DIMS];
  ObDatum *output_ids = iter_param_->id_proj_expr_->locate_batch_datums(*iter_param_->eval_ctx_);
  bool batch_boundary = false;
  bool scan_datums_restored = false;
  const auto cache_id = [&](const ObDatum &id) {
    ++input_row_cnt_;
    if (limit.is_valid() && input_row_cnt_ <= limit.offset_) {
      // OFFSET consumes IDs but does not retain their buffer references.
    } else {
      OB_ASSERT(count < real_capacity);
      output_ids[count++].set_datum(id);
      ++output_row_cnt_;
      if (limit.is_valid() && output_row_cnt_ >= limit.limit_) {
        ret = OB_ITER_END;
      }
    }
  };
  const auto compare_id = [integer_id, &ret](const ObDatum &left, const ObDatum &right) {
    if (integer_id) {
      return left.get_uint64() < right.get_uint64() ? -1
          : (left.get_uint64() > right.get_uint64() ? 1 : 0);
    }
    int comparison = 0;
    // Only non-null stored IDs with binary semantics reach here. The helper
    // keeps extend-datum handling and variable-length ordering intact.
    const int cmp_ret = compare_binary_retrieval_id(left, right, comparison, nullptr);
    if (OB_UNLIKELY(OB_SUCCESS != cmp_ret)) {
      ret = cmp_ret;
    }
    return comparison;
  };
  if (limit.is_valid() && output_row_cnt_ >= limit.limit_) {
    ret = OB_ITER_END;
  }
  while (OB_SUCC(ret) && !batch_boundary && count < real_capacity) {
    const ObDatum *minimum = nullptr;
    const ObDatum *second = nullptr;
    int64_t equal_count = 0;
    for (int64_t i = 0; OB_SUCC(ret) && i < dim_count; ++i) {
      PresenceIdBatch &batch = presence_batches_[i];
      if (!batch.ended_ && batch.pos_ == batch.count_) {
        // Returned rows alias these child-owned buffers. Once this output batch
        // retains any ID, finish it before allowing another child read so all
        // borrowed IDs stay valid until the next get_next_rows call.
        if (count > 0) {
          batch_boundary = true;
          break;
        }
        if (!scan_datums_restored) {
          // The projection and every inverted-index scan share this expression.
          // A previous output may have pointed its datums into one child's ID
          // cache, so detach all scan slots before a storage reader writes to
          // one, or that reader could overwrite another child's head.
          iter_param_->id_proj_expr_->locate_datums_for_update(
              *iter_param_->eval_ctx_, iter_param_->max_batch_size_);
          scan_datums_restored = true;
        }
        auto *token = static_cast<ObTextRetrievalDaaTTokenIter *>(dim_iters_->at(i));
        if (OB_FAIL(token->get_next_doc_id_batch(batch.ids_, batch.count_))) {
          if (OB_ITER_END == ret) {
            batch.ended_ = true;
            ret = OB_SUCCESS;
          }
        }
        batch.pos_ = 0;
      }
      if (OB_FAIL(ret) || batch.ended_) {
      } else {
        const ObDatum &head = batch.ids_[batch.pos_].get_datum();
        if (OB_UNLIKELY(head.is_null())) {
          ret = OB_ERR_UNEXPECTED;
        } else {
          const int comparison = nullptr == minimum ? -1 : compare_id(head, *minimum);
          if (comparison < 0) {
            second = minimum;
            minimum = &head;
            equal_count = 0;
          } else if (comparison > 0
              && (nullptr == second || compare_id(head, *second) < 0)) {
            second = &head;
          }
          if (comparison <= 0) {
            equal_heads[equal_count++] = i;
          }
        }
      }
    }
    if (OB_FAIL(ret) || batch_boundary) {
    } else if (nullptr == minimum) {
      ret = OB_ITER_END;
    } else if (equal_count > 1) {
      cache_id(*minimum);
      for (int64_t i = 0; i < equal_count; ++i) {
        ++presence_batches_[equal_heads[i]].pos_;
      }
    } else {
      PresenceIdBatch &batch = presence_batches_[equal_heads[0]];
      // The head was already compared against every active dimension and is
      // strictly before second, so consume it here then drain the rest of this
      // sorted run up to the second head; overlaps are deduplicated above.
      cache_id(*minimum);
      ++batch.pos_;
      while (OB_SUCC(ret) && count < real_capacity && batch.pos_ < batch.count_) {
        const ObDatum &id = batch.ids_[batch.pos_].get_datum();
        if (OB_UNLIKELY(id.is_null())) {
          ret = OB_ERR_UNEXPECTED;
        } else {
          const int comparison = nullptr == second ? -1 : compare_id(id, *second);
          if (OB_FAIL(ret)) {
          } else if (comparison >= 0) {
            break;
          } else {
            cache_id(id);
            ++batch.pos_;
          }
        }
      }
    }
  }
  if ((OB_SUCC(ret) || OB_ITER_END == ret) && count > 0) {
    project_presence_results(count);
    ret = OB_SUCCESS;
  }
  return ret;
}

void ObTextDaaTIter::project_presence_results(const int64_t count)
{
  ObEvalCtx &eval_ctx = *iter_param_->eval_ctx_;
  ObExpr &id_expr = *iter_param_->id_proj_expr_;
  id_expr.get_evaluated_flags(eval_ctx).set_all(count);
  id_expr.set_evaluated_projected(eval_ctx);
  if (iter_param_->need_project_relevance()) {
    ObExpr &presence_expr = *iter_param_->relevance_proj_expr_;
    const int64_t size = presence_expr.is_batch_result() ? count : 1;
    // set_double writes through datum.ptr_; locate_batch_datums alone does not
    // initialise it, so restore this expression's buffers first.
    ObDatum *presence_datums = presence_expr.locate_datums_for_update(eval_ctx, size);
    for (int64_t i = 0; i < size; ++i) {
      presence_datums[i].set_double(1.0);
    }
    if (presence_expr.is_batch_result()) {
      presence_expr.get_evaluated_flags(eval_ctx).set_all(count);
    }
    presence_expr.set_evaluated_projected(eval_ctx);
  }
}

ObTextBMWIter::ObTextBMWIter()
  : ObSRBMWIterImpl(),
    bm25_param_estimator_() {}

void ObTextBMWIter::reuse(const bool switch_tablet)
{
  bm25_param_estimator_.reuse(switch_tablet);
  if (OB_NOT_NULL(dim_iters_)) {
    for (int64_t i = 0; i < dim_iters_->count(); ++i) {
      static_cast<ObTextRetrievalBlockMaxIter *>(dim_iters_->at(i))->reuse();
    }
  }
  ObSRBMWIterImpl::reuse(switch_tablet);
}

void ObTextBMWIter::reset()
{
  bm25_param_estimator_.reset();
  if (OB_NOT_NULL(dim_iters_)) {
    for (int64_t i = 0; i < dim_iters_->count(); ++i) {
      static_cast<ObTextRetrievalBlockMaxIter *>(dim_iters_->at(i))->reset();
    }
  }
  ObSRBMWIterImpl::reset();
}

int ObTextBMWIter::init(const ObTextDaaTParam &param)
{
  int ret = OB_SUCCESS;
  if (OB_ISNULL(param.base_param_) || OB_ISNULL(param.dim_iters_) || OB_ISNULL(param.allocator_)
      || OB_ISNULL(param.relevance_collector_)) {
    ret = OB_INVALID_ARGUMENT;
  } else if (OB_FAIL(ObSRBMWIterImpl::init(*param.base_param_, *param.dim_iters_,
                                           *param.allocator_, *param.relevance_collector_))) {
  } else if (param.bm25_param_est_ctx_.is_valid()
             && OB_FAIL(bm25_param_estimator_.init(param.bm25_param_est_ctx_))) {
  }
  return ret;
}

int ObTextBMWIter::get_next_rows(const int64_t capacity, int64_t &count)
{
  int ret = OB_SUCCESS;
  if (IS_NOT_INIT) {
    ret = OB_NOT_INIT;
  } else if (dim_iters_->count() == 0) {
    ret = OB_ITER_END;
  } else if (bm25_param_estimator_.is_inited()
             && OB_FAIL(bm25_param_estimator_.do_estimation(*iter_param_->eval_ctx_))) {
  }

  if (FAILEDx(ObSRBMWIterImpl::get_next_rows(capacity, count))) {
    if (OB_UNLIKELY(OB_ITER_END != ret)) {
    }
  }
  return ret;
}

int ObTextBMWIter::init_before_wand_process()
{
  int ret = OB_SUCCESS;
  if (OB_UNLIKELY(!bm25_param_estimator_.is_estimated())) {
    ret = OB_ERR_UNEXPECTED;
  }

  for (int64_t i = 0; OB_SUCC(ret) && i < dim_iters_->count(); ++i) {
    ObTextRetrievalBlockMaxIter *block_max_iter = static_cast<ObTextRetrievalBlockMaxIter *>(dim_iters_->at(i));
    if (OB_FAIL(block_max_iter->init_block_max_iter(
        bm25_param_estimator_.get_total_doc_cnt(), bm25_param_estimator_.get_avg_doc_token_cnt()))) {
    }
  }
  return ret;
}

} // namespace storage
} // namespace oceanbase
