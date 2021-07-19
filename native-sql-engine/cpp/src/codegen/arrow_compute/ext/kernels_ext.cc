/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "codegen/arrow_compute/ext/kernels_ext.h"

#include <arrow/array.h>
#include <arrow/array/builder_binary.h>
#include <arrow/array/builder_primitive.h>
#include <arrow/array/concatenate.h>
#include <arrow/compute/api.h>
#include <arrow/compute/kernel.h>
#include <arrow/pretty_print.h>
#include <arrow/status.h>
#include <arrow/type.h>
#include <arrow/type_fwd.h>
#include <arrow/type_traits.h>
#include <arrow/util/bit_util.h>
#include <arrow/util/checked_cast.h>
#include <arrow/visitor_inline.h>
#include <dlfcn.h>
#include <gandiva/configuration.h>
#include <gandiva/node.h>
#include <gandiva/projector.h>
#include <gandiva/tree_expr_builder.h>

#include <chrono>
#include <cstring>
#include <fstream>
#include <iostream>
#include <unordered_map>

#include "codegen/arrow_compute/ext/actions_impl.h"
#include "codegen/arrow_compute/ext/array_item_index.h"
#include "codegen/arrow_compute/ext/codegen_common.h"
//#include "codegen/arrow_compute/ext/codegen_node_visitor.h"
#include "third_party/arrow/utils/hashing.h"
#include "utils/macros.h"

namespace sparkcolumnarplugin {
namespace codegen {
namespace arrowcompute {
namespace extra {

using ArrayList = std::vector<std::shared_ptr<arrow::Array>>;

///////////////  EncodeArray  ////////////////
class EncodeArrayKernel::Impl {
 public:
  Impl() {}
  virtual ~Impl() {}
  virtual arrow::Status Evaluate(const std::shared_ptr<arrow::Array>& in,
                                 std::shared_ptr<arrow::Array>* out) = 0;
};

template <typename InType, typename MemoTableType>
class EncodeArrayTypedImpl : public EncodeArrayKernel::Impl {
 public:
  EncodeArrayTypedImpl(arrow::compute::ExecContext* ctx) : ctx_(ctx) {
    hash_table_ = std::make_shared<MemoTableType>(ctx_->memory_pool());
    builder_ = std::make_shared<arrow::Int32Builder>(ctx_->memory_pool());
  }
  arrow::Status Evaluate(const std::shared_ptr<arrow::Array>& in,
                         std::shared_ptr<arrow::Array>* out) {
    // arrow::Datum input_datum(in);
    // RETURN_NOT_OK(arrow::compute::Group<InType>(ctx_, input_datum,
    // hash_table_, out)); we should put items into hashmap
    builder_->Reset();
    auto typed_array = std::dynamic_pointer_cast<ArrayType>(in);
    auto insert_on_found = [this](int32_t i) { builder_->Append(i); };
    auto insert_on_not_found = [this](int32_t i) { builder_->Append(i); };

    int cur_id = 0;
    int memo_index = 0;
    if (typed_array->null_count() == 0) {
      for (; cur_id < typed_array->length(); cur_id++) {
        hash_table_->GetOrInsert(typed_array->GetView(cur_id), insert_on_found,
                                 insert_on_not_found, &memo_index);
      }
    } else {
      for (; cur_id < typed_array->length(); cur_id++) {
        if (typed_array->IsNull(cur_id)) {
          hash_table_->GetOrInsertNull(insert_on_found, insert_on_not_found);
        } else {
          hash_table_->GetOrInsert(typed_array->GetView(cur_id), insert_on_found,
                                   insert_on_not_found, &memo_index);
        }
      }
    }
    RETURN_NOT_OK(builder_->Finish(out));
    return arrow::Status::OK();
  }

 private:
  using ArrayType = typename arrow::TypeTraits<InType>::ArrayType;
  arrow::compute::ExecContext* ctx_;
  std::shared_ptr<MemoTableType> hash_table_;
  std::shared_ptr<arrow::Int32Builder> builder_;
};

arrow::Status EncodeArrayKernel::Make(arrow::compute::ExecContext* ctx,
                                      std::shared_ptr<KernalBase>* out) {
  *out = std::make_shared<EncodeArrayKernel>(ctx);
  return arrow::Status::OK();
}

EncodeArrayKernel::EncodeArrayKernel(arrow::compute::ExecContext* ctx) {
  ctx_ = ctx;
  kernel_name_ = "EncodeArrayKernel";
}

#define PROCESS_SUPPORTED_TYPES(PROCESS) \
  PROCESS(arrow::BooleanType)            \
  PROCESS(arrow::UInt8Type)              \
  PROCESS(arrow::Int8Type)               \
  PROCESS(arrow::UInt16Type)             \
  PROCESS(arrow::Int16Type)              \
  PROCESS(arrow::UInt32Type)             \
  PROCESS(arrow::Int32Type)              \
  PROCESS(arrow::UInt64Type)             \
  PROCESS(arrow::Int64Type)              \
  PROCESS(arrow::FloatType)              \
  PROCESS(arrow::DoubleType)             \
  PROCESS(arrow::Date32Type)             \
  PROCESS(arrow::Date64Type)             \
  PROCESS(arrow::Time32Type)             \
  PROCESS(arrow::Time64Type)             \
  PROCESS(arrow::TimestampType)          \
  PROCESS(arrow::BinaryType)             \
  PROCESS(arrow::StringType)             \
  PROCESS(arrow::FixedSizeBinaryType)    \
  PROCESS(arrow::Decimal128Type)
arrow::Status EncodeArrayKernel::Evaluate(const std::shared_ptr<arrow::Array>& in,
                                          std::shared_ptr<arrow::Array>* out) {
  if (!impl_) {
    switch (in->type_id()) {
#define PROCESS(InType)                                                                \
  case InType::type_id: {                                                              \
    using MemoTableType = typename arrow::internal::HashTraits<InType>::MemoTableType; \
    impl_.reset(new EncodeArrayTypedImpl<InType, MemoTableType>(ctx_));                \
  } break;
      PROCESS_SUPPORTED_TYPES(PROCESS)
#undef PROCESS
      default: {
        std::cout << "EncodeArrayKernel type not found, type is " << in->type()
                  << std::endl;
      } break;
    }
  }
  return impl_->Evaluate(in, out);
}
#undef PROCESS_SUPPORTED_TYPES

///////////////  HashAggrArray  ////////////////
class HashArrayKernel::Impl {
 public:
  Impl(arrow::compute::ExecContext* ctx,
       std::vector<std::shared_ptr<arrow::DataType>> type_list)
      : ctx_(ctx) {
    // create a new result array type here
    std::vector<std::shared_ptr<gandiva::Node>> func_node_list = {};
    std::vector<std::shared_ptr<arrow::Field>> field_list = {};

    gandiva::ExpressionPtr expr;
    int index = 0;
    for (auto type : type_list) {
      auto field = arrow::field(std::to_string(index), type);
      field_list.push_back(field);
      auto field_node = gandiva::TreeExprBuilder::MakeField(field);
      auto func_node =
          gandiva::TreeExprBuilder::MakeFunction("hash64", {field_node}, arrow::int64());
      func_node_list.push_back(func_node);
      if (func_node_list.size() == 2) {
        auto shift_func_node = gandiva::TreeExprBuilder::MakeFunction(
            "multiply",
            {func_node_list[0], gandiva::TreeExprBuilder::MakeLiteral((int64_t)10)},
            arrow::int64());
        auto tmp_func_node = gandiva::TreeExprBuilder::MakeFunction(
            "add", {shift_func_node, func_node_list[1]}, arrow::int64());
        func_node_list.clear();
        func_node_list.push_back(tmp_func_node);
      }
      index++;
    }
    assert(func_node_list.size() > 0);
    expr = gandiva::TreeExprBuilder::MakeExpression(func_node_list[0],
                                                    arrow::field("res", arrow::int64()));
#ifdef DEBUG
    std::cout << expr->ToString() << std::endl;
#endif
    schema_ = arrow::schema(field_list);
    auto configuration = gandiva::ConfigurationBuilder().DefaultConfiguration();
    auto status = gandiva::Projector::Make(schema_, {expr}, configuration, &projector);
    pool_ = ctx_->memory_pool();
  }

  virtual ~Impl() {}

  arrow::Status Evaluate(const ArrayList& in, std::shared_ptr<arrow::Array>* out) {
    auto length = in[0]->length();
    auto num_columns = in.size();

    auto in_batch = arrow::RecordBatch::Make(schema_, length, in);
    // arrow::PrettyPrintOptions print_option(2, 500);
    // arrow::PrettyPrint(*in_batch.get(), print_option, &std::cout);

    arrow::ArrayVector outputs;
    RETURN_NOT_OK(projector->Evaluate(*in_batch, pool_, &outputs));
    *out = outputs[0];

    return arrow::Status::OK();
  }

 private:
  arrow::compute::ExecContext* ctx_;
  std::shared_ptr<gandiva::Projector> projector;
  std::shared_ptr<arrow::Schema> schema_;
  arrow::MemoryPool* pool_;
};

arrow::Status HashArrayKernel::Make(
    arrow::compute::ExecContext* ctx,
    std::vector<std::shared_ptr<arrow::DataType>> type_list,
    std::shared_ptr<KernalBase>* out) {
  *out = std::make_shared<HashArrayKernel>(ctx, type_list);
  return arrow::Status::OK();
}

HashArrayKernel::HashArrayKernel(
    arrow::compute::ExecContext* ctx,
    std::vector<std::shared_ptr<arrow::DataType>> type_list) {
  impl_.reset(new Impl(ctx, type_list));
  kernel_name_ = "HashArrayKernel";
}

arrow::Status HashArrayKernel::Evaluate(const ArrayList& in,
                                        std::shared_ptr<arrow::Array>* out) {
  return impl_->Evaluate(in, out);
}

///////////////  ConcatArray  ////////////////
class ConcatArrayKernel::Impl {
 public:
  Impl(arrow::compute::ExecContext* ctx,
       std::vector<std::shared_ptr<arrow::DataType>> type_list)
      : ctx_(ctx) {
    pool_ = ctx_->memory_pool();
    std::unique_ptr<arrow::ArrayBuilder> array_builder;
    arrow::MakeBuilder(ctx_->memory_pool(), arrow::utf8(), &array_builder);
    builder_.reset(
        arrow::internal::checked_cast<arrow::StringBuilder*>(array_builder.release()));
  }

  arrow::Status Evaluate(const ArrayList& in, std::shared_ptr<arrow::Array>* out) {
    auto length = in[0]->length();
    std::vector<std::shared_ptr<UnsafeArray>> payloads;
    int i = 0;
    for (auto arr : in) {
      std::shared_ptr<UnsafeArray> payload;
      RETURN_NOT_OK(MakeUnsafeArray(arr->type(), i++, arr, &payload));
      payloads.push_back(payload);
    }

    builder_->Reset();
    std::shared_ptr<UnsafeRow> payload = std::make_shared<UnsafeRow>(payloads.size());
    for (int i = 0; i < length; i++) {
      payload->reset();
      for (auto payload_arr : payloads) {
        RETURN_NOT_OK(payload_arr->Append(i, &payload));
      }
      RETURN_NOT_OK(builder_->Append(payload->data, (int64_t)payload->sizeInBytes()));
    }

    RETURN_NOT_OK(builder_->Finish(out));

    return arrow::Status::OK();
  }

 private:
  arrow::compute::ExecContext* ctx_;
  std::unique_ptr<arrow::StringBuilder> builder_;
  arrow::MemoryPool* pool_;
};

arrow::Status ConcatArrayKernel::Make(
    arrow::compute::ExecContext* ctx,
    std::vector<std::shared_ptr<arrow::DataType>> type_list,
    std::shared_ptr<KernalBase>* out) {
  *out = std::make_shared<ConcatArrayKernel>(ctx, type_list);
  return arrow::Status::OK();
}

ConcatArrayKernel::ConcatArrayKernel(
    arrow::compute::ExecContext* ctx,
    std::vector<std::shared_ptr<arrow::DataType>> type_list) {
  impl_.reset(new Impl(ctx, type_list));
  kernel_name_ = "ConcatArrayKernel";
  ctx_ = nullptr;
}

arrow::Status ConcatArrayKernel::Evaluate(const ArrayList& in,
                                          std::shared_ptr<arrow::Array>* out) {
  return impl_->Evaluate(in, out);
}

///////////////  ConcatArray  ////////////////
class CachedRelationKernel::Impl {
 public:
  Impl(arrow::compute::ExecContext* ctx, std::shared_ptr<arrow::Schema> result_schema,
       std::vector<std::shared_ptr<arrow::Field>> key_field_list, int result_type)
      : ctx_(ctx),
        result_schema_(result_schema),
        key_field_list_(key_field_list),
        result_type_(result_type) {
    pool_ = ctx_->memory_pool();
    for (auto field : key_field_list) {
      auto indices = GetIndicesFromSchemaCaseInsensitive(result_schema, field->name());
      if (indices.size() != 1) {
        std::cout << "[ERROR] CachedRelationKernel::Impl can't find key "
                  << field->ToString() << " from " << result_schema->ToString()
                  << std::endl;
        throw;
      }
      key_index_list_.push_back(indices[0]);
    }
    col_num_ = result_schema->num_fields();
  }

  arrow::Status Evaluate(ArrayList& in) {
    items_total_ += in[0]->length();
    length_list_.push_back(in[0]->length());
    if (cached_.size() < col_num_) {
      cached_.resize(col_num_);
    }
    for (int i = 0; i < col_num_; i++) {
      cached_[i].push_back(in[i]);
    }
    return arrow::Status::OK();
  }

  arrow::Status MakeResultIterator(std::shared_ptr<arrow::Schema> schema,
                                   std::shared_ptr<ResultIterator<SortRelation>>* out) {
    std::vector<std::shared_ptr<RelationColumn>> sort_relation_list;
    int idx = 0;
    for (auto field : result_schema_->fields()) {
      std::shared_ptr<RelationColumn> col_out;
      RETURN_NOT_OK(MakeRelationColumn(field->type()->id(), &col_out));
      if (cached_.size() == col_num_) {
        for (auto arr : cached_[idx]) {
          RETURN_NOT_OK(col_out->AppendColumn(arr));
        }
      }
      sort_relation_list.push_back(col_out);
      idx++;
    }
    std::vector<std::shared_ptr<RelationColumn>> key_relation_list;
    for (auto key_id : key_index_list_) {
      key_relation_list.push_back(sort_relation_list[key_id]);
    }
    auto sort_relation = std::make_shared<SortRelation>(
        ctx_, items_total_, length_list_, key_relation_list, sort_relation_list);
    *out = std::make_shared<SortRelationResultIterator>(sort_relation);
    return arrow::Status::OK();
  }

 private:
  int col_num_;
  int result_type_;
  arrow::MemoryPool* pool_;
  arrow::compute::ExecContext* ctx_;
  std::unique_ptr<arrow::StringBuilder> builder_;
  std::vector<std::shared_ptr<arrow::Field>> key_field_list_;
  std::shared_ptr<arrow::Schema> result_schema_;

  std::vector<int> key_index_list_;
  std::vector<int> length_list_;
  std::vector<arrow::ArrayVector> cached_;
  uint64_t items_total_ = 0;

  class SortRelationResultIterator : public ResultIterator<SortRelation> {
   public:
    SortRelationResultIterator(std::shared_ptr<SortRelation> sort_relation)
        : sort_relation_(sort_relation) {}
    arrow::Status Next(std::shared_ptr<SortRelation>* out) {
      *out = sort_relation_;
      return arrow::Status::OK();
    }

   private:
    std::shared_ptr<SortRelation> sort_relation_;
  };
};

arrow::Status CachedRelationKernel::Make(
    arrow::compute::ExecContext* ctx, std::shared_ptr<arrow::Schema> result_schema,
    std::vector<std::shared_ptr<arrow::Field>> key_field_list, int result_type,
    std::shared_ptr<KernalBase>* out) {
  *out = std::make_shared<CachedRelationKernel>(ctx, result_schema, key_field_list,
                                                result_type);
  return arrow::Status::OK();
}

CachedRelationKernel::CachedRelationKernel(
    arrow::compute::ExecContext* ctx, std::shared_ptr<arrow::Schema> result_schema,
    std::vector<std::shared_ptr<arrow::Field>> key_field_list, int result_type) {
  impl_.reset(new Impl(ctx, result_schema, key_field_list, result_type));
  kernel_name_ = "CachedRelationKernel";
}

arrow::Status CachedRelationKernel::Evaluate(ArrayList& in) {
  return impl_->Evaluate(in);
}

arrow::Status CachedRelationKernel::MakeResultIterator(
    std::shared_ptr<arrow::Schema> schema,
    std::shared_ptr<ResultIterator<SortRelation>>* out) {
  return impl_->MakeResultIterator(schema, out);
}

std::string CachedRelationKernel::GetSignature() { return ""; }

///////////////  ConcatArrayList  ////////////////
class ConcatArrayListKernel::Impl {
 public:
  Impl(arrow::compute::ExecContext* ctx,
       const std::vector<std::shared_ptr<arrow::Field>>& input_field_list,
       std::shared_ptr<gandiva::Node> root_node,
       const std::vector<std::shared_ptr<arrow::Field>>& output_field_list)
      : ctx_(ctx) {}
  virtual ~Impl() {}
  arrow::Status Evaluate(const ArrayList& in) {
    if (cached_.size() == 0) {
      for (int i = 0; i < in.size(); i++) {
        cached_.push_back({in[i]});
      }
    } else {
      for (int i = 0; i < in.size(); i++) {
        cached_[i].push_back(in[i]);
      }
    }
    total_num_row_ += in[0]->length();
    total_num_batch_ += 1;
    return arrow::Status::OK();
  }

  arrow::Status MakeResultIterator(
      std::shared_ptr<arrow::Schema> schema,
      std::shared_ptr<ResultIterator<arrow::RecordBatch>>* out) {
    *out = std::make_shared<ConcatArrayListResultIterator>(ctx_, schema, cached_);
    return arrow::Status::OK();
  }

 private:
  arrow::compute::ExecContext* ctx_;
  std::vector<arrow::ArrayVector> cached_;
  int total_num_row_ = 0;
  int total_num_batch_ = 0;
  class ConcatArrayListResultIterator : public ResultIterator<arrow::RecordBatch> {
   public:
    ConcatArrayListResultIterator(arrow::compute::ExecContext* ctx,
                                  std::shared_ptr<arrow::Schema> result_schema,
                                  const std::vector<arrow::ArrayVector>& cached)
        : ctx_(ctx), result_schema_(result_schema), cached_(cached) {
      batch_size_ = GetBatchSize();
      if (cached.size() > 0) {
        cached_num_batches_ = cached[0].size();
      }
    }

    std::string ToString() override { return "ConcatArrayListResultIterator"; }

    bool HasNext() override {
      if (cached_.size() == 0 || cur_batch_idx_ >= cached_num_batches_) {
        return false;
      }
      return true;
    }

    arrow::Status Next(std::shared_ptr<arrow::RecordBatch>* out) {
      if (cached_.size() == 0 || cur_batch_idx_ >= cached_num_batches_) {
        return arrow::Status::OK();
      }
      // check array length
      int tmp_len = 0;
      int end_arr_idx = cur_batch_idx_;
      for (int i = cur_batch_idx_; i < cached_num_batches_; i++) {
        auto arr = cached_[0][i];
        tmp_len += arr->length();
        end_arr_idx++;
        if (tmp_len > 20480) {
          break;
        }
      }
      arrow::ArrayVector concatenated_array_list;
      for (auto arr_list : cached_) {
        std::shared_ptr<arrow::Array> concatenated_array;
        arrow::ArrayVector to_be_concat_arr(arr_list.begin() + cur_batch_idx_,
                                            arr_list.begin() + end_arr_idx);

        RETURN_NOT_OK(arrow::Concatenate(to_be_concat_arr, ctx_->memory_pool(),
                                         &concatenated_array));
        concatenated_array_list.push_back(concatenated_array);
      }
      int length = concatenated_array_list[0]->length();

      *out = arrow::RecordBatch::Make(result_schema_, length, concatenated_array_list);
      cur_batch_idx_ = end_arr_idx;
      return arrow::Status::OK();
    }

   private:
    arrow::compute::ExecContext* ctx_;
    std::vector<arrow::ArrayVector> cached_;
    std::shared_ptr<arrow::Schema> result_schema_;
    int batch_size_;
    int cached_num_batches_ = 0;
    int cur_batch_idx_ = 0;
  };
};

arrow::Status ConcatArrayListKernel::Make(
    arrow::compute::ExecContext* ctx,
    const std::vector<std::shared_ptr<arrow::Field>>& input_field_list,
    std::shared_ptr<gandiva::Node> root_node,
    const std::vector<std::shared_ptr<arrow::Field>>& output_field_list,
    std::shared_ptr<KernalBase>* out) {
  *out = std::make_shared<ConcatArrayListKernel>(ctx, input_field_list, root_node,
                                                 output_field_list);
  return arrow::Status::OK();
}

ConcatArrayListKernel::ConcatArrayListKernel(
    arrow::compute::ExecContext* ctx,
    const std::vector<std::shared_ptr<arrow::Field>>& input_field_list,
    std::shared_ptr<gandiva::Node> root_node,
    const std::vector<std::shared_ptr<arrow::Field>>& output_field_list) {
  impl_.reset(new Impl(ctx, input_field_list, root_node, output_field_list));
  kernel_name_ = "ConcatArrayListKernel";
}

arrow::Status ConcatArrayListKernel::Evaluate(const ArrayList& in) {
  return impl_->Evaluate(in);
}

arrow::Status ConcatArrayListKernel::MakeResultIterator(
    std::shared_ptr<arrow::Schema> schema,
    std::shared_ptr<ResultIterator<arrow::RecordBatch>>* out) {
  return impl_->MakeResultIterator(schema, out);
}

}  // namespace extra
}  // namespace arrowcompute
}  // namespace codegen
}  // namespace sparkcolumnarplugin
