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

#include <arrow/array/concatenate.h>
#include <arrow/compute/api.h>
#include <arrow/filesystem/filesystem.h>
#include <arrow/filesystem/localfs.h>
#include <arrow/filesystem/path_util.h>
#include <arrow/io/file.h>
#include <arrow/ipc/api.h>
#include <arrow/type.h>
#include <arrow/type_fwd.h>
#include <arrow/type_traits.h>
#include <arrow/util/bitmap_ops.h>
#include <gandiva/projector.h>

#include <algorithm>
#include <boost/uuid/uuid_generators.hpp>
#include <boost/uuid/uuid_io.hpp>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <memory>
#include <numeric>
#include <vector>

#include "array_appender.h"
#include "cmp_function.h"
#include "codegen/arrow_compute/ext/array_item_index.h"
#include "codegen/arrow_compute/ext/code_generator_base.h"
#include "codegen/arrow_compute/ext/codegen_common.h"
#include "codegen/arrow_compute/ext/kernels_ext.h"
#include "codegen/arrow_compute/ext/typed_node_visitor.h"
#include "precompile/array.h"
#include "precompile/type.h"
#include "precompile/type_traits.h"
#include "third_party/ska_sort.hpp"
#include "third_party/timsort.hpp"
#include "utils/macros.h"

/**
                 The Overall Implementation of Sort Kernel
 * In general, there are four kenels to use when sorting for different data.
   SortArraysToIndicesKernel::Impl is the base class, and other three kernels,
including SortInplaceKernel, SortOnekeyKernel and SortMultiplekeyKernel, extend
it.
 * Usage:
   SortInplaceKernel is used when sorting for one non-string and non-bool col
without payload. SortOnekeyKernel is used when sorting for single key with
payload, and one string or bool col without payload. SortMultiplekeyKernel is
used when sorting for multiple keys and codegen is disabled.
   SortArraysToIndicesKernel::Impl is used when sorting for multiple keys and
codegen is enabled.
 * In these kernels, usually ska_sort is used for asc direciton, and std sort is
used for desc direciton. Timsort is used in multiple-key sort.
 * Before sorting, projection and partition can be conducted.
   If projection is required, it is completed before sorting, and the projected
cols are used to do comparison. If partition is required, null, NaN(for double
and float only) and valid value are partitioned before sorting.
   FIXME: 1. datatype change after projection is not supported in
SortInplaceKernel.
**/

namespace sparkcolumnarplugin {
namespace codegen {
namespace arrowcompute {
namespace extra {
using ArrayList = std::vector<std::shared_ptr<arrow::Array>>;
using namespace sparkcolumnarplugin::precompile;

template <typename T>
using is_number_bool_date = std::integral_constant<
    bool, arrow::is_number_type<T>::value || arrow::is_boolean_type<T>::value ||
              arrow::is_date_type<T>::value || arrow::is_timestamp_type<T>::value>;

///////////////  SortArraysToIndices  ////////////////
class SortArraysToIndicesKernel::Impl {
 public:
  Impl() {
    ctx_ = nullptr;
    col_num_ = 0;
    NaN_check_ = false;
  }

  Impl(arrow::compute::ExecContext* ctx, std::shared_ptr<arrow::Schema> result_schema,
       std::shared_ptr<gandiva::Projector> key_projector,
       std::vector<std::shared_ptr<arrow::DataType>> projected_types,
       std::vector<std::shared_ptr<arrow::Field>> key_field_list,
       std::vector<bool> sort_directions, std::vector<bool> nulls_order, bool NaN_check)
      : ctx_(ctx),
        result_schema_(result_schema),
        key_projector_(key_projector),
        key_field_list_(key_field_list),
        sort_directions_(sort_directions),
        nulls_order_(nulls_order),
        projected_types_(projected_types),
        NaN_check_(NaN_check) {
#ifdef DEBUG
    std::cout << "use SortArraysToIndicesKernel::Impl" << std::endl;
#endif
    for (auto field : key_field_list) {
      auto indices = GetIndicesFromSchemaCaseInsensitive(result_schema, field->name());
      if (indices.size() != 1) {
        std::cout << "[ERROR] SortArraysToIndicesKernel::Impl can't find key "
                  << field->ToString() << " from " << result_schema->ToString()
                  << std::endl;
        throw;
      }
      key_index_list_.push_back(indices[0]);
    }
    col_num_ = result_schema->num_fields();
    if (!key_projector) {
      auto status = LoadJITFunction(key_field_list);
      if (!status.ok()) {
        std::cout << "LoadJITFunction failed, msg is " << status.message() << std::endl;
        throw;
      }
    } else {
      int i = 0;
      for (auto type : projected_types) {
        auto field = arrow::field(std::to_string(i), type);
        projected_field_list_.push_back(field);
        i++;
      }
      auto status = LoadJITFunction(projected_field_list_);
      if (!status.ok()) {
        std::cout << "LoadJITFunction failed, msg is " << status.message() << std::endl;
        throw;
      }
    }
  }

  virtual ~Impl() {}

  virtual arrow::Status LoadJITFunction(
      std::vector<std::shared_ptr<arrow::Field>> key_field_list) {
    // generate ddl signature
    std::stringstream func_args_ss;
    for (int i = 0; i < sort_directions_.size(); i++) {
      func_args_ss << "[Sorter]" << (nulls_order_[i] ? "nulls_first" : "nulls_last")
                   << "|" << (sort_directions_[i] ? "asc" : "desc");
    }
    for (int i = 0; i < key_field_list.size(); i++) {
      auto field = key_field_list[i];
      func_args_ss << "[Type]" << field->type()->ToString();
    }
#ifdef DEBUG
    std::cout << "func_args_ss is " << func_args_ss.str() << std::endl;
#endif
    std::stringstream signature_ss;
    signature_ss << std::hex << std::hash<std::string>{}(func_args_ss.str());
    signature_ = signature_ss.str();

    auto file_lock = FileSpinLock();
    auto status = LoadLibrary(signature_, ctx_, &sorter_);
    if (!status.ok()) {
      // process
      auto codes = ProduceCodes();
      // compile codes
      const arrow::Status& status1 = CompileCodes(codes, signature_);
      if (!status1.ok()) {
        FileSpinUnLock(file_lock);
        return status1;
      }
      const arrow::Status& status2 = LoadLibrary(signature_, ctx_, &sorter_);
      if (!status2.ok()) {
        FileSpinUnLock(file_lock);
        return status1;
      }
    }
    FileSpinUnLock(file_lock);
    return arrow::Status::OK();
  }

  virtual arrow::Status Evaluate(ArrayList& in) {
    num_batches_++;
    if (cached_.size() <= col_num_) {
      cached_.resize(col_num_ + 1);
    }
    if (!key_projector_) {
      ArrayList key_cols;
      for (auto idx : key_index_list_) {
        key_cols.push_back(in[idx]);
      }
      sorter_->Evaluate(key_cols);
    } else {
      std::vector<std::shared_ptr<arrow::Array>> projected_batch;
      // do projection here, and the projected arrays are used for comparison
      auto length = in.size() > 0 ? in[0]->length() : 0;
      auto in_batch = arrow::RecordBatch::Make(result_schema_, length, in);
      RETURN_NOT_OK(
          key_projector_->Evaluate(*in_batch, ctx_->memory_pool(), &projected_batch));
      sorter_->Evaluate(projected_batch);
    }
    items_total_ += in[0]->length();
    length_list_.push_back(in[0]->length());

    for (int i = 0; i < col_num_; i++) {
      cached_[i].push_back(in[i]);
    }

    return arrow::Status::OK();
  }

  virtual arrow::Status Spill(int64_t* spilled_size) {
    // do spill
    local_result_iter_->SortResultSpill();

    return arrow::Status::OK();
  }

  virtual arrow::Status MakeResultIterator(
      std::shared_ptr<arrow::Schema> schema,
      std::shared_ptr<ResultIterator<arrow::RecordBatch>>* out) {
    std::shared_ptr<FixedSizeBinaryArray> indices_out;
    RETURN_NOT_OK(sorter_->FinishInternal(&indices_out));
    local_result_iter_ = std::make_shared<SorterResultIterator>(ctx_, schema, indices_out,
                                                                std::move(cached_));
    *out = local_result_iter_;
    return arrow::Status::OK();
  }

  virtual arrow::Status MakeResultIterator(
      std::shared_ptr<arrow::Schema> schema,
      std::shared_ptr<ResultIterator<SortRelation>>* out) {
    RETURN_NOT_OK(sorter_->MakeResultIterator(schema, out));
    return arrow::Status::OK();
  }

  virtual arrow::Status Finish(std::shared_ptr<arrow::Array>* out) {
    return arrow::Status::OK();
  }

  std::string GetSignature() { return signature_; }

  // TODO: move this class into common utility
  class SpillableCacheStore {
   public:
    SpillableCacheStore(std::vector<arrow::ArrayVector>& cached,
                        std::shared_ptr<arrow::Schema> result_schema)
        : cached_(cached), result_schema_(result_schema) {
      num_columns_ = cached_.size();
      local_spill_dir_ = "sort_spill_" + GenerateUUID();
    }

    ~SpillableCacheStore() {}

    arrow::Status DoSpill() {
      if (is_spilled_) {
        return arrow::Status::OK();
      }
      auto fs = std::make_shared<arrow::fs::LocalFileSystem>();
      ARROW_ASSIGN_OR_RAISE(auto path_info, fs->GetFileInfo(local_spill_dir_));
      if (path_info.type() == arrow::fs::FileType::NotFound) {
        RETURN_NOT_OK(fs->CreateDir(local_spill_dir_, true));
      }

      int num_batches = cached_[0].size();
      int64_t size = 0;
      for (auto i = 0; i < num_batches; i++) {
        ArrayList cols;
        for (auto j = 0; j < cached_.size() - 1; j++) {
          cols.push_back(cached_[j][i]);
        }

        RETURN_NOT_OK(SpillData(cols));
      }
      SetSpilled(true);
      cached_.clear();

      return arrow::Status::OK();
    }

    bool IsSpilled() { return is_spilled_; }

    void SetSpilled(bool state) { is_spilled_ = state; }

    std::string GenerateUUID() {
      boost::uuids::random_generator generator;
      return boost::uuids::to_string(generator());
    }

    std::string random_string(size_t length) {
      auto randchar = []() -> char {
        const char charset[] =
            "0123456789"
            "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            "abcdefghijklmnopqrstuvwxyz";
        const size_t max_index = (sizeof(charset) - 1);
        return charset[rand() % max_index];
      };
      std::string str(length, 0);
      std::generate_n(str.begin(), length, randchar);
      return str + ".arrow";
    }

    arrow::Status SpillData(ArrayList in) {
      arrow::ipc::IpcPayload payload;
      arrow::ipc::IpcWriteOptions options_;

      auto batch = arrow::RecordBatch::Make(result_schema_, in[0]->length(), in);

      arrow::ipc::DictionaryFieldMapper dict_file_mapper;  // unused
      std::shared_ptr<arrow::io::FileOutputStream> spilled_file_os_;
      std::string spilled_file_ =
          local_spill_dir_ + "/" + random_string(128);  // TODO(): get tmp dir

      ARROW_ASSIGN_OR_RAISE(spilled_file_os_,
                            arrow::io::FileOutputStream::Open(spilled_file_, true));
      int32_t metadata_length = -1;
      auto schema_payload_ = std::make_shared<arrow::ipc::IpcPayload>();
      RETURN_NOT_OK(arrow::ipc::GetSchemaPayload(
          *result_schema_.get(), options_, dict_file_mapper, schema_payload_.get()));
      ARROW_RETURN_NOT_OK(arrow::ipc::WriteIpcPayload(
          *schema_payload_.get(), options_, spilled_file_os_.get(), &metadata_length));
      arrow::ipc::GetRecordBatchPayload(*batch, options_, &payload);
      ARROW_RETURN_NOT_OK(arrow::ipc::WriteIpcPayload(
          payload, options_, spilled_file_os_.get(), &metadata_length));
      spill_file_list.push_back(spilled_file_);

      return arrow::Status::OK();
    }

    arrow::Result<std::shared_ptr<arrow::ipc::RecordBatchReader>>
    GetRecordBatchStreamReader(const std::string& file_name) {
      ARROW_ASSIGN_OR_RAISE(auto file_, arrow::io::ReadableFile::Open(file_name))
      ARROW_ASSIGN_OR_RAISE(auto file_reader,
                            arrow::ipc::RecordBatchStreamReader::Open(file_));
      return file_reader;
    }

    arrow::Status Fetch() {
      cached_.resize(num_columns_);
      for (const auto& file_path : spill_file_list) {
        ARROW_ASSIGN_OR_RAISE(auto file_reader, GetRecordBatchStreamReader(file_path));

        std::vector<std::shared_ptr<arrow::RecordBatch>> batches;
        RETURN_NOT_OK(file_reader->ReadAll(&batches));

        auto batch = batches[0];

        for (int i = 0; i < batch->num_columns(); i++) {
          cached_[i].push_back(batch->column(i));
        }
        std::remove(file_path.c_str());
      }
      std::remove(local_spill_dir_.c_str());
      is_spilled_ = false;
      spill_file_list.clear();
      return arrow::Status::OK();
    }

    std::vector<arrow::ArrayVector> GetCache() {
      if (is_spilled_) {
        Fetch();
      }

      return cached_;
    }

    std::string& GetSpillDir() { return local_spill_dir_; }

   private:
    std::vector<arrow::ArrayVector> cached_;
    int num_columns_;
    std::shared_ptr<arrow::Schema> result_schema_;
    std::vector<std::string> spill_file_list;
    bool is_spilled_ = false;
    std::string local_spill_dir_;
  };

  class TypedSorterCodeGenImpl {
   public:
    TypedSorterCodeGenImpl(std::string indice, std::shared_ptr<arrow::DataType> data_type)
        : indice_(indice), data_type_(data_type) {}
    std::string GetCachedVariablesDefine() {
      std::stringstream ss;
      ss << "using ArrayType_" << indice_ << " = " << GetTypeString(data_type_, "Array")
         << ";" << std::endl;
      ss << "std::vector<std::shared_ptr<ArrayType_" << indice_ << ">> cached_" << indice_
         << "_;" << std::endl;
      return ss.str();
    }

   private:
    std::string indice_;
    std::shared_ptr<arrow::DataType> data_type_;
  };

  virtual std::string ProduceCodes() {
    int indice = 0;
    std::vector<std::shared_ptr<TypedSorterCodeGenImpl>> key_typed_codegen_list;
    if (key_projector_) {
      for (auto field : projected_field_list_) {
        auto codegen = std::make_shared<TypedSorterCodeGenImpl>(std::to_string(indice),
                                                                field->type());
        key_typed_codegen_list.push_back(codegen);
        indice++;
      }
    } else {
      for (auto field : key_field_list_) {
        auto codegen = std::make_shared<TypedSorterCodeGenImpl>(std::to_string(indice),
                                                                field->type());
        key_typed_codegen_list.push_back(codegen);
        indice++;
      }
    }

    std::string cached_insert_str = GetCachedInsert();

    std::string comp_func_str = GetCompFunction(true);

    std::string comp_func_str_without_null = GetCompFunction(false);

    std::string sort_func_str = GetSortFunction();

    std::string cached_variables_define_str =
        GetCachedVariablesDefine(key_typed_codegen_list);

    return BaseCodes() + R"(
#include <arrow/buffer.h>

#include <algorithm>
#include <cmath>

#include "codegen/arrow_compute/ext/array_item_index.h"
#include "codegen/common/sort_relation.h"
#include "precompile/builder.h"
#include "precompile/type.h"
#include "third_party/ska_sort.hpp"
#include "third_party/timsort.hpp"
using namespace sparkcolumnarplugin::precompile;

class TypedSorterImpl : public CodeGenBase {
 public:
  TypedSorterImpl(arrow::compute::ExecContext* ctx) : ctx_(ctx) {}

  arrow::Status Evaluate(ArrayList& in) override {
    num_batches_++;
    )" + cached_insert_str +
           R"(
    auto cur = cached_0_[cached_0_.size() - 1];         
    items_total_ += cur->length();
    length_list_.push_back(cur->length());
    
    return arrow::Status::OK();
  }

  arrow::Status FinishInternal(std::shared_ptr<FixedSizeBinaryArray>* out) {
    // we should support nulls first and nulls last here
    // we should also support desc and asc here
    )" + comp_func_str +
           comp_func_str_without_null +
           R"(
    // initiate buffer for all arrays
    std::shared_ptr<arrow::Buffer> indices_buf;
    int64_t buf_size = items_total_ * sizeof(ArrayItemIndexS);
    auto maybe_buffer = arrow::AllocateBuffer(buf_size, ctx_->memory_pool());
    indices_buf = *std::move(maybe_buffer);

    ArrayItemIndexS* indices_begin =
        reinterpret_cast<ArrayItemIndexS*>(indices_buf->mutable_data());
    ArrayItemIndexS* indices_end = indices_begin + items_total_;

    int64_t indices_i = 0;
    for (int array_id = 0; array_id < num_batches_; array_id++) {
      for (int64_t i = 0; i < length_list_[array_id]; i++) {
        (indices_begin + indices_i)->array_id = array_id;
        (indices_begin + indices_i)->id = i;
        indices_i++;
      }
    }

    )" + sort_func_str +
           R"(
    std::shared_ptr<arrow::FixedSizeBinaryType> out_type;
    RETURN_NOT_OK(MakeFixedSizeBinaryType(sizeof(ArrayItemIndexS) / sizeof(int32_t), &out_type));
    RETURN_NOT_OK(MakeFixedSizeBinaryArray(out_type, items_total_, indices_buf, out));
    return arrow::Status::OK();
  }
  
 private:
  )" + cached_variables_define_str +
           R"(
  std::vector<int64_t> length_list_;
  arrow::compute::ExecContext* ctx_;
  uint64_t num_batches_ = 0;
  uint64_t items_total_ = 0;
  // If all batches has no null value, has_null_ will remain false
  bool has_null_ = false;
};

extern "C" void MakeCodeGen(arrow::compute::ExecContext* ctx,
                            std::shared_ptr<CodeGenBase>* out) {
  *out = std::make_shared<TypedSorterImpl>(ctx);
}

    )";
  }

  std::string GetCachedInsert() {
    std::stringstream ss;
    for (int i = 0; i < key_index_list_.size(); i++) {
      ss << "cached_" << i << "_.push_back(std::make_shared<ArrayType_" << i << ">(in["
         << i
         << "]));\n"
         // update has_null_
         << "if (!has_null_ && cached_" << i << "_[cached_" << i
         << "_.size() - 1]->null_count() > 0) {"
         << "has_null_ = true;}" << std::endl;
    }
    return ss.str();
  }

  std::string GetSortFunction() {
    std::stringstream ss;
    ss << "if (has_null_) {\n"
       << "gfx::timsort(indices_begin, indices_begin + items_total_, comp);} "
          "else {\n"
       << "gfx::timsort(indices_begin, indices_begin + items_total_, "
          "comp_without_null);}"
       << std::endl;
    return ss.str();
  }

  std::string GetCompFunction(bool has_null) {
    std::stringstream ss;
    bool projected;
    if (key_projector_) {
      projected = true;
    } else {
      projected = false;
    }
    if (has_null) {
      ss << "auto comp = [this](const ArrayItemIndexS& x, const "
            "ArrayItemIndexS& y) {"
         << GetCompFunction_(0, projected, key_field_list_, projected_types_,
                             sort_directions_, nulls_order_);
    } else {
      ss << "auto comp_without_null = "
         << "[this](const ArrayItemIndexS& x, const ArrayItemIndexS& y) {"
         << GetCompFunction_Without_Null_(0, projected, key_field_list_, projected_types_,
                                          sort_directions_);
    }
    ss << "};\n";
    return ss.str();
  }

  std::string GetCompFunction_(
      int cur_key_idx, bool projected,
      const std::vector<std::shared_ptr<arrow::Field>>& key_field_list,
      const std::vector<std::shared_ptr<arrow::DataType>>& projected_types,
      const std::vector<bool>& sort_directions, const std::vector<bool>& nulls_order) {
    std::string comp_str;
    bool asc = sort_directions[cur_key_idx];
    bool nulls_first = nulls_order[cur_key_idx];
    std::shared_ptr<arrow::DataType> data_type;
    std::string array;
    // if projected, use projected batch to compare, and use projected type
    array = "cached_";
    if (projected) {
      data_type = projected_types[cur_key_idx];
    } else {
      data_type = key_field_list[cur_key_idx]->type();
    }

    auto x_num_value =
        array + std::to_string(cur_key_idx) + "_[x.array_id]->GetView(x.id)";
    auto x_str_value =
        array + std::to_string(cur_key_idx) + "_[x.array_id]->GetString(x.id)";
    auto y_num_value =
        array + std::to_string(cur_key_idx) + "_[y.array_id]->GetView(y.id)";
    auto y_str_value =
        array + std::to_string(cur_key_idx) + "_[y.array_id]->GetString(y.id)";
    auto is_x_null = array + std::to_string(cur_key_idx) + "_[x.array_id]->IsNull(x.id)";
    auto is_y_null = array + std::to_string(cur_key_idx) + "_[y.array_id]->IsNull(y.id)";
    auto x_null_count =
        array + std::to_string(cur_key_idx) + "_[x.array_id]->null_count() > 0";
    auto y_null_count =
        array + std::to_string(cur_key_idx) + "_[y.array_id]->null_count() > 0";
    auto x_null = "(" + x_null_count + " && " + is_x_null + " )";
    auto y_null = "(" + y_null_count + " && " + is_y_null + " )";
    auto is_x_nan = "std::isnan(" + x_num_value + ")";
    auto is_y_nan = "std::isnan(" + y_num_value + ")";

    // Multiple keys sorting w/ nulls first/last is supported.
    std::stringstream ss;
    // We need to determine the position of nulls.
    ss << "if (" << x_null << ") {\n";
    // If value accessed from x is null, return true to make nulls first.
    if (nulls_first) {
      ss << "return true;\n}";
    } else {
      ss << "return false;\n}";
    }
    // If value accessed from y is null, return false to make nulls first.
    ss << " else if (" << y_null << ") {\n";
    if (nulls_first) {
      ss << "return false;\n}";
    } else {
      ss << "return true;\n}";
    }
    // If datatype is floating, we need to do partition for NaN if NaN check is
    // enabled
    if (data_type->id() == arrow::Type::DOUBLE || data_type->id() == arrow::Type::FLOAT) {
      if (NaN_check_) {
        ss << "else if (" << is_x_nan << ") {\n";
        if (asc) {
          ss << "return false;\n}";
        } else {
          ss << "return true;\n}";
        }
        ss << "else if (" << is_y_nan << ") {\n";
        if (asc) {
          ss << "return true;\n}";
        } else {
          ss << "return false;\n}";
        }
      }
    }

    // If values accessed from x and y are both not null
    ss << " else {\n";

    // Multiple keys sorting w/ different ordering is supported.
    // For string type of data, GetString should be used instead of GetView.
    if (asc) {
      if (data_type->id() == arrow::Type::STRING) {
        ss << "return " << x_str_value << " < " << y_str_value << ";\n}\n";
      } else {
        ss << "return " << x_num_value << " < " << y_num_value << ";\n}\n";
      }
    } else {
      if (data_type->id() == arrow::Type::STRING) {
        ss << "return " << x_str_value << " > " << y_str_value << ";\n}\n";
      } else {
        ss << "return " << x_num_value << " > " << y_num_value << ";\n}\n";
      }
    }
    comp_str = ss.str();
    if ((cur_key_idx + 1) == sort_directions.size()) {
      return comp_str;
    }
    // clear the contents of stringstream
    ss.str(std::string());
    if (data_type->id() == arrow::Type::STRING) {
      ss << "if ((" << x_null << " && " << y_null << ") || (" << x_str_value
         << " == " << y_str_value << ")) {";
    } else {
      if (NaN_check_ && (data_type->id() == arrow::Type::DOUBLE ||
                         data_type->id() == arrow::Type::FLOAT)) {
        // need to check NaN
        ss << "if ((" << x_null << " && " << y_null << ") || (" << is_x_nan << " && "
           << is_y_nan << ") || (" << x_num_value << " == " << y_num_value << ")) {";
      } else {
        ss << "if ((" << x_null << " && " << y_null << ") || (" << x_num_value
           << " == " << y_num_value << ")) {";
      }
    }
    ss << GetCompFunction_(cur_key_idx + 1, projected, key_field_list, projected_types,
                           sort_directions, nulls_order)
       << "} else { " << comp_str << "}";
    return ss.str();
  }

  std::string GetCompFunction_Without_Null_(
      int cur_key_idx, bool projected,
      const std::vector<std::shared_ptr<arrow::Field>>& key_field_list,
      const std::vector<std::shared_ptr<arrow::DataType>>& projected_types,
      const std::vector<bool>& sort_directions) {
    std::string comp_str;
    bool asc = sort_directions[cur_key_idx];
    std::shared_ptr<arrow::DataType> data_type;
    std::string array;
    // if projected, use projected batch to compare, and use projected type
    array = "cached_";
    if (projected) {
      data_type = projected_types[cur_key_idx];
    } else {
      data_type = key_field_list[cur_key_idx]->type();
    }

    auto x_num_value =
        array + std::to_string(cur_key_idx) + "_[x.array_id]->GetView(x.id)";
    auto x_str_value =
        array + std::to_string(cur_key_idx) + "_[x.array_id]->GetString(x.id)";
    auto y_num_value =
        array + std::to_string(cur_key_idx) + "_[y.array_id]->GetView(y.id)";
    auto y_str_value =
        array + std::to_string(cur_key_idx) + "_[y.array_id]->GetString(y.id)";
    auto is_x_nan = "std::isnan(" + x_num_value + ")";
    auto is_y_nan = "std::isnan(" + y_num_value + ")";

    // Multiple keys sorting w/ nulls first/last is supported.
    std::stringstream ss;
    // If datatype is floating, we need to do partition for NaN if NaN check is
    // enabled
    if (NaN_check_ && (data_type->id() == arrow::Type::DOUBLE ||
                       data_type->id() == arrow::Type::FLOAT)) {
      ss << "if (" << is_x_nan << ") {\n";
      if (asc) {
        ss << "return false;\n}";
      } else {
        ss << "return true;\n}";
      }
      ss << "else if (" << is_y_nan << ") {\n";
      if (asc) {
        ss << "return true;\n}";
      } else {
        ss << "return false;\n}";
      }
      // If values accessed from x and y are both not nan
      ss << " else {\n";
    }

    // Multiple keys sorting w/ different ordering is supported.
    // For string type of data, GetString should be used instead of GetView.
    if (asc) {
      if (data_type->id() == arrow::Type::STRING) {
        ss << "return " << x_str_value << " < " << y_str_value << ";\n";
      } else {
        ss << "return " << x_num_value << " < " << y_num_value << ";\n";
      }
    } else {
      if (data_type->id() == arrow::Type::STRING) {
        ss << "return " << x_str_value << " > " << y_str_value << ";\n";
      } else {
        ss << "return " << x_num_value << " > " << y_num_value << ";\n";
      }
    }
    if (NaN_check_ && (data_type->id() == arrow::Type::DOUBLE ||
                       data_type->id() == arrow::Type::FLOAT)) {
      ss << "}" << std::endl;
    }
    comp_str = ss.str();
    if ((cur_key_idx + 1) == sort_directions.size()) {
      return comp_str;
    }
    // clear the contents of stringstream
    ss.str(std::string());
    if (data_type->id() == arrow::Type::STRING) {
      ss << "if (" << x_str_value << " == " << y_str_value << ") {";
    } else {
      if (NaN_check_ && (data_type->id() == arrow::Type::DOUBLE ||
                         data_type->id() == arrow::Type::FLOAT)) {
        // need to check NaN
        ss << "if ((" << is_x_nan << " && " << is_y_nan << ") || (" << x_num_value
           << " == " << y_num_value << ")) {";
      } else {
        ss << "if (" << x_num_value << " == " << y_num_value << ") {";
      }
    }
    ss << GetCompFunction_Without_Null_(cur_key_idx + 1, projected, key_field_list,
                                        projected_types, sort_directions)
       << "} else { " << comp_str << "}";
    return ss.str();
  }

  std::string GetCachedVariablesDefine(
      std::vector<std::shared_ptr<TypedSorterCodeGenImpl>> shuffle_typed_codegen_list) {
    std::stringstream ss;
    for (auto codegen : shuffle_typed_codegen_list) {
      ss << codegen->GetCachedVariablesDefine() << std::endl;
    }
    return ss.str();
  }

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

  class SorterResultIterator : public ResultIterator<arrow::RecordBatch> {
   public:
    SorterResultIterator(arrow::compute::ExecContext* ctx,
                         std::shared_ptr<arrow::Schema> schema,
                         std::shared_ptr<FixedSizeBinaryArray> indices_in,
                         std::vector<arrow::ArrayVector> cached)
        : ctx_(ctx),
          schema_(schema),
          indices_in_cache_(indices_in),
          total_length_(indices_in->length()),
          cached_in_(std::move(cached)) {
      col_num_ = schema->num_fields();
      indices_begin_ = (ArrayItemIndexS*)indices_in->value_data();
      // appender_type won't be used
      AppenderBase::AppenderType appender_type = AppenderBase::left;
      for (int i = 0; i < col_num_; i++) {
        auto field = schema->field(i);
        std::shared_ptr<AppenderBase> appender;
        THROW_NOT_OK(MakeUnsafeAppender(ctx_, field->type(), appender_type, &appender));
        appender_list_.push_back(appender);
      }

      for (int i = 0; i < col_num_; i++) {
        int array_num = cached_in_[i].size();
        for (int array_id = 0; array_id < array_num; array_id++) {
          appender_list_[i]->AddArray(cached_in_[i][array_id]);
        }
      }
      batch_size_ = GetBatchSize();
    }

    ~SorterResultIterator() {}

    std::string ToString() override { return "SortArraysToIndicesResultIterator"; }

    arrow::Status SortResultSpill() {
      // TODO(): write to local disk

      if (!spillablecachestore_) {
        spillablecachestore_ = std::make_shared<SpillableCacheStore>(cached_in_, schema_);
      }
      std::cout << "call on: " << spillablecachestore_->GetSpillDir() << "|"
                << is_spilled_ << "\n";
      if (is_spilled_) {
        return arrow::Status::OK();
      }

      spillablecachestore_->DoSpill();

      // clean up references on cached array
      for (auto appender : appender_list_) {
        appender->ClearArrays();
      }
      cached_in_.clear();
      is_spilled_ = true;
      return arrow::Status::OK();
    }

    arrow::Status LoadCache() {
      // TODO(): more checks on the availbity/CRC?
      spillablecachestore_->Fetch();
      cached_in_ = spillablecachestore_->GetCache();

      for (int i = 0; i < col_num_; i++) {
        arrow::ArrayVector array_vector = cached_in_[i];
        int array_num = array_vector.size();
        for (int array_id = 0; array_id < array_num; array_id++) {
          auto arr = array_vector[array_id];
          appender_list_[i]->AddArray(arr);
        }
      }
      return arrow::Status::OK();
    }

    bool HasNext() override {
      if (offset_ >= total_length_) {
        return false;
      }
      return true;
    }

    arrow::Status Next(std::shared_ptr<arrow::RecordBatch>* out) {
      if (is_spilled_) {
        LoadCache();
        is_spilled_ = false;
      }
      auto length = (total_length_ - offset_) > batch_size_ ? batch_size_
                                                            : (total_length_ - offset_);
      uint64_t count = 0;
      for (int i = 0; i < col_num_; i++) {
        RETURN_NOT_OK(appender_list_[i]->Reserve(length));
        while (count < length) {
          auto item = indices_begin_ + offset_ + count++;
          RETURN_NOT_OK(appender_list_[i]->Append(item->array_id, item->id));
        }
        count = 0;
      }
      offset_ += length;
      ArrayList arrays;
      for (int i = 0; i < col_num_; i++) {
        std::shared_ptr<arrow::Array> out_array;
        RETURN_NOT_OK(appender_list_[i]->Finish(&out_array));
        arrays.push_back(out_array);
        appender_list_[i]->Reset();
      }

      *out = arrow::RecordBatch::Make(schema_, length, arrays);
      return arrow::Status::OK();
    }

   private:
    uint64_t offset_ = 0;
    const uint64_t total_length_;
    std::shared_ptr<arrow::Schema> schema_;
    arrow::compute::ExecContext* ctx_;
    uint64_t batch_size_;
    int col_num_;
    ArrayItemIndexS* indices_begin_;
    std::vector<arrow::ArrayVector> cached_in_;
    std::vector<std::shared_ptr<arrow::DataType>> type_list_;
    std::vector<std::shared_ptr<AppenderBase>> appender_list_;
    std::shared_ptr<FixedSizeBinaryArray> indices_in_cache_;
    bool is_spilled_ = false;
    std::shared_ptr<SpillableCacheStore> spillablecachestore_;
  };

 protected:
  std::shared_ptr<CodeGenBase> sorter_;
  std::vector<arrow::ArrayVector> cached_;
  arrow::compute::ExecContext* ctx_;
  std::string signature_;
  std::vector<int> key_index_list_;
  std::shared_ptr<arrow::Schema> result_schema_;
  std::shared_ptr<gandiva::Projector> key_projector_;
  std::vector<std::shared_ptr<arrow::DataType>> projected_types_;
  std::vector<std::shared_ptr<arrow::Field>> projected_field_list_;
  std::vector<std::shared_ptr<arrow::Field>> key_field_list_;
  std::vector<int64_t> length_list_;
  uint64_t num_batches_ = 0;
  uint64_t items_total_ = 0;
  // true for asc, false for desc
  std::vector<bool> sort_directions_;
  // true for nulls_first, false for nulls_last
  std::vector<bool> nulls_order_;
  bool NaN_check_;
  int col_num_ = 0;
  std::shared_ptr<SorterResultIterator> local_result_iter_;
};

///////////////  SortArraysInPlace  ////////////////
template <typename DATATYPE, typename CTYPE>
class SortInplaceKernel : public SortArraysToIndicesKernel::Impl {
 public:
  SortInplaceKernel(arrow::compute::ExecContext* ctx,
                    std::shared_ptr<arrow::Schema> result_schema,
                    std::shared_ptr<gandiva::Projector> key_projector,
                    std::vector<bool> sort_directions, std::vector<bool> nulls_order,
                    bool NaN_check)
      : ctx_(ctx),
        nulls_first_(nulls_order[0]),
        asc_(sort_directions[0]),
        result_schema_(result_schema),
        key_projector_(key_projector),
        NaN_check_(NaN_check) {}

  arrow::Status Evaluate(ArrayList& in) override {
    num_batches_++;
    // do projection here
    arrow::ArrayVector outputs;
    if (key_projector_) {
      auto length = in.size() > 0 ? in[0]->length() : 0;
      auto in_batch = arrow::RecordBatch::Make(result_schema_, length, in);
      RETURN_NOT_OK(key_projector_->Evaluate(*in_batch, ctx_->memory_pool(), &outputs));
      cached_0_.push_back(outputs[0]);
      nulls_total_ += outputs[0]->null_count();
    } else {
      cached_0_.push_back(in[0]);
      nulls_total_ += in[0]->null_count();
    }

    items_total_ += in[0]->length();
    return arrow::Status::OK();
  }

  // This function is used for float/double data without null value.
  // If NaN_check_ is true, we need to do partition for NaN before sort.
  template <typename TYPE>
  auto SortNoNull(TYPE* indices_begin, TYPE* indices_end) ->
      typename std::enable_if_t<std::is_floating_point<TYPE>::value> {
    if (asc_) {
      auto sort_end = indices_end;
      if (NaN_check_) {
        sort_end = std::partition(indices_begin, indices_end,
                                  [](TYPE i) { return !std::isnan(i); });
      }
      ska_sort(indices_begin, sort_end);
    } else {
      auto sort_begin = indices_begin;
      if (NaN_check_) {
        sort_begin = std::partition(indices_begin, indices_end,
                                    [](TYPE i) { return std::isnan(i); });
      }
      auto desc_comp = [this](TYPE& x, TYPE& y) { return x > y; };
      std::sort(sort_begin, indices_end, desc_comp);
    }
  }

  // This function is used for non-float and non-double data without null value.
  template <typename TYPE>
  auto SortNoNull(TYPE* indices_begin, TYPE* indices_end) ->
      typename std::enable_if_t<!std::is_floating_point<TYPE>::value &&
                                !std::is_same<TYPE, arrow::Decimal128>::value> {
    if (asc_) {
      ska_sort(indices_begin, indices_end);
    } else {
      auto desc_comp = [this](TYPE& x, TYPE& y) { return x > y; };
      std::sort(indices_begin, indices_end, desc_comp);
    }
  }

  // This function is used for non-float and non-double data without null value.
  template <typename TYPE>
  auto SortNoNull(TYPE* indices_begin, TYPE* indices_end) ->
      typename std::enable_if_t<std::is_same<TYPE, arrow::Decimal128>::value> {
    if (asc_) {
      auto comp = [this](TYPE& x, TYPE& y) { return x < y; };
      std::sort(indices_begin, indices_end, comp);
    } else {
      auto comp = [this](TYPE& x, TYPE& y) { return x > y; };
      std::sort(indices_begin, indices_end, comp);
    }
  }

  // This function is used for float/double data with null value.
  // We should do partition for null and NaN (if (NaN_check_ is true).
  template <typename TYPE, typename ArrayType>
  auto Sort(int64_t* indices_begin, int64_t* indices_end, const ArrayType& values) ->
      typename std::enable_if_t<std::is_floating_point<TYPE>::value> {
    std::iota(indices_begin, indices_end, 0);
    auto sort_begin = indices_begin;
    auto sort_end = indices_end;

    if (asc_) {
      if (nulls_first_) {
        sort_begin = std::partition(indices_begin, indices_end, [&values](uint64_t ind) {
          return values.IsNull(ind);
        });
        if (NaN_check_) {
          // values should be sorted to:
          // null, null, ..., valid-1, valid-2, ..., valid-3, NaN, NaN, ...
          sort_end = std::partition(sort_begin, indices_end, [&values](uint64_t ind) {
            return !std::isnan(values.GetView(ind));
          });
        }
        ska_sort(sort_begin, sort_end,
                 [&values](auto& x) -> decltype(auto) { return values.GetView(x); });
      } else {
        sort_end = std::partition(indices_begin, indices_end, [&values](uint64_t ind) {
          return !values.IsNull(ind);
        });
        if (NaN_check_) {
          // values should be sorted to:
          // valid-1, valid-2, ..., valid-3, NaN, NaN, ..., null, null, ...
          sort_end = std::partition(indices_begin, sort_end, [&values](uint64_t ind) {
            return !std::isnan(values.GetView(ind));
          });
        }
        ska_sort(indices_begin, sort_end,
                 [&values](auto& x) -> decltype(auto) { return values.GetView(x); });
      }
    } else {
      auto comp = [&values](uint64_t left, uint64_t right) {
        return values.GetView(left) > values.GetView(right);
      };
      if (nulls_first_) {
        sort_begin = std::partition(indices_begin, indices_end, [&values](uint64_t ind) {
          return values.IsNull(ind);
        });
        if (NaN_check_) {
          // values should be sorted to:
          // null, null, NaN, NaN, ..., valid-1, valid-2, ..., valid-3, ...
          sort_begin = std::partition(sort_begin, indices_end, [&values](uint64_t ind) {
            return std::isnan(values.GetView(ind));
          });
        }
        std::sort(sort_begin, indices_end, comp);
      } else {
        sort_end = std::partition(indices_begin, indices_end, [&values](uint64_t ind) {
          return !values.IsNull(ind);
        });
        if (NaN_check_) {
          // values should be sorted to:
          // NaN, NaN, ..., valid-1, valid-2, ..., valid-3, ..., null, null, ...
          sort_begin = std::partition(indices_begin, sort_end, [&values](uint64_t ind) {
            return std::isnan(values.GetView(ind));
          });
        }
        std::sort(sort_begin, sort_end, comp);
      }
    }
  }

  // This function is used for non-float and non-double data with null value.
  // We should do partition for null.
  template <typename TYPE, typename ArrayType>
  auto Sort(int64_t* indices_begin, int64_t* indices_end, const ArrayType& values) ->
      typename std::enable_if_t<!std::is_floating_point<TYPE>::value &&
                                !std::is_same<TYPE, arrow::Decimal128>::value> {
    std::iota(indices_begin, indices_end, 0);
    if (asc_) {
      if (nulls_first_) {
        auto nulls_end =
            std::partition(indices_begin, indices_end,
                           [&values](uint64_t ind) { return values.IsNull(ind); });
        ska_sort(nulls_end, indices_end,
                 [&values](auto& x) -> decltype(auto) { return values.GetView(x); });
      } else {
        auto nulls_begin =
            std::partition(indices_begin, indices_end,
                           [&values](uint64_t ind) { return !values.IsNull(ind); });
        ska_sort(indices_begin, nulls_begin,
                 [&values](auto& x) -> decltype(auto) { return values.GetView(x); });
      }
    } else {
      auto comp = [&values](uint64_t left, uint64_t right) {
        return values.GetView(left) > values.GetView(right);
      };
      if (nulls_first_) {
        auto nulls_end =
            std::partition(indices_begin, indices_end,
                           [&values](uint64_t ind) { return values.IsNull(ind); });
        std::sort(nulls_end, indices_end, comp);
      } else {
        auto nulls_begin =
            std::partition(indices_begin, indices_end,
                           [&values](uint64_t ind) { return !values.IsNull(ind); });
        std::sort(indices_begin, nulls_begin, comp);
      }
    }
  }

  template <typename TYPE, typename ArrayType>
  auto Sort(int64_t* indices_begin, int64_t* indices_end, const ArrayType& values) ->
      typename std::enable_if_t<std::is_same<TYPE, arrow::Decimal128>::value> {
    std::iota(indices_begin, indices_end, 0);
    if (asc_) {
      auto comp = [&values](uint64_t left, uint64_t right) {
        return values.GetView(left) < values.GetView(right);
      };
      if (nulls_first_) {
        auto nulls_end =
            std::partition(indices_begin, indices_end,
                           [&values](uint64_t ind) { return values.IsNull(ind); });
        std::sort(nulls_end, indices_end, comp);
      } else {
        auto nulls_begin =
            std::partition(indices_begin, indices_end,
                           [&values](uint64_t ind) { return !values.IsNull(ind); });
        std::sort(indices_begin, nulls_begin, comp);
      }
    } else {
      auto comp = [&values](uint64_t left, uint64_t right) {
        return values.GetView(left) > values.GetView(right);
      };
      if (nulls_first_) {
        auto nulls_end =
            std::partition(indices_begin, indices_end,
                           [&values](uint64_t ind) { return values.IsNull(ind); });
        std::sort(nulls_end, indices_end, comp);
      } else {
        auto nulls_begin =
            std::partition(indices_begin, indices_end,
                           [&values](uint64_t ind) { return !values.IsNull(ind); });
        std::sort(indices_begin, nulls_begin, comp);
      }
    }
  }

  arrow::Status MakeResultIterator(
      std::shared_ptr<arrow::Schema> schema,
      std::shared_ptr<ResultIterator<arrow::RecordBatch>>* out) override {
    RETURN_NOT_OK(
        arrow::Concatenate(cached_0_, ctx_->memory_pool(), &concatenated_array_));
    if (nulls_total_ == 0) {
      // Function SortNoNull is used.
      CTYPE* indices_begin = concatenated_array_->data()->GetMutableValues<CTYPE>(1);
      CTYPE* indices_end = indices_begin + concatenated_array_->length();

      SortNoNull<CTYPE>(indices_begin, indices_end);
      *out = std::make_shared<SorterResultIterator>(ctx_, schema, concatenated_array_,
                                                    nulls_first_, asc_);
    } else {
      // Function Sort is used.
      auto typed_array = std::make_shared<ArrayType_0>(concatenated_array_);
      std::shared_ptr<arrow::Array> indices_out;

      int64_t buf_size = typed_array->length() * sizeof(uint64_t);
      ARROW_ASSIGN_OR_RAISE(auto indices_buf,
                            AllocateBuffer(buf_size, ctx_->memory_pool()));
      int64_t* indices_begin = reinterpret_cast<int64_t*>(indices_buf->mutable_data());
      int64_t* indices_end = indices_begin + typed_array->length();

      Sort<CTYPE, ArrayType_0>(indices_begin, indices_end, *typed_array.get());
      indices_out = std::make_shared<arrow::UInt64Array>(typed_array->length(),
                                                         std::move(indices_buf));
      std::shared_ptr<arrow::Array> sort_out;
      arrow::compute::TakeOptions options;
      auto maybe_sort_out = arrow::compute::Take(*concatenated_array_.get(),
                                                 *indices_out.get(), options, ctx_);
      sort_out = *std::move(maybe_sort_out);
      *out = std::make_shared<SorterResultIterator>(ctx_, schema, sort_out, nulls_first_,
                                                    asc_);
    }
    return arrow::Status::OK();
  }

 private:
  using ArrayType_0 = typename TypeTraits<DATATYPE>::ArrayType;
  arrow::ArrayVector cached_0_;
  std::shared_ptr<arrow::Array> concatenated_array_;
  arrow::compute::ExecContext* ctx_;
  std::shared_ptr<arrow::Schema> result_schema_;
  std::shared_ptr<gandiva::Projector> key_projector_;
  bool nulls_first_;
  bool asc_;
  bool NaN_check_;
  uint64_t num_batches_ = 0;
  uint64_t items_total_ = 0;
  uint64_t nulls_total_ = 0;

  class SorterResultIterator : public ResultIterator<arrow::RecordBatch> {
   public:
    SorterResultIterator(arrow::compute::ExecContext* ctx,
                         std::shared_ptr<arrow::Schema> result_schema,
                         std::shared_ptr<arrow::Array> result_arr, bool nulls_first,
                         bool asc)
        : ctx_(ctx),
          result_schema_(result_schema),
          total_length_(result_arr->length()),
          nulls_total_(result_arr->null_count()),
          nulls_first_(nulls_first),
          asc_(asc),
          result_arr_(result_arr) {
      batch_size_ = GetBatchSize();
    }

    std::string ToString() override { return "SortArraysToIndicesResultIterator"; }

    bool HasNext() override {
      if (offset_ >= total_length_) {
        return false;
      }
      return true;
    }

    // This class is used to copy a piece of memory from the sorted ArrayData
    // to a result array.
    // It can be used only in sorted data because of null count calculation.
    template <typename KeyType>
    class SliceImpl {
     public:
      SliceImpl(const arrow::ArrayData& in, arrow::MemoryPool* pool, int64_t length,
                int64_t offset, int64_t null_total, bool null_first, int64_t total_length)
          : in_(in),
            pool_(pool),
            length_(length),
            offset_(offset),
            null_total_(null_total) {
        out_data_.type = in.type;
        out_data_.length = length;
        out_data_.buffers.resize(in.buffers.size());
        out_data_.child_data.resize(in.child_data.size());
        for (auto& data : out_data_.child_data) {
          data = std::make_shared<arrow::ArrayData>();
        }
        // decide null_count of this sliced array
        if (null_total == 0) {
          out_data_.null_count = 0;
        } else {
          if (null_first) {
            if ((offset + length) > null_total_) {
              out_data_.null_count =
                  (null_total_ - offset > 0) ? (null_total_ - offset) : 0;
            } else {
              out_data_.null_count = length;
            }
          } else {
            auto valid_total = total_length - null_total_;
            if ((offset + length) < valid_total) {
              out_data_.null_count = 0;
            } else {
              out_data_.null_count =
                  (offset - valid_total) > 0 ? length : (offset + length - valid_total);
            }
          }
        }
      }

      arrow::Status Slice(arrow::ArrayData* out) {
        const auto& buffer_0 = in_.buffers[0];
        const auto& buffer_1 = in_.buffers[1];
        if (out_data_.null_count) {
          SliceBitmap(buffer_0, &out_data_.buffers[0]);
        }
        SliceBuffer(buffer_1, &out_data_.buffers[1]);
        *out = std::move(out_data_);
        return arrow::Status::OK();
      }

     private:
      /// offset, length pair for representing a Range of a buffer or array
      struct Range {
        int64_t offset, length;

        Range() : offset(-1), length(0) {}
        Range(int64_t o, int64_t l) : offset(o), length(l) {}
      };

      /// non-owning view into a range of bits
      struct Bitmap {
        Bitmap() = default;
        Bitmap(const uint8_t* d, Range r) : data(d), range(r) {}
        explicit Bitmap(const std::shared_ptr<arrow::Buffer>& buffer, Range r)
            : Bitmap(buffer ? buffer->data() : nullptr, r) {}

        const uint8_t* data;
        Range range;

        bool AllSet() const { return data == nullptr; }
      };

      arrow::Status SliceBuffer(const std::shared_ptr<arrow::Buffer>& buffer,
                                std::shared_ptr<arrow::Buffer>* out) {
        ARROW_ASSIGN_OR_RAISE(*out, AllocateBuffer(size * length_, pool_));
        auto out_data = (*out)->mutable_data();
        auto data_begin = buffer->data();
        std::memcpy(out_data, data_begin + size * offset_, size * length_);
        return arrow::Status::OK();
      }

      arrow::Status SliceBitmap(const std::shared_ptr<arrow::Buffer>& buffer,
                                std::shared_ptr<arrow::Buffer>* out) {
        Range range(offset_, length_);
        Bitmap bitmap = Bitmap(buffer, range);

        auto length = bitmap.range.length;
        auto offset = bitmap.range.offset;
        ARROW_ASSIGN_OR_RAISE(*out, AllocateBitmap(length, pool_));
        uint8_t* dst = (*out)->mutable_data();

        if (bitmap.AllSet()) {
          arrow::BitUtil::SetBitsTo(dst, offset, length, true);
        } else {
          arrow::internal::CopyBitmap(bitmap.data, offset, length, dst, 0);
        }
        return arrow::Status::OK();
      }

      arrow::ArrayData in_;
      arrow::ArrayData out_data_;
      arrow::MemoryPool* pool_;
      int64_t length_;
      int64_t offset_;
      int64_t null_total_;
      int size = sizeof(KeyType);
    };

    arrow::Status Next(std::shared_ptr<arrow::RecordBatch>* out) {
      auto length = (total_length_ - offset_) > batch_size_ ? batch_size_
                                                            : (total_length_ - offset_);
      arrow::ArrayData result_data = *result_arr_->data();
      arrow::ArrayData out_data;
      SliceImpl<CTYPE>(result_data, ctx_->memory_pool(), length, offset_, nulls_total_,
                       nulls_first_, total_length_)
          .Slice(&out_data);
      std::shared_ptr<arrow::Array> out_0 =
          MakeArray(std::make_shared<arrow::ArrayData>(std::move(out_data)));
      offset_ += length;
      *out = arrow::RecordBatch::Make(result_schema_, length, {out_0});
      return arrow::Status::OK();
    }

   private:
    using ArrayType_0 = typename arrow::TypeTraits<DATATYPE>::ArrayType;
    using BuilderType_0 = typename arrow::TypeTraits<DATATYPE>::BuilderType;
    std::shared_ptr<arrow::Array> result_arr_;
    uint64_t offset_ = 0;
    const uint64_t total_length_;
    const uint64_t nulls_total_;
    std::shared_ptr<arrow::Schema> result_schema_;
    arrow::compute::ExecContext* ctx_;
    uint64_t batch_size_;
    bool nulls_first_;
    bool asc_;
  };
};

///////////////  SortArraysOneKey  ////////////////
template <typename DATATYPE>
class SortOnekeyKernel : public SortArraysToIndicesKernel::Impl {
 public:
  SortOnekeyKernel(arrow::compute::ExecContext* ctx,
                   std::shared_ptr<arrow::Schema> result_schema,
                   std::shared_ptr<gandiva::Projector> key_projector,
                   std::vector<std::shared_ptr<arrow::Field>> key_field_list,
                   std::vector<bool> sort_directions, std::vector<bool> nulls_order,
                   bool NaN_check)
      : ctx_(ctx),
        nulls_first_(nulls_order[0]),
        asc_(sort_directions[0]),
        result_schema_(result_schema),
        key_projector_(key_projector),
        NaN_check_(NaN_check) {
#ifdef DEBUG
    std::cout << "UseSortOnekeyKernel" << std::endl;
#endif
    auto indices =
        GetIndicesFromSchemaCaseInsensitive(result_schema, key_field_list[0]->name());
    if (indices.size() < 1) {
      std::cout << "[ERROR] SortOnekeyKernel for arithmetic can't find key "
                << key_field_list[0]->ToString() << " from " << result_schema->ToString()
                << std::endl;
      throw;
    }
    key_id_ = indices[0];
    col_num_ = result_schema->num_fields();
  }
  ~SortOnekeyKernel() {}

  arrow::Status Evaluate(ArrayList& in) override {
    num_batches_++;
    // do projection here
    arrow::ArrayVector outputs;
    if (key_projector_) {
      auto length = in.size() > 0 ? in[key_id_]->length() : 0;
      auto in_batch = arrow::RecordBatch::Make(result_schema_, length, in);
      RETURN_NOT_OK(key_projector_->Evaluate(*in_batch, ctx_->memory_pool(), &outputs));
      cached_key_.push_back(std::make_shared<ArrayType_key>(outputs[0]));
      nulls_total_ += outputs[0]->null_count();
    } else {
      nulls_total_ += in[key_id_]->null_count();
      cached_key_.push_back(std::make_shared<ArrayType_key>(in[key_id_]));
    }

    items_total_ += in[key_id_]->length();
    length_list_.push_back(in[key_id_]->length());
    if (cached_.size() <= col_num_) {
      cached_.resize(col_num_ + 1);
    }

    for (int i = 0; i < col_num_; i++) {
      cached_[i].push_back(std::move(in[i]));
    }

    return arrow::Status::OK();
  }

  arrow::Status Spill(int64_t* spilled_size) {
    // do spill

    local_result_iter_->SortResultSpill();

    return arrow::Status::OK();
  }

  void PartitionNulls(ArrayItemIndexS* indices_begin, ArrayItemIndexS* indices_end) {
    int64_t indices_i = 0;
    int64_t indices_null = 0;

    if (nulls_total_ == 0) {
      // if all batches have no null value,
      // we do not need to check whether the value is null
      for (int array_id = 0; array_id < num_batches_; array_id++) {
        for (int64_t i = 0; i < length_list_[array_id]; i++) {
          (indices_begin + indices_i)->array_id = array_id;
          (indices_begin + indices_i)->id = i;
          indices_i++;
        }
      }
    } else {
      // we should support nulls first and nulls last here
      for (int array_id = 0; array_id < num_batches_; array_id++) {
        if (cached_key_[array_id]->null_count() == 0) {
          // if this array has no null value,
          // we do need to check if the value is null
          for (int64_t i = 0; i < length_list_[array_id]; i++) {
            if (nulls_first_) {
              (indices_begin + nulls_total_ + indices_i)->array_id = array_id;
              (indices_begin + nulls_total_ + indices_i)->id = i;
              indices_i++;
            } else {
              (indices_begin + indices_i)->array_id = array_id;
              (indices_begin + indices_i)->id = i;
              indices_i++;
            }
          }
        } else {
          for (int64_t i = 0; i < length_list_[array_id]; i++) {
            if (nulls_first_) {
              if (!cached_key_[array_id]->IsNull(i)) {
                (indices_begin + nulls_total_ + indices_i)->array_id = array_id;
                (indices_begin + nulls_total_ + indices_i)->id = i;
                indices_i++;
              } else {
                (indices_begin + indices_null)->array_id = array_id;
                (indices_begin + indices_null)->id = i;
                indices_null++;
              }
            } else {
              if (!cached_key_[array_id]->IsNull(i)) {
                (indices_begin + indices_i)->array_id = array_id;
                (indices_begin + indices_i)->id = i;
                indices_i++;
              } else {
                (indices_end - nulls_total_ + indices_null)->array_id = array_id;
                (indices_end - nulls_total_ + indices_null)->id = i;
                indices_null++;
              }
            }
          }
        }
      }
    }
  }

  int64_t PartitionNaNs(ArrayItemIndexS* indices_begin, ArrayItemIndexS* indices_end) {
    int64_t indices_i = 0;
    int64_t indices_nan = 0;

    for (int array_id = 0; array_id < num_batches_; array_id++) {
      for (int64_t i = 0; i < length_list_[array_id]; i++) {
        if (cached_key_[array_id]->IsNull(i)) {
          continue;
        }
        if (nulls_first_) {
          if (asc_) {
            // values should be partitioned to:
            // null, null, ..., valid-1, valid-2, ..., valid-3, NaN, NaN, ...
            if (!std::isnan(cached_key_[array_id]->GetView(i))) {
              (indices_begin + nulls_total_ + indices_i)->array_id = array_id;
              (indices_begin + nulls_total_ + indices_i)->id = i;
              indices_i++;
            } else {
              (indices_end - indices_nan - 1)->array_id = array_id;
              (indices_end - indices_nan - 1)->id = i;
              indices_nan++;
            }
          } else {
            // values should be partitioned to:
            // null, null, ..., NaN, NaN, ..., valid-1, valid-2, ..., valid-3,
            // ...
            if (!std::isnan(cached_key_[array_id]->GetView(i))) {
              (indices_end - indices_i - 1)->array_id = array_id;
              (indices_end - indices_i - 1)->id = i;
              indices_i++;
            } else {
              (indices_begin + nulls_total_ + indices_nan)->array_id = array_id;
              (indices_begin + nulls_total_ + indices_nan)->id = i;
              indices_nan++;
            }
          }
        } else {
          if (asc_) {
            // values should be partitioned to:
            // valid-1, valid-2, ..., valid-3, ..., NaN, NaN, ..., null, null,
            // ...
            if (!std::isnan(cached_key_[array_id]->GetView(i))) {
              (indices_begin + indices_i)->array_id = array_id;
              (indices_begin + indices_i)->id = i;
              indices_i++;
            } else {
              (indices_end - nulls_total_ - indices_nan - 1)->array_id = array_id;
              (indices_end - nulls_total_ - indices_nan - 1)->id = i;
              indices_nan++;
            }
          } else {
            // values should be partitioned to:
            // NaN, NaN, ..., valid-1, valid-2, ..., valid-3, ..., null, null,
            // ...
            if (!std::isnan(cached_key_[array_id]->GetView(i))) {
              (indices_end - nulls_total_ - indices_i - 1)->array_id = array_id;
              (indices_end - nulls_total_ - indices_i - 1)->id = i;
              indices_i++;
            } else {
              (indices_begin + indices_nan)->array_id = array_id;
              (indices_begin + indices_nan)->id = i;
              indices_nan++;
            }
          }
        }
      }
    }
    return indices_nan;
  }

  template <typename T>
  auto Partition(ArrayItemIndexS* indices_begin, ArrayItemIndexS* indices_end,
                 int64_t& num_nan) ->
      typename std::enable_if_t<arrow::is_floating_type<T>::value> {
    PartitionNulls(indices_begin, indices_end);
    if (NaN_check_) {
      num_nan = PartitionNaNs(indices_begin, indices_end);
    }
  }

  template <typename T>
  auto Partition(ArrayItemIndexS* indices_begin, ArrayItemIndexS* indices_end,
                 int64_t& num_nan) ->
      typename std::enable_if_t<!arrow::is_floating_type<T>::value> {
    PartitionNulls(indices_begin, indices_end);
  }

  template <typename T>
  auto Sort(ArrayItemIndexS* indices_begin, ArrayItemIndexS* indices_end, int64_t num_nan)
      -> typename std::enable_if_t<is_number_bool_date<T>::value> {
    if (asc_) {
      if (nulls_first_) {
        ska_sort(indices_begin + nulls_total_, indices_begin + items_total_ - num_nan,
                 [this](auto& x) -> decltype(auto) {
                   return cached_key_[x.array_id]->GetView(x.id);
                 });
      } else {
        ska_sort(indices_begin, indices_begin + items_total_ - nulls_total_ - num_nan,
                 [this](auto& x) -> decltype(auto) {
                   return cached_key_[x.array_id]->GetView(x.id);
                 });
      }
    } else {
      auto comp = [this](const ArrayItemIndexS& x, const ArrayItemIndexS& y) {
        return cached_key_[x.array_id]->GetView(x.id) >
               cached_key_[y.array_id]->GetView(y.id);
      };
      if (nulls_first_) {
        std::sort(indices_begin + nulls_total_ + num_nan, indices_begin + items_total_,
                  comp);
      } else {
        std::sort(indices_begin + num_nan, indices_begin + items_total_ - nulls_total_,
                  comp);
      }
    }
  }

  template <typename T>
  auto Sort(ArrayItemIndexS* indices_begin, ArrayItemIndexS* indices_end, int64_t num_nan)
      -> typename std::enable_if_t<std::is_same<T, arrow::StringType>::value> {
    if (asc_) {
      auto comp = [this](const ArrayItemIndexS& x, const ArrayItemIndexS& y) {
        return cached_key_[x.array_id]->GetString(x.id) <
               cached_key_[y.array_id]->GetString(y.id);
      };
      if (nulls_first_) {
        std::sort(indices_begin + nulls_total_, indices_begin + items_total_, comp);
      } else {
        std::sort(indices_begin, indices_begin + items_total_ - nulls_total_, comp);
      }
    } else {
      auto comp = [this](const ArrayItemIndexS& x, const ArrayItemIndexS& y) {
        return cached_key_[x.array_id]->GetString(x.id) >
               cached_key_[y.array_id]->GetString(y.id);
      };
      if (nulls_first_) {
        std::sort(indices_begin + nulls_total_, indices_begin + items_total_, comp);
      } else {
        std::sort(indices_begin, indices_begin + items_total_ - nulls_total_, comp);
      }
    }
  }

  template <typename T>
  auto Sort(ArrayItemIndexS* indices_begin, ArrayItemIndexS* indices_end, int64_t num_nan)
      -> typename std::enable_if_t<arrow::is_decimal_type<T>::value> {
    if (asc_) {
      auto comp = [this](const ArrayItemIndexS& x, const ArrayItemIndexS& y) {
        return cached_key_[x.array_id]->GetView(x.id) <
               cached_key_[y.array_id]->GetView(y.id);
      };
      if (nulls_first_) {
        std::sort(indices_begin + nulls_total_, indices_begin + items_total_ - num_nan,
                  comp);
      } else {
        std::sort(indices_begin, indices_begin + items_total_ - nulls_total_ - num_nan,
                  comp);
      }
    } else {
      auto comp = [this](const ArrayItemIndexS& x, const ArrayItemIndexS& y) {
        return cached_key_[x.array_id]->GetView(x.id) >
               cached_key_[y.array_id]->GetView(y.id);
      };
      if (nulls_first_) {
        std::sort(indices_begin + nulls_total_ + num_nan, indices_begin + items_total_,
                  comp);
      } else {
        std::sort(indices_begin + num_nan, indices_begin + items_total_ - nulls_total_,
                  comp);
      }
    }
  }

  arrow::Status FinishInternal(std::shared_ptr<FixedSizeBinaryArray>* out) {
    // initiate buffer for all arrays
    std::shared_ptr<arrow::Buffer> indices_buf;
    int64_t buf_size = items_total_ * sizeof(ArrayItemIndexS);
    auto maybe_buffer = arrow::AllocateBuffer(buf_size, ctx_->memory_pool());
    indices_buf = *std::move(maybe_buffer);
    ArrayItemIndexS* indices_begin =
        reinterpret_cast<ArrayItemIndexS*>(indices_buf->mutable_data());
    ArrayItemIndexS* indices_end = indices_begin + items_total_;
    // do partition and sort here
    int64_t num_nan = 0;
    Partition<DATATYPE>(indices_begin, indices_end, num_nan);
    Sort<DATATYPE>(indices_begin, indices_end, num_nan);
    std::shared_ptr<arrow::FixedSizeBinaryType> out_type;
    RETURN_NOT_OK(
        MakeFixedSizeBinaryType(sizeof(ArrayItemIndexS) / sizeof(int32_t), &out_type));
    RETURN_NOT_OK(MakeFixedSizeBinaryArray(out_type, items_total_, indices_buf, out));
    cached_key_.clear();

    return arrow::Status::OK();
  }

  arrow::Status MakeResultIterator(
      std::shared_ptr<arrow::Schema> schema,
      std::shared_ptr<ResultIterator<arrow::RecordBatch>>* out) override {
    std::shared_ptr<FixedSizeBinaryArray> indices_out;
    RETURN_NOT_OK(FinishInternal(&indices_out));

    local_result_iter_ = std::make_shared<SorterResultIterator>(ctx_, schema, indices_out,
                                                                std::move(cached_));
    *out = local_result_iter_;
    return arrow::Status::OK();
  }

 private:
  using ArrayType_key = typename TypeTraits<DATATYPE>::ArrayType;
  std::vector<std::shared_ptr<ArrayType_key>> cached_key_;
  std::vector<arrow::ArrayVector> cached_;
  arrow::compute::ExecContext* ctx_;
  std::shared_ptr<arrow::Schema> result_schema_;
  std::shared_ptr<gandiva::Projector> key_projector_;
  bool nulls_first_;
  bool asc_;
  bool NaN_check_;
  std::vector<int64_t> length_list_;
  uint64_t num_batches_ = 0;
  uint64_t items_total_ = 0;
  uint64_t nulls_total_ = 0;
  int col_num_;
  int key_id_;
  std::shared_ptr<SorterResultIterator> local_result_iter_;
};

///////////////  SortArraysMultipleKeys  ////////////////
class SortMultiplekeyKernel : public SortArraysToIndicesKernel::Impl {
 public:
  SortMultiplekeyKernel(arrow::compute::ExecContext* ctx,
                        std::shared_ptr<arrow::Schema> result_schema,
                        std::shared_ptr<gandiva::Projector> key_projector,
                        std::vector<std::shared_ptr<arrow::DataType>> projected_types,
                        std::vector<std::shared_ptr<arrow::Field>> key_field_list,
                        std::vector<bool> sort_directions, std::vector<bool> nulls_order,
                        bool NaN_check)
      : ctx_(ctx),
        nulls_order_(nulls_order),
        sort_directions_(sort_directions),
        result_schema_(result_schema),
        key_projector_(key_projector),
        key_field_list_(key_field_list),
        NaN_check_(NaN_check) {
#ifdef DEBUG
    std::cout << "UseSortMultiplekeyKernel" << std::endl;
#endif
    for (auto field : key_field_list) {
      auto indices = GetIndicesFromSchemaCaseInsensitive(result_schema, field->name());
      if (indices.size() != 1) {
        std::cout << "[ERROR] SortArraysToIndicesKernel::Impl can't find key "
                  << field->ToString() << " from " << result_schema->ToString()
                  << std::endl;
        throw;
      }
      key_index_list_.push_back(indices[0]);
    }
    col_num_ = result_schema->num_fields();
    int i = 0;
    for (auto type : projected_types) {
      auto field = arrow::field(std::to_string(i), type);
      projected_field_list_.push_back(field);
      i++;
    }
  }
  ~SortMultiplekeyKernel() {}

  arrow::Status Evaluate(ArrayList& in) override {
    num_batches_++;
    if (cached_.size() <= col_num_) {
      cached_.resize(col_num_ + 1);
    }
    for (int i = 0; i < col_num_; i++) {
      cached_[i].push_back(in[i]);
    }
    if (key_projector_) {
      int projected_col_num = projected_field_list_.size();
      if (projected_.size() <= projected_col_num) {
        projected_.resize(projected_col_num + 1);
      }
      std::vector<std::shared_ptr<arrow::Array>> projected_batch;
      // do projection here, and the projected arrays are used for comparison
      auto length = in.size() > 0 ? in[0]->length() : 0;
      auto in_batch = arrow::RecordBatch::Make(result_schema_, length, in);
      RETURN_NOT_OK(
          key_projector_->Evaluate(*in_batch, ctx_->memory_pool(), &projected_batch));
      for (int i = 0; i < projected_col_num; i++) {
        std::shared_ptr<arrow::Array> col = projected_batch[i];
        projected_[i].push_back(col);
      }
    }
    items_total_ += in[0]->length();
    length_list_.push_back(in[0]->length());
    return arrow::Status::OK();
  }

  int compareInternal(int left_array_id, int64_t left_id, int right_array_id,
                      int64_t right_id, int keys_num) {
    int key_idx = 0;
    while (key_idx < keys_num) {
      // In comparison, 1 represents for true, 0 for false, and 2 for equal.
      int cmp_res = 2;
      cmp_functions_[key_idx](left_array_id, right_array_id, left_id, right_id, cmp_res);
      if (cmp_res != 2) {
        return cmp_res;
      }
      key_idx += 1;
    }
    return 2;
  }

  bool compareRow(int left_array_id, int64_t left_id, int right_array_id,
                  int64_t right_id, int keys_num) {
    if (compareInternal(left_array_id, left_id, right_array_id, right_id, keys_num) ==
        1) {
      return true;
    }
    return false;
  }

  void Sort(ArrayItemIndexS* indices_begin, ArrayItemIndexS* indices_end) {
    int keys_num = sort_directions_.size();
    auto comp = [this, &keys_num](const ArrayItemIndexS& x, const ArrayItemIndexS& y) {
      return compareRow(x.array_id, x.id, y.array_id, y.id, keys_num);
    };
    gfx::timsort(indices_begin, indices_begin + items_total_, comp);
  }

  void Partition(ArrayItemIndexS* indices_begin, ArrayItemIndexS* indices_end) {
    int64_t indices_i = 0;
    int64_t indices_null = 0;
    for (int array_id = 0; array_id < num_batches_; array_id++) {
      for (int64_t i = 0; i < length_list_[array_id]; i++) {
        (indices_begin + indices_i)->array_id = array_id;
        (indices_begin + indices_i)->id = i;
        indices_i++;
      }
    }
  }

  arrow::Status FinishInternal(std::shared_ptr<FixedSizeBinaryArray>* out) {
    // initiate buffer for all arrays
    std::shared_ptr<arrow::Buffer> indices_buf;
    int64_t buf_size = items_total_ * sizeof(ArrayItemIndexS);
    auto maybe_buffer = arrow::AllocateBuffer(buf_size, ctx_->memory_pool());
    indices_buf = *std::move(maybe_buffer);
    ArrayItemIndexS* indices_begin =
        reinterpret_cast<ArrayItemIndexS*>(indices_buf->mutable_data());
    ArrayItemIndexS* indices_end = indices_begin + items_total_;
    // do partition and sort here
    Partition(indices_begin, indices_end);
    if (key_projector_) {
      std::vector<int> projected_key_idx_list;
      for (int i = 0; i < projected_field_list_.size(); i++) {
        projected_key_idx_list.push_back(i);
      }
      MakeCmpFunction(projected_, projected_field_list_, projected_key_idx_list,
                      sort_directions_, nulls_order_, NaN_check_, cmp_functions_);
    } else {
      MakeCmpFunction(cached_, key_field_list_, key_index_list_, sort_directions_,
                      nulls_order_, NaN_check_, cmp_functions_);
    }
    Sort(indices_begin, indices_end);
    std::shared_ptr<arrow::FixedSizeBinaryType> out_type;
    RETURN_NOT_OK(
        MakeFixedSizeBinaryType(sizeof(ArrayItemIndexS) / sizeof(int32_t), &out_type));
    RETURN_NOT_OK(MakeFixedSizeBinaryArray(out_type, items_total_, indices_buf, out));
    return arrow::Status::OK();
  }

  arrow::Status MakeResultIterator(
      std::shared_ptr<arrow::Schema> schema,
      std::shared_ptr<ResultIterator<arrow::RecordBatch>>* out) override {
    std::shared_ptr<FixedSizeBinaryArray> indices_out;
    RETURN_NOT_OK(FinishInternal(&indices_out));
    *out = std::make_shared<SorterResultIterator>(ctx_, schema, indices_out,
                                                  std::move(cached_));
    return arrow::Status::OK();
  }

 private:
  std::vector<arrow::ArrayVector> cached_;
  std::vector<arrow::ArrayVector> projected_;
  arrow::compute::ExecContext* ctx_;
  std::shared_ptr<arrow::Schema> result_schema_;
  std::shared_ptr<gandiva::Projector> key_projector_;
  std::vector<std::shared_ptr<arrow::Field>> key_field_list_;
  std::vector<std::shared_ptr<arrow::Field>> projected_field_list_;
  std::vector<bool> nulls_order_;
  std::vector<bool> sort_directions_;
  std::vector<int> key_index_list_;
  bool NaN_check_;
  std::vector<int64_t> length_list_;
  uint64_t num_batches_ = 0;
  uint64_t items_total_ = 0;
  int col_num_;
  std::vector<std::function<void(int, int, int64_t, int64_t, int&)>> cmp_functions_;
};

arrow::Status SortArraysToIndicesKernel::Make(
    arrow::compute::ExecContext* ctx, std::shared_ptr<arrow::Schema> result_schema,
    gandiva::NodeVector sort_key_node,
    std::vector<std::shared_ptr<arrow::Field>> key_field_list,
    std::vector<bool> sort_directions, std::vector<bool> nulls_order, bool NaN_check,
    bool do_codegen, int result_type, std::shared_ptr<KernalBase>* out) {
  *out = std::make_shared<SortArraysToIndicesKernel>(
      ctx, result_schema, sort_key_node, key_field_list, sort_directions, nulls_order,
      NaN_check, do_codegen, result_type);
  return arrow::Status::OK();
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
  PROCESS(arrow::TimestampType)

SortArraysToIndicesKernel::SortArraysToIndicesKernel(
    arrow::compute::ExecContext* ctx, std::shared_ptr<arrow::Schema> result_schema,
    gandiva::NodeVector sort_key_node,
    std::vector<std::shared_ptr<arrow::Field>> key_field_list,
    std::vector<bool> sort_directions, std::vector<bool> nulls_order, bool NaN_check,
    bool do_codegen, int result_type) {
  // represents whether need to projection for sort keys
  bool pre_processed_key_ = false;
  gandiva::NodePtr key_project;
  gandiva::ExpressionVector key_project_exprs;

  for (auto node : sort_key_node) {
    std::shared_ptr<TypedNodeVisitor> node_visitor;
    THROW_NOT_OK(MakeTypedNodeVisitor(node, &node_visitor));
    if (node_visitor->GetResultType() != TypedNodeVisitor::FieldNode) {
      // if projection is needed
      pre_processed_key_ = true;
      break;
    }
  }
  // If not all key_node are FieldNode, we need to do projection
  std::shared_ptr<gandiva::Projector> key_projector;
  std::vector<std::shared_ptr<arrow::DataType>> projected_types;
  if (pre_processed_key_) {
    key_project_exprs = GetGandivaKernel(sort_key_node);
    auto configuration = gandiva::ConfigurationBuilder().DefaultConfiguration();
    THROW_NOT_OK(gandiva::Projector::Make(result_schema, key_project_exprs, configuration,
                                          &key_projector));
    for (const auto& expr : key_project_exprs) {
      auto key_type = expr->root()->return_type();
      projected_types.push_back(key_type);
    }
  }

  if (key_field_list.size() == 1 && result_schema->num_fields() == 1 &&
      key_field_list[0]->type()->id() != arrow::Type::STRING &&
      key_field_list[0]->type()->id() != arrow::Type::BOOL) {
    // Will use SortInplace when sorting for one non-string and non-boolean col
#ifdef DEBUG
    std::cout << "UseSortInplace" << std::endl;
#endif
    if (key_field_list[0]->type()->id() == arrow::Type::DECIMAL128) {
      impl_.reset(new SortInplaceKernel<arrow::Decimal128Type, arrow::Decimal128>(
          ctx, result_schema, key_projector, sort_directions, nulls_order, NaN_check));
    } else {
      switch (key_field_list[0]->type()->id()) {
#define PROCESS(InType)                                                               \
  case InType::type_id: {                                                             \
    using CType = typename arrow::TypeTraits<InType>::CType;                          \
    impl_.reset(new SortInplaceKernel<InType, CType>(                                 \
        ctx, result_schema, key_projector, sort_directions, nulls_order, NaN_check)); \
  } break;
        PROCESS_SUPPORTED_TYPES(PROCESS)
#undef PROCESS
        default: {
          std::cout << "SortInplaceKernel type not supported, type is "
                    << key_field_list[0]->type() << std::endl;
        } break;
      }
    }
  } else if (key_field_list.size() == 1 && result_schema->num_fields() >= 1) {
    // Will use SortOnekey when:
    // 1. sorting for one col with payload 2. sorting for one string col or one
    // bool col
#ifdef DEBUG
    std::cout << "UseSortOneKey" << std::endl;
#endif
    if (pre_processed_key_) {
      // if needs projection, will use projected type for key col
      if (projected_types[0]->id() == arrow::Type::STRING) {
        impl_.reset(new SortOnekeyKernel<arrow::StringType>(
            ctx, result_schema, key_projector, key_field_list, sort_directions,
            nulls_order, NaN_check));
      } else if (projected_types[0]->id() == arrow::Type::DECIMAL128) {
        impl_.reset(new SortOnekeyKernel<arrow::Decimal128Type>(
            ctx, result_schema, key_projector, key_field_list, sort_directions,
            nulls_order, NaN_check));
      } else {
        switch (projected_types[0]->id()) {
#define PROCESS(InType)                                                         \
  case InType::type_id: {                                                       \
    using CType = typename arrow::TypeTraits<InType>::CType;                    \
    impl_.reset(new SortOnekeyKernel<InType>(ctx, result_schema, key_projector, \
                                             key_field_list, sort_directions,   \
                                             nulls_order, NaN_check));          \
  } break;
          PROCESS_SUPPORTED_TYPES(PROCESS)
#undef PROCESS
          default: {
            std::cout << "SortOnekeyKernel type not supported, type is "
                      << key_field_list[0]->type() << std::endl;
          } break;
        }
      }
    } else {
      // if no projection, will use the original type for key col
      if (key_field_list[0]->type()->id() == arrow::Type::STRING) {
        impl_.reset(new SortOnekeyKernel<arrow::StringType>(
            ctx, result_schema, key_projector, key_field_list, sort_directions,
            nulls_order, NaN_check));
      } else if (key_field_list[0]->type()->id() == arrow::Type::DECIMAL128) {
        impl_.reset(new SortOnekeyKernel<arrow::Decimal128Type>(
            ctx, result_schema, key_projector, key_field_list, sort_directions,
            nulls_order, NaN_check));
      } else {
        switch (key_field_list[0]->type()->id()) {
#define PROCESS(InType)                                                         \
  case InType::type_id: {                                                       \
    using CType = typename arrow::TypeTraits<InType>::CType;                    \
    impl_.reset(new SortOnekeyKernel<InType>(ctx, result_schema, key_projector, \
                                             key_field_list, sort_directions,   \
                                             nulls_order, NaN_check));          \
  } break;
          PROCESS_SUPPORTED_TYPES(PROCESS)
#undef PROCESS
          default: {
            std::cout << "SortOnekeyKernel type not supported, type is "
                      << key_field_list[0]->type() << std::endl;
          } break;
        }
      }
    }
  } else {
    if (do_codegen) {
      // Will use Sort with Codegen for multiple-key sort
      impl_.reset(new Impl(ctx, result_schema, key_projector, projected_types,
                           key_field_list, sort_directions, nulls_order, NaN_check));
    } else {
      // Will use Sort without Codegen for multiple-key sort
      impl_.reset(new SortMultiplekeyKernel(ctx, result_schema, key_projector,
                                            projected_types, key_field_list,
                                            sort_directions, nulls_order, NaN_check));
    }
  }
  kernel_name_ = "SortArraysToIndicesKernel";
}
#undef PROCESS_SUPPORTED_TYPES

arrow::Status SortArraysToIndicesKernel::Evaluate(ArrayList& in) {
  return impl_->Evaluate(in);
}

arrow::Status SortArraysToIndicesKernel::Spill(int64_t* spilled_size) {
  return impl_->Spill(spilled_size);
}

arrow::Status SortArraysToIndicesKernel::MakeResultIterator(
    std::shared_ptr<arrow::Schema> schema,
    std::shared_ptr<ResultIterator<arrow::RecordBatch>>* out) {
  return impl_->MakeResultIterator(schema, out);
}

arrow::Status SortArraysToIndicesKernel::MakeResultIterator(
    std::shared_ptr<arrow::Schema> schema,
    std::shared_ptr<ResultIterator<SortRelation>>* out) {
  return impl_->MakeResultIterator(schema, out);
}

std::string SortArraysToIndicesKernel::GetSignature() { return impl_->GetSignature(); }

}  // namespace extra
}  // namespace arrowcompute
}  // namespace codegen
}  // namespace sparkcolumnarplugin
