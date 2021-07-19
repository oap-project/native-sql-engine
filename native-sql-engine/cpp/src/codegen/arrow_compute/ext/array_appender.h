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

#pragma once

#include <arrow/status.h>

#include <cstdint>
#include <vector>

#include "codegen/arrow_compute/ext/array_item_index.h"
#include "precompile/type_traits.h"

using sparkcolumnarplugin::precompile::enable_if_decimal;

namespace sparkcolumnarplugin {
namespace codegen {
namespace arrowcompute {
namespace extra {

class AppenderBase {
 public:
  virtual ~AppenderBase() {}

  enum AppenderType { left, right, exist };
  virtual AppenderType GetType() { return left; }

  virtual arrow::Status AddArray(const std::shared_ptr<arrow::Array>& arr) {
    return arrow::Status::NotImplemented("AppenderBase AddArray is abstract.");
  }

  virtual arrow::Status PopArray() {
    return arrow::Status::NotImplemented("AppenderBase PopArray is abstract.");
  }

  virtual arrow::Status ClearArrays() {
    return arrow::Status::NotImplemented("AppenderBase ClearArrays is abstract.");
  }

  virtual arrow::Status Append(const uint16_t& array_id, const uint16_t& item_id) {
    return arrow::Status::NotImplemented("AppenderBase Append is abstract.");
  }

  virtual arrow::Status Append(const uint16_t& array_id, const uint16_t& item_id,
                               int repeated) {
    return arrow::Status::NotImplemented("AppenderBase Append is abstract.");
  }

  virtual arrow::Status Append(const std::vector<ArrayItemIndex>& index_list) {
    return arrow::Status::NotImplemented("AppenderBase Append is abstract.");
  }

  virtual arrow::Status AppendNull() {
    return arrow::Status::NotImplemented("AppenderBase AppendNull is abstract.");
  }

  virtual arrow::Status Finish(std::shared_ptr<arrow::Array>* out_) {
    return arrow::Status::NotImplemented("AppenderBase Finish is abstract.");
  }

  virtual arrow::Status Reserve(uint64_t) {
    return arrow::Status::NotImplemented("AppenderBase Reset is abstract.");
  }

  virtual arrow::Status Reset() {
    return arrow::Status::NotImplemented("AppenderBase Reset is abstract.");
  }

  virtual arrow::Status AppendExistence(bool is_exist) {
    return arrow::Status::NotImplemented("AppenderBase AppendExistence is abstract.");
  }
};

template <typename DataType, typename Enable = void>
class ArrayAppender {};

template <typename T>
using is_number_or_date = std::integral_constant<bool, arrow::is_number_type<T>::value ||
                                                           arrow::is_date_type<T>::value>;
template <typename T>
using is_timestamp = std::integral_constant<bool, arrow::is_timestamp_type<T>::value>;

template <typename DataType, typename R = void>
using enable_if_timestamp = std::enable_if_t<is_timestamp<DataType>::value, R>;

template <typename DataType>
class ArrayAppender<DataType, enable_if_timestamp<DataType>> : public AppenderBase {
 public:
  ArrayAppender(arrow::compute::ExecContext* ctx,
                std::shared_ptr<arrow::DataType> data_type, AppenderType type = left)
      : ctx_(ctx), data_type_(data_type), type_(type) {
    std::unique_ptr<arrow::ArrayBuilder> array_builder;
    arrow::MakeBuilder(ctx_->memory_pool(), data_type, &array_builder);
    builder_.reset(arrow::internal::checked_cast<BuilderType_*>(array_builder.release()));
  }
  ~ArrayAppender() {}

  AppenderType GetType() override { return type_; }
  arrow::Status AddArray(const std::shared_ptr<arrow::Array>& arr) override {
    auto typed_arr_ = std::dynamic_pointer_cast<ArrayType_>(arr);
    cached_arr_.push_back(typed_arr_);
    if (typed_arr_->null_count() > 0) has_null_ = true;
    return arrow::Status::OK();
  }

  arrow::Status PopArray() override {
    cached_arr_.pop_back();
    has_null_ = false;
    return arrow::Status::OK();
  }

  arrow::Status ClearArrays() override {
    cached_arr_.clear();
    has_null_ = false;
    return arrow::Status::OK();
  }

  arrow::Status Append(const uint16_t& array_id, const uint16_t& item_id) override {
    if (has_null_ && cached_arr_[array_id]->null_count() > 0 &&
        cached_arr_[array_id]->IsNull(item_id)) {
      RETURN_NOT_OK(builder_->AppendNull());
    } else {
      RETURN_NOT_OK(builder_->Append(cached_arr_[array_id]->GetView(item_id)));
    }
    return arrow::Status::OK();
  }

  arrow::Status Append(const uint16_t& array_id, const uint16_t& item_id,
                       int repeated) override {
    if (repeated == 0) return arrow::Status::OK();
    if (has_null_ && cached_arr_[array_id]->null_count() > 0 &&
        cached_arr_[array_id]->IsNull(item_id)) {
      RETURN_NOT_OK(builder_->AppendNulls(repeated));
    } else {
      auto val = cached_arr_[array_id]->GetView(item_id);
      std::vector<CType> values;
      values.resize(repeated, val);
      RETURN_NOT_OK(builder_->AppendValues(values.data(), repeated));
    }
    return arrow::Status::OK();
  }

  arrow::Status Append(const std::vector<ArrayItemIndex>& index_list) {
    for (auto tmp : index_list) {
      if (has_null_ && cached_arr_[tmp.array_id]->null_count() > 0 &&
          cached_arr_[tmp.array_id]->IsNull(tmp.id)) {
        RETURN_NOT_OK(builder_->AppendNull());
      } else {
        RETURN_NOT_OK(builder_->Append(cached_arr_[tmp.array_id]->GetView(tmp.id)));
      }
    }
    return arrow::Status::OK();
  }

  arrow::Status AppendNull() override { return builder_->AppendNull(); }

  arrow::Status Finish(std::shared_ptr<arrow::Array>* out_) override {
    auto status = builder_->Finish(out_);
    return status;
  }

  arrow::Status Reset() override {
    builder_->Reset();
    return arrow::Status::OK();
  }

 private:
  using BuilderType_ = typename arrow::TypeTraits<DataType>::BuilderType;
  using ArrayType_ = typename arrow::TypeTraits<DataType>::ArrayType;
  using CType = typename arrow::TypeTraits<DataType>::CType;
  std::unique_ptr<BuilderType_> builder_;
  std::vector<std::shared_ptr<ArrayType_>> cached_arr_;
  std::shared_ptr<arrow::DataType> data_type_;
  arrow::compute::ExecContext* ctx_;
  AppenderType type_;
  bool has_null_ = false;
};

template <typename DataType, typename R = void>
using enable_if_number_or_date = std::enable_if_t<is_number_or_date<DataType>::value, R>;

template <typename DataType>
class ArrayAppender<DataType, enable_if_number_or_date<DataType>> : public AppenderBase {
 public:
  ArrayAppender(arrow::compute::ExecContext* ctx, AppenderType type = left)
      : ctx_(ctx), type_(type) {
    std::unique_ptr<arrow::ArrayBuilder> array_builder;
    arrow::MakeBuilder(ctx_->memory_pool(), arrow::TypeTraits<DataType>::type_singleton(),
                       &array_builder);
    builder_.reset(arrow::internal::checked_cast<BuilderType_*>(array_builder.release()));
  }
  ~ArrayAppender() {}

  AppenderType GetType() override { return type_; }
  arrow::Status AddArray(const std::shared_ptr<arrow::Array>& arr) override {
    auto typed_arr_ = std::dynamic_pointer_cast<ArrayType_>(arr);
    cached_arr_.push_back(typed_arr_);
    if (typed_arr_->null_count() > 0) has_null_ = true;
    return arrow::Status::OK();
  }

  arrow::Status PopArray() override {
    cached_arr_.pop_back();
    has_null_ = false;
    return arrow::Status::OK();
  }
  arrow::Status ClearArrays() override {
    cached_arr_.clear();
    has_null_ = false;
    return arrow::Status::OK();
  }
  arrow::Status Append(const uint16_t& array_id, const uint16_t& item_id) override {
    if (has_null_ && cached_arr_[array_id]->null_count() > 0 &&
        cached_arr_[array_id]->IsNull(item_id)) {
      RETURN_NOT_OK(builder_->AppendNull());
    } else {
      RETURN_NOT_OK(builder_->Append(cached_arr_[array_id]->GetView(item_id)));
    }
    return arrow::Status::OK();
  }

  arrow::Status Append(const uint16_t& array_id, const uint16_t& item_id,
                       int repeated) override {
    if (repeated == 0) return arrow::Status::OK();
    if (has_null_ && cached_arr_[array_id]->null_count() > 0 &&
        cached_arr_[array_id]->IsNull(item_id)) {
      RETURN_NOT_OK(builder_->AppendNulls(repeated));
    } else {
      auto val = cached_arr_[array_id]->GetView(item_id);
      std::vector<CType> values;
      values.resize(repeated, val);
      RETURN_NOT_OK(builder_->AppendValues(values.data(), repeated));
    }
    return arrow::Status::OK();
  }

  arrow::Status Append(const std::vector<ArrayItemIndex>& index_list) {
    for (auto tmp : index_list) {
      if (has_null_ && cached_arr_[tmp.array_id]->null_count() > 0 &&
          cached_arr_[tmp.array_id]->IsNull(tmp.id)) {
        RETURN_NOT_OK(builder_->AppendNull());
      } else {
        RETURN_NOT_OK(builder_->Append(cached_arr_[tmp.array_id]->GetView(tmp.id)));
      }
    }
    return arrow::Status::OK();
  }

  arrow::Status AppendNull() override { return builder_->AppendNull(); }

  arrow::Status Finish(std::shared_ptr<arrow::Array>* out_) override {
    auto status = builder_->Finish(out_);
    return status;
  }

  arrow::Status Reset() override {
    builder_->Reset();
    return arrow::Status::OK();
  }

 private:
  using BuilderType_ = typename arrow::TypeTraits<DataType>::BuilderType;
  using ArrayType_ = typename arrow::TypeTraits<DataType>::ArrayType;
  using CType = typename arrow::TypeTraits<DataType>::CType;
  std::unique_ptr<BuilderType_> builder_;
  std::vector<std::shared_ptr<ArrayType_>> cached_arr_;
  arrow::compute::ExecContext* ctx_;
  AppenderType type_;
  bool has_null_ = false;
};

template <typename DataType>
class ArrayAppender<DataType, arrow::enable_if_string_like<DataType>>
    : public AppenderBase {
 public:
  ArrayAppender(arrow::compute::ExecContext* ctx, AppenderType type = left)
      : ctx_(ctx), type_(type) {
    std::unique_ptr<arrow::ArrayBuilder> array_builder;
    arrow::MakeBuilder(ctx_->memory_pool(), arrow::TypeTraits<DataType>::type_singleton(),
                       &array_builder);
    builder_.reset(arrow::internal::checked_cast<BuilderType_*>(array_builder.release()));
  }
  ~ArrayAppender() {}

  AppenderType GetType() override { return type_; }
  arrow::Status AddArray(const std::shared_ptr<arrow::Array>& arr) override {
    auto typed_arr_ = std::dynamic_pointer_cast<ArrayType_>(arr);
    cached_arr_.push_back(typed_arr_);
    if (typed_arr_->null_count() > 0) has_null_ = true;
    return arrow::Status::OK();
  }

  arrow::Status PopArray() override {
    cached_arr_.pop_back();
    has_null_ = false;
    return arrow::Status::OK();
  }
  arrow::Status ClearArrays() override {
    cached_arr_.clear();
    has_null_ = false;
    return arrow::Status::OK();
  }
  arrow::Status Append(const uint16_t& array_id, const uint16_t& item_id) override {
    if (has_null_ && cached_arr_[array_id]->null_count() > 0 &&
        cached_arr_[array_id]->IsNull(item_id)) {
      RETURN_NOT_OK(builder_->AppendNull());
    } else {
      RETURN_NOT_OK(builder_->Append(cached_arr_[array_id]->GetView(item_id)));
    }
    return arrow::Status::OK();
  }

  arrow::Status Append(const uint16_t& array_id, const uint16_t& item_id,
                       int repeated) override {
    if (repeated == 0) return arrow::Status::OK();
    if (has_null_ && cached_arr_[array_id]->null_count() > 0 &&
        cached_arr_[array_id]->IsNull(item_id)) {
      RETURN_NOT_OK(builder_->AppendNulls(repeated));
    } else {
      auto val = cached_arr_[array_id]->GetView(item_id);
      for (int i = 0; i < repeated; i++) {
        RETURN_NOT_OK(builder_->Append(val));
      }
    }
    return arrow::Status::OK();
  }

  arrow::Status Append(const std::vector<ArrayItemIndex>& index_list) {
    for (auto tmp : index_list) {
      if (has_null_ && cached_arr_[tmp.array_id]->null_count() > 0 &&
          cached_arr_[tmp.array_id]->IsNull(tmp.id)) {
        RETURN_NOT_OK(builder_->AppendNull());
      } else {
        RETURN_NOT_OK(builder_->Append(cached_arr_[tmp.array_id]->GetView(tmp.id)));
      }
    }
    return arrow::Status::OK();
  }

  arrow::Status AppendNull() override { return builder_->AppendNull(); }

  arrow::Status Finish(std::shared_ptr<arrow::Array>* out_) override {
    auto status = builder_->Finish(out_);
    return status;
  }

  arrow::Status Reset() override {
    builder_->Reset();
    return arrow::Status::OK();
  }

 private:
  using BuilderType_ = typename arrow::TypeTraits<DataType>::BuilderType;
  using ArrayType_ = typename arrow::TypeTraits<DataType>::ArrayType;
  std::unique_ptr<BuilderType_> builder_;
  std::vector<std::shared_ptr<ArrayType_>> cached_arr_;
  arrow::compute::ExecContext* ctx_;
  AppenderType type_;
  bool has_null_ = false;
};

template <typename DataType>
class ArrayAppender<DataType, arrow::enable_if_boolean<DataType>> : public AppenderBase {
 public:
  ArrayAppender(arrow::compute::ExecContext* ctx, AppenderType type = left)
      : ctx_(ctx), type_(type) {
    std::unique_ptr<arrow::ArrayBuilder> array_builder;
    arrow::MakeBuilder(ctx_->memory_pool(), arrow::TypeTraits<DataType>::type_singleton(),
                       &array_builder);
    builder_.reset(arrow::internal::checked_cast<BuilderType_*>(array_builder.release()));
  }
  ~ArrayAppender() {}

  AppenderType GetType() override { return type_; }
  arrow::Status AddArray(const std::shared_ptr<arrow::Array>& arr) override {
    auto typed_arr_ = std::dynamic_pointer_cast<ArrayType_>(arr);
    cached_arr_.push_back(typed_arr_);
    if (typed_arr_->null_count() > 0) has_null_ = true;
    return arrow::Status::OK();
  }

  arrow::Status PopArray() override {
    cached_arr_.pop_back();
    has_null_ = false;
    return arrow::Status::OK();
  }
  arrow::Status ClearArrays() override {
    cached_arr_.clear();
    has_null_ = false;
    return arrow::Status::OK();
  }
  arrow::Status Append(const uint16_t& array_id, const uint16_t& item_id) override {
    if (has_null_ && cached_arr_[array_id]->null_count() > 0 &&
        cached_arr_[array_id]->IsNull(item_id)) {
      RETURN_NOT_OK(builder_->AppendNull());
    } else {
      RETURN_NOT_OK(builder_->Append(cached_arr_[array_id]->GetView(item_id)));
    }
    return arrow::Status::OK();
  }

  arrow::Status Append(const uint16_t& array_id, const uint16_t& item_id,
                       int repeated) override {
    if (repeated == 0) return arrow::Status::OK();
    if (has_null_ && cached_arr_[array_id]->null_count() > 0 &&
        cached_arr_[array_id]->IsNull(item_id)) {
      RETURN_NOT_OK(builder_->AppendNulls(repeated));
    } else {
      auto val = cached_arr_[array_id]->GetView(item_id);
      for (int i = 0; i < repeated; i++) {
        RETURN_NOT_OK(builder_->Append(val));
      }
    }
    return arrow::Status::OK();
  }

  arrow::Status Append(const std::vector<ArrayItemIndex>& index_list) {
    for (auto tmp : index_list) {
      if (has_null_ && cached_arr_[tmp.array_id]->null_count() > 0 &&
          cached_arr_[tmp.array_id]->IsNull(tmp.id)) {
        RETURN_NOT_OK(builder_->AppendNull());
      } else {
        RETURN_NOT_OK(builder_->Append(cached_arr_[tmp.array_id]->GetView(tmp.id)));
      }
    }
    return arrow::Status::OK();
  }

  arrow::Status AppendNull() override { return builder_->AppendNull(); }

  arrow::Status AppendExistence(bool is_exist) { return builder_->Append(is_exist); }

  arrow::Status Finish(std::shared_ptr<arrow::Array>* out_) override {
    auto status = builder_->Finish(out_);
    return status;
  }

  arrow::Status Reset() override {
    builder_->Reset();
    return arrow::Status::OK();
  }

 private:
  using BuilderType_ = typename arrow::TypeTraits<DataType>::BuilderType;
  using ArrayType_ = typename arrow::TypeTraits<DataType>::ArrayType;
  std::unique_ptr<BuilderType_> builder_;
  std::vector<std::shared_ptr<ArrayType_>> cached_arr_;
  arrow::compute::ExecContext* ctx_;
  AppenderType type_;
  bool has_null_ = false;
};

template <typename DataType>
class ArrayAppender<DataType, enable_if_decimal<DataType>> : public AppenderBase {
 public:
  ArrayAppender(arrow::compute::ExecContext* ctx,
                std::shared_ptr<arrow::DataType> data_type, AppenderType type = left)
      : ctx_(ctx), type_(type) {
    std::unique_ptr<arrow::ArrayBuilder> array_builder;
    arrow::MakeBuilder(ctx_->memory_pool(), data_type, &array_builder);
    builder_.reset(arrow::internal::checked_cast<BuilderType_*>(array_builder.release()));
  }
  ~ArrayAppender() {}

  AppenderType GetType() override { return type_; }
  arrow::Status AddArray(const std::shared_ptr<arrow::Array>& arr) override {
    auto typed_arr_ = std::dynamic_pointer_cast<ArrayType_>(arr);
    cached_arr_.push_back(typed_arr_);
    if (typed_arr_->null_count() > 0) has_null_ = true;
    return arrow::Status::OK();
  }

  arrow::Status PopArray() override {
    cached_arr_.pop_back();
    has_null_ = false;
    return arrow::Status::OK();
  }
  arrow::Status ClearArrays() override {
    cached_arr_.clear();
    has_null_ = false;
    return arrow::Status::OK();
  }
  arrow::Status Append(const uint16_t& array_id, const uint16_t& item_id) override {
    if (has_null_ && cached_arr_[array_id]->IsNull(item_id)) {
      RETURN_NOT_OK(builder_->AppendNull());
    } else {
      RETURN_NOT_OK(builder_->Append(cached_arr_[array_id]->GetView(item_id)));
    }
    return arrow::Status::OK();
  }

  arrow::Status Append(const uint16_t& array_id, const uint16_t& item_id,
                       int repeated) override {
    if (repeated == 0) return arrow::Status::OK();
    if (has_null_ && cached_arr_[array_id]->IsNull(item_id)) {
      RETURN_NOT_OK(builder_->AppendNulls(repeated));
    } else {
      auto val = cached_arr_[array_id]->GetView(item_id);
      for (int i = 0; i < repeated; i++) {
        RETURN_NOT_OK(builder_->Append(val));
      }
    }
    return arrow::Status::OK();
  }

  arrow::Status Append(const std::vector<ArrayItemIndex>& index_list) {
    for (auto tmp : index_list) {
      if (has_null_ && cached_arr_[tmp.array_id]->IsNull(tmp.id)) {
        RETURN_NOT_OK(builder_->AppendNull());
      } else {
        RETURN_NOT_OK(builder_->Append(cached_arr_[tmp.array_id]->GetView(tmp.id)));
      }
    }
    return arrow::Status::OK();
  }

  arrow::Status AppendNull() override { return builder_->AppendNull(); }

  arrow::Status Finish(std::shared_ptr<arrow::Array>* out_) override {
    auto status = builder_->Finish(out_);
    return status;
  }

  arrow::Status Reset() override {
    builder_->Reset();
    return arrow::Status::OK();
  }

 private:
  using BuilderType_ = typename arrow::TypeTraits<DataType>::BuilderType;
  using ArrayType_ = typename arrow::TypeTraits<DataType>::ArrayType;
  std::unique_ptr<BuilderType_> builder_;
  std::vector<std::shared_ptr<ArrayType_>> cached_arr_;
  arrow::compute::ExecContext* ctx_;
  AppenderType type_;
  bool has_null_ = false;
};

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
  PROCESS(arrow::StringType)
static arrow::Status MakeAppender(arrow::compute::ExecContext* ctx,
                                  std::shared_ptr<arrow::DataType> type,
                                  AppenderBase::AppenderType appender_type,
                                  std::shared_ptr<AppenderBase>* out) {
  switch (type->id()) {
#define PROCESS(InType)                                                         \
  case InType::type_id: {                                                       \
    auto app_ptr = std::make_shared<ArrayAppender<InType>>(ctx, appender_type); \
    *out = std::dynamic_pointer_cast<AppenderBase>(app_ptr);                    \
  } break;
    PROCESS_SUPPORTED_TYPES(PROCESS)
#undef PROCESS
    case arrow::Decimal128Type::type_id: {
      auto app_ptr = std::make_shared<ArrayAppender<arrow::Decimal128Type>>(
          ctx, type, appender_type);
      *out = std::dynamic_pointer_cast<AppenderBase>(app_ptr);
    } break;
    case arrow::TimestampType::type_id: {
      auto app_ptr =
          std::make_shared<ArrayAppender<arrow::TimestampType>>(ctx, type, appender_type);
      *out = std::dynamic_pointer_cast<AppenderBase>(app_ptr);
    } break;
    default: {
      return arrow::Status::NotImplemented("MakeAppender type not supported, type is ",
                                           type->ToString());
    } break;
  }
  return arrow::Status::OK();
}
#undef PROCESS_SUPPORTED_TYPES

/// unsafe appender ////
template <typename DataType, typename Enable = void>
class UnsafeArrayAppender {};

template <typename DataType>
class UnsafeArrayAppender<DataType, enable_if_timestamp<DataType>> : public AppenderBase {
 public:
  UnsafeArrayAppender(arrow::compute::ExecContext* ctx,
                      std::shared_ptr<arrow::DataType> data_type,
                      AppenderType type = left)
      : ctx_(ctx), data_type_(data_type), type_(type) {
    std::unique_ptr<arrow::ArrayBuilder> array_builder;
    arrow::MakeBuilder(ctx_->memory_pool(), data_type, &array_builder);
    builder_.reset(arrow::internal::checked_cast<BuilderType_*>(array_builder.release()));
  }
  ~UnsafeArrayAppender() {}

  AppenderType GetType() override { return type_; }
  arrow::Status AddArray(const std::shared_ptr<arrow::Array>& arr) override {
    auto typed_arr_ = std::dynamic_pointer_cast<ArrayType_>(arr);
    cached_arr_.push_back(typed_arr_);
    if (typed_arr_->null_count() > 0) has_null_ = true;
    return arrow::Status::OK();
  }

  arrow::Status PopArray() override {
    cached_arr_.pop_back();
    has_null_ = false;
    return arrow::Status::OK();
  }
  arrow::Status ClearArrays() override {
    cached_arr_.clear();
    has_null_ = false;
    return arrow::Status::OK();
  }
  arrow::Status Append(const uint16_t& array_id, const uint16_t& item_id) override {
    if (has_null_ && cached_arr_[array_id]->null_count() > 0 &&
        cached_arr_[array_id]->IsNull(item_id)) {
      builder_->UnsafeAppendNull();
    } else {
      builder_->UnsafeAppend(cached_arr_[array_id]->GetView(item_id));
    }
    return arrow::Status::OK();
  }

  arrow::Status Append(const uint16_t& array_id, const uint16_t& item_id,
                       int repeated) override {
    if (repeated == 0) return arrow::Status::OK();
    if (has_null_ && cached_arr_[array_id]->null_count() > 0 &&
        cached_arr_[array_id]->IsNull(item_id)) {
      // TODO: unloop here and use unsafeappend
      RETURN_NOT_OK(builder_->AppendNulls(repeated));
    } else {
      auto val = cached_arr_[array_id]->GetView(item_id);
      std::vector<CType> values;
      values.resize(repeated, val);
      // TODO: unloop here and use unsafeappend
      RETURN_NOT_OK(builder_->AppendValues(values.data(), repeated));
    }
    return arrow::Status::OK();
  }

  arrow::Status Append(const std::vector<ArrayItemIndex>& index_list) {
    for (auto tmp : index_list) {
      if (has_null_ && cached_arr_[tmp.array_id]->null_count() > 0 &&
          cached_arr_[tmp.array_id]->IsNull(tmp.id)) {
        builder_->AppendNull();
      } else {
        builder_->UnsafeAppend(cached_arr_[tmp.array_id]->GetView(tmp.id));
      }
    }
    return arrow::Status::OK();
  }

  arrow::Status AppendNull() override {
    // TODO: use unsafe append
    return builder_->AppendNull();
  }

  arrow::Status Finish(std::shared_ptr<arrow::Array>* out_) override {
    auto status = builder_->Finish(out_);
    return status;
  }

  arrow::Status Reserve(uint64_t len) override {
    builder_->Reserve(len);
    return arrow::Status::OK();
  }

  arrow::Status Reset() override {
    builder_->Reset();
    return arrow::Status::OK();
  }

 private:
  using BuilderType_ = typename arrow::TypeTraits<DataType>::BuilderType;
  using ArrayType_ = typename arrow::TypeTraits<DataType>::ArrayType;
  using CType = typename arrow::TypeTraits<DataType>::CType;
  std::unique_ptr<BuilderType_> builder_;
  std::vector<std::shared_ptr<ArrayType_>> cached_arr_;
  std::shared_ptr<arrow::DataType> data_type_;
  arrow::compute::ExecContext* ctx_;
  AppenderType type_;
  bool has_null_ = false;
};

template <typename DataType>
class UnsafeArrayAppender<DataType, enable_if_number_or_date<DataType>>
    : public AppenderBase {
 public:
  UnsafeArrayAppender(arrow::compute::ExecContext* ctx, AppenderType type = left)
      : ctx_(ctx), type_(type) {
    std::unique_ptr<arrow::ArrayBuilder> array_builder;
    arrow::MakeBuilder(ctx_->memory_pool(), arrow::TypeTraits<DataType>::type_singleton(),
                       &array_builder);
    builder_.reset(arrow::internal::checked_cast<BuilderType_*>(array_builder.release()));
  }
  ~UnsafeArrayAppender() {}

  AppenderType GetType() override { return type_; }

  arrow::Status AddArray(const std::shared_ptr<arrow::Array>& arr) override {
    auto typed_arr_ = std::dynamic_pointer_cast<ArrayType_>(arr);
    if (typed_arr_->null_count() > 0) has_null_ = true;
    cached_arr_.push_back(typed_arr_);
    return arrow::Status::OK();
  }

  arrow::Status PopArray() override {
    cached_arr_.pop_back();
    has_null_ = false;
    return arrow::Status::OK();
  }
  arrow::Status ClearArrays() override {
    cached_arr_.clear();
    has_null_ = false;
    return arrow::Status::OK();
  }
  arrow::Status Append(const uint16_t& array_id, const uint16_t& item_id) override {
    if (has_null_ && cached_arr_[array_id]->null_count() > 0 &&
        cached_arr_[array_id]->IsNull(item_id)) {
      builder_->UnsafeAppendNull();
    } else {
      builder_->UnsafeAppend(cached_arr_[array_id]->GetView(item_id));
    }
    return arrow::Status::OK();
  }

  arrow::Status Append(const uint16_t& array_id, const uint16_t& item_id,
                       int repeated) override {
    if (repeated == 0) return arrow::Status::OK();
    if (has_null_ && cached_arr_[array_id]->null_count() > 0 &&
        cached_arr_[array_id]->IsNull(item_id)) {
      // TODO: unloop here and use unsafeappend
      RETURN_NOT_OK(builder_->AppendNulls(repeated));
    } else {
      auto val = cached_arr_[array_id]->GetView(item_id);
      std::vector<CType> values;
      values.resize(repeated, val);
      // TODO: unloop here and use unsafeappend
      RETURN_NOT_OK(builder_->AppendValues(values.data(), repeated));
    }
    return arrow::Status::OK();
  }

  arrow::Status Append(const std::vector<ArrayItemIndex>& index_list) {
    for (auto tmp : index_list) {
      if (has_null_ && cached_arr_[tmp.array_id]->null_count() > 0 &&
          cached_arr_[tmp.array_id]->IsNull(tmp.id)) {
        builder_->AppendNull();
      } else {
        builder_->UnsafeAppend(cached_arr_[tmp.array_id]->GetView(tmp.id));
      }
    }
    return arrow::Status::OK();
  }

  arrow::Status AppendNull() override {
    // TODO: use unsafe append
    return builder_->AppendNull();
  }

  arrow::Status Finish(std::shared_ptr<arrow::Array>* out_) override {
    auto status = builder_->Finish(out_);
    return status;
  }

  arrow::Status Reserve(uint64_t len) override {
    builder_->Reserve(len);
    return arrow::Status::OK();
  }

  arrow::Status Reset() override {
    builder_->Reset();
    return arrow::Status::OK();
  }

 private:
  using BuilderType_ = typename arrow::TypeTraits<DataType>::BuilderType;
  using ArrayType_ = typename arrow::TypeTraits<DataType>::ArrayType;
  using CType = typename arrow::TypeTraits<DataType>::CType;
  std::unique_ptr<BuilderType_> builder_;
  std::vector<std::shared_ptr<ArrayType_>> cached_arr_;
  arrow::compute::ExecContext* ctx_;
  AppenderType type_;
  bool has_null_ = false;
};

// TODO(): this is a fake unsafeappende for string array
template <typename DataType>
class UnsafeArrayAppender<DataType, arrow::enable_if_string_like<DataType>>
    : public AppenderBase {
 public:
  UnsafeArrayAppender(arrow::compute::ExecContext* ctx, AppenderType type = left)
      : ctx_(ctx), type_(type) {
    std::unique_ptr<arrow::ArrayBuilder> array_builder;
    arrow::MakeBuilder(ctx_->memory_pool(), arrow::TypeTraits<DataType>::type_singleton(),
                       &array_builder);
    builder_.reset(arrow::internal::checked_cast<BuilderType_*>(array_builder.release()));
  }
  ~UnsafeArrayAppender() {}

  AppenderType GetType() override { return type_; }
  arrow::Status AddArray(const std::shared_ptr<arrow::Array>& arr) override {
    auto typed_arr_ = std::dynamic_pointer_cast<ArrayType_>(arr);
    cached_arr_.push_back(typed_arr_);
    if (typed_arr_->null_count() > 0) has_null_ = true;
    return arrow::Status::OK();
  }

  arrow::Status PopArray() override {
    cached_arr_.pop_back();
    has_null_ = false;
    return arrow::Status::OK();
  }
  arrow::Status ClearArrays() override {
    cached_arr_.clear();
    has_null_ = false;
    return arrow::Status::OK();
  }
  arrow::Status Append(const uint16_t& array_id, const uint16_t& item_id) override {
    if (has_null_ && cached_arr_[array_id]->null_count() > 0 &&
        cached_arr_[array_id]->IsNull(item_id)) {
      RETURN_NOT_OK(builder_->AppendNull());
    } else {
      RETURN_NOT_OK(builder_->Append(cached_arr_[array_id]->GetView(item_id)));
    }
    return arrow::Status::OK();
  }

  arrow::Status Append(const uint16_t& array_id, const uint16_t& item_id,
                       int repeated) override {
    if (repeated == 0) return arrow::Status::OK();
    if (has_null_ && cached_arr_[array_id]->null_count() > 0 &&
        cached_arr_[array_id]->IsNull(item_id)) {
      RETURN_NOT_OK(builder_->AppendNulls(repeated));
    } else {
      auto val = cached_arr_[array_id]->GetView(item_id);
      for (int i = 0; i < repeated; i++) {
        RETURN_NOT_OK(builder_->Append(val));
      }
    }
    return arrow::Status::OK();
  }

  arrow::Status Append(const std::vector<ArrayItemIndex>& index_list) {
    for (auto tmp : index_list) {
      if (has_null_ && cached_arr_[tmp.array_id]->null_count() > 0 &&
          cached_arr_[tmp.array_id]->IsNull(tmp.id)) {
        RETURN_NOT_OK(builder_->AppendNull());
      } else {
        RETURN_NOT_OK(builder_->Append(cached_arr_[tmp.array_id]->GetView(tmp.id)));
      }
    }
    return arrow::Status::OK();
  }

  arrow::Status AppendNull() override { return builder_->AppendNull(); }

  arrow::Status Finish(std::shared_ptr<arrow::Array>* out_) override {
    auto status = builder_->Finish(out_);
    return status;
  }

  arrow::Status Reserve(uint64_t len) override {
    // builder_->Reserve(len);
    return arrow::Status::OK();
  }

  arrow::Status Reset() override {
    builder_->Reset();
    return arrow::Status::OK();
  }

 private:
  using BuilderType_ = typename arrow::TypeTraits<DataType>::BuilderType;
  using ArrayType_ = typename arrow::TypeTraits<DataType>::ArrayType;
  std::unique_ptr<BuilderType_> builder_;
  std::vector<std::shared_ptr<ArrayType_>> cached_arr_;
  arrow::compute::ExecContext* ctx_;
  AppenderType type_;
  bool has_null_ = false;
};

// TOOD(): this is a fake unsafeappender for boolean array
template <typename DataType>
class UnsafeArrayAppender<DataType, arrow::enable_if_boolean<DataType>>
    : public AppenderBase {
 public:
  UnsafeArrayAppender(arrow::compute::ExecContext* ctx, AppenderType type = left)
      : ctx_(ctx), type_(type) {
    std::unique_ptr<arrow::ArrayBuilder> array_builder;
    arrow::MakeBuilder(ctx_->memory_pool(), arrow::TypeTraits<DataType>::type_singleton(),
                       &array_builder);
    builder_.reset(arrow::internal::checked_cast<BuilderType_*>(array_builder.release()));
  }
  ~UnsafeArrayAppender() {}

  AppenderType GetType() override { return type_; }
  arrow::Status AddArray(const std::shared_ptr<arrow::Array>& arr) override {
    auto typed_arr_ = std::dynamic_pointer_cast<ArrayType_>(arr);
    cached_arr_.push_back(typed_arr_);
    if (typed_arr_->null_count() > 0) has_null_ = true;
    return arrow::Status::OK();
  }

  arrow::Status PopArray() override {
    cached_arr_.pop_back();
    has_null_ = false;
    return arrow::Status::OK();
  }
  arrow::Status ClearArrays() override {
    cached_arr_.clear();
    has_null_ = false;
    return arrow::Status::OK();
  }
  arrow::Status Append(const uint16_t& array_id, const uint16_t& item_id) override {
    if (has_null_ && cached_arr_[array_id]->null_count() > 0 &&
        cached_arr_[array_id]->IsNull(item_id)) {
      RETURN_NOT_OK(builder_->AppendNull());
    } else {
      RETURN_NOT_OK(builder_->Append(cached_arr_[array_id]->GetView(item_id)));
    }
    return arrow::Status::OK();
  }

  arrow::Status Append(const uint16_t& array_id, const uint16_t& item_id,
                       int repeated) override {
    if (repeated == 0) return arrow::Status::OK();
    if (has_null_ && cached_arr_[array_id]->null_count() > 0 &&
        cached_arr_[array_id]->IsNull(item_id)) {
      RETURN_NOT_OK(builder_->AppendNulls(repeated));
    } else {
      auto val = cached_arr_[array_id]->GetView(item_id);
      for (int i = 0; i < repeated; i++) {
        RETURN_NOT_OK(builder_->Append(val));
      }
    }
    return arrow::Status::OK();
  }

  arrow::Status Append(const std::vector<ArrayItemIndex>& index_list) {
    for (auto tmp : index_list) {
      if (has_null_ && cached_arr_[tmp.array_id]->null_count() > 0 &&
          cached_arr_[tmp.array_id]->IsNull(tmp.id)) {
        RETURN_NOT_OK(builder_->AppendNull());
      } else {
        RETURN_NOT_OK(builder_->Append(cached_arr_[tmp.array_id]->GetView(tmp.id)));
      }
    }
    return arrow::Status::OK();
  }

  arrow::Status AppendNull() override { return builder_->AppendNull(); }

  arrow::Status AppendExistence(bool is_exist) { return builder_->Append(is_exist); }

  arrow::Status Finish(std::shared_ptr<arrow::Array>* out_) override {
    auto status = builder_->Finish(out_);
    return status;
  }

  arrow::Status Reserve(uint64_t len) override {
    // builder_->Reserve(len);
    return arrow::Status::OK();
  }

  arrow::Status Reset() override {
    builder_->Reset();
    return arrow::Status::OK();
  }

 private:
  using BuilderType_ = typename arrow::TypeTraits<DataType>::BuilderType;
  using ArrayType_ = typename arrow::TypeTraits<DataType>::ArrayType;
  std::unique_ptr<BuilderType_> builder_;
  std::vector<std::shared_ptr<ArrayType_>> cached_arr_;
  arrow::compute::ExecContext* ctx_;
  AppenderType type_;
  bool has_null_ = false;
};

template <typename DataType>
class UnsafeArrayAppender<DataType, enable_if_decimal<DataType>> : public AppenderBase {
 public:
  UnsafeArrayAppender(arrow::compute::ExecContext* ctx,
                      std::shared_ptr<arrow::DataType> data_type,
                      AppenderType type = left)
      : ctx_(ctx), type_(type) {
    std::unique_ptr<arrow::ArrayBuilder> array_builder;
    arrow::MakeBuilder(ctx_->memory_pool(), data_type, &array_builder);
    builder_.reset(arrow::internal::checked_cast<BuilderType_*>(array_builder.release()));
  }
  ~UnsafeArrayAppender() {}

  AppenderType GetType() override { return type_; }
  arrow::Status AddArray(const std::shared_ptr<arrow::Array>& arr) override {
    auto typed_arr_ = std::dynamic_pointer_cast<ArrayType_>(arr);
    cached_arr_.push_back(typed_arr_);
    if (typed_arr_->null_count() > 0) has_null_ = true;
    return arrow::Status::OK();
  }

  arrow::Status PopArray() override {
    cached_arr_.pop_back();
    has_null_ = false;
    return arrow::Status::OK();
  }
  arrow::Status ClearArrays() override {
    cached_arr_.clear();
    has_null_ = false;
    return arrow::Status::OK();
  }
  arrow::Status Append(const uint16_t& array_id, const uint16_t& item_id) override {
    if (has_null_ && cached_arr_[array_id]->IsNull(item_id)) {
      builder_->UnsafeAppendNull();
    } else {
      builder_->UnsafeAppend(cached_arr_[array_id]->GetView(item_id));
    }
    return arrow::Status::OK();
  }

  arrow::Status Append(const uint16_t& array_id, const uint16_t& item_id,
                       int repeated) override {
    if (repeated == 0) return arrow::Status::OK();
    if (has_null_ && cached_arr_[array_id]->IsNull(item_id)) {
      RETURN_NOT_OK(builder_->AppendNulls(repeated));
    } else {
      auto val = cached_arr_[array_id]->GetView(item_id);
      for (int i = 0; i < repeated; i++) {
        RETURN_NOT_OK(builder_->Append(val));
      }
    }
    return arrow::Status::OK();
  }

  arrow::Status Append(const std::vector<ArrayItemIndex>& index_list) {
    for (auto tmp : index_list) {
      if (has_null_ && cached_arr_[tmp.array_id]->IsNull(tmp.id)) {
        RETURN_NOT_OK(builder_->AppendNull());
      } else {
        RETURN_NOT_OK(builder_->Append(cached_arr_[tmp.array_id]->GetView(tmp.id)));
      }
    }
    return arrow::Status::OK();
  }

  arrow::Status AppendNull() override { return builder_->AppendNull(); }

  arrow::Status Finish(std::shared_ptr<arrow::Array>* out_) override {
    auto status = builder_->Finish(out_);
    return status;
  }

  arrow::Status Reserve(uint64_t len) override {
    builder_->Reserve(len);
    return arrow::Status::OK();
  }

  arrow::Status Reset() override {
    builder_->Reset();
    return arrow::Status::OK();
  }

 private:
  using BuilderType_ = typename arrow::TypeTraits<DataType>::BuilderType;
  using ArrayType_ = typename arrow::TypeTraits<DataType>::ArrayType;
  std::unique_ptr<BuilderType_> builder_;
  std::vector<std::shared_ptr<ArrayType_>> cached_arr_;
  arrow::compute::ExecContext* ctx_;
  AppenderType type_;
  bool has_null_ = false;
};

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
  PROCESS(arrow::StringType)
static arrow::Status MakeUnsafeAppender(arrow::compute::ExecContext* ctx,
                                        std::shared_ptr<arrow::DataType> type,
                                        AppenderBase::AppenderType appender_type,
                                        std::shared_ptr<AppenderBase>* out) {
  switch (type->id()) {
#define PROCESS(InType)                                                               \
  case InType::type_id: {                                                             \
    auto app_ptr = std::make_shared<UnsafeArrayAppender<InType>>(ctx, appender_type); \
    *out = std::dynamic_pointer_cast<AppenderBase>(app_ptr);                          \
  } break;
    PROCESS_SUPPORTED_TYPES(PROCESS)
#undef PROCESS
    case arrow::Decimal128Type::type_id: {
      auto app_ptr = std::make_shared<UnsafeArrayAppender<arrow::Decimal128Type>>(
          ctx, type, appender_type);
      *out = std::dynamic_pointer_cast<AppenderBase>(app_ptr);
    } break;
    case arrow::TimestampType::type_id: {
      auto app_ptr = std::make_shared<UnsafeArrayAppender<arrow::TimestampType>>(
          ctx, type, appender_type);
      *out = std::dynamic_pointer_cast<AppenderBase>(app_ptr);
    } break;
    default: {
      return arrow::Status::NotImplemented("MakeAppender type not supported, type is ",
                                           type->ToString());
    } break;
  }
  return arrow::Status::OK();
}
#undef PROCESS_SUPPORTED_TYPES

}  // namespace extra
}  // namespace arrowcompute
}  // namespace codegen
}  // namespace sparkcolumnarplugin
