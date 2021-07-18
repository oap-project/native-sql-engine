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

#include <arrow/compute/api.h>
#include <arrow/status.h>
#include <arrow/type_fwd.h>
#include <arrow/util/iterator.h>
#include <arrow/record_batch.h>

#include "precompile/type_traits.h"
#include "iostream"

using sparkcolumnarplugin::precompile::enable_if_number_or_decimal;
using sparkcolumnarplugin::precompile::enable_if_string_like;
using sparkcolumnarplugin::precompile::StringArray;
using sparkcolumnarplugin::precompile::TypeTraits;

class LazyBatchIterator {
 public:
  LazyBatchIterator(arrow::RecordBatchIterator in) {
    this->in_ = std::move(in);
  }

  std::shared_ptr<arrow::RecordBatch> GetBatch(int32_t batch_id) {
    for (; current_batch_id_ <= batch_id; current_batch_id_++) {
      std::shared_ptr<arrow::RecordBatch> next = in_.Next().ValueOrDie();
      if (next == nullptr) {
        return nullptr;
      }
      cache_.push_back(next);
      ref_cnts_.push_back(0);
    }
    return cache_[batch_id];
  }

  void RetainBatch(int32_t batch_id) {
    ref_cnts_[batch_id] = ref_cnts_[batch_id] + 1;
  }

  void ReleaseBatch(int32_t batch_id) {
    ref_cnts_[batch_id] = ref_cnts_[batch_id] - 1;
    if (ref_cnts_[batch_id] <= 0) {
      cache_[batch_id] = nullptr;
    }
  }

 private:
  arrow::RecordBatchIterator in_;
  std::vector<std::shared_ptr<arrow::RecordBatch>> cache_;
  std::vector<int32_t> ref_cnts_;
  int32_t current_batch_id_ = 0;
};

class RelationColumn {
 public:
  virtual bool IsNull(int array_id, int id) = 0;
  virtual bool IsEqualTo(int x_array_id, int x_id, int y_array_id, int y_id) = 0;
  virtual arrow::Status AppendColumn(std::shared_ptr<arrow::Array> in) {
    return arrow::Status::NotImplemented("RelationColumn AppendColumn is abstract.");
  };
  virtual arrow::Status FromLazyBatchIterator(std::shared_ptr<LazyBatchIterator> in, int field_id) {
    return arrow::Status::NotImplemented("RelationColumn AppendColumn is abstract.");
  };
  virtual arrow::Status Advance(int array_id) {
    return arrow::Status::NotImplemented("RelationColumn Advance is abstract.");
  };
  virtual arrow::Status ReleaseArray(int array_id) = 0;
  virtual arrow::Status GetArrayVector(std::vector<std::shared_ptr<arrow::Array>>* out) {
    return arrow::Status::NotImplemented("RelationColumn GetArrayVector is abstract.");
  }
  virtual bool HasNull() = 0;
};

template <typename T, typename Enable = void>
class TypedRelationColumn {};

template <typename DataType>
class TypedRelationColumn<DataType, enable_if_number_or_decimal<DataType>>
    : public RelationColumn {
 public:
  using T = typename TypeTraits<DataType>::CType;
  TypedRelationColumn() = default;
  bool IsNull(int array_id, int id) override {
    return (!has_null_) ? false : array_vector_[array_id]->IsNull(id);
  }
  bool IsEqualTo(int x_array_id, int x_id, int y_array_id, int y_id) {
    if (!has_null_) return GetValue(x_array_id, x_id) == GetValue(y_array_id, y_id);
    auto is_null_x = IsNull(x_array_id, x_id);
    auto is_null_y = IsNull(y_array_id, y_id);
    if (is_null_x && is_null_y) return true;
    if (is_null_x || is_null_y) return false;
    return GetValue(x_array_id, x_id) == GetValue(y_array_id, y_id);
  }
  arrow::Status AppendColumn(std::shared_ptr<arrow::Array> in) override {
    auto typed_in = std::make_shared<ArrayType>(in);
    if (typed_in->null_count() > 0) has_null_ = true;
    array_vector_.push_back(typed_in);
    return arrow::Status::OK();
  }
  arrow::Status ReleaseArray(int array_id) {
    array_vector_[array_id] = nullptr; // fixme using reset()?
    return arrow::Status::OK();
  }
  arrow::Status GetArrayVector(std::vector<std::shared_ptr<arrow::Array>>* out) override {
    for (auto arr : array_vector_) {
      (*out).push_back(arr->cache_);
    }
    return arrow::Status::OK();
  }
  T GetValue(int array_id, int id) { return array_vector_[array_id]->GetView(id); }
  bool HasNull() { return has_null_; }

 private:
  using ArrayType = typename TypeTraits<DataType>::ArrayType;
  std::vector<std::shared_ptr<ArrayType>> array_vector_;
  bool has_null_ = false;
};

template <typename DataType>
class TypedRelationColumn<DataType, enable_if_string_like<DataType>>
    : public RelationColumn {
 public:
  TypedRelationColumn() {}
  bool IsNull(int array_id, int id) override {
    return (!has_null_) ? false : array_vector_[array_id]->IsNull(id);
  }
  bool IsEqualTo(int x_array_id, int x_id, int y_array_id, int y_id) {
    if (!has_null_) return GetValue(x_array_id, x_id) == GetValue(y_array_id, y_id);
    auto is_null_x = IsNull(x_array_id, x_id);
    auto is_null_y = IsNull(y_array_id, y_id);
    if (is_null_x && is_null_y) return true;
    if (is_null_x || is_null_y) return false;
    return GetValue(x_array_id, x_id) == GetValue(y_array_id, y_id);
  }
  arrow::Status AppendColumn(std::shared_ptr<arrow::Array> in) override {
    auto typed_in = std::make_shared<StringArray>(in);
    if (typed_in->null_count() > 0) has_null_ = true;
    array_vector_.push_back(typed_in);
    return arrow::Status::OK();
  }
  arrow::Status ReleaseArray(int array_id) {
    array_vector_[array_id] = nullptr;
    return arrow::Status::OK();
  }
  arrow::Status GetArrayVector(std::vector<std::shared_ptr<arrow::Array>>* out) override {
    for (auto arr : array_vector_) {
      (*out).push_back(arr->cache_);
    }
    return arrow::Status::OK();
  }
  std::string GetValue(int array_id, int id) {
    return array_vector_[array_id]->GetString(id);
  }
  bool HasNull() { return has_null_; }

 private:
  std::vector<std::shared_ptr<StringArray>> array_vector_;
  bool has_null_ = false;
};

arrow::Status MakeRelationColumn(uint32_t data_type_id,
                                 std::shared_ptr<RelationColumn>* out);

template <typename T, typename Enable = void>
class TypedLazyLoadRelationColumn {};

template <typename DataType>
class TypedLazyLoadRelationColumn<DataType, enable_if_number_or_decimal<DataType>>
    : public RelationColumn {
 public:
  using T = typename TypeTraits<DataType>::CType;
  TypedLazyLoadRelationColumn() = default;

  bool IsNull(int array_id, int id) override {
    Advance(array_id);
    return delegated.IsNull(array_id, id);
  }

  bool IsEqualTo(int x_array_id, int x_id, int y_array_id, int y_id) override {
    Advance(x_array_id);
    Advance(y_array_id);
    return delegated.IsEqualTo(x_array_id, x_id, y_array_id, y_id);
  }

  arrow::Status FromLazyBatchIterator(std::shared_ptr<LazyBatchIterator> in, int field_id) override {
    in_ = in;
    field_id_ = field_id;
    return arrow::Status::OK();
  };

  arrow::Status Advance(int array_id) override {
    if (array_id < current_array_id_) {
      return arrow::Status::OK();
    }
    for (int i = current_array_id_; i <= array_id; i++) {
      std::shared_ptr<arrow::RecordBatch> batch = in_->GetBatch(i);
      std::shared_ptr<arrow::Array> array = batch->column(field_id_);
      delegated.AppendColumn(array);
      in_->RetainBatch(i);
      array_released.push_back(false);
    }
    current_array_id_ = array_id + 1;
    return arrow::Status::OK();
  }

  arrow::Status ReleaseArray(int array_id) override {
    if (array_released[array_id]) {
      return arrow::Status::OK();
    }
    RETURN_NOT_OK(delegated.ReleaseArray(array_id));
    in_->ReleaseBatch(array_id);
    array_released[array_id] = true;
    return arrow::Status::OK();
  }

  arrow::Status GetArrayVector(std::vector<std::shared_ptr<arrow::Array>>* out) override {
    return delegated.GetArrayVector(out);
  }
  
  T GetValue(int array_id, int id) {
    Advance(array_id);
    if (array_id >= 1) {
      ReleaseArray(array_id - 1);
    }
    return delegated.GetValue(array_id, id);
  }
  bool HasNull() override { return has_null_; }

 private:
  std::shared_ptr<LazyBatchIterator> in_;
  TypedRelationColumn<DataType> delegated;
  int current_array_id_ = 0;
  int field_id_ = -1;
  std::vector<bool> array_released;
  bool has_null_ = true; // fixme always true
};

template <typename DataType>
class TypedLazyLoadRelationColumn<DataType, enable_if_string_like<DataType>>
    : public RelationColumn {
 public:
  TypedLazyLoadRelationColumn() = default;
  bool IsNull(int array_id, int id) override {
    Advance(array_id);
    return delegated.IsNull(array_id, id);
  }
  bool IsEqualTo(int x_array_id, int x_id, int y_array_id, int y_id) override {
    Advance(x_array_id);
    Advance(y_array_id);
    return delegated.IsEqualTo(x_array_id, x_id, y_array_id, y_id);
  }

  arrow::Status FromLazyBatchIterator(std::shared_ptr<LazyBatchIterator> in, int field_id) override {
    in_ = in;
    field_id_ = field_id;
    return arrow::Status::OK();
  };

  arrow::Status Advance(int array_id) override {
    if (array_id < current_array_id_) {
      return arrow::Status::OK();
    }
    for (int i = current_array_id_; i <= array_id; i++) {
      std::shared_ptr<arrow::RecordBatch> batch = in_->GetBatch(i);
      std::shared_ptr<arrow::Array> array = batch->column(field_id_);
      delegated.AppendColumn(array);
      in_->RetainBatch(i);
      array_released.push_back(false);
    }
    current_array_id_ = array_id + 1;
    return arrow::Status::OK();
  }

  arrow::Status ReleaseArray(int array_id) override {
    if (array_released[array_id]) {
      return arrow::Status::OK();
    }
    RETURN_NOT_OK(delegated.ReleaseArray(array_id));
    in_->ReleaseBatch(array_id);
    array_released[array_id] = true;
    return arrow::Status::OK();
  }
  
  arrow::Status GetArrayVector(std::vector<std::shared_ptr<arrow::Array>>* out) override {
    return delegated.GetArrayVector(out);
  }
  
  std::string GetValue(int array_id, int id) {
    Advance(array_id);
    if (array_id >= 1) {
      ReleaseArray(array_id - 1);
    }
    return delegated.GetValue(array_id, id);
  }
  
  bool HasNull() override { return has_null_; }

 private:
  std::shared_ptr<LazyBatchIterator> in_;
  TypedRelationColumn<DataType> delegated;
  int32_t current_array_id_ = 0;
  int32_t field_id_ = -1;
  std::vector<bool> array_released;
  bool has_null_ = true; // fixme always true
};

arrow::Status MakeLazyLoadRelationColumn(uint32_t data_type_id,
                                 std::shared_ptr<RelationColumn>* out);

