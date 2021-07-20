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

#include <arrow/compute/api.h>
#include <arrow/datum.h>
#include <arrow/io/api.h>
#include <arrow/ipc/reader.h>
#include <arrow/ipc/util.h>
#include <arrow/record_batch.h>
#include <arrow/util/io_util.h>
#include <gtest/gtest.h>

#include <iostream>

#include "operators/unsafe_row_writer_and_reader.h"
#include "tests/test_utils.h"

namespace sparkcolumnarplugin {
namespace unsaferow {

class MyMemoryPool : public arrow::MemoryPool {
 public:
  explicit MyMemoryPool(int64_t capacity) : capacity_(capacity) {}

  Status Allocate(int64_t size, uint8_t** out) override {
    if (bytes_allocated() + size > capacity_) {
      return Status::OutOfMemory("malloc of size ", size, " failed");
    }
    RETURN_NOT_OK(pool_->Allocate(size, out));
    stats_.UpdateAllocatedBytes(size);
    return arrow::Status::OK();
  }

  Status Reallocate(int64_t old_size, int64_t new_size, uint8_t** ptr) override {
    if (new_size > capacity_) {
      return Status::OutOfMemory("malloc of size ", new_size, " failed");
    }
    RETURN_NOT_OK(pool_->Reallocate(old_size, new_size, ptr));
    stats_.UpdateAllocatedBytes(new_size - old_size);
    return arrow::Status::OK();
  }

  void Free(uint8_t* buffer, int64_t size) override {
    pool_->Free(buffer, size);
    stats_.UpdateAllocatedBytes(-size);
  }

  int64_t bytes_allocated() const override { return stats_.bytes_allocated(); }

  int64_t max_memory() const override { return pool_->max_memory(); }

  std::string backend_name() const override { return pool_->backend_name(); }

 private:
  MemoryPool* pool_ = arrow::default_memory_pool();
  int64_t capacity_;
  arrow::internal::MemoryPoolStats stats_;
};

class UnsaferowTest : public ::testing::Test {
 protected:
  void SetUp() {
    auto f_int8 = field("f_int8_a", arrow::int8());
    auto f_int16 = field("f_int16", arrow::int16());
    auto f_int32 = field("f_int32", arrow::int32());
    auto f_int64 = field("f_int64", arrow::int64());
    auto f_double = field("f_double", arrow::float64());
    auto f_float = field("f_float", arrow::float32());
    auto f_bool = field("f_bool", arrow::boolean());
    auto f_string = field("f_string", arrow::utf8());
    auto f_binary = field("f_binary", arrow::binary());
    auto f_decimal = field("f_decimal128", arrow::decimal(10, 2));

    schema_ = arrow::schema({f_bool, f_int8, f_int16, f_int32, f_int64, f_float, f_double,
                             f_binary, f_decimal});

    MakeInputBatch(input_data_, schema_, &input_batch_);
    ConstructNullInputBatch(&nullable_input_batch_);
  }

  static const std::vector<std::string> input_data_1;
  static const std::vector<std::string> input_data_;

  std::shared_ptr<arrow::Schema> schema_;

  std::shared_ptr<arrow::RecordBatch> input_batch_;
  std::shared_ptr<arrow::RecordBatch> nullable_input_batch_;
};

const std::vector<std::string> UnsaferowTest::input_data_ = {"[true, true]",
                                                             "[1, 1]",
                                                             "[1, 1]",
                                                             "[1, 1]",
                                                             "[1, 1]",
                                                             "[3.5, 3.5]",
                                                             "[1, 1]",
                                                             R"(["abc", "abc"])",
                                                             R"(["100.00", "100.00"])"};

TEST_F(UnsaferowTest, TestNullTypeCheck) {
  std::shared_ptr<arrow::MemoryPool> pool = std::make_shared<MyMemoryPool>(4000);
  std::shared_ptr<UnsafeRowWriterAndReader> unsafe_row_writer_reader =
      std::make_shared<UnsafeRowWriterAndReader>(nullable_input_batch_, pool.get());

  unsafe_row_writer_reader->Init();
  unsafe_row_writer_reader->Write();

  long expected[2][3] = {{1, 0, 1}, {2, 1, 0}};
  int32_t count = 0;
  while (unsafe_row_writer_reader->HasNext()) {
    int64_t length;
    std::shared_ptr<arrow::ResizableBuffer> buffer;
    unsafe_row_writer_reader->Next(&length, &buffer);

    auto data = buffer->mutable_data();
    long value = 0;
    long result[3] = {0, 0, 0};
    int32_t k = 0;
    for (int32_t i = 0; i < length; i += sizeof(long)) {
      memcpy(&value, data + i, sizeof(long));
      result[k++] = value;
    }

    int32_t result_size = sizeof(result) / sizeof(result[0]);
    ASSERT_EQ(result_size, 3);

    for (int32_t i = 0; i < result_size; i++) {
      ASSERT_EQ(result[i], expected[count][i]);
    }
    count++;
  }
}

TEST_F(UnsaferowTest, TestUnsaferowWriterandReader) {
  std::shared_ptr<arrow::MemoryPool> pool = std::make_shared<MyMemoryPool>(4000);
  std::shared_ptr<UnsafeRowWriterAndReader> unsafe_row_writer_reader =
      std::make_shared<UnsafeRowWriterAndReader>(input_batch_, pool.get());

  unsafe_row_writer_reader->Init();
  unsafe_row_writer_reader->Write();

  while (unsafe_row_writer_reader->HasNext()) {
    int64_t length;
    std::shared_ptr<arrow::ResizableBuffer> buffer;
    unsafe_row_writer_reader->Next(&length, &buffer);
  }
}
}  // namespace unsaferow
}  // namespace sparkcolumnarplugin
