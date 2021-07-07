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

#include <arrow/record_batch.h>
#include <arrow/type.h>
#include <arrow/buffer.h>
#include <arrow/util/bit_util.h>
#include <arrow/array/array_base.h>
#include <arrow/array/array_primitive.h>
#include <arrow/array/array_binary.h>
#include "gandiva/decimal_type_util.h"

namespace sparkcolumnarplugin {
namespace unsaferow {

class UnsafeRowWriterAndReader {

 public:
   UnsafeRowWriterAndReader(std::shared_ptr<arrow::RecordBatch> rb, arrow::MemoryPool* memory_pool)
   : rb_(rb), memory_pool_(memory_pool){}
   arrow::Status Init();
   arrow::Status Write();
   bool HasNext();
   arrow::Status  Next(int64_t* length, std::shared_ptr<arrow::ResizableBuffer>* buffer);
   int64_t GetNumCols() { return num_cols_; }

 protected:
  std::vector<std::shared_ptr<arrow::ResizableBuffer>> buffers_;
  std::vector<int64_t> buffer_cursor_;
  std::shared_ptr<arrow::RecordBatch> rb_;
  int64_t nullBitsetWidthInBytes_;
  int64_t row_cursor_;
  int64_t num_cols_;
  int64_t num_rows_;
  arrow::MemoryPool* memory_pool_ = arrow::default_memory_pool(); 
};

}  // namespace unsaferow
}  // namespace sparkcolumnarplugin