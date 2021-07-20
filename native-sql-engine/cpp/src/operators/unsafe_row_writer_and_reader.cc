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

#include "operators/unsafe_row_writer_and_reader.h"

namespace sparkcolumnarplugin {
namespace unsaferow {

int64_t CalculateBitSetWidthInBytes(int32_t numFields) {
  return ((numFields + 63) / 64) * 8;
}

int32_t GetVariableColsNum(std::shared_ptr<arrow::Schema> schema, int64_t num_cols) {
  std::vector<std::shared_ptr<arrow::Field>> fields = schema->fields();
  int32_t count = 0;
  for (auto i = 0; i < num_cols; i++) {
    auto type_id = fields[i]->type()->id();
    if ((type_id == arrow::BinaryType::type_id) ||
        (type_id == arrow::StringType::type_id) ||
        (type_id == arrow::Decimal128Type::type_id))
      count++;
  }
  return count;
}

arrow::Status UnsafeRowWriterAndReader::Init() {
  num_rows_ = rb_->num_rows();
  num_cols_ = rb_->num_columns();

  // Calculate the initial size
  nullBitsetWidthInBytes_ = CalculateBitSetWidthInBytes(num_cols_);

  int64_t fixedSize =
      nullBitsetWidthInBytes_ + 8 * num_cols_;  // not contain the variable size

  int32_t num_variable_cols = GetVariableColsNum(rb_->schema(), num_cols_);

  // Initialize the buffers_ with the initial size.
  for (auto i = 0; i < num_rows_; i++) {
    int64_t new_size =
        fixedSize + num_variable_cols * 32;  // 32 is same with vanilla spark.
    std::shared_ptr<arrow::ResizableBuffer> buffer;
    ARROW_ASSIGN_OR_RAISE(
        buffer, arrow::AllocateResizableBuffer(arrow::BitUtil::BytesForBits(new_size * 8),
                                               memory_pool_));
    memset(buffer->mutable_data(), 0, new_size);
    buffer_cursor_.push_back(fixedSize);
    buffers_.push_back(std::move(buffer));
  }

  row_cursor_ = 0;
  return arrow::Status::OK();
}

bool UnsafeRowWriterAndReader::HasNext() {
  if (row_cursor_ < num_rows_) {
    return true;
  } else {
    return false;
  }
}

arrow::Status UnsafeRowWriterAndReader::Next(
    int64_t* length, std::shared_ptr<arrow::ResizableBuffer>* buffer) {
  *buffer = buffers_[row_cursor_];
  *length = buffer_cursor_[row_cursor_];
  row_cursor_++;
  return arrow::Status::OK();
}

void BitSet(uint8_t* data, int32_t index) {
  int64_t mask = 1L << (index & 0x3f);  // mod 64 and shift
  int64_t wordOffset = (index >> 6) * 8;
  int64_t word;
  memcpy(&word, data + wordOffset, sizeof(int64_t));
  int64_t value = word | mask;
  memcpy(data + wordOffset, &value, sizeof(int64_t));
}

int64_t GetFieldOffset(int64_t nullBitsetWidthInBytes, int32_t index) {
  return nullBitsetWidthInBytes + 8L * index;
}

void SetNullAt(uint8_t* data, int64_t offset, int32_t col_index) {
  BitSet(data, col_index);
  // set the value to 0
  memset(data + offset, 0, sizeof(int64_t));
  return;
}

int64_t RoundNumberOfBytesToNearestWord(int64_t numBytes) {
  int64_t remainder = numBytes & 0x07;  // This is equivalent to `numBytes % 8`
  if (remainder == 0) {
    return numBytes;
  } else {
    return numBytes + (8 - remainder);
  }
}

int32_t FirstNonzeroLongNum(int32_t* mag, int32_t length) {
  int32_t fn = 0;
  int32_t i;
  for (i = length - 1; i >= 0 && mag[i] == 0; i--)
    ;
  fn = length - i - 1;
  return fn;
}

int32_t GetInt(int32_t n, int32_t sig, int32_t* mag, int32_t length) {
  if (n < 0) return 0;
  if (n >= length) return sig < 0 ? -1 : 0;

  int32_t magInt = mag[length - n - 1];
  return (sig >= 0 ? magInt
                   : (n <= FirstNonzeroLongNum(mag, length) ? -magInt : ~magInt));
}

int32_t GetNumberOfLeadingZeros(uint32_t i) {
  // HD, Figure 5-6
  if (i == 0) return 32;
  int32_t n = 1;
  if (i >> 16 == 0) {
    n += 16;
    i <<= 16;
  }
  if (i >> 24 == 0) {
    n += 8;
    i <<= 8;
  }
  if (i >> 28 == 0) {
    n += 4;
    i <<= 4;
  }
  if (i >> 30 == 0) {
    n += 2;
    i <<= 2;
  }
  n -= i >> 31;
  return n;
}

int32_t GetBitLengthForInt(uint32_t n) { return 32 - GetNumberOfLeadingZeros(n); }

int32_t GetBitCount(uint32_t i) {
  // HD, Figure 5-2
  i = i - ((i >> 1) & 0x55555555);
  i = (i & 0x33333333) + ((i >> 2) & 0x33333333);
  i = (i + (i >> 4)) & 0x0f0f0f0f;
  i = i + (i >> 8);
  i = i + (i >> 16);
  return i & 0x3f;
}

int32_t GetBitLength(int32_t sig, int32_t* mag, int32_t len) {
  int32_t n = -1;
  if (len == 0) {
    n = 0;
  } else {
    // Calculate the bit length of the magnitude
    int32_t mag_bit_length = ((len - 1) << 5) + GetBitLengthForInt((uint32_t)mag[0]);
    if (sig < 0) {
      // Check if magnitude is a power of two
      bool pow2 = (GetBitCount((uint32_t)mag[0]) == 1);
      for (int i = 1; i < len && pow2; i++) pow2 = (mag[i] == 0);

      n = (pow2 ? mag_bit_length - 1 : mag_bit_length);
    } else {
      n = mag_bit_length;
    }
  }
  return n;
}

uint32_t* ConvertMagArray(int64_t new_high, uint64_t new_low, int32_t* size) {
  // convert the new_high and new_low to 4 int value.
  uint32_t* mag = new uint32_t[4];
  mag[3] = (uint32_t)new_low;
  mag[2] = new_low >>= 32;
  mag[1] = (uint32_t)new_high;
  mag[0] = new_high >>= 32;

  int32_t start = 0;
  // remove the front 0
  for (int32_t i = 0; i < 4; i++) {
    if (mag[i] == 0) start++;
    if (mag[i] != 0) break;
  }

  int32_t length = 4 - start;
  uint32_t* new_mag = new uint32_t[length];
  int32_t k = 0;
  // get the mag after remove the high 0
  for (int32_t i = start; i < 4; i++) {
    new_mag[k++] = mag[i];
  }

  delete mag;
  *size = length;
  return new_mag;
}

/*
 *  This method refer to the BigInterger#toByteArray() method in Java side.
 */
std::array<uint8_t, 16> ToByteArray(arrow::Decimal128 value, int32_t* length) {
  int64_t high = value.high_bits();
  uint64_t low = value.low_bits();
  arrow::Decimal128 new_value;
  int32_t sig;
  if (value > 0) {
    new_value = value;
    sig = 1;
  } else if (value < 0) {
    new_value = value.Abs();
    sig = -1;
  } else {
    new_value = value;
    sig = 0;
  }

  int64_t new_high = new_value.high_bits();
  uint64_t new_low = new_value.low_bits();

  uint32_t* mag;
  int32_t size;
  mag = ConvertMagArray(new_high, new_low, &size);

  int32_t* final_mag = new int32_t[size];
  memcpy(final_mag, mag, size * 4);

  int32_t byte_length = GetBitLength(sig, final_mag, size) / 8 + 1;

  std::array<uint8_t, 16> out{{0}};
  uint32_t next_int = 0;
  for (int32_t i = byte_length - 1, bytes_copied = 4, int_index = 0; i >= 0; i--) {
    if (bytes_copied == 4) {
      next_int = GetInt(int_index++, sig, final_mag, size);
      bytes_copied = 1;
    } else {
      next_int >>= 8;
      bytes_copied++;
    }

    out[i] = (uint8_t)next_int;
  }
  *length = byte_length;

  delete mag, final_mag;
  return out;
}

arrow::Status WriteValue(std::shared_ptr<arrow::ResizableBuffer> buffer, int64_t offset,
                         int32_t row_index, std::shared_ptr<arrow::Array> array,
                         int64_t currentCursor, int32_t col_index,
                         int64_t* updatedCursor) {
  auto data = buffer->mutable_data();
  // Write the value into the buffer
  switch (array->type_id()) {
    case arrow::BooleanType::type_id: {
      // Boolean type
      auto boolArray = std::static_pointer_cast<arrow::BooleanArray>(array);
      auto value = boolArray->Value(row_index);
      memcpy(data + offset, &value, sizeof(bool));
      break;
    }
    case arrow::Int8Type::type_id: {
      // Byte type
      auto int8Array = std::static_pointer_cast<arrow::Int8Array>(array);
      auto value = int8Array->Value(row_index);
      memcpy(data + offset, &value, sizeof(int8_t));
      break;
    }
    case arrow::Int16Type::type_id: {
      // Short type
      auto int16Array = std::static_pointer_cast<arrow::Int16Array>(array);
      auto value = int16Array->Value(row_index);
      memcpy(data + offset, &value, sizeof(int16_t));
      break;
    }
    case arrow::Int32Type::type_id: {
      // Integer type
      auto int32Array = std::static_pointer_cast<arrow::Int32Array>(array);
      auto value = int32Array->Value(row_index);
      memcpy(data + offset, &value, sizeof(int32_t));
      break;
    }
    case arrow::Int64Type::type_id: {
      // Long type
      auto int64Array = std::static_pointer_cast<arrow::Int64Array>(array);
      auto value = int64Array->Value(row_index);
      memcpy(data + offset, &value, sizeof(int64_t));
      break;
    }
    case arrow::FloatType::type_id: {
      // Float type
      auto floatArray = std::static_pointer_cast<arrow::FloatArray>(array);
      auto value = floatArray->Value(row_index);
      memcpy(data + offset, &value, sizeof(float));
      break;
    }
    case arrow::DoubleType::type_id: {
      // Double type
      auto doubleArray = std::static_pointer_cast<arrow::DoubleArray>(array);
      auto value = doubleArray->Value(row_index);
      memcpy(data + offset, &value, sizeof(double));
      break;
    }
    case arrow::BinaryType::type_id: {
      // Binary type
      auto binaryArray = std::static_pointer_cast<arrow::BinaryArray>(array);
      using offset_type = typename arrow::BinaryType::offset_type;
      offset_type length;
      auto value = binaryArray->GetValue(row_index, &length);
      int64_t roundedSize = RoundNumberOfBytesToNearestWord(length);
      if (roundedSize > 32) {
        int64_t new_size = buffer->size() + roundedSize;
        RETURN_NOT_OK(buffer->Resize(new_size));
      }
      // After resize buffer, the data address is changed and the value is reset to 0.
      auto new_data = buffer->mutable_data();
      // write the variable value
      memcpy(new_data + currentCursor, value, length);
      // write the offset and size
      int64_t offsetAndSize = (currentCursor << 32) | length;
      memcpy(new_data + offset, &offsetAndSize, sizeof(int64_t));
      // Update the cursor of the buffer.
      *updatedCursor = currentCursor + roundedSize;
      break;
    }
    case arrow::StringType::type_id: {
      // String type
      auto stringArray = std::static_pointer_cast<arrow::StringArray>(array);
      using offset_type = typename arrow::StringType::offset_type;
      offset_type length;
      auto value = stringArray->GetValue(row_index, &length);
      int64_t roundedSize = RoundNumberOfBytesToNearestWord(length);

      if (roundedSize > 32) {
        int64_t new_size = buffer->size() + roundedSize;
        RETURN_NOT_OK(buffer->Resize(new_size));
      }

      // After resize buffer, the data address is changed and the value is reset to 0.
      auto new_data = buffer->mutable_data();
      // write the variable value
      memcpy(new_data + currentCursor, value, length);
      // write the offset and size
      int64_t offsetAndSize = (currentCursor << 32) | length;
      memcpy(new_data + offset, &offsetAndSize, sizeof(int64_t));
      // Update the cursor of the buffer.
      *updatedCursor = currentCursor + roundedSize;
      break;
    }
    case arrow::Decimal128Type::type_id: {
      auto out_array = dynamic_cast<arrow::Decimal128Array*>(array.get());
      auto dtype = dynamic_cast<arrow::Decimal128Type*>(out_array->type().get());

      int32_t precision = dtype->precision();
      int32_t scale = dtype->scale();

      const arrow::Decimal128 out_value(out_array->GetValue(row_index));
      bool flag = out_array->IsNull(row_index);

      if (precision <= 18) {
        if (!flag) {
          // Get the long value and write the long value
          // Refer to the int64_t() method of Decimal128
          int64_t long_value = static_cast<int64_t>(out_value.low_bits());
          memcpy(data + offset, &long_value, sizeof(long));
        } else {
          SetNullAt(data, offset, col_index);
        }
      } else {
        if (flag) {
          SetNullAt(data, offset, col_index);
        } else {
          int32_t size;
          auto out = ToByteArray(out_value, &size);
          assert(size <= 16);

          // write the variable value
          memcpy(data + currentCursor, &out[0], size);
          // write the offset and size
          int64_t offsetAndSize = (currentCursor << 32) | size;
          memcpy(data + offset, &offsetAndSize, sizeof(int64_t));
        }

        // Update the cursor of the buffer.
        *updatedCursor = currentCursor + 16;
      }
      break;
    }
    case arrow::Date32Type::type_id: {
      auto date32Array = std::static_pointer_cast<arrow::Date32Array>(array);
      auto value = date32Array->Value(row_index);
      memcpy(data + offset, &value, sizeof(int32_t));
      break;
    }
    default:
      return arrow::Status::Invalid("Unsupported data type: " + array->type_id());
  }
  return arrow::Status::OK();
  ;
}

arrow::Status UnsafeRowWriterAndReader::Write() {
  // Get each row value and write to the buffer
  for (auto i = 0; i < num_rows_; i++) {
    auto buffer = buffers_[i];
    for (auto j = 0; j < num_cols_; j++) {
      uint8_t* data = buffer->mutable_data();
      auto array = rb_->column(j);
      // for each column, get the current row  value, check whether it is null, and then
      // write it to data.
      bool is_null = array->IsNull(i);
      int64_t offset = GetFieldOffset(nullBitsetWidthInBytes_, j);
      if (is_null) {
        SetNullAt(data, offset, j);
      } else {
        // Write the value to the buffer
        int64_t updatedCursor = buffer_cursor_[i];
        WriteValue(buffer, offset, i, array, buffer_cursor_[i], j, &updatedCursor);
        buffer_cursor_[i] = updatedCursor;
      }
    }
  }
  return arrow::Status::OK();
}

}  // namespace unsaferow
}  // namespace sparkcolumnarplugin
