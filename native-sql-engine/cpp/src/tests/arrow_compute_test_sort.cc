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

#include <arrow/array.h>
#include <arrow/ipc/json_simple.h>
#include <arrow/record_batch.h>
#include <gtest/gtest.h>

#include <memory>

#include "codegen/code_generator.h"
#include "codegen/code_generator_factory.h"
#include "tests/test_utils.h"

namespace sparkcolumnarplugin {
namespace codegen {

TEST(TestArrowComputeSort, SortTestInPlaceNullsFirstAsc) {
  ////////////////////// prepare expr_vector ///////////////////////
  auto f0 = field("f0", float64());
  auto arg_0 = TreeExprBuilder::MakeField(f0);
  auto true_literal = TreeExprBuilder::MakeLiteral(true);
  auto false_literal = TreeExprBuilder::MakeLiteral(false);

  auto f_res = field("res", uint32());
  auto indices_type = std::make_shared<FixedSizeBinaryType>(16);
  auto f_indices = field("indices", indices_type);

  auto n_key_func = TreeExprBuilder::MakeFunction("key_function", {arg_0}, uint32());
  auto n_key_field = TreeExprBuilder::MakeFunction("key_field", {arg_0}, uint32());
  auto n_dir = TreeExprBuilder::MakeFunction("sort_directions", {true_literal}, uint32());
  auto n_nulls_order =
      TreeExprBuilder::MakeFunction("sort_nulls_order", {true_literal}, uint32());
  auto NaN_check = TreeExprBuilder::MakeFunction("NaN_check", {true_literal}, uint32());
  auto do_codegen = TreeExprBuilder::MakeFunction("codegen", {false_literal}, uint32());
  auto n_sort_to_indices = TreeExprBuilder::MakeFunction(
      "sortArraysToIndices",
      {n_key_func, n_key_field, n_dir, n_nulls_order, NaN_check, do_codegen}, uint32());
  auto n_sort =
      TreeExprBuilder::MakeFunction("standalone", {n_sort_to_indices}, uint32());
  auto sortArrays_expr = TreeExprBuilder::MakeExpression(n_sort, f_res);

  auto sch = arrow::schema({f0});
  std::vector<std::shared_ptr<Field>> ret_types = {f0};
  ///////////////////// Calculation //////////////////
  std::shared_ptr<CodeGenerator> sort_expr;
  arrow::compute::ExecContext ctx;
  ASSERT_NOT_OK(CreateCodeGenerator(ctx.memory_pool(), sch, {sortArrays_expr}, ret_types,
                                    &sort_expr, true));

  std::shared_ptr<arrow::RecordBatch> input_batch;
  std::vector<std::shared_ptr<arrow::RecordBatch>> input_batch_list;
  std::vector<std::shared_ptr<arrow::RecordBatch>> dummy_result_batches;
  std::shared_ptr<ResultIteratorBase> sort_result_iterator_base;

  std::vector<std::string> input_data_string = {"[10, 12, 4, 50, 52, 32, 11]"};
  MakeInputBatch(input_data_string, sch, &input_batch);
  input_batch_list.push_back(input_batch);

  std::vector<std::string> input_data_string_2 = {"[1, NaN, 43, 42, 6, null, 2]"};
  MakeInputBatch(input_data_string_2, sch, &input_batch);
  input_batch_list.push_back(input_batch);

  std::vector<std::string> input_data_string_3 = {"[3, 64, 15, 7, 9, 19, 33]"};
  MakeInputBatch(input_data_string_3, sch, &input_batch);
  input_batch_list.push_back(input_batch);

  std::vector<std::string> input_data_string_4 = {"[23, 17, NaN, 18, 20, 35, 30]"};
  MakeInputBatch(input_data_string_4, sch, &input_batch);
  input_batch_list.push_back(input_batch);

  std::vector<std::string> input_data_string_5 = {"[37, null, NaN, 13, 8, 59, 21]"};
  MakeInputBatch(input_data_string_5, sch, &input_batch);
  input_batch_list.push_back(input_batch);

  ////////////////////////////////// calculation
  //////////////////////////////////////
  std::shared_ptr<arrow::RecordBatch> expected_result;
  std::vector<std::string> expected_result_string = {
      "[null, null, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 15, 17, 18, 19, "
      "20, 21, "
      "23, 30, 32, 33, 35, 37, 42, 43, 50, 52, 59, 64, NaN, NaN, NaN]"};
  MakeInputBatch(expected_result_string, sch, &expected_result);

  for (auto batch : input_batch_list) {
    ASSERT_NOT_OK(sort_expr->evaluate(batch, &dummy_result_batches));
  }
  ASSERT_NOT_OK(sort_expr->finish(&sort_result_iterator_base));

  auto sort_result_iterator =
      std::dynamic_pointer_cast<ResultIterator<arrow::RecordBatch>>(
          sort_result_iterator_base);

  std::shared_ptr<arrow::RecordBatch> dummy_result_batch;
  std::shared_ptr<arrow::RecordBatch> result_batch;

  if (sort_result_iterator->HasNext()) {
    ASSERT_NOT_OK(sort_result_iterator->Next(&result_batch));
    ASSERT_NOT_OK(Equals(*expected_result.get(), *result_batch.get()));
  }
}

TEST(TestArrowComputeSort, SortTestInplaceNullsLastAsc) {
  ////////////////////// prepare expr_vector ///////////////////////
  auto f0 = field("f0", float64());
  auto arg_0 = TreeExprBuilder::MakeField(f0);
  auto true_literal = TreeExprBuilder::MakeLiteral(true);
  auto false_literal = TreeExprBuilder::MakeLiteral(false);

  auto f_res = field("res", uint32());
  auto indices_type = std::make_shared<FixedSizeBinaryType>(16);
  auto f_indices = field("indices", indices_type);

  auto n_key_func = TreeExprBuilder::MakeFunction("key_function", {arg_0}, uint32());
  auto n_key_field = TreeExprBuilder::MakeFunction("key_field", {arg_0}, uint32());
  auto n_dir = TreeExprBuilder::MakeFunction("sort_directions", {true_literal}, uint32());
  auto n_nulls_order =
      TreeExprBuilder::MakeFunction("sort_nulls_order", {false_literal}, uint32());
  auto NaN_check = TreeExprBuilder::MakeFunction("NaN_check", {true_literal}, uint32());
  auto do_codegen = TreeExprBuilder::MakeFunction("codegen", {false_literal}, uint32());
  auto n_sort_to_indices = TreeExprBuilder::MakeFunction(
      "sortArraysToIndices",
      {n_key_func, n_key_field, n_dir, n_nulls_order, NaN_check, do_codegen}, uint32());
  auto n_sort =
      TreeExprBuilder::MakeFunction("standalone", {n_sort_to_indices}, uint32());
  auto sortArrays_expr = TreeExprBuilder::MakeExpression(n_sort, f_res);

  auto sch = arrow::schema({f0});
  std::vector<std::shared_ptr<Field>> ret_types = {f0};
  ///////////////////// Calculation //////////////////
  std::shared_ptr<CodeGenerator> sort_expr;
  arrow::compute::ExecContext ctx;
  ASSERT_NOT_OK(CreateCodeGenerator(ctx.memory_pool(), sch, {sortArrays_expr}, ret_types,
                                    &sort_expr, true));

  std::shared_ptr<arrow::RecordBatch> input_batch;
  std::vector<std::shared_ptr<arrow::RecordBatch>> input_batch_list;
  std::vector<std::shared_ptr<arrow::RecordBatch>> dummy_result_batches;
  std::shared_ptr<ResultIteratorBase> sort_result_iterator_base;

  std::vector<std::string> input_data_string = {"[10, 12, 4, 50, 52, 32, 11]"};
  MakeInputBatch(input_data_string, sch, &input_batch);
  input_batch_list.push_back(input_batch);

  std::vector<std::string> input_data_string_2 = {"[1, NaN, 43, 42, 6, null, 2]"};
  MakeInputBatch(input_data_string_2, sch, &input_batch);
  input_batch_list.push_back(input_batch);

  std::vector<std::string> input_data_string_3 = {"[3, 64, 15, 7, 9, 19, 33]"};
  MakeInputBatch(input_data_string_3, sch, &input_batch);
  input_batch_list.push_back(input_batch);

  std::vector<std::string> input_data_string_4 = {"[23, 17, NaN, 18, 20, 35, 30]"};
  MakeInputBatch(input_data_string_4, sch, &input_batch);
  input_batch_list.push_back(input_batch);

  std::vector<std::string> input_data_string_5 = {"[37, null, NaN, 13, 8, 59, 21]"};
  MakeInputBatch(input_data_string_5, sch, &input_batch);
  input_batch_list.push_back(input_batch);

  ////////////////////////////////// calculation
  //////////////////////////////////////
  std::shared_ptr<arrow::RecordBatch> expected_result;
  std::vector<std::string> expected_result_string = {
      "[1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 15, 17, 18, 19, 20, 21, "
      "23, 30, 32, 33, 35, 37, 42, 43, 50, 52, 59, 64, NaN, NaN, NaN, null, "
      "null]"};
  MakeInputBatch(expected_result_string, sch, &expected_result);

  for (auto batch : input_batch_list) {
    ASSERT_NOT_OK(sort_expr->evaluate(batch, &dummy_result_batches));
  }
  ASSERT_NOT_OK(sort_expr->finish(&sort_result_iterator_base));

  auto sort_result_iterator =
      std::dynamic_pointer_cast<ResultIterator<arrow::RecordBatch>>(
          sort_result_iterator_base);

  std::shared_ptr<arrow::RecordBatch> dummy_result_batch;
  std::shared_ptr<arrow::RecordBatch> result_batch;

  if (sort_result_iterator->HasNext()) {
    ASSERT_NOT_OK(sort_result_iterator->Next(&result_batch));
    ASSERT_NOT_OK(Equals(*expected_result.get(), *result_batch.get()));
  }
}

TEST(TestArrowComputeSort, SortTestInplaceNullsFirstDesc) {
  ////////////////////// prepare expr_vector ///////////////////////
  auto f0 = field("f0", float64());
  auto arg_0 = TreeExprBuilder::MakeField(f0);
  auto true_literal = TreeExprBuilder::MakeLiteral(true);
  auto false_literal = TreeExprBuilder::MakeLiteral(false);

  auto f_res = field("res", uint32());
  auto indices_type = std::make_shared<FixedSizeBinaryType>(16);
  auto f_indices = field("indices", indices_type);

  auto n_key_func = TreeExprBuilder::MakeFunction("key_function", {arg_0}, uint32());
  auto n_key_field = TreeExprBuilder::MakeFunction("key_field", {arg_0}, uint32());
  auto n_dir =
      TreeExprBuilder::MakeFunction("sort_directions", {false_literal}, uint32());
  auto n_nulls_order =
      TreeExprBuilder::MakeFunction("sort_nulls_order", {true_literal}, uint32());
  auto NaN_check = TreeExprBuilder::MakeFunction("NaN_check", {true_literal}, uint32());
  auto do_codegen = TreeExprBuilder::MakeFunction("codegen", {false_literal}, uint32());
  auto n_sort_to_indices = TreeExprBuilder::MakeFunction(
      "sortArraysToIndices",
      {n_key_func, n_key_field, n_dir, n_nulls_order, NaN_check, do_codegen}, uint32());
  auto n_sort =
      TreeExprBuilder::MakeFunction("standalone", {n_sort_to_indices}, uint32());
  auto sortArrays_expr = TreeExprBuilder::MakeExpression(n_sort, f_res);

  auto sch = arrow::schema({f0});
  std::vector<std::shared_ptr<Field>> ret_types = {f0};
  ///////////////////// Calculation //////////////////
  std::shared_ptr<CodeGenerator> sort_expr;
  arrow::compute::ExecContext ctx;
  ASSERT_NOT_OK(CreateCodeGenerator(ctx.memory_pool(), sch, {sortArrays_expr}, ret_types,
                                    &sort_expr, true));

  std::shared_ptr<arrow::RecordBatch> input_batch;
  std::vector<std::shared_ptr<arrow::RecordBatch>> input_batch_list;
  std::vector<std::shared_ptr<arrow::RecordBatch>> dummy_result_batches;
  std::shared_ptr<ResultIteratorBase> sort_result_iterator_base;

  std::vector<std::string> input_data_string = {"[10, 12, 4, 50, 52, 32, 11]"};
  MakeInputBatch(input_data_string, sch, &input_batch);
  input_batch_list.push_back(input_batch);

  std::vector<std::string> input_data_string_2 = {"[1, NaN, 43, 42, 6, null, 2]"};
  MakeInputBatch(input_data_string_2, sch, &input_batch);
  input_batch_list.push_back(input_batch);

  std::vector<std::string> input_data_string_3 = {"[3, 64, 15, 7, 9, 19, 33]"};
  MakeInputBatch(input_data_string_3, sch, &input_batch);
  input_batch_list.push_back(input_batch);

  std::vector<std::string> input_data_string_4 = {"[23, 17, NaN, 18, 20, 35, 30]"};
  MakeInputBatch(input_data_string_4, sch, &input_batch);
  input_batch_list.push_back(input_batch);

  std::vector<std::string> input_data_string_5 = {"[37, null, NaN, 13, 8, 59, 21]"};
  MakeInputBatch(input_data_string_5, sch, &input_batch);
  input_batch_list.push_back(input_batch);

  ////////////////////////////////// calculation
  //////////////////////////////////////
  std::shared_ptr<arrow::RecordBatch> expected_result;
  std::vector<std::string> expected_result_string = {
      "[null, null, NaN, NaN, NaN, 64, 59, 52, 50, 43, 42, 37, 35, 33, 32, 30, "
      "23, "
      "21, 20, 19, 18, 17, 15, 13, 12, 11, 10, 9, 8, 7, 6, 4, 3, 2, 1]"};
  MakeInputBatch(expected_result_string, sch, &expected_result);

  for (auto batch : input_batch_list) {
    ASSERT_NOT_OK(sort_expr->evaluate(batch, &dummy_result_batches));
  }
  ASSERT_NOT_OK(sort_expr->finish(&sort_result_iterator_base));

  auto sort_result_iterator =
      std::dynamic_pointer_cast<ResultIterator<arrow::RecordBatch>>(
          sort_result_iterator_base);

  std::shared_ptr<arrow::RecordBatch> dummy_result_batch;
  std::shared_ptr<arrow::RecordBatch> result_batch;

  if (sort_result_iterator->HasNext()) {
    ASSERT_NOT_OK(sort_result_iterator->Next(&result_batch));
    ASSERT_NOT_OK(Equals(*expected_result.get(), *result_batch.get()));
  }
}

TEST(TestArrowComputeSort, SortTestInplaceNullsLastDesc) {
  ////////////////////// prepare expr_vector ///////////////////////
  auto f0 = field("f0", float64());
  auto arg_0 = TreeExprBuilder::MakeField(f0);
  auto true_literal = TreeExprBuilder::MakeLiteral(true);
  auto false_literal = TreeExprBuilder::MakeLiteral(false);

  auto f_res = field("res", uint32());
  auto indices_type = std::make_shared<FixedSizeBinaryType>(16);
  auto f_indices = field("indices", indices_type);

  auto n_key_func = TreeExprBuilder::MakeFunction("key_function", {arg_0}, uint32());
  auto n_key_field = TreeExprBuilder::MakeFunction("key_field", {arg_0}, uint32());
  auto n_dir =
      TreeExprBuilder::MakeFunction("sort_directions", {false_literal}, uint32());
  auto n_nulls_order =
      TreeExprBuilder::MakeFunction("sort_nulls_order", {false_literal}, uint32());
  auto NaN_check = TreeExprBuilder::MakeFunction("NaN_check", {true_literal}, uint32());
  auto do_codegen = TreeExprBuilder::MakeFunction("codegen", {false_literal}, uint32());
  auto n_sort_to_indices = TreeExprBuilder::MakeFunction(
      "sortArraysToIndices",
      {n_key_func, n_key_field, n_dir, n_nulls_order, NaN_check, do_codegen}, uint32());
  auto n_sort =
      TreeExprBuilder::MakeFunction("standalone", {n_sort_to_indices}, uint32());
  auto sortArrays_expr = TreeExprBuilder::MakeExpression(n_sort, f_res);

  auto sch = arrow::schema({f0});
  std::vector<std::shared_ptr<Field>> ret_types = {f0};
  ///////////////////// Calculation //////////////////
  std::shared_ptr<CodeGenerator> sort_expr;
  arrow::compute::ExecContext ctx;
  ASSERT_NOT_OK(CreateCodeGenerator(ctx.memory_pool(), sch, {sortArrays_expr}, ret_types,
                                    &sort_expr, true));

  std::shared_ptr<arrow::RecordBatch> input_batch;
  std::vector<std::shared_ptr<arrow::RecordBatch>> input_batch_list;
  std::vector<std::shared_ptr<arrow::RecordBatch>> dummy_result_batches;
  std::shared_ptr<ResultIteratorBase> sort_result_iterator_base;

  std::vector<std::string> input_data_string = {"[10, 12, 4, 50, 52, 32, 11]"};
  MakeInputBatch(input_data_string, sch, &input_batch);
  input_batch_list.push_back(input_batch);

  std::vector<std::string> input_data_string_2 = {"[1, NaN, 43, 42, 6, null, 2]"};
  MakeInputBatch(input_data_string_2, sch, &input_batch);
  input_batch_list.push_back(input_batch);

  std::vector<std::string> input_data_string_3 = {"[3, 64, 15, 7, 9, 19, 33]"};
  MakeInputBatch(input_data_string_3, sch, &input_batch);
  input_batch_list.push_back(input_batch);

  std::vector<std::string> input_data_string_4 = {"[23, 17, NaN, 18, 20, 35, 30]"};
  MakeInputBatch(input_data_string_4, sch, &input_batch);
  input_batch_list.push_back(input_batch);

  std::vector<std::string> input_data_string_5 = {"[37, null, NaN, 13, 8, 59, 21]"};
  MakeInputBatch(input_data_string_5, sch, &input_batch);
  input_batch_list.push_back(input_batch);

  ////////////////////////////////// calculation
  //////////////////////////////////////
  std::shared_ptr<arrow::RecordBatch> expected_result;
  std::vector<std::string> expected_result_string = {
      "[NaN, NaN, NaN, 64, 59, 52, 50, 43, 42, 37, 35, 33, 32, 30, 23, 21, 20, "
      "19, 18, 17, 15, 13, 12, 11, 10, 9, 8, 7, 6, 4, 3, 2, 1, null, null]"};
  MakeInputBatch(expected_result_string, sch, &expected_result);

  for (auto batch : input_batch_list) {
    ASSERT_NOT_OK(sort_expr->evaluate(batch, &dummy_result_batches));
  }
  ASSERT_NOT_OK(sort_expr->finish(&sort_result_iterator_base));

  auto sort_result_iterator =
      std::dynamic_pointer_cast<ResultIterator<arrow::RecordBatch>>(
          sort_result_iterator_base);

  std::shared_ptr<arrow::RecordBatch> dummy_result_batch;
  std::shared_ptr<arrow::RecordBatch> result_batch;

  if (sort_result_iterator->HasNext()) {
    ASSERT_NOT_OK(sort_result_iterator->Next(&result_batch));
    ASSERT_NOT_OK(Equals(*expected_result.get(), *result_batch.get()));
  }
}

TEST(TestArrowComputeSort, SortTestInplaceAsc) {
  ////////////////////// prepare expr_vector ///////////////////////
  auto f0 = field("f0", float64());
  auto arg_0 = TreeExprBuilder::MakeField(f0);
  auto true_literal = TreeExprBuilder::MakeLiteral(true);
  auto false_literal = TreeExprBuilder::MakeLiteral(false);

  auto f_res = field("res", uint32());
  auto indices_type = std::make_shared<FixedSizeBinaryType>(16);
  auto f_indices = field("indices", indices_type);

  auto n_key_func = TreeExprBuilder::MakeFunction("key_function", {arg_0}, uint32());
  auto n_key_field = TreeExprBuilder::MakeFunction("key_field", {arg_0}, uint32());
  auto n_dir = TreeExprBuilder::MakeFunction("sort_directions", {true_literal}, uint32());
  auto n_nulls_order =
      TreeExprBuilder::MakeFunction("sort_nulls_order", {false_literal}, uint32());
  auto NaN_check = TreeExprBuilder::MakeFunction("NaN_check", {true_literal}, uint32());
  auto do_codegen = TreeExprBuilder::MakeFunction("codegen", {false_literal}, uint32());
  auto n_sort_to_indices = TreeExprBuilder::MakeFunction(
      "sortArraysToIndices",
      {n_key_func, n_key_field, n_dir, n_nulls_order, NaN_check, do_codegen}, uint32());
  auto n_sort =
      TreeExprBuilder::MakeFunction("standalone", {n_sort_to_indices}, uint32());
  auto sortArrays_expr = TreeExprBuilder::MakeExpression(n_sort, f_res);

  auto sch = arrow::schema({f0});
  std::vector<std::shared_ptr<Field>> ret_types = {f0};
  ///////////////////// Calculation //////////////////
  std::shared_ptr<CodeGenerator> sort_expr;
  arrow::compute::ExecContext ctx;
  ASSERT_NOT_OK(CreateCodeGenerator(ctx.memory_pool(), sch, {sortArrays_expr}, ret_types,
                                    &sort_expr, true));

  std::shared_ptr<arrow::RecordBatch> input_batch;
  std::vector<std::shared_ptr<arrow::RecordBatch>> input_batch_list;
  std::vector<std::shared_ptr<arrow::RecordBatch>> dummy_result_batches;
  std::shared_ptr<ResultIteratorBase> sort_result_iterator_base;

  std::vector<std::string> input_data_string = {"[10, NaN, 4, 50, 52, 32, 11]"};
  MakeInputBatch(input_data_string, sch, &input_batch);
  input_batch_list.push_back(input_batch);

  std::vector<std::string> input_data_string_2 = {"[1, 14, 43, 42, 6, 45, 2]"};
  MakeInputBatch(input_data_string_2, sch, &input_batch);
  input_batch_list.push_back(input_batch);

  std::vector<std::string> input_data_string_3 = {"[3, 64, NaN, 7, 9, 19, 33]"};
  MakeInputBatch(input_data_string_3, sch, &input_batch);
  input_batch_list.push_back(input_batch);

  std::vector<std::string> input_data_string_4 = {"[23, 17, 41, 18, 20, 35, 30]"};
  MakeInputBatch(input_data_string_4, sch, &input_batch);
  input_batch_list.push_back(input_batch);

  std::vector<std::string> input_data_string_5 = {"[37, 12, 22, 13, 8, 59, 21]"};
  MakeInputBatch(input_data_string_5, sch, &input_batch);
  input_batch_list.push_back(input_batch);

  ////////////////////////////////// calculation
  //////////////////////////////////////
  std::shared_ptr<arrow::RecordBatch> expected_result;
  std::vector<std::string> expected_result_string = {
      "[1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 17, 18, 19, 20, 21, 22, "
      "23, "
      "30, 32, 33, 35, 37, 41, 42, 43, 45, 50, 52, 59, 64, NaN, NaN]"};
  MakeInputBatch(expected_result_string, sch, &expected_result);

  for (auto batch : input_batch_list) {
    ASSERT_NOT_OK(sort_expr->evaluate(batch, &dummy_result_batches));
  }
  ASSERT_NOT_OK(sort_expr->finish(&sort_result_iterator_base));

  auto sort_result_iterator =
      std::dynamic_pointer_cast<ResultIterator<arrow::RecordBatch>>(
          sort_result_iterator_base);

  std::shared_ptr<arrow::RecordBatch> dummy_result_batch;
  std::shared_ptr<arrow::RecordBatch> result_batch;

  if (sort_result_iterator->HasNext()) {
    ASSERT_NOT_OK(sort_result_iterator->Next(&result_batch));
    ASSERT_NOT_OK(Equals(*expected_result.get(), *result_batch.get()));
  }
}

TEST(TestArrowComputeSort, SortTestInplaceDesc) {
  ////////////////////// prepare expr_vector ///////////////////////
  auto f0 = field("f0", float64());
  auto arg_0 = TreeExprBuilder::MakeField(f0);
  auto true_literal = TreeExprBuilder::MakeLiteral(true);
  auto false_literal = TreeExprBuilder::MakeLiteral(false);

  auto f_res = field("res", uint32());
  auto indices_type = std::make_shared<FixedSizeBinaryType>(16);
  auto f_indices = field("indices", indices_type);

  auto n_key_func = TreeExprBuilder::MakeFunction("key_function", {arg_0}, uint32());
  auto n_key_field = TreeExprBuilder::MakeFunction("key_field", {arg_0}, uint32());
  auto n_dir =
      TreeExprBuilder::MakeFunction("sort_directions", {false_literal}, uint32());
  auto n_nulls_order =
      TreeExprBuilder::MakeFunction("sort_nulls_order", {false_literal}, uint32());
  auto NaN_check = TreeExprBuilder::MakeFunction("NaN_check", {true_literal}, uint32());
  auto do_codegen = TreeExprBuilder::MakeFunction("codegen", {false_literal}, uint32());
  auto n_sort_to_indices = TreeExprBuilder::MakeFunction(
      "sortArraysToIndices",
      {n_key_func, n_key_field, n_dir, n_nulls_order, NaN_check, do_codegen}, uint32());
  auto n_sort =
      TreeExprBuilder::MakeFunction("standalone", {n_sort_to_indices}, uint32());
  auto sortArrays_expr = TreeExprBuilder::MakeExpression(n_sort, f_res);

  auto sch = arrow::schema({f0});
  std::vector<std::shared_ptr<Field>> ret_types = {f0};
  ///////////////////// Calculation //////////////////
  std::shared_ptr<CodeGenerator> sort_expr;
  arrow::compute::ExecContext ctx;
  ASSERT_NOT_OK(CreateCodeGenerator(ctx.memory_pool(), sch, {sortArrays_expr}, ret_types,
                                    &sort_expr, true));

  std::shared_ptr<arrow::RecordBatch> input_batch;
  std::vector<std::shared_ptr<arrow::RecordBatch>> input_batch_list;
  std::vector<std::shared_ptr<arrow::RecordBatch>> dummy_result_batches;
  std::shared_ptr<ResultIteratorBase> sort_result_iterator_base;

  std::vector<std::string> input_data_string = {"[10, NaN, 4, 50, 52, 32, 11]"};
  MakeInputBatch(input_data_string, sch, &input_batch);
  input_batch_list.push_back(input_batch);

  std::vector<std::string> input_data_string_2 = {"[1, 14, 43, 42, 6, 45, 2]"};
  MakeInputBatch(input_data_string_2, sch, &input_batch);
  input_batch_list.push_back(input_batch);

  std::vector<std::string> input_data_string_3 = {"[3, 64, NaN, 7, 9, 19, 33]"};
  MakeInputBatch(input_data_string_3, sch, &input_batch);
  input_batch_list.push_back(input_batch);

  std::vector<std::string> input_data_string_4 = {"[23, 17, 41, 18, 20, 35, 30]"};
  MakeInputBatch(input_data_string_4, sch, &input_batch);
  input_batch_list.push_back(input_batch);

  std::vector<std::string> input_data_string_5 = {"[37, 12, 22, 13, 8, 59, 21]"};
  MakeInputBatch(input_data_string_5, sch, &input_batch);
  input_batch_list.push_back(input_batch);

  ////////////////////////////////// calculation
  //////////////////////////////////////
  std::shared_ptr<arrow::RecordBatch> expected_result;
  std::vector<std::string> expected_result_string = {
      "[NaN, NaN, 64, 59, 52, 50, 45, 43, 42, 41, 37, 35, 33, 32, 30, 23, "
      "22, 21, 20, 19, 18, 17, 14, 13, 12, 11, 10, 9, 8, 7, 6, 4, 3, 2, 1]"};
  MakeInputBatch(expected_result_string, sch, &expected_result);

  for (auto batch : input_batch_list) {
    ASSERT_NOT_OK(sort_expr->evaluate(batch, &dummy_result_batches));
  }
  ASSERT_NOT_OK(sort_expr->finish(&sort_result_iterator_base));

  auto sort_result_iterator =
      std::dynamic_pointer_cast<ResultIterator<arrow::RecordBatch>>(
          sort_result_iterator_base);

  std::shared_ptr<arrow::RecordBatch> dummy_result_batch;
  std::shared_ptr<arrow::RecordBatch> result_batch;

  if (sort_result_iterator->HasNext()) {
    ASSERT_NOT_OK(sort_result_iterator->Next(&result_batch));
    ASSERT_NOT_OK(Equals(*expected_result.get(), *result_batch.get()));
  }
}
TEST(TestArrowComputeSort, SortTestOnekeyNullsFirstAscWithSpill) {
  ////////////////////// prepare expr_vector ///////////////////////
  auto f0 = field("f0", float64());
  auto f1 = field("f1", uint32());
  auto arg_0 = TreeExprBuilder::MakeField(f0);
  auto arg_1 = TreeExprBuilder::MakeField(f1);
  auto f_res = field("res", uint32());
  auto indices_type = std::make_shared<FixedSizeBinaryType>(16);
  auto f_indices = field("indices", indices_type);

  auto n_key_func = TreeExprBuilder::MakeFunction("key_function", {arg_0}, uint32());
  auto n_key_field = TreeExprBuilder::MakeFunction("key_field", {arg_0}, uint32());
  auto n_dir = TreeExprBuilder::MakeFunction(
      "sort_directions", {TreeExprBuilder::MakeLiteral(true)}, uint32());
  auto n_nulls_order = TreeExprBuilder::MakeFunction(
      "sort_nulls_order", {TreeExprBuilder::MakeLiteral(true)}, uint32());
  auto NaN_check = TreeExprBuilder::MakeFunction(
      "NaN_check", {TreeExprBuilder::MakeLiteral(true)}, uint32());
  auto do_codegen = TreeExprBuilder::MakeFunction(
      "codegen", {TreeExprBuilder::MakeLiteral(false)}, uint32());
  auto n_sort_to_indices = TreeExprBuilder::MakeFunction(
      "sortArraysToIndices",
      {n_key_func, n_key_field, n_dir, n_nulls_order, NaN_check, do_codegen}, uint32());
  auto n_sort =
      TreeExprBuilder::MakeFunction("standalone", {n_sort_to_indices}, uint32());
  auto sortArrays_expr = TreeExprBuilder::MakeExpression(n_sort, f_res);

  auto sch = arrow::schema({f0, f1});
  std::vector<std::shared_ptr<Field>> ret_types = {f0, f1};
  ///////////////////// Calculation //////////////////
  std::shared_ptr<CodeGenerator> sort_expr;
  arrow::compute::ExecContext ctx;
  ASSERT_NOT_OK(CreateCodeGenerator(ctx.memory_pool(), sch, {sortArrays_expr}, ret_types,
                                    &sort_expr, true));
  std::shared_ptr<arrow::RecordBatch> input_batch;
  std::vector<std::shared_ptr<arrow::RecordBatch>> input_batch_list;
  std::vector<std::shared_ptr<arrow::RecordBatch>> dummy_result_batches;
  auto before = arrow::default_memory_pool()->bytes_allocated();
  std::vector<std::string> input_data_string = {"[10, NaN, 4, 50, 52, 32, 11]",
                                                "[11, 13, 5, 51, null, 33, 12]"};
  MakeInputBatch(input_data_string, sch, &input_batch);
  input_batch_list.push_back(std::move(input_batch));

  std::vector<std::string> input_data_string_2 = {"[1, NaN, 43, 42, 6, null, 2]",
                                                  "[2, null, 44, 43, 7, 34, 3]"};
  MakeInputBatch(input_data_string_2, sch, &input_batch);
  input_batch_list.push_back(std::move(input_batch));

  std::vector<std::string> input_data_string_3 = {"[3, 64, 15, 7, 9, 19, 33]",
                                                  "[4, 65, 16, 8, 10, 20, 34]"};
  MakeInputBatch(input_data_string_3, sch, &input_batch);
  input_batch_list.push_back(std::move(input_batch));

  std::vector<std::string> input_data_string_4 = {"[23, 17, 41, 18, NaN, 35, 30]",
                                                  "[24, 18, 42, 19, 21, 36, 31]"};
  MakeInputBatch(input_data_string_4, sch, &input_batch);
  input_batch_list.push_back(std::move(input_batch));

  std::vector<std::string> input_data_string_5 = {"[37, null, 22, 13, 8, 59, 21]",
                                                  "[38, 67, 23, 14, 9, 60, 22]"};
  MakeInputBatch(input_data_string_5, sch, &input_batch);
  input_batch_list.push_back(std::move(input_batch));
  auto after = arrow::default_memory_pool()->bytes_allocated();
  auto gap = after - before;
  ////////////////////////////////// calculation
  //////////////////////////////////////
  std::shared_ptr<arrow::RecordBatch> expected_result;
  std::vector<std::string> expected_result_string1 = {
      "[null, null, 1, 2, 3, 4, 6, 7, 8, 9]", "[34, 67, 2, 3, 4, 5, 7, 8, 9, 10]"};
  std::vector<std::string> expected_result_string2 = {
      "[10, 11, 13, 15, 17, 18, 19, 21, "
      "22, 23]",
      "[11, 12, 14, 16, 18, 19, 20, 22, "
      "23, 24]"};
  std::vector<std::string> expected_result_string3 = {
      "[30, "
      "32, 33, 35, 37, 41, 42, 43, 50, 52]",
      "["
      "31, 33, 34, 36, 38, 42, 43, 44, 51, null]"};
  std::vector<std::string> expected_result_string4 = {"[59, 64, NaN, NaN, NaN]",
                                                      "[60, 65, 21, null, 13]"};
  std::vector<std::shared_ptr<arrow::RecordBatch>> result_batch_list;
  MakeInputBatch(expected_result_string1, sch, &expected_result);
  result_batch_list.push_back(std::move(expected_result));

  MakeInputBatch(expected_result_string2, sch, &expected_result);
  result_batch_list.push_back(std::move(expected_result));

  MakeInputBatch(expected_result_string3, sch, &expected_result);
  result_batch_list.push_back(std::move(expected_result));

  MakeInputBatch(expected_result_string4, sch, &expected_result);
  result_batch_list.push_back(std::move(expected_result));

  for (auto& batch : input_batch_list) {
    ASSERT_NOT_OK(sort_expr->evaluate(batch, &dummy_result_batches));
  }
  setenv("NATIVESQL_BATCH_SIZE", "10", 0);
  std::shared_ptr<ResultIterator<arrow::RecordBatch>> sort_result_iterator;
  std::shared_ptr<ResultIteratorBase> sort_result_iterator_base;
  ASSERT_NOT_OK(sort_expr->finish(&sort_result_iterator_base));

  sort_result_iterator = std::dynamic_pointer_cast<ResultIterator<arrow::RecordBatch>>(
      sort_result_iterator_base);

  std::shared_ptr<arrow::RecordBatch> dummy_result_batch;
  std::shared_ptr<arrow::RecordBatch> result_batch;
  int64_t size;
  bool firstspill = true;
  auto result_iter = result_batch_list.begin();
  while (sort_result_iterator->HasNext()) {
    ASSERT_NOT_OK(sort_result_iterator->Next(&result_batch));
    sort_expr->Spill(100, false, &size);
    if (firstspill) {
      EXPECT_TRUE(gap >= size);
    } else {
      EXPECT_TRUE(0 == size);
    }
    firstspill = false;
    expected_result = *result_iter;
    ASSERT_NOT_OK(Equals(*expected_result.get(), *result_batch.get()));
    result_iter++;
  }
  unsetenv("NATIVESQL_BATCH_SIZE");
}

TEST(TestArrowComputeSort, SortTestOnekeyNullsFirstAsc) {
  ////////////////////// prepare expr_vector ///////////////////////
  auto f0 = field("f0", float64());
  auto f1 = field("f1", uint32());
  auto arg_0 = TreeExprBuilder::MakeField(f0);
  auto arg_1 = TreeExprBuilder::MakeField(f1);
  auto f_res = field("res", uint32());
  auto indices_type = std::make_shared<FixedSizeBinaryType>(16);
  auto f_indices = field("indices", indices_type);

  auto n_key_func = TreeExprBuilder::MakeFunction("key_function", {arg_0}, uint32());
  auto n_key_field = TreeExprBuilder::MakeFunction("key_field", {arg_0}, uint32());
  auto n_dir = TreeExprBuilder::MakeFunction(
      "sort_directions", {TreeExprBuilder::MakeLiteral(true)}, uint32());
  auto n_nulls_order = TreeExprBuilder::MakeFunction(
      "sort_nulls_order", {TreeExprBuilder::MakeLiteral(true)}, uint32());
  auto NaN_check = TreeExprBuilder::MakeFunction(
      "NaN_check", {TreeExprBuilder::MakeLiteral(true)}, uint32());
  auto do_codegen = TreeExprBuilder::MakeFunction(
      "codegen", {TreeExprBuilder::MakeLiteral(false)}, uint32());
  auto n_sort_to_indices = TreeExprBuilder::MakeFunction(
      "sortArraysToIndices",
      {n_key_func, n_key_field, n_dir, n_nulls_order, NaN_check, do_codegen}, uint32());
  auto n_sort =
      TreeExprBuilder::MakeFunction("standalone", {n_sort_to_indices}, uint32());
  auto sortArrays_expr = TreeExprBuilder::MakeExpression(n_sort, f_res);

  auto sch = arrow::schema({f0, f1});
  std::vector<std::shared_ptr<Field>> ret_types = {f0, f1};
  ///////////////////// Calculation //////////////////
  std::shared_ptr<CodeGenerator> sort_expr;
  arrow::compute::ExecContext ctx;
  ASSERT_NOT_OK(CreateCodeGenerator(ctx.memory_pool(), sch, {sortArrays_expr}, ret_types,
                                    &sort_expr, true));
  std::shared_ptr<arrow::RecordBatch> input_batch;
  std::vector<std::shared_ptr<arrow::RecordBatch>> input_batch_list;
  std::vector<std::shared_ptr<arrow::RecordBatch>> dummy_result_batches;

  std::vector<std::string> input_data_string = {"[10, NaN, 4, 50, 52, 32, 11]",
                                                "[11, 13, 5, 51, null, 33, 12]"};
  MakeInputBatch(input_data_string, sch, &input_batch);
  input_batch_list.push_back(input_batch);

  std::vector<std::string> input_data_string_2 = {"[1, NaN, 43, 42, 6, null, 2]",
                                                  "[2, null, 44, 43, 7, 34, 3]"};
  MakeInputBatch(input_data_string_2, sch, &input_batch);
  input_batch_list.push_back(input_batch);

  std::vector<std::string> input_data_string_3 = {"[3, 64, 15, 7, 9, 19, 33]",
                                                  "[4, 65, 16, 8, 10, 20, 34]"};
  MakeInputBatch(input_data_string_3, sch, &input_batch);
  input_batch_list.push_back(input_batch);

  std::vector<std::string> input_data_string_4 = {"[23, 17, 41, 18, NaN, 35, 30]",
                                                  "[24, 18, 42, 19, 21, 36, 31]"};
  MakeInputBatch(input_data_string_4, sch, &input_batch);
  input_batch_list.push_back(input_batch);

  std::vector<std::string> input_data_string_5 = {"[37, null, 22, 13, 8, 59, 21]",
                                                  "[38, 67, 23, 14, 9, 60, 22]"};
  MakeInputBatch(input_data_string_5, sch, &input_batch);
  input_batch_list.push_back(input_batch);

  ////////////////////////////////// calculation
  //////////////////////////////////////
  std::shared_ptr<arrow::RecordBatch> expected_result;
  std::vector<std::string> expected_result_string = {
      "[null, null, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 13, 15, 17, 18, 19, 21, "
      "22, 23, 30, "
      "32, 33, 35, 37, 41, 42, 43, 50, 52, 59, 64, NaN, NaN, NaN]",
      "[34, 67, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 14, 16, 18, 19, 20, 22, "
      "23, 24, "
      "31, 33, 34, 36, 38, 42, 43, 44, 51, null, 60, 65, 21, null, 13]"};
  MakeInputBatch(expected_result_string, sch, &expected_result);

  for (auto batch : input_batch_list) {
    ASSERT_NOT_OK(sort_expr->evaluate(batch, &dummy_result_batches));
  }
  std::shared_ptr<ResultIterator<arrow::RecordBatch>> sort_result_iterator;
  std::shared_ptr<ResultIteratorBase> sort_result_iterator_base;
  ASSERT_NOT_OK(sort_expr->finish(&sort_result_iterator_base));
  sort_result_iterator = std::dynamic_pointer_cast<ResultIterator<arrow::RecordBatch>>(
      sort_result_iterator_base);

  std::shared_ptr<arrow::RecordBatch> dummy_result_batch;
  std::shared_ptr<arrow::RecordBatch> result_batch;

  if (sort_result_iterator->HasNext()) {
    ASSERT_NOT_OK(sort_result_iterator->Next(&result_batch));
    ASSERT_NOT_OK(Equals(*expected_result.get(), *result_batch.get()));
  }
}

TEST(TestArrowComputeSort, SortTestOnekeyNullsLastAsc) {
  ////////////////////// prepare expr_vector ///////////////////////
  auto f0 = field("f0", float64());
  auto f1 = field("f1", uint32());
  auto arg_0 = TreeExprBuilder::MakeField(f0);
  auto arg_1 = TreeExprBuilder::MakeField(f1);
  auto f_res = field("res", uint32());
  auto indices_type = std::make_shared<FixedSizeBinaryType>(16);
  auto f_indices = field("indices", indices_type);

  auto n_key_func = TreeExprBuilder::MakeFunction("key_function", {arg_0}, uint32());
  auto n_key_field = TreeExprBuilder::MakeFunction("key_field", {arg_0}, uint32());
  auto n_dir = TreeExprBuilder::MakeFunction(
      "sort_directions", {TreeExprBuilder::MakeLiteral(true)}, uint32());
  auto n_nulls_order = TreeExprBuilder::MakeFunction(
      "sort_nulls_order", {TreeExprBuilder::MakeLiteral(false)}, uint32());
  auto NaN_check = TreeExprBuilder::MakeFunction(
      "NaN_check", {TreeExprBuilder::MakeLiteral(true)}, uint32());
  auto do_codegen = TreeExprBuilder::MakeFunction(
      "codegen", {TreeExprBuilder::MakeLiteral(false)}, uint32());
  auto n_sort_to_indices = TreeExprBuilder::MakeFunction(
      "sortArraysToIndices",
      {n_key_func, n_key_field, n_dir, n_nulls_order, NaN_check, do_codegen}, uint32());
  auto n_sort =
      TreeExprBuilder::MakeFunction("standalone", {n_sort_to_indices}, uint32());
  auto sortArrays_expr = TreeExprBuilder::MakeExpression(n_sort, f_res);

  auto sch = arrow::schema({f0, f1});
  std::vector<std::shared_ptr<Field>> ret_types = {f0, f1};
  ///////////////////// Calculation //////////////////
  std::shared_ptr<CodeGenerator> sort_expr;
  arrow::compute::ExecContext ctx;
  ASSERT_NOT_OK(CreateCodeGenerator(ctx.memory_pool(), sch, {sortArrays_expr}, ret_types,
                                    &sort_expr, true));

  std::shared_ptr<arrow::RecordBatch> input_batch;
  std::vector<std::shared_ptr<arrow::RecordBatch>> input_batch_list;
  std::vector<std::shared_ptr<arrow::RecordBatch>> dummy_result_batches;

  std::vector<std::string> input_data_string = {"[10, NaN, 4, 50, 52, 32, 11]",
                                                "[11, 13, 5, 51, null, 33, 12]"};
  MakeInputBatch(input_data_string, sch, &input_batch);
  input_batch_list.push_back(input_batch);

  std::vector<std::string> input_data_string_2 = {"[1, NaN, 43, 42, 6, null, 2]",
                                                  "[2, null, 44, 43, 7, 34, 3]"};
  MakeInputBatch(input_data_string_2, sch, &input_batch);
  input_batch_list.push_back(input_batch);

  std::vector<std::string> input_data_string_3 = {"[3, 64, 15, 7, 9, 19, 33]",
                                                  "[4, 65, 16, 8, 10, 20, 34]"};
  MakeInputBatch(input_data_string_3, sch, &input_batch);
  input_batch_list.push_back(input_batch);

  std::vector<std::string> input_data_string_4 = {"[23, 17, 41, 18, NaN, 35, 30]",
                                                  "[24, 18, 42, 19, 21, 36, 31]"};
  MakeInputBatch(input_data_string_4, sch, &input_batch);
  input_batch_list.push_back(input_batch);

  std::vector<std::string> input_data_string_5 = {"[37, null, 22, 13, 8, 59, 21]",
                                                  "[38, 67, 23, 14, 9, 60, 22]"};
  MakeInputBatch(input_data_string_5, sch, &input_batch);
  input_batch_list.push_back(input_batch);

  ////////////////////////////////// calculation
  //////////////////////////////////////
  std::shared_ptr<arrow::RecordBatch> expected_result;
  std::vector<std::string> expected_result_string = {
      "[1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 13, 15, 17, 18, 19, 21, 22, 23, 30, "
      "32, 33, 35, 37, 41, 42, 43, 50, 52, 59, 64, NaN, NaN, NaN, null, null]",
      "[2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 14, 16, 18, 19, 20, 22, 23, 24,"
      "31, 33, 34, 36, 38, 42, 43, 44, 51, null, 60, 65, 21, null, 13, 34, "
      "67]"};
  MakeInputBatch(expected_result_string, sch, &expected_result);

  for (auto batch : input_batch_list) {
    ASSERT_NOT_OK(sort_expr->evaluate(batch, &dummy_result_batches));
  }
  std::shared_ptr<ResultIterator<arrow::RecordBatch>> sort_result_iterator;
  std::shared_ptr<ResultIteratorBase> sort_result_iterator_base;
  ASSERT_NOT_OK(sort_expr->finish(&sort_result_iterator_base));
  sort_result_iterator = std::dynamic_pointer_cast<ResultIterator<arrow::RecordBatch>>(
      sort_result_iterator_base);

  std::shared_ptr<arrow::RecordBatch> dummy_result_batch;
  std::shared_ptr<arrow::RecordBatch> result_batch;

  if (sort_result_iterator->HasNext()) {
    ASSERT_NOT_OK(sort_result_iterator->Next(&result_batch));
    ASSERT_NOT_OK(Equals(*expected_result.get(), *result_batch.get()));
  }
}

TEST(TestArrowComputeSort, SortTestOnekeyNullsFirstDesc) {
  ////////////////////// prepare expr_vector ///////////////////////
  auto f0 = field("f0", float64());
  auto f1 = field("f1", uint32());
  auto arg_0 = TreeExprBuilder::MakeField(f0);
  auto arg_1 = TreeExprBuilder::MakeField(f1);
  auto f_res = field("res", uint32());
  auto indices_type = std::make_shared<FixedSizeBinaryType>(16);
  auto f_indices = field("indices", indices_type);

  auto n_key_func = TreeExprBuilder::MakeFunction("key_function", {arg_0}, uint32());
  auto n_key_field = TreeExprBuilder::MakeFunction("key_field", {arg_0}, uint32());
  auto n_dir = TreeExprBuilder::MakeFunction(
      "sort_directions", {TreeExprBuilder::MakeLiteral(false)}, uint32());
  auto n_nulls_order = TreeExprBuilder::MakeFunction(
      "sort_nulls_order", {TreeExprBuilder::MakeLiteral(true)}, uint32());
  auto NaN_check = TreeExprBuilder::MakeFunction(
      "NaN_check", {TreeExprBuilder::MakeLiteral(true)}, uint32());
  auto do_codegen = TreeExprBuilder::MakeFunction(
      "codegen", {TreeExprBuilder::MakeLiteral(false)}, uint32());
  auto n_sort_to_indices = TreeExprBuilder::MakeFunction(
      "sortArraysToIndices",
      {n_key_func, n_key_field, n_dir, n_nulls_order, NaN_check, do_codegen}, uint32());
  auto n_sort =
      TreeExprBuilder::MakeFunction("standalone", {n_sort_to_indices}, uint32());
  auto sortArrays_expr = TreeExprBuilder::MakeExpression(n_sort, f_res);

  auto sch = arrow::schema({f0, f1});
  std::vector<std::shared_ptr<Field>> ret_types = {f0, f1};
  ///////////////////// Calculation //////////////////
  std::shared_ptr<CodeGenerator> sort_expr;
  arrow::compute::ExecContext ctx;
  ASSERT_NOT_OK(CreateCodeGenerator(ctx.memory_pool(), sch, {sortArrays_expr}, ret_types,
                                    &sort_expr, true));

  std::shared_ptr<arrow::RecordBatch> input_batch;
  std::vector<std::shared_ptr<arrow::RecordBatch>> input_batch_list;
  std::vector<std::shared_ptr<arrow::RecordBatch>> dummy_result_batches;

  std::vector<std::string> input_data_string = {"[10, NaN, 4, 50, 52, 32, 11]",
                                                "[11, 13, 5, 51, null, 33, 12]"};
  MakeInputBatch(input_data_string, sch, &input_batch);
  input_batch_list.push_back(input_batch);

  std::vector<std::string> input_data_string_2 = {"[1, NaN, 43, 42, 6, null, 2]",
                                                  "[2, null, 44, 43, 7, 34, 3]"};
  MakeInputBatch(input_data_string_2, sch, &input_batch);
  input_batch_list.push_back(input_batch);

  std::vector<std::string> input_data_string_3 = {"[3, 64, 15, 7, 9, 19, 33]",
                                                  "[4, 65, 16, 8, 10, 20, 34]"};
  MakeInputBatch(input_data_string_3, sch, &input_batch);
  input_batch_list.push_back(input_batch);

  std::vector<std::string> input_data_string_4 = {"[23, 17, 41, 18, NaN, 35, 30]",
                                                  "[24, 18, 42, 19, 21, 36, 31]"};
  MakeInputBatch(input_data_string_4, sch, &input_batch);
  input_batch_list.push_back(input_batch);

  std::vector<std::string> input_data_string_5 = {"[37, null, 22, 13, 8, 59, 21]",
                                                  "[38, 67, 23, 14, 9, 60, 22]"};
  MakeInputBatch(input_data_string_5, sch, &input_batch);
  input_batch_list.push_back(input_batch);

  ////////////////////////////////// calculation
  //////////////////////////////////////
  std::shared_ptr<arrow::RecordBatch> expected_result;
  std::vector<std::string> expected_result_string = {
      "[null ,null , NaN, NaN, NaN, 64 ,59 ,52 ,50 ,43 ,42 ,41 ,37 ,35 ,33 ,32 "
      ",30 "
      ",23 ,22 ,21 ,19 ,18 ,17 ,15 ,13 , 11 ,10 ,9 ,8 ,7 ,6 ,4 ,3 ,2 ,1]",
      "[34 ,67 ,13, null, 21, 65 ,60 ,null ,51 ,44 ,43 ,42 ,38 ,36 ,34 ,33 ,31 "
      ",24 "
      ",23 ,22 , 20 ,19 ,18 ,16 ,14 ,12 ,11 ,10 ,9 ,8 ,7 ,5 ,4 ,3 ,2]"};
  MakeInputBatch(expected_result_string, sch, &expected_result);

  for (auto batch : input_batch_list) {
    ASSERT_NOT_OK(sort_expr->evaluate(batch, &dummy_result_batches));
  }
  std::shared_ptr<ResultIterator<arrow::RecordBatch>> sort_result_iterator;
  std::shared_ptr<ResultIteratorBase> sort_result_iterator_base;
  ASSERT_NOT_OK(sort_expr->finish(&sort_result_iterator_base));
  sort_result_iterator = std::dynamic_pointer_cast<ResultIterator<arrow::RecordBatch>>(
      sort_result_iterator_base);

  std::shared_ptr<arrow::RecordBatch> dummy_result_batch;
  std::shared_ptr<arrow::RecordBatch> result_batch;

  if (sort_result_iterator->HasNext()) {
    ASSERT_NOT_OK(sort_result_iterator->Next(&result_batch));
    ASSERT_NOT_OK(Equals(*expected_result.get(), *result_batch.get()));
  }
}

TEST(TestArrowComputeSort, SortTestOnekeyNullsLastDesc) {
  ////////////////////// prepare expr_vector ///////////////////////
  auto f0 = field("f0", float64());
  auto f1 = field("f1", uint32());
  auto arg_0 = TreeExprBuilder::MakeField(f0);
  auto arg_1 = TreeExprBuilder::MakeField(f1);
  auto f_res = field("res", uint32());
  auto indices_type = std::make_shared<FixedSizeBinaryType>(16);
  auto f_indices = field("indices", indices_type);

  auto n_key_func = TreeExprBuilder::MakeFunction("key_function", {arg_0}, uint32());
  auto n_key_field = TreeExprBuilder::MakeFunction("key_field", {arg_0}, uint32());
  auto n_dir = TreeExprBuilder::MakeFunction(
      "sort_directions", {TreeExprBuilder::MakeLiteral(false)}, uint32());
  auto n_nulls_order = TreeExprBuilder::MakeFunction(
      "sort_nulls_order", {TreeExprBuilder::MakeLiteral(false)}, uint32());
  auto NaN_check = TreeExprBuilder::MakeFunction(
      "NaN_check", {TreeExprBuilder::MakeLiteral(true)}, uint32());
  auto do_codegen = TreeExprBuilder::MakeFunction(
      "codegen", {TreeExprBuilder::MakeLiteral(false)}, uint32());
  auto n_sort_to_indices = TreeExprBuilder::MakeFunction(
      "sortArraysToIndices",
      {n_key_func, n_key_field, n_dir, n_nulls_order, NaN_check, do_codegen}, uint32());
  auto n_sort =
      TreeExprBuilder::MakeFunction("standalone", {n_sort_to_indices}, uint32());
  auto sortArrays_expr = TreeExprBuilder::MakeExpression(n_sort, f_res);

  auto sch = arrow::schema({f0, f1});
  std::vector<std::shared_ptr<Field>> ret_types = {f0, f1};
  ///////////////////// Calculation //////////////////
  std::shared_ptr<CodeGenerator> sort_expr;
  arrow::compute::ExecContext ctx;
  ASSERT_NOT_OK(CreateCodeGenerator(ctx.memory_pool(), sch, {sortArrays_expr}, ret_types,
                                    &sort_expr, true));

  std::shared_ptr<arrow::RecordBatch> input_batch;
  std::vector<std::shared_ptr<arrow::RecordBatch>> input_batch_list;
  std::vector<std::shared_ptr<arrow::RecordBatch>> dummy_result_batches;

  std::vector<std::string> input_data_string = {"[10, NaN, 4, 50, 52, 32, 11]",
                                                "[11, 13, 5, 51, null, 33, 12]"};
  MakeInputBatch(input_data_string, sch, &input_batch);
  input_batch_list.push_back(input_batch);

  std::vector<std::string> input_data_string_2 = {"[1, NaN, 43, 42, 6, null, 2]",
                                                  "[2, null, 44, 43, 7, 34, 3]"};
  MakeInputBatch(input_data_string_2, sch, &input_batch);
  input_batch_list.push_back(input_batch);

  std::vector<std::string> input_data_string_3 = {"[3, 64, 15, 7, 9, 19, 33]",
                                                  "[4, 65, 16, 8, 10, 20, 34]"};
  MakeInputBatch(input_data_string_3, sch, &input_batch);
  input_batch_list.push_back(input_batch);

  std::vector<std::string> input_data_string_4 = {"[23, 17, 41, 18, NaN, 35, 30]",
                                                  "[24, 18, 42, 19, 21, 36, 31]"};
  MakeInputBatch(input_data_string_4, sch, &input_batch);
  input_batch_list.push_back(input_batch);

  std::vector<std::string> input_data_string_5 = {"[37, null, 22, 13, 8, 59, 21]",
                                                  "[38, 67, 23, 14, 9, 60, 22]"};
  MakeInputBatch(input_data_string_5, sch, &input_batch);
  input_batch_list.push_back(input_batch);

  ////////////////////////////////// calculation
  //////////////////////////////////////
  std::shared_ptr<arrow::RecordBatch> expected_result;
  std::vector<std::string> expected_result_string = {
      "[NaN, NaN, NaN, 64 ,59 ,52 ,50 ,43 ,42 ,41 ,37 ,35 ,33 ,32 ,30 ,23 ,22 "
      ",21 "
      ",19 ,18 ,17 ,15 ,13 , 11 ,10 ,9 ,8 ,7 ,6 ,4 ,3 ,2 ,1, null, null]",
      "[13, null, 21, 65 ,60 ,null ,51 ,44 ,43 ,42 ,38 ,36 ,34 ,33 ,31 ,24 ,23 "
      ",22 "
      ", 20 ,19 ,18 ,16 ,14 ,12 ,11 ,10 ,9 ,8 ,7 ,5 ,4 ,3 ,2, 34, 67]"};
  MakeInputBatch(expected_result_string, sch, &expected_result);

  for (auto batch : input_batch_list) {
    ASSERT_NOT_OK(sort_expr->evaluate(batch, &dummy_result_batches));
  }
  std::shared_ptr<ResultIterator<arrow::RecordBatch>> sort_result_iterator;
  std::shared_ptr<ResultIteratorBase> sort_result_iterator_base;
  ASSERT_NOT_OK(sort_expr->finish(&sort_result_iterator_base));
  sort_result_iterator = std::dynamic_pointer_cast<ResultIterator<arrow::RecordBatch>>(
      sort_result_iterator_base);

  std::shared_ptr<arrow::RecordBatch> dummy_result_batch;
  std::shared_ptr<arrow::RecordBatch> result_batch;

  if (sort_result_iterator->HasNext()) {
    ASSERT_NOT_OK(sort_result_iterator->Next(&result_batch));
    ASSERT_NOT_OK(Equals(*expected_result.get(), *result_batch.get()));
  }
}

TEST(TestArrowComputeSort, SortTestOnekeyBooleanDesc) {
  ////////////////////// prepare expr_vector ///////////////////////
  auto f0 = field("f0", boolean());
  auto f1 = field("f1", uint32());
  auto arg_0 = TreeExprBuilder::MakeField(f0);
  auto arg_1 = TreeExprBuilder::MakeField(f1);
  auto true_literal = TreeExprBuilder::MakeLiteral(true);
  auto false_literal = TreeExprBuilder::MakeLiteral(false);

  auto f_res = field("res", uint32());
  auto indices_type = std::make_shared<FixedSizeBinaryType>(16);
  auto f_indices = field("indices", indices_type);

  auto n_key_func = TreeExprBuilder::MakeFunction("key_function", {arg_0}, uint32());
  auto n_key_field = TreeExprBuilder::MakeFunction("key_field", {arg_0}, uint32());
  auto n_dir =
      TreeExprBuilder::MakeFunction("sort_directions", {false_literal}, uint32());
  auto n_nulls_order =
      TreeExprBuilder::MakeFunction("sort_nulls_order", {true_literal}, uint32());
  auto NaN_check = TreeExprBuilder::MakeFunction("NaN_check", {true_literal}, uint32());
  auto do_codegen = TreeExprBuilder::MakeFunction("codegen", {false_literal}, uint32());
  auto n_sort_to_indices = TreeExprBuilder::MakeFunction(
      "sortArraysToIndices",
      {n_key_func, n_key_field, n_dir, n_nulls_order, NaN_check, do_codegen}, uint32());
  auto n_sort =
      TreeExprBuilder::MakeFunction("standalone", {n_sort_to_indices}, uint32());
  auto sortArrays_expr = TreeExprBuilder::MakeExpression(n_sort, f_res);

  auto sch = arrow::schema({f0, f1});
  std::vector<std::shared_ptr<Field>> ret_types = {f0, f1};
  ///////////////////// Calculation //////////////////
  std::shared_ptr<CodeGenerator> sort_expr;
  arrow::compute::ExecContext ctx;
  ASSERT_NOT_OK(CreateCodeGenerator(ctx.memory_pool(), sch, {sortArrays_expr}, ret_types,
                                    &sort_expr, true));

  std::shared_ptr<arrow::RecordBatch> input_batch;
  std::vector<std::shared_ptr<arrow::RecordBatch>> input_batch_list;
  std::vector<std::shared_ptr<arrow::RecordBatch>> dummy_result_batches;
  std::shared_ptr<ResultIteratorBase> sort_result_iterator_base;

  std::vector<std::string> input_data_string = {
      "[true, false, false, false, true, true, false]", "[1, 2, 3, 4, 5, 6, 7]"};
  MakeInputBatch(input_data_string, sch, &input_batch);
  input_batch_list.push_back(input_batch);

  std::vector<std::string> input_data_string_2 = {
      "[true, true, false, false, false, true, false]", "[4, 2, 6, 0, 1, 4, 12]"};
  MakeInputBatch(input_data_string_2, sch, &input_batch);
  input_batch_list.push_back(input_batch);

  std::vector<std::string> input_data_string_3 = {
      "[true, true, false, false, false, true, false]", "[6, 12, 16, 10, 11, 41, 2]"};
  MakeInputBatch(input_data_string_3, sch, &input_batch);
  input_batch_list.push_back(input_batch);

  std::vector<std::string> input_data_string_4 = {
      "[true, true, false, false, false, true, false]", "[8, 22, 45, 12, 78, 12, 32]"};
  MakeInputBatch(input_data_string_4, sch, &input_batch);
  input_batch_list.push_back(input_batch);

  std::vector<std::string> input_data_string_5 = {
      "[true, true, false, false, false, true, false]", "[18, 5, 6, 78, 11, 2, 12]"};
  MakeInputBatch(input_data_string_5, sch, &input_batch);
  input_batch_list.push_back(input_batch);

  ////////////////////////////////// calculation
  //////////////////////////////////////
  std::shared_ptr<arrow::RecordBatch> expected_result;
  std::vector<std::string> expected_result_string = {
      "[true, true, true, true, true, true, true, true, true, true, true, "
      "true, true, true, true, "
      "false, false, false, false, false, false, false, false, false, false, "
      "false, false, false, "
      "false, false, false, false, false, false, false]",
      "[18, 22, 8, 41, 1, 12, 12, 6, 4, 5, 2, 4, 6, 5, 2, 6, 10, 32, 78, 78, "
      "11, 12, 12, 45, 2, "
      "11, 16, 12, 1, 0, 6, 7, 4, 3, 2]"};
  MakeInputBatch(expected_result_string, sch, &expected_result);

  for (auto batch : input_batch_list) {
    ASSERT_NOT_OK(sort_expr->evaluate(batch, &dummy_result_batches));
  }
  ASSERT_NOT_OK(sort_expr->finish(&sort_result_iterator_base));

  auto sort_result_iterator =
      std::dynamic_pointer_cast<ResultIterator<arrow::RecordBatch>>(
          sort_result_iterator_base);

  std::shared_ptr<arrow::RecordBatch> dummy_result_batch;
  std::shared_ptr<arrow::RecordBatch> result_batch;

  if (sort_result_iterator->HasNext()) {
    ASSERT_NOT_OK(sort_result_iterator->Next(&result_batch));
    ASSERT_NOT_OK(Equals(*expected_result.get(), *result_batch.get()));
  }
}

TEST(TestArrowComputeSort, SortTestOneKeyStr) {
  ////////////////////// prepare expr_vector ///////////////////////
  auto f0 = field("f0", utf8());
  auto f1 = field("f1", utf8());
  auto arg_0 = TreeExprBuilder::MakeField(f0);
  auto arg_1 = TreeExprBuilder::MakeField(f1);
  auto true_literal = TreeExprBuilder::MakeLiteral(true);
  auto false_literal = TreeExprBuilder::MakeLiteral(false);

  auto f_res = field("res", uint32());
  auto indices_type = std::make_shared<FixedSizeBinaryType>(16);
  auto f_indices = field("indices", indices_type);

  auto n_key_func = TreeExprBuilder::MakeFunction("key_function", {arg_0}, uint32());
  auto n_key_field = TreeExprBuilder::MakeFunction("key_field", {arg_0}, uint32());
  auto n_dir = TreeExprBuilder::MakeFunction("sort_directions", {true_literal}, uint32());
  auto n_nulls_order =
      TreeExprBuilder::MakeFunction("sort_nulls_order", {false_literal}, uint32());
  auto NaN_check = TreeExprBuilder::MakeFunction("NaN_check", {true_literal}, uint32());
  auto do_codegen = TreeExprBuilder::MakeFunction("codegen", {false_literal}, uint32());
  auto n_sort_to_indices = TreeExprBuilder::MakeFunction(
      "sortArraysToIndices",
      {n_key_func, n_key_field, n_dir, n_nulls_order, NaN_check, do_codegen}, uint32());
  auto n_sort =
      TreeExprBuilder::MakeFunction("standalone", {n_sort_to_indices}, uint32());
  auto sortArrays_expr = TreeExprBuilder::MakeExpression(n_sort, f_res);

  auto sch = arrow::schema({f0, f1});
  std::vector<std::shared_ptr<Field>> ret_types = {f0, f1};
  ///////////////////// Calculation //////////////////
  std::shared_ptr<CodeGenerator> sort_expr;
  arrow::compute::ExecContext ctx;
  ASSERT_NOT_OK(CreateCodeGenerator(ctx.memory_pool(), sch, {sortArrays_expr}, ret_types,
                                    &sort_expr, true));
  std::shared_ptr<arrow::RecordBatch> input_batch;
  std::vector<std::shared_ptr<arrow::RecordBatch>> input_batch_list;
  std::vector<std::shared_ptr<arrow::RecordBatch>> dummy_result_batches;
  std::shared_ptr<ResultIteratorBase> sort_result_iterator_base;
  std::vector<std::string> input_data_string = {
      R"(["b", "q", "s", "t", null, null, "a"])",
      R"(["a", "c", "e", "f", "g", null, "h"])"};
  MakeInputBatch(input_data_string, sch, &input_batch);
  input_batch_list.push_back(input_batch);
  std::vector<std::string> input_data_string_2 = {
      R"([null, "f", "q", "d", "r", null, "g"])",
      R"(["a", "c", "e", "f", null, "j", "h"])"};
  MakeInputBatch(input_data_string_2, sch, &input_batch);
  input_batch_list.push_back(input_batch);
  std::vector<std::string> input_data_string_3 = {
      R"(["p", "q", "o", "e", null, null, "l"])",
      R"(["a", "c", "e", "f", "g","j", null])"};
  MakeInputBatch(input_data_string_3, sch, &input_batch);
  input_batch_list.push_back(input_batch);
  std::vector<std::string> input_data_string_4 = {
      R"(["q", "w", "z", "x", "y", null, "u"])", R"(["a", "c", "e", "f", "g","j", "h"])"};
  MakeInputBatch(input_data_string_4, sch, &input_batch);
  input_batch_list.push_back(input_batch);
  std::vector<std::string> input_data_string_5 = {
      R"(["a", "c", "b", "d", null, null, null])",
      R"(["a", null, "e", "f", "g","j", "h"])"};
  MakeInputBatch(input_data_string_5, sch, &input_batch);
  input_batch_list.push_back(input_batch);
  ////////////////////////////////// calculation
  //////////////////////////////////////
  std::shared_ptr<arrow::RecordBatch> expected_result;
  std::vector<std::string> expected_result_string = {
      R"(["a","a","b","b","c","d","d","e","f","g","l","o","p","q","q","q","q","r","s","t","u","w","x","y","z",null,null,null,null,null,null,null,null,null,null])",
      R"(["h","a","e","a",null,"f","f","f","c","h",null,"e","a","c","a","e","c",null,"e","f","h","c","f","g","e","g",null,"a","j","g","j","j","g","j","h"])"};
  MakeInputBatch(expected_result_string, sch, &expected_result);
  for (auto batch : input_batch_list) {
    ASSERT_NOT_OK(sort_expr->evaluate(batch, &dummy_result_batches));
  }
  ASSERT_NOT_OK(sort_expr->finish(&sort_result_iterator_base));
  auto sort_result_iterator =
      std::dynamic_pointer_cast<ResultIterator<arrow::RecordBatch>>(
          sort_result_iterator_base);
  std::shared_ptr<arrow::RecordBatch> dummy_result_batch;
  std::shared_ptr<arrow::RecordBatch> result_batch;
  if (sort_result_iterator->HasNext()) {
    ASSERT_NOT_OK(sort_result_iterator->Next(&result_batch));
    ASSERT_NOT_OK(Equals(*expected_result.get(), *result_batch.get()));
  }
}

TEST(TestArrowComputeSort, SortTestOneKeyWithProjection) {
  ////////////////////// prepare expr_vector ///////////////////////
  auto f0 = field("f0", utf8());
  auto f1 = field("f1", utf8());
  auto arg_0 = TreeExprBuilder::MakeField(f0);
  auto arg_1 = TreeExprBuilder::MakeField(f1);
  auto true_literal = TreeExprBuilder::MakeLiteral(true);
  auto false_literal = TreeExprBuilder::MakeLiteral(false);

  auto f_res = field("res", uint32());
  auto indices_type = std::make_shared<FixedSizeBinaryType>(16);
  auto f_indices = field("indices", indices_type);

  auto n_projection = TreeExprBuilder::MakeFunction("upper", {arg_0}, utf8());
  auto n_key_func =
      TreeExprBuilder::MakeFunction("key_function", {n_projection}, uint32());
  auto n_key_field = TreeExprBuilder::MakeFunction("key_field", {arg_0}, uint32());
  auto n_dir = TreeExprBuilder::MakeFunction("sort_directions", {true_literal}, uint32());
  auto n_nulls_order =
      TreeExprBuilder::MakeFunction("sort_nulls_order", {false_literal}, uint32());
  auto NaN_check = TreeExprBuilder::MakeFunction("NaN_check", {true_literal}, uint32());
  auto do_codegen = TreeExprBuilder::MakeFunction("codegen", {false_literal}, uint32());
  auto n_sort_to_indices = TreeExprBuilder::MakeFunction(
      "sortArraysToIndices",
      {n_key_func, n_key_field, n_dir, n_nulls_order, NaN_check, do_codegen}, uint32());
  auto n_sort =
      TreeExprBuilder::MakeFunction("standalone", {n_sort_to_indices}, uint32());
  auto sortArrays_expr = TreeExprBuilder::MakeExpression(n_sort, f_res);

  auto sch = arrow::schema({f0, f1});
  std::vector<std::shared_ptr<Field>> ret_types = {f0, f1};
  ///////////////////// Calculation //////////////////
  std::shared_ptr<CodeGenerator> sort_expr;
  arrow::compute::ExecContext ctx;
  ASSERT_NOT_OK(CreateCodeGenerator(ctx.memory_pool(), sch, {sortArrays_expr}, ret_types,
                                    &sort_expr, true));
  std::shared_ptr<arrow::RecordBatch> input_batch;
  std::vector<std::shared_ptr<arrow::RecordBatch>> input_batch_list;
  std::vector<std::shared_ptr<arrow::RecordBatch>> dummy_result_batches;
  std::shared_ptr<ResultIteratorBase> sort_result_iterator_base;
  std::vector<std::string> input_data_string = {
      R"(["B", "q", "s", "T", null, null, "a"])",
      R"(["a", "c", "e", "f", "g", null, "h"])"};
  MakeInputBatch(input_data_string, sch, &input_batch);
  input_batch_list.push_back(input_batch);
  std::vector<std::string> input_data_string_2 = {
      R"([null, "F", "Q", "d", "r", null, "g"])",
      R"(["a", "c", "e", "f", null, "j", "h"])"};
  MakeInputBatch(input_data_string_2, sch, &input_batch);
  input_batch_list.push_back(input_batch);
  std::vector<std::string> input_data_string_3 = {
      R"(["p", "q", "o", "E", null, null, "l"])",
      R"(["a", "c", "e", "f", "g","j", null])"};
  MakeInputBatch(input_data_string_3, sch, &input_batch);
  input_batch_list.push_back(input_batch);
  std::vector<std::string> input_data_string_4 = {
      R"(["q", "W", "Z", "x", "y", null, "u"])", R"(["a", "c", "e", "f", "g","j", "h"])"};
  MakeInputBatch(input_data_string_4, sch, &input_batch);
  input_batch_list.push_back(input_batch);
  std::vector<std::string> input_data_string_5 = {
      R"(["a", "C", "b", "D", null, null, null])",
      R"(["a", null, "e", "f", "g","j", "h"])"};
  MakeInputBatch(input_data_string_5, sch, &input_batch);
  input_batch_list.push_back(input_batch);
  ////////////////////////////////// calculation
  //////////////////////////////////////
  std::shared_ptr<arrow::RecordBatch> expected_result;
  std::vector<std::string> expected_result_string = {
      R"(["a","a","b","B","C","D","d","E","F","g","l","o","p","q","q","Q","q","r","s","T","u","W","x","y","Z",null,null,null,null,null,null,null,null,null,null])",
      R"(["h","a","e","a",null,"f","f","f","c","h",null,"e","a","c","a","e","c",null,"e","f","h","c","f","g","e","g",null,"a","j","g","j","j","g","j","h"])"};
  MakeInputBatch(expected_result_string, sch, &expected_result);
  for (auto batch : input_batch_list) {
    ASSERT_NOT_OK(sort_expr->evaluate(batch, &dummy_result_batches));
  }
  ASSERT_NOT_OK(sort_expr->finish(&sort_result_iterator_base));
  auto sort_result_iterator =
      std::dynamic_pointer_cast<ResultIterator<arrow::RecordBatch>>(
          sort_result_iterator_base);
  std::shared_ptr<arrow::RecordBatch> dummy_result_batch;
  std::shared_ptr<arrow::RecordBatch> result_batch;
  if (sort_result_iterator->HasNext()) {
    ASSERT_NOT_OK(sort_result_iterator->Next(&result_batch));
    ASSERT_NOT_OK(Equals(*expected_result.get(), *result_batch.get()));
  }
}

TEST(TestArrowComputeSort, SortTestMultipleKeysNaN) {
  ////////////////////// prepare expr_vector ///////////////////////
  auto f0 = field("f0", float64());
  auto f1 = field("f1", utf8());
  auto f2 = field("f2", float64());
  auto f3 = field("f3", uint32());
  auto arg_0 = TreeExprBuilder::MakeField(f0);
  auto arg_1 = TreeExprBuilder::MakeField(f1);
  auto arg_2 = TreeExprBuilder::MakeField(f2);
  auto true_literal = TreeExprBuilder::MakeLiteral(true);
  auto false_literal = TreeExprBuilder::MakeLiteral(false);
  auto f_res = field("res", uint32());
  auto indices_type = std::make_shared<FixedSizeBinaryType>(16);
  auto f_indices = field("indices", indices_type);

  auto n_key_func =
      TreeExprBuilder::MakeFunction("key_function", {arg_0, arg_1, arg_2}, uint32());
  auto n_key_field =
      TreeExprBuilder::MakeFunction("key_field", {arg_0, arg_1, arg_2}, uint32());
  auto n_dir = TreeExprBuilder::MakeFunction(
      "sort_directions", {true_literal, false_literal, true_literal}, uint32());
  auto n_nulls_order = TreeExprBuilder::MakeFunction(
      "sort_nulls_order", {false_literal, true_literal, true_literal}, uint32());
  auto NaN_check = TreeExprBuilder::MakeFunction("NaN_check", {true_literal}, uint32());
  auto do_codegen = TreeExprBuilder::MakeFunction("codegen", {true_literal}, uint32());
  auto n_sort_to_indices = TreeExprBuilder::MakeFunction(
      "sortArraysToIndices",
      {n_key_func, n_key_field, n_dir, n_nulls_order, NaN_check, do_codegen}, uint32());
  auto n_sort =
      TreeExprBuilder::MakeFunction("standalone", {n_sort_to_indices}, uint32());
  auto sortArrays_expr = TreeExprBuilder::MakeExpression(n_sort, f_res);

  auto sch = arrow::schema({f0, f1, f2, f3});
  std::vector<std::shared_ptr<Field>> ret_types = {f0, f1, f2, f3};
  ///////////////////// Calculation //////////////////
  std::shared_ptr<CodeGenerator> sort_expr;
  arrow::compute::ExecContext ctx;
  ASSERT_NOT_OK(CreateCodeGenerator(ctx.memory_pool(), sch, {sortArrays_expr}, ret_types,
                                    &sort_expr, true));

  std::shared_ptr<arrow::RecordBatch> input_batch;
  std::vector<std::shared_ptr<arrow::RecordBatch>> input_batch_list;
  std::vector<std::shared_ptr<arrow::RecordBatch>> dummy_result_batches;

  std::vector<std::string> input_data_string = {
      "[8, NaN, 4, 50, 52, 32, 11]", R"([null, "a", "a", "b", "b","b", "b"])",
      "[11, NaN, 5, 51, null, 33, 12]", "[1, 3, 5, 10, null, 13, 2]"};
  MakeInputBatch(input_data_string, sch, &input_batch);
  input_batch_list.push_back(input_batch);

  std::vector<std::string> input_data_string_2 = {
      "[1, 14, NaN, 42, 6, null, 2]", R"(["a", "a", null, "b", "b", "a", "b"])",
      "[2, null, 44, 43, 7, 34, 3]", "[9, 7, 5, 1, 5, null, 17]"};
  MakeInputBatch(input_data_string_2, sch, &input_batch);
  input_batch_list.push_back(input_batch);

  std::vector<std::string> input_data_string_3 = {
      "[3, 64, 8, 7, 9, 8, NaN]", R"(["a", "a", "b", "b", "b","b", "b"])",
      "[4, 65, 16, 8, 10, 20, 34]", "[8, 6, 2, 3, 10, 12, 15]"};
  MakeInputBatch(input_data_string_3, sch, &input_batch);
  input_batch_list.push_back(input_batch);

  std::vector<std::string> input_data_string_4 = {
      "[23, 17, 41, 18, 20, 35, 30]", R"(["a", "a", "a", "b", "b","b", "b"])",
      "[24, 18, 42, NaN, 21, 36, 31]", "[15, 16, 2, 51, null, 33, 12]"};
  MakeInputBatch(input_data_string_4, sch, &input_batch);
  input_batch_list.push_back(input_batch);

  std::vector<std::string> input_data_string_5 = {
      "[37, null, 22, 13, 8, 59, 21]", R"(["a", "b", "a", "b", "b","b", "b"])",
      "[38, 67, 23, 14, null, 60, 22]", "[16, 17, 5, 15, 9, null, 19]"};
  MakeInputBatch(input_data_string_5, sch, &input_batch);
  input_batch_list.push_back(input_batch);

  ////////////////////////////////// calculation
  //////////////////////////////////////
  std::shared_ptr<arrow::RecordBatch> expected_result;
  std::vector<std::string> expected_result_string = {
      "[1, 2, 3, 4, 6, 7, 8, 8, 8, 8, 9, 11, 13, 14, 17, 18, 20, 21, "
      "22, 23, 30, 32, 35, 37, 41, 42, 50, 52, 59, 64, NaN, NaN, NaN, null, "
      "null]",
      R"(["a","b","a","a","b","b", null,"b","b","b","b","b","b","a","a","b","b","b","a","a","b","b","b","a","a","b","b","b","b","a",null,"b","a","b","a"])",
      "[2, 3, 4, 5, 7, 8, 11, null, 16, 20, 10, 12, 14, null, 18, NaN, 21, 22, "
      "23, "
      "24, 31, 33, 36, 38, 42, 43, 51, null, 60, 65, 44, 34, NaN, 67, 34]",
      "[9, 17, 8, 5, 5, 3, 1, 9, 2, 12, 10, 2, 15, 7, 16, 51, null, 19, 5, "
      "15, 12, 13, 33, 16, 2, 1, 10, null, null, 6, 5, 15, 3, 17, null]"};

  MakeInputBatch(expected_result_string, sch, &expected_result);

  for (auto batch : input_batch_list) {
    ASSERT_NOT_OK(sort_expr->evaluate(batch, &dummy_result_batches));
  }
  std::shared_ptr<ResultIterator<arrow::RecordBatch>> sort_result_iterator;
  std::shared_ptr<ResultIteratorBase> sort_result_iterator_base;
  ASSERT_NOT_OK(sort_expr->finish(&sort_result_iterator_base));
  sort_result_iterator = std::dynamic_pointer_cast<ResultIterator<arrow::RecordBatch>>(
      sort_result_iterator_base);

  std::shared_ptr<arrow::RecordBatch> dummy_result_batch;
  std::shared_ptr<arrow::RecordBatch> result_batch;

  if (sort_result_iterator->HasNext()) {
    ASSERT_NOT_OK(sort_result_iterator->Next(&result_batch));
    ASSERT_NOT_OK(Equals(*expected_result.get(), *result_batch.get()));
  }
}

TEST(TestArrowComputeSort, SortTestMultipleKeysWithProjection) {
  ////////////////////// prepare expr_vector ///////////////////////
  auto f0 = field("f0", uint32());
  auto f1 = field("f1", utf8());
  auto f2 = field("f2", uint32());
  auto f3 = field("f3", uint32());
  auto arg_0 = TreeExprBuilder::MakeField(f0);
  auto arg_1 = TreeExprBuilder::MakeField(f1);
  auto arg_2 = TreeExprBuilder::MakeField(f2);
  auto true_literal = TreeExprBuilder::MakeLiteral(true);
  auto false_literal = TreeExprBuilder::MakeLiteral(false);
  auto f_res = field("res", uint32());
  auto f_bool = field("res", arrow::boolean());
  auto indices_type = std::make_shared<FixedSizeBinaryType>(16);
  auto f_indices = field("indices", indices_type);

  auto uint32_node = TreeExprBuilder::MakeLiteral((uint32_t)0);
  auto str_node = TreeExprBuilder::MakeStringLiteral("");

  auto isnotnull_0 = TreeExprBuilder::MakeFunction(
      "isnotnull", {TreeExprBuilder::MakeField(f0)}, arrow::boolean());
  auto coalesce_0 = TreeExprBuilder::MakeIf(isnotnull_0, TreeExprBuilder::MakeField(f0),
                                            uint32_node, uint32());
  auto isnull_0 = TreeExprBuilder::MakeFunction("isnull", {arg_0}, arrow::boolean());

  auto isnotnull_1 = TreeExprBuilder::MakeFunction(
      "isnotnull", {TreeExprBuilder::MakeField(f1)}, arrow::boolean());
  auto coalesce_1 = TreeExprBuilder::MakeIf(isnotnull_1, TreeExprBuilder::MakeField(f1),
                                            str_node, utf8());
  auto isnull_1 = TreeExprBuilder::MakeFunction("isnull", {arg_1}, arrow::boolean());

  auto isnotnull_2 = TreeExprBuilder::MakeFunction(
      "isnotnull", {TreeExprBuilder::MakeField(f2)}, arrow::boolean());
  auto coalesce_2 = TreeExprBuilder::MakeIf(isnotnull_2, TreeExprBuilder::MakeField(f2),
                                            uint32_node, uint32());
  auto isnull_2 = TreeExprBuilder::MakeFunction("isnull", {arg_2}, arrow::boolean());

  auto n_key_func = TreeExprBuilder::MakeFunction(
      "key_function", {coalesce_0, isnull_0, coalesce_1, isnull_1, coalesce_2, isnull_2},
      uint32());
  auto n_key_field = TreeExprBuilder::MakeFunction(
      "key_field", {arg_0, arg_0, arg_1, arg_1, arg_2, arg_2}, uint32());
  auto n_dir = TreeExprBuilder::MakeFunction("sort_directions",
                                             {
                                                 true_literal,
                                                 true_literal,
                                                 false_literal,
                                                 false_literal,
                                                 true_literal,
                                                 true_literal,
                                             },
                                             uint32());
  auto n_nulls_order =
      TreeExprBuilder::MakeFunction("sort_nulls_order",
                                    {false_literal, false_literal, true_literal,
                                     true_literal, true_literal, true_literal},
                                    uint32());
  auto NaN_check = TreeExprBuilder::MakeFunction("NaN_check", {true_literal}, uint32());
  auto do_codegen = TreeExprBuilder::MakeFunction("codegen", {true_literal}, uint32());
  auto n_sort_to_indices = TreeExprBuilder::MakeFunction(
      "sortArraysToIndices",
      {n_key_func, n_key_field, n_dir, n_nulls_order, NaN_check, do_codegen}, uint32());
  auto n_sort =
      TreeExprBuilder::MakeFunction("standalone", {n_sort_to_indices}, uint32());
  auto sortArrays_expr = TreeExprBuilder::MakeExpression(n_sort, f_res);

  auto sch = arrow::schema({f0, f1, f2, f3});
  std::vector<std::shared_ptr<Field>> ret_types = {f0, f1, f2, f3};
  auto ret_schema = arrow::schema(ret_types);
  ///////////////////// Calculation //////////////////
  std::shared_ptr<CodeGenerator> sort_expr;
  arrow::compute::ExecContext ctx;
  ASSERT_NOT_OK(CreateCodeGenerator(ctx.memory_pool(), sch, {sortArrays_expr}, ret_types,
                                    &sort_expr, true));

  std::shared_ptr<arrow::RecordBatch> input_batch;
  std::vector<std::shared_ptr<arrow::RecordBatch>> input_batch_list;
  std::vector<std::shared_ptr<arrow::RecordBatch>> dummy_result_batches;

  std::vector<std::string> input_data_string = {
      "[8, 8, 4, 50, 52, 32, 11]", R"([null, "b", "a", "b", "b","b", "b"])",
      "[11, 10, 5, 51, null, 33, 12]", "[1, 3, 5, 10, null, 13, 2]"};
  MakeInputBatch(input_data_string, sch, &input_batch);
  input_batch_list.push_back(input_batch);

  std::vector<std::string> input_data_string_2 = {
      "[1, 14, 8, 42, 6, null, 2]", R"(["a", "a", null, "b", "b","b", "b"])",
      "[2, null, 44, 43, 7, 34, 3]", "[9, 7, 5, 1, 5, null, 17]"};
  MakeInputBatch(input_data_string_2, sch, &input_batch);
  input_batch_list.push_back(input_batch);

  std::vector<std::string> input_data_string_3 = {
      "[3, 64, 8, 7, 9, 8, 33]", R"(["a", "a", "a", "b", "b","b", "b"])",
      "[4, 65, 16, 8, 10, 20, 34]", "[8, 6, 2, 3, 10, 12, 15]"};
  MakeInputBatch(input_data_string_3, sch, &input_batch);
  input_batch_list.push_back(input_batch);

  std::vector<std::string> input_data_string_4 = {
      "[23, 17, 41, 18, 20, 35, 30]", R"(["a", "a", "a", "b", "b","b", "b"])",
      "[24, 18, 42, 19, 21, 36, 31]", "[15, 16, 2, 51, null, 33, 12]"};
  MakeInputBatch(input_data_string_4, sch, &input_batch);
  input_batch_list.push_back(input_batch);

  std::vector<std::string> input_data_string_5 = {
      "[37, null, 22, 13, 8, 59, 21]", R"(["a", "a", "a", "b", "b","b", "b"])",
      "[38, 67, 23, 14, null, 60, 22]", "[16, 17, 5, 15, 9, null, 19]"};
  MakeInputBatch(input_data_string_5, sch, &input_batch);
  input_batch_list.push_back(input_batch);

  ////////////////////////////////// calculation
  //////////////////////////////////////
  std::shared_ptr<arrow::RecordBatch> expected_result;
  std::vector<std::string> expected_result_string = {
      "[null, null, 1, 2, 3, 4, 6, 7, 8, 8, 8, 8, 8, 8, 9, 11, 13, 14, 17, 18, "
      "20, 21, "
      "22, 23, 30, 32, 33, 35, 37, 41, 42, 50, 52, 59, 64]",
      R"(["b","a","a","b","a","a","b","b","b","b","b","a", null, null,"b","b","b","a","a","b","b","b","a","a","b","b","b","b","a","a","b","b","b","b","a"])",
      "[34, 67, 2, 3, 4, 5, 7, 8, null, 10, 20, 16, 11, 44, 10, 12, 14, null, "
      "18, 19, 21, 22, 23, "
      "24, 31, 33, 34, 36, 38, 42, 43, 51, null, 60, 65]",
      "[null, 17, 9, 17, 8, 5, 5, 3, 9, 3, 12, 2, 1, 5, 10, 2, 15, 7, 16, 51, "
      "null, 19, 5, "
      "15, 12, 13, 15, 33, 16, 2, 1, 10, null, null, 6]"};

  MakeInputBatch(expected_result_string, ret_schema, &expected_result);

  for (auto batch : input_batch_list) {
    ASSERT_NOT_OK(sort_expr->evaluate(batch, &dummy_result_batches));
  }
  std::shared_ptr<ResultIterator<arrow::RecordBatch>> sort_result_iterator;
  std::shared_ptr<ResultIteratorBase> sort_result_iterator_base;
  ASSERT_NOT_OK(sort_expr->finish(&sort_result_iterator_base));
  sort_result_iterator = std::dynamic_pointer_cast<ResultIterator<arrow::RecordBatch>>(
      sort_result_iterator_base);

  std::shared_ptr<arrow::RecordBatch> dummy_result_batch;
  std::shared_ptr<arrow::RecordBatch> result_batch;

  if (sort_result_iterator->HasNext()) {
    ASSERT_NOT_OK(sort_result_iterator->Next(&result_batch));
    ASSERT_NOT_OK(Equals(*expected_result.get(), *result_batch.get()));
  }
}

TEST(TestArrowComputeSort, SortTestMultipleKeysWithoutCodegen) {
  ////////////////////// prepare expr_vector ///////////////////////
  auto f0 = field("f0", float32());
  auto f1 = field("f1", utf8());
  auto f2 = field("f2", uint32());
  auto f3 = field("f3", float64());
  auto arg_0 = TreeExprBuilder::MakeField(f0);
  auto arg_1 = TreeExprBuilder::MakeField(f1);
  auto arg_2 = TreeExprBuilder::MakeField(f2);
  auto true_literal = TreeExprBuilder::MakeLiteral(true);
  auto false_literal = TreeExprBuilder::MakeLiteral(false);
  auto f_res = field("res", uint32());
  auto indices_type = std::make_shared<FixedSizeBinaryType>(16);
  auto f_indices = field("indices", indices_type);

  auto n_key_func =
      TreeExprBuilder::MakeFunction("key_function", {arg_0, arg_1, arg_2}, uint32());
  auto n_key_field =
      TreeExprBuilder::MakeFunction("key_field", {arg_0, arg_1, arg_2}, uint32());
  auto n_dir = TreeExprBuilder::MakeFunction(
      "sort_directions", {true_literal, false_literal, true_literal}, uint32());
  auto n_nulls_order = TreeExprBuilder::MakeFunction(
      "sort_nulls_order", {false_literal, true_literal, true_literal}, uint32());
  auto NaN_check = TreeExprBuilder::MakeFunction("NaN_check", {true_literal}, uint32());
  auto do_codegen = TreeExprBuilder::MakeFunction("codegen", {false_literal}, uint32());
  auto n_sort_to_indices = TreeExprBuilder::MakeFunction(
      "sortArraysToIndices",
      {n_key_func, n_key_field, n_dir, n_nulls_order, NaN_check, do_codegen}, uint32());
  auto n_sort =
      TreeExprBuilder::MakeFunction("standalone", {n_sort_to_indices}, uint32());
  auto sortArrays_expr = TreeExprBuilder::MakeExpression(n_sort, f_res);

  auto sch = arrow::schema({f0, f1, f2, f3});
  std::vector<std::shared_ptr<Field>> ret_types = {f0, f1, f2, f3};
  ///////////////////// Calculation //////////////////
  std::shared_ptr<CodeGenerator> sort_expr;
  arrow::compute::ExecContext ctx;
  ASSERT_NOT_OK(CreateCodeGenerator(ctx.memory_pool(), sch, {sortArrays_expr}, ret_types,
                                    &sort_expr, true));

  std::shared_ptr<arrow::RecordBatch> input_batch;
  std::vector<std::shared_ptr<arrow::RecordBatch>> input_batch_list;
  std::vector<std::shared_ptr<arrow::RecordBatch>> dummy_result_batches;

  std::vector<std::string> input_data_string = {
      "[8, 9, 4, 50, 52, 32, 11]", R"([null, "a", "a", "b", "b","b", "b"])",
      "[11, 3, 5, 51, null, 33, 12]", "[1, 3, 5, 10, null, 13, 2]"};
  MakeInputBatch(input_data_string, sch, &input_batch);
  input_batch_list.push_back(input_batch);

  std::vector<std::string> input_data_string_2 = {
      "[1, 14, 6, 42, 6, null, 2]", R"(["a", "a", null, "b", "b", "a", "b"])",
      "[2, null, 44, 43, 7, 34, 3]", "[9, 7, 5, 1, 5, null, 17]"};
  MakeInputBatch(input_data_string_2, sch, &input_batch);
  input_batch_list.push_back(input_batch);

  std::vector<std::string> input_data_string_3 = {
      "[3, 64, 8, 7, 9, 8, 12]", R"(["a", "a", "b", "b", "b","b", "b"])",
      "[4, 65, 16, 8, 10, 20, 34]", "[8, 6, 2, 3, 10, 12, 15]"};
  MakeInputBatch(input_data_string_3, sch, &input_batch);
  input_batch_list.push_back(input_batch);

  std::vector<std::string> input_data_string_4 = {
      "[23, 17, 41, 18, 20, 35, 30]", R"(["a", "a", "a", "b", "b","b", "b"])",
      "[24, 18, 42, 15, 21, 36, 31]", "[15, 16, 2, 51, null, 33, 12]"};
  MakeInputBatch(input_data_string_4, sch, &input_batch);
  input_batch_list.push_back(input_batch);

  std::vector<std::string> input_data_string_5 = {
      "[37, null, 22, 13, 8, 59, 21]", R"(["a", "b", "a", "b", "a","b", "b"])",
      "[38, 67, 23, 14, null, 60, 22]", "[16, 17, 5, 15, 9, null, 19]"};
  MakeInputBatch(input_data_string_5, sch, &input_batch);
  input_batch_list.push_back(input_batch);

  ////////////////////////////////// calculation
  //////////////////////////////////////
  std::shared_ptr<arrow::RecordBatch> expected_result;
  std::vector<std::string> expected_result_string = {
      "[1, 2, 3, 4, 6, 6, 7, 8, 8, 8, 8, 9, 9, 11, 12, 13, 14, 17, 18, 20, 21, "
      "22, 23, 30, 32, 35, 37, 41, 42, 50, 52, 59, 64, null, null]",
      R"(["a","b","a","a",null,"b","b",null,"b","b","a","b","a","b","b","b","a","a","b","b","b","a","a","b","b","b","a","a","b","b","b","b","a","b","a"])",
      "[2, 3, 4, 5, 44, 7, 8, 11, 16, 20, null, 10, 3, 12, 34, 14, null, 18, "
      "15, 21, 22, "
      "23, 24, 31, 33, 36, 38, 42, 43, 51, null, 60, 65, 67, 34]",
      "[9, 17, 8, 5, 5, 5, 3, 1, 2, 12, 9, 10, 3, 2, 15, 15, 7, 16, 51, null, "
      "19, 5, "
      "15, 12, 13, 33, 16, 2, 1, 10, null, null, 6, 17, null]"};

  MakeInputBatch(expected_result_string, sch, &expected_result);

  for (auto batch : input_batch_list) {
    ASSERT_NOT_OK(sort_expr->evaluate(batch, &dummy_result_batches));
  }
  std::shared_ptr<ResultIterator<arrow::RecordBatch>> sort_result_iterator;
  std::shared_ptr<ResultIteratorBase> sort_result_iterator_base;
  ASSERT_NOT_OK(sort_expr->finish(&sort_result_iterator_base));
  sort_result_iterator = std::dynamic_pointer_cast<ResultIterator<arrow::RecordBatch>>(
      sort_result_iterator_base);

  std::shared_ptr<arrow::RecordBatch> dummy_result_batch;
  std::shared_ptr<arrow::RecordBatch> result_batch;

  if (sort_result_iterator->HasNext()) {
    ASSERT_NOT_OK(sort_result_iterator->Next(&result_batch));
    ASSERT_NOT_OK(Equals(*expected_result.get(), *result_batch.get()));
  }
}

TEST(TestArrowComputeSort, SortTestMultipleKeysWithoutCodegenWithProjection) {
  ////////////////////// prepare expr_vector ///////////////////////
  auto f0 = field("f0", uint32());
  auto f1 = field("f1", utf8());
  auto f2 = field("f2", uint32());
  auto f3 = field("f3", uint32());
  auto arg_0 = TreeExprBuilder::MakeField(f0);
  auto arg_1 = TreeExprBuilder::MakeField(f1);
  auto arg_2 = TreeExprBuilder::MakeField(f2);
  auto true_literal = TreeExprBuilder::MakeLiteral(true);
  auto false_literal = TreeExprBuilder::MakeLiteral(false);
  auto f_res = field("res", uint32());
  auto f_bool = field("res", arrow::boolean());
  auto indices_type = std::make_shared<FixedSizeBinaryType>(16);
  auto f_indices = field("indices", indices_type);

  auto uint32_node = TreeExprBuilder::MakeLiteral((uint32_t)0);
  auto str_node = TreeExprBuilder::MakeStringLiteral("");

  auto isnotnull_0 = TreeExprBuilder::MakeFunction(
      "isnotnull", {TreeExprBuilder::MakeField(f0)}, arrow::boolean());
  auto coalesce_0 = TreeExprBuilder::MakeIf(isnotnull_0, TreeExprBuilder::MakeField(f0),
                                            uint32_node, uint32());
  auto isnull_0 = TreeExprBuilder::MakeFunction("isnull", {arg_0}, arrow::boolean());

  auto isnotnull_1 = TreeExprBuilder::MakeFunction(
      "isnotnull", {TreeExprBuilder::MakeField(f1)}, arrow::boolean());
  auto coalesce_1 = TreeExprBuilder::MakeIf(isnotnull_1, TreeExprBuilder::MakeField(f1),
                                            str_node, utf8());
  auto isnull_1 = TreeExprBuilder::MakeFunction("isnull", {arg_1}, arrow::boolean());

  auto isnotnull_2 = TreeExprBuilder::MakeFunction(
      "isnotnull", {TreeExprBuilder::MakeField(f2)}, arrow::boolean());
  auto coalesce_2 = TreeExprBuilder::MakeIf(isnotnull_2, TreeExprBuilder::MakeField(f2),
                                            uint32_node, uint32());
  auto isnull_2 = TreeExprBuilder::MakeFunction("isnull", {arg_2}, arrow::boolean());

  auto n_key_func = TreeExprBuilder::MakeFunction(
      "key_function", {coalesce_0, isnull_0, coalesce_1, isnull_1, coalesce_2, isnull_2},
      uint32());
  auto n_key_field = TreeExprBuilder::MakeFunction(
      "key_field", {arg_0, arg_0, arg_1, arg_1, arg_2, arg_2}, uint32());
  auto n_dir = TreeExprBuilder::MakeFunction("sort_directions",
                                             {
                                                 true_literal,
                                                 true_literal,
                                                 false_literal,
                                                 false_literal,
                                                 true_literal,
                                                 true_literal,
                                             },
                                             uint32());
  auto n_nulls_order =
      TreeExprBuilder::MakeFunction("sort_nulls_order",
                                    {false_literal, false_literal, true_literal,
                                     true_literal, true_literal, true_literal},
                                    uint32());
  auto NaN_check = TreeExprBuilder::MakeFunction("NaN_check", {true_literal}, uint32());
  auto do_codegen = TreeExprBuilder::MakeFunction("codegen", {false_literal}, uint32());
  auto n_sort_to_indices = TreeExprBuilder::MakeFunction(
      "sortArraysToIndices",
      {n_key_func, n_key_field, n_dir, n_nulls_order, NaN_check, do_codegen}, uint32());
  auto n_sort =
      TreeExprBuilder::MakeFunction("standalone", {n_sort_to_indices}, uint32());
  auto sortArrays_expr = TreeExprBuilder::MakeExpression(n_sort, f_res);

  auto sch = arrow::schema({f0, f1, f2, f3});
  std::vector<std::shared_ptr<Field>> ret_types = {f0, f1, f2, f3};
  auto ret_schema = arrow::schema(ret_types);
  ///////////////////// Calculation //////////////////
  std::shared_ptr<CodeGenerator> sort_expr;
  arrow::compute::ExecContext ctx;
  ASSERT_NOT_OK(CreateCodeGenerator(ctx.memory_pool(), sch, {sortArrays_expr}, ret_types,
                                    &sort_expr, true));

  std::shared_ptr<arrow::RecordBatch> input_batch;
  std::vector<std::shared_ptr<arrow::RecordBatch>> input_batch_list;
  std::vector<std::shared_ptr<arrow::RecordBatch>> dummy_result_batches;

  std::vector<std::string> input_data_string = {
      "[8, 8, 4, 50, 52, 32, 11]", R"([null, "b", "a", "b", "b","b", "b"])",
      "[11, 10, 5, 51, null, 33, 12]", "[1, 3, 5, 10, null, 13, 2]"};
  MakeInputBatch(input_data_string, sch, &input_batch);
  input_batch_list.push_back(input_batch);

  std::vector<std::string> input_data_string_2 = {
      "[1, 14, 8, 42, 6, null, 2]", R"(["a", "a", null, "b", "b","b", "b"])",
      "[2, null, 44, 43, 7, 34, 3]", "[9, 7, 5, 1, 5, null, 17]"};
  MakeInputBatch(input_data_string_2, sch, &input_batch);
  input_batch_list.push_back(input_batch);

  std::vector<std::string> input_data_string_3 = {
      "[3, 64, 8, 7, 9, 8, 33]", R"(["a", "a", "a", "b", "b","b", "b"])",
      "[4, 65, 16, 8, 10, 20, 34]", "[8, 6, 2, 3, 10, 12, 15]"};
  MakeInputBatch(input_data_string_3, sch, &input_batch);
  input_batch_list.push_back(input_batch);

  std::vector<std::string> input_data_string_4 = {
      "[23, 17, 41, 18, 20, 35, 30]", R"(["a", "a", "a", "b", "b","b", "b"])",
      "[24, 18, 42, 19, 21, 36, 31]", "[15, 16, 2, 51, null, 33, 12]"};
  MakeInputBatch(input_data_string_4, sch, &input_batch);
  input_batch_list.push_back(input_batch);

  std::vector<std::string> input_data_string_5 = {
      "[37, null, 22, 13, 8, 59, 21]", R"(["a", "a", "a", "b", "b","b", "b"])",
      "[38, 67, 23, 14, null, 60, 22]", "[16, 17, 5, 15, 9, null, 19]"};
  MakeInputBatch(input_data_string_5, sch, &input_batch);
  input_batch_list.push_back(input_batch);

  ////////////////////////////////// calculation
  //////////////////////////////////////
  std::shared_ptr<arrow::RecordBatch> expected_result;
  std::vector<std::string> expected_result_string = {
      "[null, null, 1, 2, 3, 4, 6, 7, 8, 8, 8, 8, 8, 8, 9, 11, 13, 14, 17, 18, "
      "20, 21, "
      "22, 23, 30, 32, 33, 35, 37, 41, 42, 50, 52, 59, 64]",
      R"(["b","a","a","b","a","a","b","b","b","b","b","a", null, null,"b","b","b","a","a","b","b","b","a","a","b","b","b","b","a","a","b","b","b","b","a"])",
      "[34, 67, 2, 3, 4, 5, 7, 8, null, 10, 20, 16, 11, 44, 10, 12, 14, null, "
      "18, 19, 21, 22, 23, "
      "24, 31, 33, 34, 36, 38, 42, 43, 51, null, 60, 65]",
      "[null, 17, 9, 17, 8, 5, 5, 3, 9, 3, 12, 2, 1, 5, 10, 2, 15, 7, 16, 51, "
      "null, 19, 5, "
      "15, 12, 13, 15, 33, 16, 2, 1, 10, null, null, 6]"};

  MakeInputBatch(expected_result_string, ret_schema, &expected_result);

  for (auto batch : input_batch_list) {
    ASSERT_NOT_OK(sort_expr->evaluate(batch, &dummy_result_batches));
  }
  std::shared_ptr<ResultIterator<arrow::RecordBatch>> sort_result_iterator;
  std::shared_ptr<ResultIteratorBase> sort_result_iterator_base;
  ASSERT_NOT_OK(sort_expr->finish(&sort_result_iterator_base));
  sort_result_iterator = std::dynamic_pointer_cast<ResultIterator<arrow::RecordBatch>>(
      sort_result_iterator_base);

  std::shared_ptr<arrow::RecordBatch> dummy_result_batch;
  std::shared_ptr<arrow::RecordBatch> result_batch;

  if (sort_result_iterator->HasNext()) {
    ASSERT_NOT_OK(sort_result_iterator->Next(&result_batch));
    ASSERT_NOT_OK(Equals(*expected_result.get(), *result_batch.get()));
  }
}

TEST(TestArrowComputeSort, SortTestMultipleKeysNaNWithoutCodegen) {
  ////////////////////// prepare expr_vector ///////////////////////
  auto f0 = field("f0", float64());
  auto f1 = field("f1", utf8());
  auto f2 = field("f2", float64());
  auto f3 = field("f3", uint32());
  auto arg_0 = TreeExprBuilder::MakeField(f0);
  auto arg_1 = TreeExprBuilder::MakeField(f1);
  auto arg_2 = TreeExprBuilder::MakeField(f2);
  auto true_literal = TreeExprBuilder::MakeLiteral(true);
  auto false_literal = TreeExprBuilder::MakeLiteral(false);
  auto f_res = field("res", uint32());
  auto indices_type = std::make_shared<FixedSizeBinaryType>(16);
  auto f_indices = field("indices", indices_type);

  auto n_key_func =
      TreeExprBuilder::MakeFunction("key_function", {arg_0, arg_1, arg_2}, uint32());
  auto n_key_field =
      TreeExprBuilder::MakeFunction("key_field", {arg_0, arg_1, arg_2}, uint32());
  auto n_dir = TreeExprBuilder::MakeFunction(
      "sort_directions", {true_literal, false_literal, true_literal}, uint32());
  auto n_nulls_order = TreeExprBuilder::MakeFunction(
      "sort_nulls_order", {false_literal, true_literal, true_literal}, uint32());
  auto NaN_check = TreeExprBuilder::MakeFunction("NaN_check", {true_literal}, uint32());
  auto do_codegen = TreeExprBuilder::MakeFunction("codegen", {false_literal}, uint32());
  auto n_sort_to_indices = TreeExprBuilder::MakeFunction(
      "sortArraysToIndices",
      {n_key_func, n_key_field, n_dir, n_nulls_order, NaN_check, do_codegen}, uint32());
  auto n_sort =
      TreeExprBuilder::MakeFunction("standalone", {n_sort_to_indices}, uint32());
  auto sortArrays_expr = TreeExprBuilder::MakeExpression(n_sort, f_res);

  auto sch = arrow::schema({f0, f1, f2, f3});
  std::vector<std::shared_ptr<Field>> ret_types = {f0, f1, f2, f3};
  ///////////////////// Calculation //////////////////
  std::shared_ptr<CodeGenerator> sort_expr;
  arrow::compute::ExecContext ctx;
  ASSERT_NOT_OK(CreateCodeGenerator(ctx.memory_pool(), sch, {sortArrays_expr}, ret_types,
                                    &sort_expr, true));

  std::shared_ptr<arrow::RecordBatch> input_batch;
  std::vector<std::shared_ptr<arrow::RecordBatch>> input_batch_list;
  std::vector<std::shared_ptr<arrow::RecordBatch>> dummy_result_batches;

  std::vector<std::string> input_data_string = {
      "[8, NaN, 4, 50, 52, 32, 11]", R"([null, "a", "a", "b", "b","b", "b"])",
      "[11, NaN, 5, 51, null, 33, 12]", "[1, 3, 5, 10, null, 13, 2]"};
  MakeInputBatch(input_data_string, sch, &input_batch);
  input_batch_list.push_back(input_batch);

  std::vector<std::string> input_data_string_2 = {
      "[1, 14, NaN, 42, 6, null, 2]", R"(["a", "a", null, "b", "b", "a", "b"])",
      "[2, null, 44, 43, 7, 34, 3]", "[9, 7, 5, 1, 5, null, 17]"};
  MakeInputBatch(input_data_string_2, sch, &input_batch);
  input_batch_list.push_back(input_batch);

  std::vector<std::string> input_data_string_3 = {
      "[3, 64, 8, 7, 9, 8, NaN]", R"(["a", "a", "b", "b", "b","b", "b"])",
      "[4, 65, 16, 8, 10, 20, 34]", "[8, 6, 2, 3, 10, 12, 15]"};
  MakeInputBatch(input_data_string_3, sch, &input_batch);
  input_batch_list.push_back(input_batch);

  std::vector<std::string> input_data_string_4 = {
      "[23, 17, 41, 18, 20, 35, 30]", R"(["a", "a", "a", "b", "b","b", "b"])",
      "[24, 18, 42, NaN, 21, 36, 31]", "[15, 16, 2, 51, null, 33, 12]"};
  MakeInputBatch(input_data_string_4, sch, &input_batch);
  input_batch_list.push_back(input_batch);

  std::vector<std::string> input_data_string_5 = {
      "[37, null, 22, 13, 8, 59, 21]", R"(["a", "b", "a", "b", "b","b", "b"])",
      "[38, 67, 23, 14, null, 60, 22]", "[16, 17, 5, 15, 9, null, 19]"};
  MakeInputBatch(input_data_string_5, sch, &input_batch);
  input_batch_list.push_back(input_batch);

  ////////////////////////////////// calculation
  //////////////////////////////////////
  std::shared_ptr<arrow::RecordBatch> expected_result;
  std::vector<std::string> expected_result_string = {
      "[1, 2, 3, 4, 6, 7, 8, 8, 8, 8, 9, 11, 13, 14, 17, 18, 20, 21, "
      "22, 23, 30, 32, 35, 37, 41, 42, 50, 52, 59, 64, NaN, NaN, NaN, null, "
      "null]",
      R"(["a","b","a","a","b","b", null,"b","b","b","b","b","b","a","a","b","b","b","a","a","b","b","b","a","a","b","b","b","b","a",null,"b","a","b","a"])",
      "[2, 3, 4, 5, 7, 8, 11, null, 16, 20, 10, 12, 14, null, 18, NaN, 21, 22, "
      "23, "
      "24, 31, 33, 36, 38, 42, 43, 51, null, 60, 65, 44, 34, NaN, 67, 34]",
      "[9, 17, 8, 5, 5, 3, 1, 9, 2, 12, 10, 2, 15, 7, 16, 51, null, 19, 5, "
      "15, 12, 13, 33, 16, 2, 1, 10, null, null, 6, 5, 15, 3, 17, null]"};

  MakeInputBatch(expected_result_string, sch, &expected_result);

  for (auto batch : input_batch_list) {
    ASSERT_NOT_OK(sort_expr->evaluate(batch, &dummy_result_batches));
  }
  std::shared_ptr<ResultIterator<arrow::RecordBatch>> sort_result_iterator;
  std::shared_ptr<ResultIteratorBase> sort_result_iterator_base;
  ASSERT_NOT_OK(sort_expr->finish(&sort_result_iterator_base));
  sort_result_iterator = std::dynamic_pointer_cast<ResultIterator<arrow::RecordBatch>>(
      sort_result_iterator_base);

  std::shared_ptr<arrow::RecordBatch> dummy_result_batch;
  std::shared_ptr<arrow::RecordBatch> result_batch;

  if (sort_result_iterator->HasNext()) {
    ASSERT_NOT_OK(sort_result_iterator->Next(&result_batch));
    ASSERT_NOT_OK(Equals(*expected_result.get(), *result_batch.get()));
  }
}

TEST(TestArrowComputeSort, SortTestOneKeyDecimal) {
  ////////////////////// prepare expr_vector ///////////////////////
  auto f0 = field("f0", decimal128(10, 4));
  auto f1 = field("f1", decimal128(16, 5));
  auto arg_0 = TreeExprBuilder::MakeField(f0);
  auto arg_1 = TreeExprBuilder::MakeField(f1);
  auto true_literal = TreeExprBuilder::MakeLiteral(true);
  auto false_literal = TreeExprBuilder::MakeLiteral(false);

  auto f_res = field("res", uint32());
  auto indices_type = std::make_shared<FixedSizeBinaryType>(16);
  auto f_indices = field("indices", indices_type);

  auto n_key_func = TreeExprBuilder::MakeFunction("key_function", {arg_0}, uint32());
  auto n_key_field = TreeExprBuilder::MakeFunction("key_field", {arg_0}, uint32());
  auto n_dir = TreeExprBuilder::MakeFunction("sort_directions", {true_literal}, uint32());
  auto n_nulls_order =
      TreeExprBuilder::MakeFunction("sort_nulls_order", {false_literal}, uint32());
  auto NaN_check = TreeExprBuilder::MakeFunction("NaN_check", {false_literal}, uint32());
  auto do_codegen = TreeExprBuilder::MakeFunction("codegen", {false_literal}, uint32());
  auto n_sort_to_indices = TreeExprBuilder::MakeFunction(
      "sortArraysToIndices",
      {n_key_func, n_key_field, n_dir, n_nulls_order, NaN_check, do_codegen}, uint32());
  auto n_sort =
      TreeExprBuilder::MakeFunction("standalone", {n_sort_to_indices}, uint32());
  auto sortArrays_expr = TreeExprBuilder::MakeExpression(n_sort, f_res);

  auto sch = arrow::schema({f0, f1});
  std::vector<std::shared_ptr<Field>> ret_types = {f0, f1};
  ///////////////////// Calculation //////////////////
  std::shared_ptr<CodeGenerator> sort_expr;
  arrow::compute::ExecContext ctx;
  ASSERT_NOT_OK(CreateCodeGenerator(ctx.memory_pool(), sch, {sortArrays_expr}, ret_types,
                                    &sort_expr, true));
  std::shared_ptr<arrow::RecordBatch> input_batch;
  std::vector<std::shared_ptr<arrow::RecordBatch>> input_batch_list;
  std::vector<std::shared_ptr<arrow::RecordBatch>> dummy_result_batches;
  std::shared_ptr<ResultIteratorBase> sort_result_iterator_base;
  std::vector<std::string> input_data_string = {
      R"(["132311.7856", "1311.7556", null, "311.2656", null, "811.3656", "532311.7986"])",
      R"(["132361.65356", "1211.12256", "3311.45256", "3191.96156", "211.16536", "341.36526", "5311.56736"])"};
  MakeInputBatch(input_data_string, sch, &input_batch);
  input_batch_list.push_back(input_batch);
  std::vector<std::string> input_data_string_2 = {
      R"(["832312.2656", "5511.7856", "324311.8956", "11.1666", "121.5657", "861.6656", "6311.1236"])",
      R"(["6761.19356", null, "50311.53256", "2591.26156", "451.16536", "2341.66526", "1211.78626"])"};
  MakeInputBatch(input_data_string_2, sch, &input_batch);
  input_batch_list.push_back(input_batch);
  std::vector<std::string> input_data_string_3 = {
      R"(["1573.5343", "1678.6556", null, "355.7626", null, "1911.8426", "453113.3556"])",
      R"(["132361.44356", "1211.44256", "3311.44256", "3191.46156", "211.46536", "341.46526", "5311.44446"])"};
  MakeInputBatch(input_data_string_3, sch, &input_batch);
  input_batch_list.push_back(input_batch);
  std::vector<std::string> input_data_string_4 = {
      R"(["5467.4224", "12345.6546", "435.2543", "643.0000", "643.0001", "42342.5642", "42663.2675"])",
      R"(["2545326.54763", "2456.63765", "56734.43767", "2364457.23545", "57648.45773", "356.04500", "36.46522"])"};
  MakeInputBatch(input_data_string_4, sch, &input_batch);
  input_batch_list.push_back(input_batch);
  std::vector<std::string> input_data_string_5 = {
      R"([null, "43556.3466", "245.2455", "6423.2562", "6342.0001", "75783.4757", "747487.2365"])",
      R"(["3452321.54346", "6351.53632", "36546.54356", "87584.53763", "45753.54676", "23.56743", "2.54732"])"};
  MakeInputBatch(input_data_string_5, sch, &input_batch);
  input_batch_list.push_back(input_batch);
  ////////////////////////////////// calculation
  //////////////////////////////////////
  std::shared_ptr<arrow::RecordBatch> expected_result;
  std::vector<std::string> expected_result_string = {
      R"(["11.1666", "121.5657", "245.2455", "311.2656", "355.7626", "435.2543", "643.0000", "643.0001", 
        "811.3656", "861.6656", "1311.7556", "1573.5343", "1678.6556", "1911.8426", "5467.4224", "5511.7856", 
        "6311.1236", "6342.0001", "6423.2562", "12345.6546", "42342.5642", "42663.2675", "43556.3466", 
        "75783.4757", "132311.7856", "324311.8956", "453113.3556", "532311.7986", "747487.2365", "832312.2656", 
         null, null, null, null, null])",
      R"(["2591.26156", "451.16536", "36546.54356", "3191.96156", "3191.46156", "56734.43767", "2364457.23545", 
        "57648.45773", "341.36526", "2341.66526", "1211.12256", "132361.44356", "1211.44256", "341.46526", 
        "2545326.54763", null, "1211.78626", "45753.54676", "87584.53763", "2456.63765", "356.04500", "36.46522", 
        "6351.53632", "23.56743", "132361.65356", "50311.53256", "5311.44446", "5311.56736", "2.54732", "6761.19356", 
         "3311.45256", "211.16536", "3311.44256", "211.46536", "3452321.54346"])"};
  MakeInputBatch(expected_result_string, sch, &expected_result);
  for (auto batch : input_batch_list) {
    ASSERT_NOT_OK(sort_expr->evaluate(batch, &dummy_result_batches));
  }
  ASSERT_NOT_OK(sort_expr->finish(&sort_result_iterator_base));
  auto sort_result_iterator =
      std::dynamic_pointer_cast<ResultIterator<arrow::RecordBatch>>(
          sort_result_iterator_base);
  std::shared_ptr<arrow::RecordBatch> dummy_result_batch;
  std::shared_ptr<arrow::RecordBatch> result_batch;
  if (sort_result_iterator->HasNext()) {
    ASSERT_NOT_OK(sort_result_iterator->Next(&result_batch));
    ASSERT_NOT_OK(Equals(*expected_result.get(), *result_batch.get()));
  }
}

TEST(TestArrowComputeSort, SortTestMulKeyDecimalCodegen) {
  ////////////////////// prepare expr_vector ///////////////////////
  auto f0 = field("f0", decimal128(10, 4));
  auto f1 = field("f1", decimal128(16, 5));
  auto f2 = field("f2", decimal128(12, 3));
  auto f3 = field("f3", decimal128(14, 2));
  auto arg_0 = TreeExprBuilder::MakeField(f0);
  auto arg_1 = TreeExprBuilder::MakeField(f1);
  auto arg_2 = TreeExprBuilder::MakeField(f2);
  auto arg_3 = TreeExprBuilder::MakeField(f3);
  auto true_literal = TreeExprBuilder::MakeLiteral(true);
  auto false_literal = TreeExprBuilder::MakeLiteral(false);

  auto f_res = field("res", uint32());

  auto n_key_func =
      TreeExprBuilder::MakeFunction("key_function", {arg_0, arg_1}, uint32());
  auto n_key_field = TreeExprBuilder::MakeFunction("key_field", {arg_0, arg_1}, uint32());
  auto n_dir = TreeExprBuilder::MakeFunction("sort_directions",
                                             {true_literal, false_literal}, uint32());
  auto n_nulls_order = TreeExprBuilder::MakeFunction(
      "sort_nulls_order", {false_literal, true_literal}, uint32());
  auto NaN_check = TreeExprBuilder::MakeFunction("NaN_check", {false_literal}, uint32());
  auto do_codegen = TreeExprBuilder::MakeFunction("codegen", {true_literal}, uint32());
  auto n_sort_to_indices = TreeExprBuilder::MakeFunction(
      "sortArraysToIndices",
      {n_key_func, n_key_field, n_dir, n_nulls_order, NaN_check, do_codegen}, uint32());
  auto n_sort =
      TreeExprBuilder::MakeFunction("standalone", {n_sort_to_indices}, uint32());
  auto sortArrays_expr = TreeExprBuilder::MakeExpression(n_sort, f_res);

  auto sch = arrow::schema({f0, f1, f2, f3});
  std::vector<std::shared_ptr<Field>> ret_types = {f0, f1, f2, f3};
  ///////////////////// Calculation //////////////////
  std::shared_ptr<CodeGenerator> sort_expr;
  arrow::compute::ExecContext ctx;
  ASSERT_NOT_OK(CreateCodeGenerator(ctx.memory_pool(), sch, {sortArrays_expr}, ret_types,
                                    &sort_expr, true));
  std::shared_ptr<arrow::RecordBatch> input_batch;
  std::vector<std::shared_ptr<arrow::RecordBatch>> input_batch_list;
  std::vector<std::shared_ptr<arrow::RecordBatch>> dummy_result_batches;
  std::shared_ptr<ResultIteratorBase> sort_result_iterator_base;
  std::vector<std::string> input_data_string = {
      R"(["132311.7856", "861.6656", null, "311.2656", null, "811.3656", "532311.7986"])",
      R"(["132361.65356", "1211.12256", "3311.45256", "3191.96156", "211.16536", "341.36526", "5311.56736"])",
      R"(["143451.436", "1415.345", "1345.636", "42651.345", "212351.162", "3241.421", "2351.235"])",
      R"(["1244213.66", "23545.52", "5251.56", "2351.96", "3631.76", "52.52", "3456.23"])"};
  MakeInputBatch(input_data_string, sch, &input_batch);
  input_batch_list.push_back(input_batch);
  std::vector<std::string> input_data_string_2 = {
      R"(["832312.2656", "5511.7856", "324311.8956", "11.1666", "121.5657", "861.6656", "6311.1236"])",
      R"(["6761.19356", null, "50311.53256", "2591.26156", "451.16536", "2341.66526", "1211.78626"])",
      R"(["67261.156", null, "32542.352", "3251.226", "124.252", "5647.290", "3252.679"])",
      R"(["26.11", null, "325.98", "51.86", "451.56", "53.52", "151.56"])"};
  MakeInputBatch(input_data_string_2, sch, &input_batch);
  input_batch_list.push_back(input_batch);
  std::vector<std::string> input_data_string_3 = {
      R"(["861.6656", "861.6656", null, "355.7626", null, "1911.8426", "453113.3556"])",
      R"(["132361.44356", null, null, "3191.46156", "211.46536", "341.46526", "5311.44446"])",
      R"(["34521.562", "42421.522", "4622.561", "3466.145", "22251.432", "2652.543", "52662.424"])",
      R"(["535.23", "4241.34", "452.60", "542.66", "241.66", "421.96", "41.26"])"};
  MakeInputBatch(input_data_string_3, sch, &input_batch);
  input_batch_list.push_back(input_batch);
  std::vector<std::string> input_data_string_4 = {
      R"(["5467.4224", null, "435.2543", "643.0000", "643.0001", "42342.5642", "42663.2675"])",
      R"(["2545326.54763", null, "56734.43767", "2364457.23545", "57648.45773", "356.04500", "36.46522"])",
      R"(["4352.432", "241.321", "46536.432", "6875.452", "6432.412", "141.664", "41.465"])",
      R"(["42521.52", "21453.63", "6342.41", "63213.46", "63451.86", "2521.76", "2441.23"])"};
  MakeInputBatch(input_data_string_4, sch, &input_batch);
  input_batch_list.push_back(input_batch);
  std::vector<std::string> input_data_string_5 = {
      R"([null, "43556.3466", "245.2455", "6423.2562", "6342.0001", "75783.4757", "747487.2365"])",
      R"(["3452321.54346", "6351.53632", "36546.54356", "87584.53763", "45753.54676", "23.56743", "2.54732"])",
      R"(["4531.563", "642.674", "3526.756", "6436.234", "634.675", "532.875", "632.865"])",
      R"(["653.86", "524.98", "632.97", "865.98", "867.96", "7554.43", "24.80"])"};
  MakeInputBatch(input_data_string_5, sch, &input_batch);
  input_batch_list.push_back(input_batch);
  ////////////////////////////////// calculation
  //////////////////////////////////////
  std::shared_ptr<arrow::RecordBatch> expected_result;
  std::vector<std::string> expected_result_string = {
      R"(["11.1666", "121.5657", "245.2455", "311.2656", "355.7626", "435.2543", "643.0000", "643.0001", 
        "811.3656", "861.6656", "861.6656", "861.6656", "861.6656", "1911.8426", "5467.4224", 
        "5511.7856", "6311.1236", "6342.0001", "6423.2562", "42342.5642", "42663.2675", "43556.3466", 
        "75783.4757", "132311.7856", "324311.8956", "453113.3556", "532311.7986", "747487.2365", "832312.2656", 
         null, null, null, null, null, null])",
      R"(["2591.26156", "451.16536", "36546.54356", "3191.96156", "3191.46156", "56734.43767", "2364457.23545", "57648.45773", 
        "341.36526", null, "132361.44356", "2341.66526", "1211.12256", "341.46526", "2545326.54763", 
         null, "1211.78626", "45753.54676", "87584.53763", "356.04500", "36.46522", "6351.53632", 
         "23.56743", "132361.65356", "50311.53256", "5311.44446", "5311.56736", "2.54732", "6761.19356",
         null, null, "3452321.54346", "3311.45256", "211.46536", "211.16536"])",
      R"(["3251.226", "124.252", "3526.756", "42651.345", "3466.145", "46536.432", "6875.452", "6432.412", 
        "3241.421", "42421.522", "34521.562", "5647.290", "1415.345", "2652.543", "4352.432", 
        null, "3252.679", "634.675", "6436.234", "141.664", "41.465", "642.674", 
        "532.875", "143451.436", "32542.352", "52662.424", "2351.235", "632.865", "67261.156", 
        "241.321", "4622.561", "4531.563", "1345.636", "22251.432", "212351.162"])",
      R"(["51.86", "451.56", "632.97", "2351.96", "542.66", "6342.41", "63213.46", "63451.86", 
        "52.52", "4241.34", "535.23", "53.52", "23545.52", "421.96", "42521.52", 
        null, "151.56", "867.96", "865.98", "2521.76", "2441.23", "524.98", 
        "7554.43", "1244213.66", "325.98", "41.26", "3456.23", "24.80", "26.11", 
         "21453.63", "452.60", "653.86", "5251.56", "241.66", "3631.76"])"};
  MakeInputBatch(expected_result_string, sch, &expected_result);
  for (auto batch : input_batch_list) {
    ASSERT_NOT_OK(sort_expr->evaluate(batch, &dummy_result_batches));
  }
  ASSERT_NOT_OK(sort_expr->finish(&sort_result_iterator_base));
  auto sort_result_iterator =
      std::dynamic_pointer_cast<ResultIterator<arrow::RecordBatch>>(
          sort_result_iterator_base);
  std::shared_ptr<arrow::RecordBatch> dummy_result_batch;
  std::shared_ptr<arrow::RecordBatch> result_batch;
  if (sort_result_iterator->HasNext()) {
    ASSERT_NOT_OK(sort_result_iterator->Next(&result_batch));
    ASSERT_NOT_OK(Equals(*expected_result.get(), *result_batch.get()));
  }
}

TEST(TestArrowComputeSort, SortTestMulKeyDecimalWithoutCodegen) {
  ////////////////////// prepare expr_vector ///////////////////////
  auto f0 = field("f0", decimal128(10, 4));
  auto f1 = field("f1", decimal128(16, 5));
  auto f2 = field("f2", decimal128(12, 3));
  auto f3 = field("f3", decimal128(14, 2));
  auto arg_0 = TreeExprBuilder::MakeField(f0);
  auto arg_1 = TreeExprBuilder::MakeField(f1);
  auto arg_2 = TreeExprBuilder::MakeField(f2);
  auto arg_3 = TreeExprBuilder::MakeField(f3);
  auto true_literal = TreeExprBuilder::MakeLiteral(true);
  auto false_literal = TreeExprBuilder::MakeLiteral(false);

  auto f_res = field("res", uint32());

  auto n_key_func =
      TreeExprBuilder::MakeFunction("key_function", {arg_0, arg_1}, uint32());
  auto n_key_field = TreeExprBuilder::MakeFunction("key_field", {arg_0, arg_1}, uint32());
  auto n_dir = TreeExprBuilder::MakeFunction("sort_directions",
                                             {true_literal, false_literal}, uint32());
  auto n_nulls_order = TreeExprBuilder::MakeFunction(
      "sort_nulls_order", {false_literal, true_literal}, uint32());
  auto NaN_check = TreeExprBuilder::MakeFunction("NaN_check", {false_literal}, uint32());
  auto do_codegen = TreeExprBuilder::MakeFunction("codegen", {false_literal}, uint32());
  auto n_sort_to_indices = TreeExprBuilder::MakeFunction(
      "sortArraysToIndices",
      {n_key_func, n_key_field, n_dir, n_nulls_order, NaN_check, do_codegen}, uint32());
  auto n_sort =
      TreeExprBuilder::MakeFunction("standalone", {n_sort_to_indices}, uint32());
  auto sortArrays_expr = TreeExprBuilder::MakeExpression(n_sort, f_res);

  auto sch = arrow::schema({f0, f1, f2, f3});
  std::vector<std::shared_ptr<Field>> ret_types = {f0, f1, f2, f3};
  ///////////////////// Calculation //////////////////
  std::shared_ptr<CodeGenerator> sort_expr;
  arrow::compute::ExecContext ctx;
  ASSERT_NOT_OK(CreateCodeGenerator(ctx.memory_pool(), sch, {sortArrays_expr}, ret_types,
                                    &sort_expr, true));
  std::shared_ptr<arrow::RecordBatch> input_batch;
  std::vector<std::shared_ptr<arrow::RecordBatch>> input_batch_list;
  std::vector<std::shared_ptr<arrow::RecordBatch>> dummy_result_batches;
  std::shared_ptr<ResultIteratorBase> sort_result_iterator_base;
  std::vector<std::string> input_data_string = {
      R"(["132311.7856", "861.6656", null, "311.2656", null, "811.3656", "532311.7986"])",
      R"(["132361.65356", "1211.12256", "3311.45256", "3191.96156", "211.16536", "341.36526", "5311.56736"])",
      R"(["143451.436", "1415.345", "1345.636", "42651.345", "212351.162", "3241.421", "2351.235"])",
      R"(["1244213.66", "23545.52", "5251.56", "2351.96", "3631.76", "52.52", "3456.23"])"};
  MakeInputBatch(input_data_string, sch, &input_batch);
  input_batch_list.push_back(input_batch);
  std::vector<std::string> input_data_string_2 = {
      R"(["832312.2656", "5511.7856", "324311.8956", "11.1666", "121.5657", "861.6656", "6311.1236"])",
      R"(["6761.19356", null, "50311.53256", "2591.26156", "451.16536", "2341.66526", "1211.78626"])",
      R"(["67261.156", null, "32542.352", "3251.226", "124.252", "5647.290", "3252.679"])",
      R"(["26.11", null, "325.98", "51.86", "451.56", "53.52", "151.56"])"};
  MakeInputBatch(input_data_string_2, sch, &input_batch);
  input_batch_list.push_back(input_batch);
  std::vector<std::string> input_data_string_3 = {
      R"(["861.6656", "861.6656", null, "355.7626", null, "1911.8426", "453113.3556"])",
      R"(["132361.44356", null, null, "3191.46156", "211.46536", "341.46526", "5311.44446"])",
      R"(["34521.562", "42421.522", "4622.561", "3466.145", "22251.432", "2652.543", "52662.424"])",
      R"(["535.23", "4241.34", "452.60", "542.66", "241.66", "421.96", "41.26"])"};
  MakeInputBatch(input_data_string_3, sch, &input_batch);
  input_batch_list.push_back(input_batch);
  std::vector<std::string> input_data_string_4 = {
      R"(["5467.4224", null, "435.2543", "643.0000", "643.0001", "42342.5642", "42663.2675"])",
      R"(["2545326.54763", null, "56734.43767", "2364457.23545", "57648.45773", "356.04500", "36.46522"])",
      R"(["4352.432", "241.321", "46536.432", "6875.452", "6432.412", "141.664", "41.465"])",
      R"(["42521.52", "21453.63", "6342.41", "63213.46", "63451.86", "2521.76", "2441.23"])"};
  MakeInputBatch(input_data_string_4, sch, &input_batch);
  input_batch_list.push_back(input_batch);
  std::vector<std::string> input_data_string_5 = {
      R"([null, "43556.3466", "245.2455", "6423.2562", "6342.0001", "75783.4757", "747487.2365"])",
      R"(["3452321.54346", "6351.53632", "36546.54356", "87584.53763", "45753.54676", "23.56743", "2.54732"])",
      R"(["4531.563", "642.674", "3526.756", "6436.234", "634.675", "532.875", "632.865"])",
      R"(["653.86", "524.98", "632.97", "865.98", "867.96", "7554.43", "24.80"])"};
  MakeInputBatch(input_data_string_5, sch, &input_batch);
  input_batch_list.push_back(input_batch);
  ////////////////////////////////// calculation
  //////////////////////////////////////
  std::shared_ptr<arrow::RecordBatch> expected_result;
  std::vector<std::string> expected_result_string = {
      R"(["11.1666", "121.5657", "245.2455", "311.2656", "355.7626", "435.2543", "643.0000", "643.0001", 
        "811.3656", "861.6656", "861.6656", "861.6656", "861.6656", "1911.8426", "5467.4224", 
        "5511.7856", "6311.1236", "6342.0001", "6423.2562", "42342.5642", "42663.2675", "43556.3466", 
        "75783.4757", "132311.7856", "324311.8956", "453113.3556", "532311.7986", "747487.2365", "832312.2656", 
         null, null, null, null, null, null])",
      R"(["2591.26156", "451.16536", "36546.54356", "3191.96156", "3191.46156", "56734.43767", "2364457.23545", "57648.45773", 
        "341.36526", null, "132361.44356", "2341.66526", "1211.12256", "341.46526", "2545326.54763", 
         null, "1211.78626", "45753.54676", "87584.53763", "356.04500", "36.46522", "6351.53632", 
         "23.56743", "132361.65356", "50311.53256", "5311.44446", "5311.56736", "2.54732", "6761.19356",
         null, null, "3452321.54346", "3311.45256", "211.46536", "211.16536"])",
      R"(["3251.226", "124.252", "3526.756", "42651.345", "3466.145", "46536.432", "6875.452", "6432.412", 
        "3241.421", "42421.522", "34521.562", "5647.290", "1415.345", "2652.543", "4352.432", 
        null, "3252.679", "634.675", "6436.234", "141.664", "41.465", "642.674", 
        "532.875", "143451.436", "32542.352", "52662.424", "2351.235", "632.865", "67261.156", 
        "4622.561", "241.321", "4531.563", "1345.636", "22251.432", "212351.162"])",
      R"(["51.86", "451.56", "632.97", "2351.96", "542.66", "6342.41", "63213.46", "63451.86", 
        "52.52", "4241.34", "535.23", "53.52", "23545.52", "421.96", "42521.52", 
        null, "151.56", "867.96", "865.98", "2521.76", "2441.23", "524.98", 
        "7554.43", "1244213.66", "325.98", "41.26", "3456.23", "24.80", "26.11", 
        "452.60", "21453.63", "653.86", "5251.56", "241.66", "3631.76"])"};
  MakeInputBatch(expected_result_string, sch, &expected_result);
  for (auto batch : input_batch_list) {
    ASSERT_NOT_OK(sort_expr->evaluate(batch, &dummy_result_batches));
  }
  ASSERT_NOT_OK(sort_expr->finish(&sort_result_iterator_base));
  auto sort_result_iterator =
      std::dynamic_pointer_cast<ResultIterator<arrow::RecordBatch>>(
          sort_result_iterator_base);
  std::shared_ptr<arrow::RecordBatch> dummy_result_batch;
  std::shared_ptr<arrow::RecordBatch> result_batch;
  if (sort_result_iterator->HasNext()) {
    ASSERT_NOT_OK(sort_result_iterator->Next(&result_batch));
    ASSERT_NOT_OK(Equals(*expected_result.get(), *result_batch.get()));
  }
}

TEST(TestArrowComputeSort, SortTestInplaceDecimal) {
  ////////////////////// prepare expr_vector ///////////////////////
  auto f0 = field("f0", decimal128(10, 4));
  auto arg_0 = TreeExprBuilder::MakeField(f0);
  auto true_literal = TreeExprBuilder::MakeLiteral(true);
  auto false_literal = TreeExprBuilder::MakeLiteral(false);

  auto f_res = field("res", uint32());

  auto n_key_func = TreeExprBuilder::MakeFunction("key_function", {arg_0}, uint32());
  auto n_key_field = TreeExprBuilder::MakeFunction("key_field", {arg_0}, uint32());
  auto n_dir = TreeExprBuilder::MakeFunction("sort_directions", {true_literal}, uint32());
  auto n_nulls_order =
      TreeExprBuilder::MakeFunction("sort_nulls_order", {false_literal}, uint32());
  auto NaN_check = TreeExprBuilder::MakeFunction("NaN_check", {false_literal}, uint32());
  auto do_codegen = TreeExprBuilder::MakeFunction("codegen", {false_literal}, uint32());
  auto n_sort_to_indices = TreeExprBuilder::MakeFunction(
      "sortArraysToIndices",
      {n_key_func, n_key_field, n_dir, n_nulls_order, NaN_check, do_codegen}, uint32());
  auto n_sort =
      TreeExprBuilder::MakeFunction("standalone", {n_sort_to_indices}, uint32());
  auto sortArrays_expr = TreeExprBuilder::MakeExpression(n_sort, f_res);

  auto sch = arrow::schema({f0});
  std::vector<std::shared_ptr<Field>> ret_types = {f0};
  ///////////////////// Calculation //////////////////
  std::shared_ptr<CodeGenerator> sort_expr;
  arrow::compute::ExecContext ctx;
  ASSERT_NOT_OK(CreateCodeGenerator(ctx.memory_pool(), sch, {sortArrays_expr}, ret_types,
                                    &sort_expr, true));
  std::shared_ptr<arrow::RecordBatch> input_batch;
  std::vector<std::shared_ptr<arrow::RecordBatch>> input_batch_list;
  std::vector<std::shared_ptr<arrow::RecordBatch>> dummy_result_batches;
  std::shared_ptr<ResultIteratorBase> sort_result_iterator_base;
  std::vector<std::string> input_data_string = {
      R"(["132311.7856", "1311.7556", null, "311.2656", null, "811.3656", "532311.7986"])"};
  MakeInputBatch(input_data_string, sch, &input_batch);
  input_batch_list.push_back(input_batch);
  std::vector<std::string> input_data_string_2 = {
      R"(["832312.2656", "5511.7856", "324311.8956", "11.1666", "121.5657", "861.6656", "6311.1236"])"};
  MakeInputBatch(input_data_string_2, sch, &input_batch);
  input_batch_list.push_back(input_batch);
  std::vector<std::string> input_data_string_3 = {
      R"(["1573.5343", "1678.6556", null, "355.7626", null, "1911.8426", "453113.3556"])"};
  MakeInputBatch(input_data_string_3, sch, &input_batch);
  input_batch_list.push_back(input_batch);
  std::vector<std::string> input_data_string_4 = {
      R"(["5467.4224", "12345.6546", "435.2543", "643.0000", "643.0001", "42342.5642", "42663.2675"])"};
  MakeInputBatch(input_data_string_4, sch, &input_batch);
  input_batch_list.push_back(input_batch);
  std::vector<std::string> input_data_string_5 = {
      R"([null, "43556.3466", "245.2455", "6423.2562", "6342.0001", "75783.4757", "747487.2365"])"};
  MakeInputBatch(input_data_string_5, sch, &input_batch);
  input_batch_list.push_back(input_batch);
  ////////////////////////////////// calculation
  //////////////////////////////////////
  std::shared_ptr<arrow::RecordBatch> expected_result;
  std::vector<std::string> expected_result_string = {
      R"(["11.1666", "121.5657", "245.2455", "311.2656", "355.7626", "435.2543", "643.0000", "643.0001", 
        "811.3656", "861.6656", "1311.7556", "1573.5343", "1678.6556", "1911.8426", "5467.4224", "5511.7856", 
        "6311.1236", "6342.0001", "6423.2562", "12345.6546", "42342.5642", "42663.2675", "43556.3466", 
        "75783.4757", "132311.7856", "324311.8956", "453113.3556", "532311.7986", "747487.2365", "832312.2656", 
         null, null, null, null, null])"};
  MakeInputBatch(expected_result_string, sch, &expected_result);
  for (auto batch : input_batch_list) {
    ASSERT_NOT_OK(sort_expr->evaluate(batch, &dummy_result_batches));
  }
  ASSERT_NOT_OK(sort_expr->finish(&sort_result_iterator_base));
  auto sort_result_iterator =
      std::dynamic_pointer_cast<ResultIterator<arrow::RecordBatch>>(
          sort_result_iterator_base);
  std::shared_ptr<arrow::RecordBatch> dummy_result_batch;
  std::shared_ptr<arrow::RecordBatch> result_batch;
  if (sort_result_iterator->HasNext()) {
    ASSERT_NOT_OK(sort_result_iterator->Next(&result_batch));
    ASSERT_NOT_OK(Equals(*expected_result.get(), *result_batch.get()));
  }
}

}  // namespace codegen
}  // namespace sparkcolumnarplugin
