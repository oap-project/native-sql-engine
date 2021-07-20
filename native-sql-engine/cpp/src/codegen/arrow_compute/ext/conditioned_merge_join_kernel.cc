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
#include <arrow/compute/api.h>
#include <arrow/pretty_print.h>
#include <arrow/record_batch.h>
#include <arrow/status.h>
#include <arrow/type.h>
#include <arrow/type_traits.h>
#include <arrow/util/bit_util.h>
#include <gandiva/node.h>
#include <gandiva/projector.h>

#include <chrono>
#include <cstring>
#include <fstream>
#include <functional>
#include <iostream>
#include <unordered_map>

#include "codegen/arrow_compute/ext/codegen_common.h"
#include "codegen/arrow_compute/ext/expression_codegen_visitor.h"
#include "codegen/arrow_compute/ext/kernels_ext.h"
#include "codegen/arrow_compute/ext/typed_node_visitor.h"
#include "utils/macros.h"

namespace sparkcolumnarplugin {
namespace codegen {
namespace arrowcompute {
namespace extra {

using ArrayList = std::vector<std::shared_ptr<arrow::Array>>;

///////////////  ConditionedProbe  ////////////////
class ConditionedMergeJoinKernel::Impl {
 public:
  Impl(arrow::compute::ExecContext* ctx, const gandiva::NodeVector& left_key_node_list,
       const gandiva::NodeVector& right_key_node_list,
       const gandiva::NodeVector& left_schema_node_list,
       const gandiva::NodeVector& right_schema_node_list,
       const gandiva::NodePtr& condition, int join_type,
       const gandiva::NodeVector& result_node_list, std::vector<int> hash_relation_idx)
      : ctx_(ctx),
        join_type_(join_type),
        condition_(condition),
        relation_id_(hash_relation_idx) {
    for (auto node : left_schema_node_list) {
      left_field_list_.push_back(
          std::dynamic_pointer_cast<gandiva::FieldNode>(node)->field());
    }
    for (auto node : right_schema_node_list) {
      right_field_list_.push_back(
          std::dynamic_pointer_cast<gandiva::FieldNode>(node)->field());
    }
    for (auto node : result_node_list) {
      result_schema_.push_back(
          std::dynamic_pointer_cast<gandiva::FieldNode>(node)->field());
    }

    left_key_project_codegen_ = GetGandivaKernel(left_key_node_list);

    right_key_project_codegen_ = GetGandivaKernel(right_key_node_list);

    /////////// map result_schema to input schema /////////////
    THROW_NOT_OK(
        GetIndexList(result_schema_, left_field_list_, &left_shuffle_index_list_));
    THROW_NOT_OK(
        GetIndexList(result_schema_, right_field_list_, &right_shuffle_index_list_));
    if (join_type != 4) {
      THROW_NOT_OK(GetIndexList(result_schema_, left_field_list_, right_field_list_,
                                false, &exist_index_, &result_schema_index_list_));
    } else {
      THROW_NOT_OK(GetIndexList(result_schema_, left_field_list_, right_field_list_, true,
                                &exist_index_, &result_schema_index_list_));
    }

    pool_ = nullptr;
  }

  arrow::Status MakeResultIterator(
      std::shared_ptr<arrow::Schema> schema,
      std::shared_ptr<ResultIterator<arrow::RecordBatch>>* out) {
    return arrow::Status::OK();
  }

  std::string GetSignature() { return ""; }

  arrow::Status DoCodeGen(
      int level,
      std::vector<std::pair<std::pair<std::string, std::string>, gandiva::DataTypePtr>>
          input,
      std::shared_ptr<CodeGenContext>* codegen_ctx_out, int* var_id) {
    auto codegen_ctx = std::make_shared<CodeGenContext>();
    bool use_relation_for_stream = input.empty();

    codegen_ctx->header_codes.push_back(
        R"(#include "codegen/arrow_compute/ext/array_item_index.h")");

    std::vector<std::string> prepare_list;
    bool cond_check = false;
    if (condition_) cond_check = true;
    // 1.0 prepare sort relation columns, and output
    std::stringstream sort_prepare_ss;
    std::stringstream sort_define_ss;
    std::vector<gandiva::FieldVector> field_list = {left_field_list_, right_field_list_};

    codegen_ctx->header_codes.push_back(R"(#include "codegen/common/sort_relation.h")");
    int idx = 0;
    for (auto relation_id : relation_id_) {
      auto relation_list_name = "sort_relation_" + std::to_string(relation_id) + "_";
      sort_prepare_ss << "auto typed_dependent_iter_list_" << relation_id
                      << " = "
                         "std::dynamic_pointer_cast<ResultIterator<SortRelation>>("
                         "dependent_iter_list["
                      << relation_id << "]);" << std::endl;
      sort_prepare_ss << "RETURN_NOT_OK(typed_dependent_iter_list_" << relation_id
                      << "->Next("
                      << "&" << relation_list_name << "));" << std::endl;
      sort_define_ss << "std::shared_ptr<SortRelation> " << relation_list_name << ";"
                     << std::endl;
      for (int i = 0; i < field_list[idx].size(); i++) {
        std::stringstream relation_col_name_ss;
        relation_col_name_ss << "sort_relation_" << relation_id << "_" << i;
        auto relation_col_name = relation_col_name_ss.str();
        auto relation_col_type = field_list[idx][i]->type();
        sort_define_ss << "std::shared_ptr<"
                       << GetTemplateString(relation_col_type, "TypedRelationColumn",
                                            "Type", "arrow::")
                       << "> " << relation_col_name << ";" << std::endl;
        sort_define_ss << "bool " << relation_col_name << "_has_null;" << std::endl;
        sort_prepare_ss << "RETURN_NOT_OK(" << relation_list_name << "->GetColumn(" << i
                        << ", &" << relation_col_name << "));" << std::endl;
        sort_prepare_ss << relation_col_name << "_has_null = " << relation_col_name
                        << "->HasNull();" << std::endl;
      }
      idx++;
    }
    codegen_ctx->relation_prepare_codes = sort_prepare_ss.str();
    codegen_ctx->definition_codes = sort_define_ss.str();

    ///// Prepare compare function /////
    // 1. do pre process for all merge join keys
    std::stringstream prepare_ss;

    std::vector<gandiva::ExpressionVector> project_codegen_list = {
        left_key_project_codegen_, right_key_project_codegen_};
    std::vector<std::vector<std::pair<std::string, std::string>>> project_output_list;
    project_output_list.resize(2);

    std::vector<std::string> input_list;
    for (int i = 0; i < 2; i++) {
      for (auto expr : project_codegen_list[i]) {
        std::shared_ptr<ExpressionCodegenVisitor> project_node_visitor;
        auto is_local = true;
        RETURN_NOT_OK(MakeExpressionCodegenVisitor(
            expr->root(), &input, field_list, relation_id_[0], var_id, is_local,
            &input_list, &project_node_visitor, true));
        prepare_ss << project_node_visitor->GetPrepare();
        auto key_name = project_node_visitor->GetResult();
        auto validity_name = project_node_visitor->GetPreCheck();

        project_output_list[i].push_back(std::make_pair(key_name, validity_name));
        for (auto header : project_node_visitor->GetHeaders()) {
          if (std::find(codegen_ctx->header_codes.begin(),
                        codegen_ctx->header_codes.end(),
                        header) == codegen_ctx->header_codes.end()) {
            codegen_ctx->header_codes.push_back(header);
          }
        }
        idx++;
      }
    }

    std::stringstream function_define_ss;
    std::vector<std::string> output_value_list;
    for (auto pair : project_output_list[0]) output_value_list.push_back(pair.first);
    auto left_paramater = GetParameterList(output_value_list, false);
    output_value_list.clear();
    for (auto pair : project_output_list[0]) output_value_list.push_back(pair.second);
    auto left_validity_paramater = GetParameterList(output_value_list, false, " && ");
    output_value_list.clear();
    for (auto pair : project_output_list[1]) output_value_list.push_back(pair.first);
    auto right_paramater = GetParameterList(output_value_list, false);
    output_value_list.clear();
    for (auto pair : project_output_list[1]) output_value_list.push_back(pair.second);
    auto right_validity_paramater = GetParameterList(output_value_list, false, " && ");

    function_define_ss << "inline int JoinCompare_" << relation_id_[0] << "() {"
                       << std::endl;
    function_define_ss << "if (!sort_relation_" << relation_id_[0]
                       << "_->CheckRangeBound(0)) return -1;" << std::endl;
    function_define_ss << "auto idx_0 = sort_relation_" << relation_id_[0]
                       << "_->GetItemIndexWithShift(0);" << std::endl;
    if (use_relation_for_stream) {
      function_define_ss << "if (!sort_relation_" << relation_id_[1]
                         << "_->CheckRangeBound(0)) return 1;" << std::endl;
      function_define_ss << "auto idx_1 = sort_relation_" << relation_id_[1]
                         << "_->GetItemIndexWithShift(0);" << std::endl;
    }
    function_define_ss << prepare_ss.str() << std::endl;
    function_define_ss << "if (!(" << left_validity_paramater << ")) return -1;"
                       << std::endl;
    function_define_ss << "if (!(" << right_validity_paramater << ")) return 1;"
                       << std::endl;
    auto left_tuple_name = left_paramater;
    auto right_tuple_name = right_paramater;
    if (project_output_list[0].size() > 1) {
      function_define_ss << "auto left_tuple = std::forward_as_tuple(" << left_paramater
                         << " );" << std::endl;
      left_tuple_name = "left_tuple";
    }
    if (project_output_list[1].size() > 1) {
      function_define_ss << "auto right_tuple = std::forward_as_tuple(" << right_paramater
                         << " );" << std::endl;
      right_tuple_name = "right_tuple";
    }
    function_define_ss << "return " << left_tuple_name << " == " << right_tuple_name
                       << " ? 0 : (" << left_tuple_name << " < " << right_tuple_name
                       << " ? -1 : 1);" << std::endl;
    function_define_ss << "}" << std::endl;
    auto compare_function = function_define_ss.str();
    codegen_ctx->function_list.push_back(compare_function);
    function_define_ss.str("");

    /////   inside loop  //////
    // 2. do merge join
    RETURN_NOT_OK(GetProcessJoin(join_type_, cond_check, input, &codegen_ctx));
    // 3. do continue if not exists
    if (cond_check) {
      std::shared_ptr<ExpressionCodegenVisitor> condition_node_visitor;
      auto is_local = true;
      RETURN_NOT_OK(MakeExpressionCodegenVisitor(
          condition_, &input, field_list, relation_id_[0], var_id, is_local,
          &prepare_list, &condition_node_visitor, true));
      auto function_name = "ConditionCheck_" + std::to_string(relation_id_[0]);
      if (use_relation_for_stream) {
        function_define_ss << "inline bool " << function_name
                           << "(ArrayItemIndexS idx_0, ArrayItemIndexS idx_1) {"
                           << std::endl;
      } else {
        function_define_ss << "inline bool " << function_name
                           << "(ArrayItemIndexS idx_0) {" << std::endl;
      }
      function_define_ss << condition_node_visitor->GetPrepare() << std::endl;
      function_define_ss << "return " << condition_node_visitor->GetResult() << ";"
                         << std::endl;
      function_define_ss << "}" << std::endl;
      codegen_ctx->function_list.push_back(function_define_ss.str());
      for (auto header : condition_node_visitor->GetHeaders()) {
        if (std::find(codegen_ctx->header_codes.begin(), codegen_ctx->header_codes.end(),
                      header) == codegen_ctx->header_codes.end()) {
          codegen_ctx->header_codes.push_back(header);
        }
      }
    }
    // set join output list for next kernel.
    ///////////////////////////////
    *codegen_ctx_out = codegen_ctx;

    return arrow::Status::OK();
  }

 private:
  arrow::compute::ExecContext* ctx_;
  arrow::MemoryPool* pool_;
  std::string signature_;
  int join_type_;

  gandiva::NodePtr condition_;

  gandiva::ExpressionVector left_key_project_codegen_;
  gandiva::ExpressionVector right_key_project_codegen_;
  gandiva::FieldVector key_hash_field_list_;

  bool left_pre_processed_key_ = true;
  bool right_pre_processed_key_ = true;
  gandiva::FieldVector left_field_list_;
  gandiva::FieldVector right_field_list_;
  gandiva::FieldVector result_schema_;
  std::vector<int> right_key_index_list_;
  std::vector<int> left_shuffle_index_list_;
  std::vector<int> right_shuffle_index_list_;
  std::vector<std::pair<int, int>> result_schema_index_list_;
  int exist_index_ = -1;
  std::vector<int> relation_id_;
  std::vector<arrow::ArrayVector> cached_;

  arrow::Status GetInnerJoin(bool cond_check, bool use_relation_for_stream,
                             bool cache_right, std::shared_ptr<CodeGenContext>* output) {
    std::stringstream shuffle_ss;
    std::stringstream codes_ss;
    std::stringstream finish_codes_ss;
    auto relation_id = std::to_string(relation_id_[0]);
    auto function_name = "JoinCompare_" + relation_id;
    auto condition_name = "ConditionCheck_" + relation_id;
    auto range_name = "range_" + relation_id;
    auto range_id = "range_" + relation_id + "_i";
    auto streamed_range_name = "streamed_range_" + relation_id;
    auto streamed_range_id = "streamed_range_" + relation_id + "_i";
    auto build_relation = "sort_relation_" + relation_id + "_";
    auto streamed_relation = "sort_relation_" + std::to_string(relation_id_[1]) + "_";
    auto left_index_name = "left_index_" + relation_id;
    auto right_index_name = "right_index_" + relation_id;

    ///// Get Matched row /////
    codes_ss << "int " << range_name << " = 0;" << std::endl;
    codes_ss << "auto " << function_name << "_res = " << function_name << "();"
             << std::endl;
    codes_ss << "while (" << function_name << "_res < 0) {" << std::endl;
    codes_ss << "if ((should_stop_ = !" << build_relation << "->NextNewKey())) break;"
             << std::endl;
    codes_ss << function_name << "_res = " << function_name << "();" << std::endl;
    codes_ss << "}" << std::endl;
    codes_ss << "if (should_stop_) break;" << std::endl;
    codes_ss << "if (" << function_name << "_res == 0) {" << std::endl;
    codes_ss << range_name << " = " << build_relation << "->GetSameKeyRange();"
             << std::endl;
    codes_ss << "} else { /*stream relation should step forward*/" << std::endl;
    if (use_relation_for_stream) {
      codes_ss << "if ((should_stop_ = !" << streamed_relation
               << "->NextNewKey())) break;" << std::endl;
    }
    codes_ss << "continue;" << std::endl;
    codes_ss << "}" << std::endl;
    ///////////////////////////

    std::stringstream right_for_loop_codes;
    if (use_relation_for_stream) {
      codes_ss << "auto " << streamed_range_name << " = " << streamed_relation
               << "->GetSameKeyRange();" << std::endl;
      right_for_loop_codes << "for (int " << streamed_range_id << " = 0; "
                           << streamed_range_id << " < " << streamed_range_name << "; "
                           << streamed_range_id << "++) {" << std::endl;
      right_for_loop_codes << right_index_name << " = " << streamed_relation
                           << "->GetItemIndexWithShift(" << streamed_range_id << ");"
                           << std::endl;
      std::stringstream prepare_ss;
      prepare_ss << "ArrayItemIndexS " << right_index_name << ";" << std::endl;
      (*output)->definition_codes += prepare_ss.str();
    }
    std::stringstream prepare_ss;
    prepare_ss << "ArrayItemIndexS " << left_index_name << ";" << std::endl;
    (*output)->definition_codes += prepare_ss.str();
    if (cache_right) {
      codes_ss << right_for_loop_codes.str();
      codes_ss << "auto is_smj_" << relation_id << " = false;" << std::endl;
    }
    codes_ss << "for (int " << range_id << " = 0; " << range_id << " < " << range_name
             << "; " << range_id << "++) {" << std::endl;
    codes_ss << left_index_name << " = " << build_relation << "->GetItemIndexWithShift("
             << range_id << ");" << std::endl;
    if (!cache_right) {
      codes_ss << "auto is_smj_" << relation_id << " = false;" << std::endl;
      codes_ss << right_for_loop_codes.str();
    }
    if (cond_check) {
      auto condition_name = "ConditionCheck_" + std::to_string(relation_id_[0]);
      if (use_relation_for_stream) {
        codes_ss << "if (!" << condition_name << "(" << left_index_name << ", "
                 << right_index_name << ")) {" << std::endl;
      } else {
        codes_ss << "if (!" << condition_name << "(" << left_index_name << ")) {"
                 << std::endl;
      }
      codes_ss << "  continue;" << std::endl;
      codes_ss << "}" << std::endl;
    }
    finish_codes_ss << "} // end of Inner Join" << std::endl;
    if (use_relation_for_stream) {
      finish_codes_ss << "} // end of Inner Join" << std::endl;
      finish_codes_ss << "should_stop_ = !" << streamed_relation << "->NextNewKey();"
                      << std::endl;
    }
    (*output)->process_codes += codes_ss.str();
    (*output)->finish_codes += finish_codes_ss.str();

    return arrow::Status::OK();
  }
  arrow::Status GetOuterJoin(bool cond_check, bool use_relation_for_stream,
                             bool cache_right, std::shared_ptr<CodeGenContext>* output) {
    std::stringstream shuffle_ss;
    std::stringstream codes_ss;
    std::stringstream finish_codes_ss;
    auto relation_id = std::to_string(relation_id_[0]);
    auto function_name = "JoinCompare_" + relation_id;
    auto condition_name = "ConditionCheck_" + relation_id;
    auto range_name = "range_" + relation_id;
    auto fill_null_name = "is_outer_null_" + relation_id;
    auto range_id = "range_" + relation_id + "_i";
    auto streamed_range_name = "streamed_range_" + relation_id;
    auto streamed_range_id = "streamed_range_" + relation_id + "_i";
    auto build_relation = "sort_relation_" + relation_id + "_";
    auto streamed_relation = "sort_relation_" + std::to_string(relation_id_[1]) + "_";
    auto left_index_name = "left_index_" + relation_id;
    auto right_index_name = "right_index_" + relation_id;
    auto outer_join_num_matches_name = "outer_join_num_matches_" + relation_id;

    ///// Get Matched row /////
    codes_ss << "int " << range_name << " = 0;" << std::endl;
    codes_ss << fill_null_name << " = false;" << std::endl;
    codes_ss << "auto " << function_name << "_res = " << function_name << "();"
             << std::endl;
    codes_ss << "while (" << function_name << "_res < 0) {" << std::endl;
    codes_ss << "if ((!" << build_relation << "->NextNewKey())) break;" << std::endl;
    codes_ss << function_name << "_res = " << function_name << "();" << std::endl;
    codes_ss << "}" << std::endl;
    codes_ss << "if (" << function_name << "_res == 0) {" << std::endl;
    codes_ss << range_name << " = " << build_relation << "->GetSameKeyRange();"
             << std::endl;
    codes_ss << "} else { /*put null for build side*/" << std::endl;
    codes_ss << range_name << " = 1;" << std::endl;
    codes_ss << fill_null_name << " = true;" << std::endl;
    codes_ss << "}" << std::endl;
    ///////////////////////////

    std::stringstream right_for_loop_codes;
    if (use_relation_for_stream) {
      codes_ss << "auto " << streamed_range_name << " = " << streamed_relation
               << "->GetSameKeyRange();" << std::endl;
      right_for_loop_codes << "for (int " << streamed_range_id << " = 0; "
                           << streamed_range_id << " < " << streamed_range_name << "; "
                           << streamed_range_id << "++) {" << std::endl;
      right_for_loop_codes << right_index_name << " = " << streamed_relation
                           << "->GetItemIndexWithShift(" << streamed_range_id << ");"
                           << std::endl;
      std::stringstream prepare_ss;
      prepare_ss << "ArrayItemIndexS " << right_index_name << ";" << std::endl;
      (*output)->definition_codes += prepare_ss.str();
    }
    std::stringstream prepare_ss;
    prepare_ss << "ArrayItemIndexS " << left_index_name << ";" << std::endl;
    prepare_ss << "bool " << fill_null_name << ";" << std::endl;
    (*output)->definition_codes += prepare_ss.str();
    if (cache_right) {
      codes_ss << right_for_loop_codes.str();
      codes_ss << "auto is_smj_" << relation_id << " = false;" << std::endl;
    }
    codes_ss << "int " << outer_join_num_matches_name << " = 0;" << std::endl;
    codes_ss << "for (int " << range_id << " = 0; " << range_id << " < " << range_name
             << "; " << range_id << "++) {" << std::endl;
    codes_ss << "if(" << function_name << "_res == 0"
             << ") {" << std::endl;
    codes_ss << left_index_name << " = " << build_relation << "->GetItemIndexWithShift("
             << range_id << ");" << std::endl;
    codes_ss << fill_null_name << " = false;" << std::endl;
    if (cond_check) {
      auto condition_name = "ConditionCheck_" + std::to_string(relation_id_[0]);
      if (use_relation_for_stream) {
        codes_ss << "if (!" << condition_name << "(" << left_index_name << ", "
                 << right_index_name << ")) {" << std::endl;
      } else {
        codes_ss << "if (!" << condition_name << "(" << left_index_name << ")) {"
                 << std::endl;
      }
      codes_ss << "if ((" << range_id << " + 1) == " << range_name << " && "
               << outer_join_num_matches_name << " == 0) {" << std::endl;
      codes_ss << fill_null_name << " = true; } else {" << std::endl;
      codes_ss << "continue;" << std::endl;
      codes_ss << "}}" << std::endl;
    }
    codes_ss << "} else {" << std::endl;
    codes_ss << fill_null_name << " =  true;}" << std::endl;
    codes_ss << outer_join_num_matches_name << " += 1;" << std::endl;
    if (!cache_right) {
      codes_ss << "auto is_smj_" << relation_id << " = false;" << std::endl;
      codes_ss << right_for_loop_codes.str();
    }
    finish_codes_ss << "} // end of Outer Join" << std::endl;
    if (use_relation_for_stream) {
      finish_codes_ss << "} // end of Outer Join" << std::endl;
      finish_codes_ss << "should_stop_ = !" << streamed_relation << "->NextNewKey();"
                      << std::endl;
    }
    (*output)->process_codes += codes_ss.str();
    (*output)->finish_codes += finish_codes_ss.str();

    return arrow::Status::OK();
  }
  arrow::Status GetAntiJoin(bool cond_check, bool use_relation_for_stream,
                            std::shared_ptr<CodeGenContext>* output) {
    std::stringstream shuffle_ss;
    std::stringstream codes_ss;
    std::stringstream finish_codes_ss;
    auto relation_id = std::to_string(relation_id_[0]);
    auto function_name = "JoinCompare_" + relation_id;
    auto condition_name = "ConditionCheck_" + relation_id;
    auto range_name = "range_" + relation_id;
    auto found_match_name = "found_" + relation_id;
    auto range_id = "range_" + relation_id + "_i";
    auto streamed_range_name = "streamed_range_" + relation_id;
    auto streamed_range_id = "streamed_range_" + relation_id + "_i";
    auto build_relation = "sort_relation_" + relation_id + "_";
    auto streamed_relation = "sort_relation_" + std::to_string(relation_id_[1]) + "_";
    auto left_index_name = "left_index_" + relation_id;
    auto right_index_name = "right_index_" + relation_id;

    ///// Get Matched row /////
    codes_ss << "int " << range_name << " = 1;" << std::endl;
    codes_ss << "bool " << found_match_name << " = false;" << std::endl;
    codes_ss << "auto " << function_name << "_res = " << function_name << "();"
             << std::endl;
    codes_ss << "while (" << function_name << "_res < 0) {" << std::endl;
    codes_ss << "if ((!" << build_relation << "->NextNewKey())) break;" << std::endl;
    codes_ss << function_name << "_res = " << function_name << "();" << std::endl;
    codes_ss << "}" << std::endl;
    codes_ss << "if (" << function_name << "_res == 0) {" << std::endl;
    if (cond_check) {
      codes_ss << found_match_name << " = true;" << std::endl;
      codes_ss << range_name << " = " << build_relation << "->GetSameKeyRange();"
               << std::endl;
    } else {
      if (use_relation_for_stream) {
        codes_ss << "if ((should_stop_ = !" << streamed_relation
                 << "->NextNewKey())) break;" << std::endl;
      }
      codes_ss << "continue;" << std::endl;
    }
    codes_ss << "}" << std::endl;
    ///////////////////////////

    if (use_relation_for_stream) {
      codes_ss << "auto " << streamed_range_name << " = " << streamed_relation
               << "->GetSameKeyRange();" << std::endl;
      codes_ss << "for (int " << streamed_range_id << " = 0; " << streamed_range_id
               << " < " << streamed_range_name << "; " << streamed_range_id << "++) {"
               << std::endl;
      codes_ss << right_index_name << " = " << streamed_relation
               << "->GetItemIndexWithShift(" << streamed_range_id << ");" << std::endl;
      std::stringstream prepare_ss;
      prepare_ss << "ArrayItemIndexS " << right_index_name << ";" << std::endl;
      (*output)->definition_codes += prepare_ss.str();
    }
    std::stringstream prepare_ss;
    prepare_ss << "ArrayItemIndexS " << left_index_name << ";" << std::endl;
    (*output)->definition_codes += prepare_ss.str();
    codes_ss << "for (int " << range_id << " = 0; " << range_id << " < 1;" << range_id
             << "++) {" << std::endl;
    if (cond_check) {
      auto condition_name = "ConditionCheck_" + std::to_string(relation_id_[0]);
      codes_ss << "if (" << found_match_name << ") {" << std::endl;
      codes_ss << found_match_name << " = false;" << std::endl;
      codes_ss << "for (int j = 0; j < " << range_name << "; j++) {" << std::endl;
      codes_ss << left_index_name << " = " << build_relation
               << "->GetItemIndexWithShift(j);" << std::endl;
      if (use_relation_for_stream) {
        codes_ss << "if (" << condition_name << "(" << left_index_name << ", "
                 << right_index_name << ")) {" << std::endl;
      } else {
        codes_ss << "if (" << condition_name << "(" << left_index_name << ")) {"
                 << std::endl;
      }
      codes_ss << found_match_name << " = true;" << std::endl;
      codes_ss << "break;" << std::endl;
      codes_ss << "}" << std::endl;
      codes_ss << "}" << std::endl;
      codes_ss << "if (" << found_match_name << ") {" << std::endl;
      codes_ss << "continue;" << std::endl;
      codes_ss << "}" << std::endl;
      codes_ss << "}" << std::endl;
    }
    finish_codes_ss << "} // end of anti Join" << std::endl;
    if (use_relation_for_stream) {
      finish_codes_ss << "} // end of anti Join" << std::endl;
      finish_codes_ss << "should_stop_ = !" << streamed_relation << "->NextNewKey();"
                      << std::endl;
    }
    (*output)->process_codes += codes_ss.str();
    (*output)->finish_codes += finish_codes_ss.str();

    return arrow::Status::OK();
  }
  arrow::Status GetSemiJoin(bool cond_check, bool use_relation_for_stream,
                            std::shared_ptr<CodeGenContext>* output) {
    std::stringstream shuffle_ss;
    std::stringstream codes_ss;
    std::stringstream finish_codes_ss;
    auto relation_id = std::to_string(relation_id_[0]);
    auto function_name = "JoinCompare_" + relation_id;
    auto condition_name = "ConditionCheck_" + relation_id;
    auto range_name = "range_" + relation_id;
    auto found_match_name = "found_" + relation_id;
    auto range_id = "range_" + relation_id + "_i";
    auto streamed_range_name = "streamed_range_" + relation_id;
    auto streamed_range_id = "streamed_range_" + relation_id + "_i";
    auto build_relation = "sort_relation_" + relation_id + "_";
    auto streamed_relation = "sort_relation_" + std::to_string(relation_id_[1]) + "_";
    auto left_index_name = "left_index_" + relation_id;
    auto right_index_name = "right_index_" + relation_id;

    ///// Get Matched row /////
    codes_ss << "int " << range_name << " = 1;" << std::endl;
    codes_ss << "bool " << found_match_name << " = true;" << std::endl;
    codes_ss << "auto " << function_name << "_res = " << function_name << "();"
             << std::endl;
    codes_ss << "while (" << function_name << "_res < 0) {" << std::endl;
    codes_ss << "if ((should_stop_ = !" << build_relation << "->NextNewKey())) break;"
             << std::endl;
    codes_ss << function_name << "_res = " << function_name << "();" << std::endl;
    codes_ss << "}" << std::endl;
    codes_ss << "if (should_stop_) break;" << std::endl;
    codes_ss << "if (" << function_name << "_res != 0) {" << std::endl;
    if (use_relation_for_stream) {
      codes_ss << "if ((should_stop_ = !" << streamed_relation
               << "->NextNewKey())) break;" << std::endl;
    }
    codes_ss << "continue;" << std::endl;
    codes_ss << "}" << std::endl;
    if (cond_check) {
      codes_ss << range_name << " = " << build_relation << "->GetSameKeyRange();"
               << std::endl;
    }
    ///////////////////////////

    if (use_relation_for_stream) {
      codes_ss << "auto " << streamed_range_name << " = " << streamed_relation
               << "->GetSameKeyRange();" << std::endl;
      codes_ss << "for (int " << streamed_range_id << " = 0; " << streamed_range_id
               << " < " << streamed_range_name << "; " << streamed_range_id << "++) {"
               << std::endl;
      codes_ss << right_index_name << " = " << streamed_relation
               << "->GetItemIndexWithShift(" << streamed_range_id << ");" << std::endl;
      std::stringstream prepare_ss;
      prepare_ss << "ArrayItemIndexS " << right_index_name << ";" << std::endl;
      (*output)->definition_codes += prepare_ss.str();
    }
    std::stringstream prepare_ss;
    prepare_ss << "ArrayItemIndexS " << left_index_name << ";" << std::endl;
    (*output)->definition_codes += prepare_ss.str();
    codes_ss << "for (int " << range_id << " = 0; " << range_id << " < 1;" << range_id
             << "++) {" << std::endl;
    if (cond_check) {
      auto condition_name = "ConditionCheck_" + std::to_string(relation_id_[0]);
      codes_ss << found_match_name << " = false;" << std::endl;
      codes_ss << "for (int j = 0; j < " << range_name << "; j++) {" << std::endl;
      codes_ss << left_index_name << " = " << build_relation
               << "->GetItemIndexWithShift(j);" << std::endl;
      if (use_relation_for_stream) {
        codes_ss << "if (" << condition_name << "(" << left_index_name << ", "
                 << right_index_name << ")) {" << std::endl;
      } else {
        codes_ss << "if (" << condition_name << "(" << left_index_name << ")) {"
                 << std::endl;
      }
      codes_ss << found_match_name << " = true;" << std::endl;
      codes_ss << "break;" << std::endl;
      codes_ss << "}" << std::endl;
      codes_ss << "}" << std::endl;
      codes_ss << "if (!" << found_match_name << ") {" << std::endl;
      codes_ss << "continue;" << std::endl;
      codes_ss << "}" << std::endl;
    }
    finish_codes_ss << "} // end of semi Join" << std::endl;
    if (use_relation_for_stream) {
      finish_codes_ss << "} // end of semi Join" << std::endl;
      finish_codes_ss << "should_stop_ = !" << streamed_relation << "->NextNewKey();"
                      << std::endl;
    }
    (*output)->process_codes += codes_ss.str();
    (*output)->finish_codes += finish_codes_ss.str();

    return arrow::Status::OK();
  }

  arrow::Status GetExistenceJoin(bool cond_check, bool use_relation_for_stream,
                                 std::shared_ptr<CodeGenContext>* output) {
    std::stringstream shuffle_ss;
    std::stringstream codes_ss;
    std::stringstream finish_codes_ss;
    auto relation_id = std::to_string(relation_id_[0]);
    auto function_name = "JoinCompare_" + relation_id;
    auto condition_name = "ConditionCheck_" + relation_id;
    auto range_name = "range_" + relation_id;
    auto found_match_name = "found_" + relation_id;
    auto range_id = "range_" + relation_id + "_i";
    auto streamed_range_name = "streamed_range_" + relation_id;
    auto streamed_range_id = "streamed_range_" + relation_id + "_i";
    auto build_relation = "sort_relation_" + relation_id + "_";
    auto streamed_relation = "sort_relation_" + std::to_string(relation_id_[1]) + "_";
    auto left_index_name = "left_index_" + relation_id;
    auto right_index_name = "right_index_" + relation_id;
    auto exist_name = "sort_relation_" + relation_id + "_existence_value";
    auto exist_validity = exist_name + "_validity";

    ///// Get Matched row /////
    codes_ss << "int " << range_name << " = 1;" << std::endl;
    codes_ss << "bool " << found_match_name << " = true;" << std::endl;
    codes_ss << "auto " << function_name << "_res = " << function_name << "();"
             << std::endl;
    codes_ss << "while (" << function_name << "_res < 0) {" << std::endl;
    codes_ss << "if ((!" << build_relation << "->NextNewKey())) break;" << std::endl;
    codes_ss << function_name << "_res = " << function_name << "();" << std::endl;
    codes_ss << "}" << std::endl;
    codes_ss << "if (" << function_name << "_res != 0) {" << std::endl;
    codes_ss << found_match_name << " = false;" << std::endl;
    codes_ss << "} else {" << std::endl;
    codes_ss << found_match_name << " = true;" << std::endl;
    codes_ss << "}" << std::endl;
    if (cond_check) {
      codes_ss << range_name << " = " << build_relation << "->GetSameKeyRange();"
               << std::endl;
    }
    ///////////////////////////
    if (use_relation_for_stream) {
      codes_ss << "auto " << streamed_range_name << " = " << streamed_relation
               << "->GetSameKeyRange();" << std::endl;
      codes_ss << "for (int " << streamed_range_id << " = 0; " << streamed_range_id
               << " < " << streamed_range_name << "; " << streamed_range_id << "++) {"
               << std::endl;
      codes_ss << right_index_name << " = " << streamed_relation
               << "->GetItemIndexWithShift(" << streamed_range_id << ");" << std::endl;
      std::stringstream prepare_ss;
      prepare_ss << "ArrayItemIndexS " << right_index_name << ";" << std::endl;
      (*output)->definition_codes += prepare_ss.str();
    }
    std::stringstream prepare_ss;
    prepare_ss << "ArrayItemIndexS " << left_index_name << ";" << std::endl;
    (*output)->definition_codes += prepare_ss.str();
    codes_ss << "for (int " << range_id << " = 0; " << range_id << " < 1;" << range_id
             << "++) {" << std::endl;
    if (cond_check) {
      auto condition_name = "ConditionCheck_" + std::to_string(relation_id_[0]);
      codes_ss << "if (" << found_match_name << ") {" << std::endl;
      codes_ss << found_match_name << " = false;" << std::endl;
      codes_ss << "for (int j = 0; j < " << range_name << "; j++) {" << std::endl;
      codes_ss << left_index_name << " = " << build_relation
               << "->GetItemIndexWithShift(j);" << std::endl;
      if (use_relation_for_stream) {
        codes_ss << "if (" << condition_name << "(" << left_index_name << ", "
                 << right_index_name << ")) {" << std::endl;
      } else {
        codes_ss << "if (" << condition_name << "(" << left_index_name << ")) {"
                 << std::endl;
      }
      codes_ss << found_match_name << " = true;" << std::endl;
      codes_ss << "break;" << std::endl;
      codes_ss << "}" << std::endl;
      codes_ss << "}" << std::endl;
      codes_ss << "}" << std::endl;
      codes_ss << "auto " << exist_name << " = " << found_match_name << ";" << std::endl;
      codes_ss << "bool " << exist_validity << " = true;" << std::endl;
    } else {
      codes_ss << "auto " << exist_name << " = " << found_match_name << ";" << std::endl;
      codes_ss << "bool " << exist_validity << " = true;" << std::endl;
    }
    finish_codes_ss << "} // end of Existence Join" << std::endl;
    if (use_relation_for_stream) {
      finish_codes_ss << "} // end of Existence Join" << std::endl;
      finish_codes_ss << "should_stop_ = !" << streamed_relation << "->NextNewKey();"
                      << std::endl;
    }
    (*output)->process_codes += codes_ss.str();
    (*output)->finish_codes += finish_codes_ss.str();

    return arrow::Status::OK();
  }

  arrow::Status GetProcessJoin(
      int join_type, bool cond_check,
      std::vector<std::pair<std::pair<std::string, std::string>, gandiva::DataTypePtr>>
          input,
      std::shared_ptr<CodeGenContext>* output) {
    // General codes when found matched rows
    auto relation_id = std::to_string(relation_id_[0]);
    auto left_index_name = "left_index_" + relation_id;
    auto right_index_name = "right_index_" + relation_id;
    auto fill_null_name = "is_outer_null_" + relation_id;
    bool use_relation_for_stream = input.empty();

    // define output list here, which will also be defined in class variables
    // definition

    int right_index_shift = 0;
    std::vector<int> left_output_idx_list;
    std::vector<int> right_output_idx_list;
    std::stringstream define_ss;

    for (int idx = 0; idx < result_schema_index_list_.size(); idx++) {
      std::string name;
      std::string arguments;
      std::shared_ptr<arrow::DataType> type;
      std::stringstream valid_ss;
      auto output_name = "sort_merge_join_" + std::to_string(relation_id_[0]) +
                         "_output_col_" + std::to_string(idx);
      auto output_validity = output_name + "_validity";
      auto i = result_schema_index_list_[idx].second;
      if (result_schema_index_list_[idx].first == 0) { /*left(streamed) table*/
        if (join_type != 0 && join_type != 1) continue;
        name =
            "sort_relation_" + std::to_string(relation_id_[0]) + "_" + std::to_string(i);
        type = left_field_list_[i]->type();
        arguments = left_index_name + ".array_id, " + left_index_name + ".id";
        if (join_type == 1) {
          valid_ss << output_validity << " = !" << fill_null_name << " && !(" << name
                   << "_has_null && " << name << "->IsNull(" << arguments << "));"
                   << std::endl;
        } else {
          valid_ss << output_validity << " = !(" << name << "_has_null && " << name
                   << "->IsNull(" << arguments << "));" << std::endl;
        }
        valid_ss << "if (" << output_validity << ")" << std::endl;
        valid_ss << output_name << " = " << name << "->GetValue(" << arguments << ");"
                 << std::endl;

        left_output_idx_list.push_back(idx);
        define_ss << "bool " << output_name << "_validity = true;" << std::endl;
        define_ss << GetCTypeString(type) << " " << output_name << ";" << std::endl;
      } else {                         /*right(streamed) table*/
        if (use_relation_for_stream) { /* use sort relation in streamed side*/
          if (exist_index_ != -1 && exist_index_ == i) {
            name =
                "sort_relation_" + std::to_string(relation_id_[0]) + "_existence_value";
            type = arrow::boolean();
            valid_ss << output_validity << " = " << name << "_validity;" << std::endl;
            valid_ss << output_name << " = " << name << ";" << std::endl;
            right_index_shift = -1;
          } else {
            i += right_index_shift;
            name = "sort_relation_" + std::to_string(relation_id_[1]) + "_" +
                   std::to_string(i);
            type = right_field_list_[i]->type();
            arguments = right_index_name + ".array_id, " + right_index_name + ".id";
            valid_ss << output_validity << " = !(" << name << "_has_null && " << name
                     << "->IsNull(" << arguments << "));" << std::endl;
            valid_ss << "if (" << output_validity << ")" << std::endl;
            valid_ss << output_name << " = " << name << "->GetValue(" << arguments << ");"
                     << std::endl;
          }
          right_output_idx_list.push_back(idx);
          define_ss << "bool " << output_name << "_validity = true;" << std::endl;
          define_ss << GetCTypeString(type) << " " << output_name << ";" << std::endl;
        } else { /* use previous output in streamed side*/
          if (exist_index_ != -1 && exist_index_ == i) {
            name =
                "sort_relation_" + std::to_string(relation_id_[0]) + "_existence_value";
            valid_ss << output_validity << " = " << name << "_validity;" << std::endl;
            valid_ss << output_name << " = " << name << ";" << std::endl;
            type = arrow::boolean();
            right_index_shift = -1;
            define_ss << "bool " << output_name << "_validity = true;" << std::endl;
            define_ss << GetCTypeString(type) << " " << output_name << ";" << std::endl;
          } else {
            i += right_index_shift;
            output_name = input[i].first.first;
            output_validity = output_name + "_validity";
            valid_ss << input[i].first.second;
            type = input[i].second;
          }
        }
      }
      (*output)->output_list.push_back(
          std::make_pair(std::make_pair(output_name, valid_ss.str()), type));
    }
    std::stringstream process_ss;
    bool cache_right = true;
    if (left_output_idx_list.size() > right_output_idx_list.size()) cache_right = false;

    switch (join_type) {
      case 0: { /* inner join */
        RETURN_NOT_OK(
            GetInnerJoin(cond_check, use_relation_for_stream, cache_right, output));
      } break;
      case 1: { /* Outer join */
        RETURN_NOT_OK(
            GetOuterJoin(cond_check, use_relation_for_stream, cache_right, output));
      } break;
      case 2: { /* Anti join */
        RETURN_NOT_OK(GetAntiJoin(cond_check, use_relation_for_stream, output));
      } break;
      case 3: { /* Semi join */
        RETURN_NOT_OK(GetSemiJoin(cond_check, use_relation_for_stream, output));
      } break;
      case 4: { /* Existence join */
        RETURN_NOT_OK(GetExistenceJoin(cond_check, use_relation_for_stream, output));
      } break;
      default: {
      } break;
    }
    (*output)->process_codes += process_ss.str();
    (*output)->definition_codes += define_ss.str();

    return arrow::Status::OK();
  }
};

arrow::Status ConditionedMergeJoinKernel::Make(
    arrow::compute::ExecContext* ctx, const gandiva::NodeVector& left_key_list,
    const gandiva::NodeVector& right_key_list,
    const gandiva::NodeVector& left_schema_list,
    const gandiva::NodeVector& right_schema_list, const gandiva::NodePtr& condition,
    int join_type, const gandiva::NodeVector& result_schema,
    std::vector<int> hash_relation_idx, std::shared_ptr<KernalBase>* out) {
  *out = std::make_shared<ConditionedMergeJoinKernel>(
      ctx, left_key_list, right_key_list, left_schema_list, right_schema_list, condition,
      join_type, result_schema, hash_relation_idx);
  return arrow::Status::OK();
}

ConditionedMergeJoinKernel::ConditionedMergeJoinKernel(
    arrow::compute::ExecContext* ctx, const gandiva::NodeVector& left_key_list,
    const gandiva::NodeVector& right_key_list,
    const gandiva::NodeVector& left_schema_list,
    const gandiva::NodeVector& right_schema_list, const gandiva::NodePtr& condition,
    int join_type, const gandiva::NodeVector& result_schema,
    std::vector<int> hash_relation_idx) {
  impl_.reset(new Impl(ctx, left_key_list, right_key_list, left_schema_list,
                       right_schema_list, condition, join_type, result_schema,
                       hash_relation_idx));
  kernel_name_ = "ConditionedMergeJoinKernel";
}

arrow::Status ConditionedMergeJoinKernel::MakeResultIterator(
    std::shared_ptr<arrow::Schema> schema,
    std::shared_ptr<ResultIterator<arrow::RecordBatch>>* out) {
  return impl_->MakeResultIterator(schema, out);
}

std::string ConditionedMergeJoinKernel::GetSignature() { return impl_->GetSignature(); }

arrow::Status ConditionedMergeJoinKernel::DoCodeGen(
    int level,
    std::vector<std::pair<std::pair<std::string, std::string>, gandiva::DataTypePtr>>
        input,
    std::shared_ptr<CodeGenContext>* codegen_ctx_out, int* var_id) {
  return impl_->DoCodeGen(level, input, codegen_ctx_out, var_id);
}

}  // namespace extra
}  // namespace arrowcompute
}  // namespace codegen
}  // namespace sparkcolumnarplugin
