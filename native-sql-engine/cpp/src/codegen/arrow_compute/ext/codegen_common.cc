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

#include "codegen/arrow_compute/ext/codegen_common.h"

#include <dlfcn.h>
#include <fcntl.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <chrono>
#include <fstream>
#include <iostream>
#include <sstream>

#include "utils/macros.h"

namespace sparkcolumnarplugin {
namespace codegen {
namespace arrowcompute {
namespace extra {

std::string BaseCodes() {
  return R"(
#include <arrow/compute/api.h>
#include <arrow/record_batch.h>

#include "codegen/arrow_compute/ext/code_generator_base.h"
#include "precompile/array.h"
using namespace sparkcolumnarplugin::codegen::arrowcompute::extra;
)";
}

std::string valueOfTimeUnit(arrow::TimeUnit::type unit) {
  switch (unit) {
    case arrow::TimeUnit::SECOND:
      return "SECOND";
    case arrow::TimeUnit::MILLI:
      return "MILLI";
    case arrow::TimeUnit::MICRO:
      return "MICRO";
    case arrow::TimeUnit::NANO:
      return "NANO";
    default:
      return "ERROR";
  }
}

std::string GetArrowTypeDefString(std::shared_ptr<arrow::DataType> type) {
  switch (type->id()) {
    case arrow::UInt8Type::type_id:
      return "uint8()";
    case arrow::Int8Type::type_id:
      return "int8()";
    case arrow::UInt16Type::type_id:
      return "uint16()";
    case arrow::Int16Type::type_id:
      return "int16()";
    case arrow::UInt32Type::type_id:
      return "uint32()";
    case arrow::Int32Type::type_id:
      return "int32()";
    case arrow::UInt64Type::type_id:
      return "uint64()";
    case arrow::Int64Type::type_id:
      return "int64()";
    case arrow::FloatType::type_id:
      return "float632()";
    case arrow::DoubleType::type_id:
      return "float64()";
    case arrow::Date32Type::type_id:
      return "date32()";
    case arrow::Date64Type::type_id:
      return "date64()";
    case arrow::StringType::type_id:
      return "utf8()";
    case arrow::BooleanType::type_id:
      return "boolean()";
    case arrow::TimestampType::type_id: {
      std::stringstream ss;
      auto* ts_type = arrow::internal::checked_cast<arrow::TimestampType*>(type.get());
      ss << "timestamp(";
      ss << "arrow::TimeUnit::";
      ss << valueOfTimeUnit(ts_type->unit());
      ss << ",";
      ss << "\"" << ts_type->timezone() << "\"";
      ss << ")";
      return ss.str();
    }
    case arrow::Decimal128Type::type_id:
      return type->ToString();
    default:
      std::cout << "GetArrowTypeString can't convert " << type->ToString() << std::endl;
      throw;
  }
}
std::string GetCTypeString(std::shared_ptr<arrow::DataType> type) {
  switch (type->id()) {
    case arrow::UInt8Type::type_id:
      return "uint8_t";
    case arrow::Int8Type::type_id:
      return "int8_t";
    case arrow::UInt16Type::type_id:
      return "uint16_t";
    case arrow::Int16Type::type_id:
      return "int16_t";
    case arrow::UInt32Type::type_id:
      return "uint32_t";
    case arrow::Int32Type::type_id:
      return "int32_t";
    case arrow::UInt64Type::type_id:
      return "uint64_t";
    case arrow::Int64Type::type_id:
      return "int64_t";
    case arrow::FloatType::type_id:
      return "float";
    case arrow::DoubleType::type_id:
      return "double";
    case arrow::Date32Type::type_id:
      return "int32_t";
    case arrow::Date64Type::type_id:
      return "int64_t";
    case arrow::TimestampType::type_id:
      return "int64_t";
    case arrow::StringType::type_id:
      return "std::string";
    case arrow::BooleanType::type_id:
      return "bool";
    case arrow::Decimal128Type::type_id:
      return "arrow::Decimal128";
    default:
      std::cout << "GetCTypeString can't convert " << type->ToString() << std::endl;
      throw;
  }
}
std::string GetTypeString(std::shared_ptr<arrow::DataType> type, std::string tail) {
  switch (type->id()) {
    case arrow::UInt8Type::type_id:
      return "UInt8" + tail;
    case arrow::Int8Type::type_id:
      return "Int8" + tail;
    case arrow::UInt16Type::type_id:
      return "UInt16" + tail;
    case arrow::Int16Type::type_id:
      return "Int16" + tail;
    case arrow::UInt32Type::type_id:
      return "UInt32" + tail;
    case arrow::Int32Type::type_id:
      return "Int32" + tail;
    case arrow::UInt64Type::type_id:
      return "UInt64" + tail;
    case arrow::Int64Type::type_id:
      return "Int64" + tail;
    case arrow::FloatType::type_id:
      return "Float" + tail;
    case arrow::DoubleType::type_id:
      return "Double" + tail;
    case arrow::Date32Type::type_id:
      return "Date32" + tail;
    case arrow::Date64Type::type_id:
      return "Date64" + tail;
    case arrow::TimestampType::type_id:
      return "Timestamp" + tail;
    case arrow::StringType::type_id:
      return "String" + tail;
    case arrow::BooleanType::type_id:
      return "Boolean" + tail;
    case arrow::Decimal128Type::type_id:
      return "Decimal128" + tail;
    default:
      std::cout << "GetTypeString can't convert " << type->ToString() << std::endl;
      throw;
  }
}
std::string GetTemplateString(std::shared_ptr<arrow::DataType> type,
                              std::string template_name, std::string tail,
                              std::string prefix) {
  switch (type->id()) {
    case arrow::UInt8Type::type_id:
      if (tail.empty())
        return template_name + "<uint8_t>";
      else
        return template_name + "<" + prefix + "UInt8" + tail + ">";
    case arrow::Int8Type::type_id:
      if (tail.empty())
        return template_name + "<int8_t>";
      else
        return template_name + "<" + prefix + "Int8" + tail + ">";
    case arrow::UInt16Type::type_id:
      if (tail.empty())
        return template_name + "<uint16_t>";
      else
        return template_name + "<" + prefix + "UInt16" + tail + ">";
    case arrow::Int16Type::type_id:
      if (tail.empty())
        return template_name + "<int16_t>";
      else
        return template_name + "<" + prefix + "Int16" + tail + ">";
    case arrow::UInt32Type::type_id:
      if (tail.empty())
        return template_name + "<uint32_t>";
      else
        return template_name + "<" + prefix + "UInt32" + tail + ">";
    case arrow::Int32Type::type_id:
      if (tail.empty())
        return template_name + "<int32_t>";
      else
        return template_name + "<" + prefix + "Int32" + tail + ">";
    case arrow::UInt64Type::type_id:
      if (tail.empty())
        return template_name + "<uint64_t>";
      else
        return template_name + "<" + prefix + "UInt64" + tail + ">";
    case arrow::Int64Type::type_id:
      if (tail.empty())
        return template_name + "<int64_t>";
      else
        return template_name + "<" + prefix + "Int64" + tail + ">";
    case arrow::FloatType::type_id:
      if (tail.empty())
        return template_name + "<float>";
      else
        return template_name + "<" + prefix + "Float" + tail + ">";
    case arrow::DoubleType::type_id:
      if (tail.empty())
        return template_name + "<double>";
      else
        return template_name + "<" + prefix + "Double" + tail + ">";
    case arrow::Date32Type::type_id:
      if (tail.empty())
        return template_name + "<uint32_t>";
      else
        return template_name + "<" + prefix + "Date32" + tail + ">";
    case arrow::Date64Type::type_id:
      if (tail.empty())
        return template_name + "<uint64_t>";
      else
        return template_name + "<" + prefix + "Date64" + tail + ">";
    case arrow::TimestampType::type_id:
      if (tail.empty())
        return template_name + "<uint64_t>";
      else
        return template_name + "<" + prefix + "Timestamp" + tail + ">";
    case arrow::StringType::type_id:
      if (tail.empty())
        return template_name + "<std::string>";
      else
        return template_name + "<" + prefix + "String" + tail + ">";
    case arrow::BooleanType::type_id:
      if (tail.empty())
        return template_name + "<bool>";
      else
        return template_name + "<" + prefix + "Boolean" + tail + ">";
    case arrow::Decimal128Type::type_id:
      if (tail.empty())
        return template_name + "<arrow::Decimal128>";
      else
        return template_name + "<" + prefix + "Decimal128" + tail + ">";
    default:
      std::cout << "GetTemplateString can't convert " << type->ToString() << std::endl;
      throw;
  }
}

std::string GetParameterList(std::vector<std::string> parameter_list_in, bool comma_ahead,
                             std::string split) {
  std::vector<std::string> parameter_list;
  for (auto s : parameter_list_in) {
    if (s != "") {
      parameter_list.push_back(s);
    }
  }

  std::stringstream ss;
  for (int i = 0; i < parameter_list.size(); i++) {
    if (i != (parameter_list.size() - 1)) {
      ss << parameter_list[i] << split;
    } else {
      ss << parameter_list[i] << "";
    }
  }
  auto ret = ss.str();
  if (!comma_ahead) {
    return ret;
  }
  if (ret.empty()) {
    return ret;
  } else {
    return split + ret;
  }
}

std::string str_tolower(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(),
                 [](unsigned char c) { return std::tolower(c); }  // correct
  );
  return s;
}

std::pair<int, int> GetFieldIndex(gandiva::FieldPtr target_field,
                                  std::vector<gandiva::FieldVector> field_list_v) {
  int arg_id = 0;
  int index = 0;
  bool found = false;
  for (auto field_list : field_list_v) {
    arg_id = 0;
    for (auto field : field_list) {
      if (str_tolower(field->name()) == str_tolower(target_field->name())) {
        found = true;
        break;
      }
      arg_id++;
    }
    if (found) {
      break;
    }
    index += 1;
  }
  if (!found) {
    return std::make_pair(-1, -1);
  }
  return std::make_pair(index, arg_id);
}

gandiva::ExpressionVector GetGandivaKernel(std::vector<gandiva::NodePtr> key_list) {
  gandiva::ExpressionVector project_list;
  int idx = 0;
  for (auto key : key_list) {
    auto expr = gandiva::TreeExprBuilder::MakeExpression(
        key, arrow::field("projection_key_" + std::to_string(idx++), key->return_type()));
    project_list.push_back(expr);
  }
  return project_list;
}

gandiva::ExpressionPtr GetHash32Kernel(std::vector<gandiva::NodePtr> key_list,
                                       std::vector<int> key_index_list) {
  // This Project should be do upon GetGandivaKernel
  // So we need to treat inside functionNode as fieldNode.
  std::vector<std::shared_ptr<gandiva::Node>> func_node_list = {};
  std::shared_ptr<arrow::DataType> ret_type;
  auto seed = gandiva::TreeExprBuilder::MakeLiteral((int32_t)0);
  gandiva::NodePtr func_node;
  ret_type = arrow::int32();
  int idx = 0;
  for (auto key : key_list) {
    auto field_node = gandiva::TreeExprBuilder::MakeField(arrow::field(
        "projection_key_" + std::to_string(key_index_list[idx++]), key->return_type()));
    func_node =
        gandiva::TreeExprBuilder::MakeFunction("hash32", {field_node, seed}, ret_type);
    seed = func_node;
  }
  func_node_list.push_back(func_node);
  return gandiva::TreeExprBuilder::MakeExpression(func_node_list.at(0),
                                                  arrow::field("hash_key", ret_type));
}

gandiva::ExpressionPtr GetHash32Kernel(std::vector<gandiva::NodePtr> key_list) {
  // This Project should be do upon GetGandivaKernel
  // So we need to treat inside functionNode as fieldNode.
  std::vector<std::shared_ptr<gandiva::Node>> func_node_list = {};
  std::shared_ptr<arrow::DataType> ret_type;
  auto seed = gandiva::TreeExprBuilder::MakeLiteral((int32_t)0);
  gandiva::NodePtr func_node;
  ret_type = arrow::int32();
  int idx = 0;
  for (auto key : key_list) {
    auto field_node = gandiva::TreeExprBuilder::MakeField(
        arrow::field("projection_key_" + std::to_string(idx++), key->return_type()));
    func_node =
        gandiva::TreeExprBuilder::MakeFunction("hash32", {field_node, seed}, ret_type);
    seed = func_node;
  }
  func_node_list.push_back(func_node);
  return gandiva::TreeExprBuilder::MakeExpression(func_node_list[0],
                                                  arrow::field("hash_key", ret_type));
}

gandiva::ExpressionPtr GetConcatedKernel(std::vector<gandiva::NodePtr> key_list) {
  std::vector<std::shared_ptr<gandiva::Node>> func_node_list = {};
  std::shared_ptr<arrow::DataType> ret_type;
  if (key_list.size() >= 2) {
    ret_type = arrow::int64();
    for (auto key : key_list) {
      auto field_node = key;
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
    }
  } else {
    auto node = key_list[0];
    ret_type = node->return_type();
    func_node_list.push_back(node);
  }
  return gandiva::TreeExprBuilder::MakeExpression(
      func_node_list[0], arrow::field("projection_key", ret_type));
}

arrow::Status GetIndexList(const std::vector<std::shared_ptr<arrow::Field>>& target_list,
                           const std::vector<std::shared_ptr<arrow::Field>>& source_list,
                           std::vector<int>* out) {
  bool found = false;
  for (auto key_field : target_list) {
    int i = 0;
    found = false;
    for (auto field : source_list) {
      if (key_field->name() == field->name()) {
        found = true;
        break;
      }
      i++;
    }
    if (found) (*out).push_back(i);
  }
  return arrow::Status::OK();
}

arrow::Status GetIndexList(
    const std::vector<std::shared_ptr<arrow::Field>>& target_list,

    const std::vector<std::shared_ptr<arrow::Field>>& left_field_list,
    const std::vector<std::shared_ptr<arrow::Field>>& right_field_list,
    const bool isExistJoin, int* exist_index,
    std::vector<std::pair<int, int>>* result_schema_index_list) {
  int i = 0;
  bool found = false;
  int target_index = -1;
  int right_found = 0;
  for (auto target_field : target_list) {
    target_index++;
    i = 0;
    found = false;
    for (auto field : left_field_list) {
      if (target_field->name() == field->name()) {
        (*result_schema_index_list).push_back(std::make_pair(0, i));
        found = true;
        break;
      }
      i++;
    }
    if (found == true) continue;
    i = 0;
    for (auto field : right_field_list) {
      if (target_field->name() == field->name()) {
        (*result_schema_index_list).push_back(std::make_pair(1, i));
        found = true;
        right_found++;
        break;
      }
      i++;
    }
    if (found == true) continue;
    if (isExistJoin) *exist_index = target_index;
  }
  // Add one more col if join_type is ExistenceJoin
  if (isExistJoin) {
    (*result_schema_index_list).push_back(std::make_pair(1, right_found));
  }
  return arrow::Status::OK();
}

arrow::Status GetIndexListFromSchema(
    const std::shared_ptr<arrow::Schema>& result_schema,
    const std::vector<std::shared_ptr<arrow::Field>>& field_list,
    std::vector<int>* index_list) {
  int i = 0;
  for (auto field : field_list) {
    auto indices = result_schema->GetAllFieldIndices(field->name());
    if (indices.size() >= 1) {
      (*index_list).push_back(i);
    }
    i++;
  }
  return arrow::Status::OK();
}

std::string GetTempPath() {
  std::string tmp_dir_;
  const char* env_tmp_dir = std::getenv("NATIVESQL_TMP_DIR");
  if (env_tmp_dir != nullptr) {
    tmp_dir_ = std::string(env_tmp_dir);
  } else {
#ifdef NATIVESQL_SRC_PATH
    tmp_dir_ = NATIVESQL_SRC_PATH;
#else
    std::cerr << "envioroment variable NATIVESQL_TMP_DIR is not set" << std::endl;
    throw;
#endif
  }
  return tmp_dir_;
}

bool GetEnableTimeMetrics() {
  bool is_enable = false;
  const char* env_enable_time_metrics = std::getenv("NATIVESQL_METRICS_TIME");
  if (env_enable_time_metrics != nullptr) {
    auto is_enable_str = std::string(env_enable_time_metrics);
    if (is_enable_str.compare("true") == 0) is_enable = true;
  }
  return is_enable;
}

int GetBatchSize() {
  int batch_size;
  const char* env_batch_size = std::getenv("NATIVESQL_BATCH_SIZE");
  if (env_batch_size != nullptr) {
    batch_size = atoi(env_batch_size);
  } else {
    batch_size = 10000;
  }
  return batch_size;
}

int FileSpinLock() {
  std::string lockfile = GetTempPath() + "/nativesql_compile.lock";

  auto fd = open(lockfile.c_str(), O_CREAT, S_IRWXU | S_IRWXG);
  flock(fd, LOCK_EX);

  return fd;
}

void FileSpinUnLock(int fd) {
  flock(fd, LOCK_UN);
  close(fd);
}

arrow::Status CompileCodes(std::string codes, std::string signature) {
  // temporary cpp/library output files
  srand(time(NULL));
  std::string outpath = GetTempPath() + "/tmp";
  mkdir(outpath.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
  std::string prefix = "/spark-columnar-plugin-codegen-";
  std::string cppfile = outpath + prefix + signature + ".cc";
  std::string libfile = outpath + prefix + signature + ".so";
  std::string jarfile = outpath + prefix + signature + ".jar";
  std::string logfile = outpath + prefix + signature + ".log";
  std::ofstream out(cppfile.c_str(), std::ofstream::out);

  // output code to file
  if (out.bad()) {
    std::cout << "cannot open " << cppfile << std::endl;
    exit(EXIT_FAILURE);
  }
  out << codes;
#ifdef DEBUG
  std::cout << "BatchSize is " << GetBatchSize() << std::endl;
  std::cout << codes << std::endl;
#endif
  out.flush();
  out.close();

  // compile the code
  const char* env_gcc_ = std::getenv("CC");
  if (env_gcc_ == nullptr) {
    env_gcc_ = "gcc";
  }
  std::string env_gcc = std::string(env_gcc_);

  char* env_codegen_option_ = std::getenv("CODEGEN_OPTION");

  if (env_codegen_option_ == nullptr) {
    env_codegen_option_ = " -O3 -march=native ";
  }
  std::string env_codegen_option = std::string(env_codegen_option_);

  const char* env_arrow_dir = std::getenv("LIBARROW_DIR");
  std::string arrow_header;
  std::string arrow_lib, arrow_lib2;
  std::string nativesql_header = " -I" + GetTempPath() + "/nativesql_include/ ";
  std::string nativesql_header_2 = " -I" + GetTempPath() + "/include/ ";
  std::string nativesql_lib = " -L" + GetTempPath() + " ";
  if (env_arrow_dir != nullptr) {
    arrow_header = " -I" + std::string(env_arrow_dir) + "/include ";
    arrow_lib = " -L" + std::string(env_arrow_dir) + "/lib64 ";
    // incase there's a different location for libarrow.so
    arrow_lib2 = " -L" + std::string(env_arrow_dir) + "/lib ";
  }
  // compile the code
  std::string cmd = env_gcc + " -std=c++14 -Wno-deprecated-declarations " + arrow_header +
                    arrow_lib + arrow_lib2 + nativesql_header + nativesql_header_2 +
                    nativesql_lib + cppfile + " -o " + libfile + env_codegen_option +
                    " -shared -fPIC -lspark_columnar_jni 2> " + logfile;
#ifdef DEBUG
  std::cout << cmd << std::endl;
#endif
  int ret;
  int elapse_time = 0;
  TIME_MICRO(elapse_time, ret, system(cmd.c_str()));
#ifdef DEBUG
  std::cout << "CodeGeneration took " << TIME_TO_STRING(elapse_time) << std::endl;
#endif
  if (WEXITSTATUS(ret) != EXIT_SUCCESS) {
    std::cout << "compilation failed, see " << logfile << std::endl;
    std::cout << cmd << std::endl;
    return arrow::Status::Invalid("compilation failed, see ", logfile);
  }
  cmd = "cd " + outpath + "; jar -cf spark-columnar-plugin-codegen-precompile-" +
        signature + ".jar spark-columnar-plugin-codegen-" + signature + ".so";
#ifdef DEBUG
  std::cout << cmd << std::endl;
#endif
  ret = system(cmd.c_str());
  if (WEXITSTATUS(ret) != EXIT_SUCCESS) {
    exit(EXIT_FAILURE);
  }

  struct stat tstat;
  ret = stat(libfile.c_str(), &tstat);
  if (ret == -1) {
    std::cout << "stat failed: " << strerror(errno) << std::endl;
    exit(EXIT_FAILURE);
  }

  return arrow::Status::OK();
}

std::string exec(const char* cmd) {
  std::array<char, 128> buffer;
  std::string result;
  FILE* file = popen(cmd, "r");
  std::unique_ptr<FILE, decltype(&pclose)> pipe(file, pclose);
  if (!pipe) {
    pclose(file);
    throw std::runtime_error("popen() failed!");
  }
  try {
    while (fgets(buffer.data(), sizeof buffer, pipe.get()) != nullptr) {
      result += buffer.data();
    }
  } catch (...) {
    pclose(file);
    throw;
  }
  return result;
}

arrow::Status LoadLibrary(std::string signature, arrow::compute::ExecContext* ctx,
                          std::shared_ptr<CodeGenBase>* out) {
  std::string outpath = GetTempPath() + "/tmp";
  std::string prefix = "/spark-columnar-plugin-codegen-";
  std::string libfile = outpath + prefix + signature + ".so";
  // load dynamic library
  void* dynlib = dlopen(libfile.c_str(), RTLD_LAZY);
  if (!dynlib) {
    std::stringstream ss;
    ss << "LoadLibrary " << libfile
       << " failed. \nCur dir has contents "
          "as below."
       << std::endl;
    auto cmd = "ls -l " + GetTempPath() + ";";
    ss << exec(cmd.c_str()) << std::endl;
    return arrow::Status::Invalid(libfile,
                                  " is not generated, failed msg as below: ", ss.str());
  }

  // loading symbol from library and assign to pointer
  // (to be cast to function pointer later)

  void (*MakeCodeGen)(arrow::compute::ExecContext * ctx,
                      std::shared_ptr<CodeGenBase> * out);
  *(void**)(&MakeCodeGen) = dlsym(dynlib, "MakeCodeGen");
  const char* dlsym_error = dlerror();
  if (dlsym_error != NULL) {
    std::stringstream ss;
    ss << "error loading symbol:\n" << dlsym_error << std::endl;
    return arrow::Status::Invalid(ss.str());
  }

  MakeCodeGen(ctx, out);
  return arrow::Status::OK();
}
}  // namespace extra
}  // namespace arrowcompute
}  // namespace codegen
}  // namespace sparkcolumnarplugin
