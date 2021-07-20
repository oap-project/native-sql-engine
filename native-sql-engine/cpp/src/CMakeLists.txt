cmake_minimum_required(VERSION 3.16)

project(spark_columnar_jni)

include(ExternalProject)
include(FindPkgConfig)
include(GNUInstallDirs)
include(CheckCXXCompilerFlag)

set(CMAKE_CXX_STANDARD 14)

set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_BUILD_TYPE  "Release")

set(ARROW_ROOT "/usr/local" CACHE PATH "Arrow Root dir")
set(ARROW_BFS_INSTALL_DIR "/usr/local" CACHE PATH "Arrow Build from Source dir")
set(ARROW_LIB_NAME arrow)
set(GANDIVA_LIB_NAME gandiva)
set(ARROW_SHARED_LIBRARY_SUFFIX ".so")

option(BUILD_ARROW "Build Arrow from Source" ON)
option(STATIC_ARROW "Build Arrow with Static Libraries" OFF)
option(BUILD_PROTOBUF "Build Protobuf from Source" ON)
option(USE_AVX512 "Build with AVX-512 optimizations" OFF)
option(TESTS "Build the tests" OFF)
option(BENCHMARKS "Build the benchmarks" OFF)
option(DEBUG "Enable Debug Info" OFF)

set(BOOST_MIN_VERSION "1.42.0")
find_package(Boost REQUIRED)
INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIRS})

find_package(JNI REQUIRED)
set(source_root_directory ${CMAKE_CURRENT_SOURCE_DIR})

find_program(CCACHE_FOUND ccache)
if(CCACHE_FOUND)
  set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE ccache)
  set_property(GLOBAL PROPERTY RULE_LAUNCH_LINK ccache)
endif(CCACHE_FOUND)

# Building Protobuf
macro(build_protobuf)
  message(STATUS "Building Protocol Buffers from Source")
  set (PROTOBUF_SOURCE_URL
       "https://github.com/protocolbuffers/protobuf/releases/download/v3.13.0/protobuf-all-3.13.0.tar.gz"
        "https://github.com/ursa-labs/thirdparty/releases/download/latest/protobuf-v3.13.0.tar.gz"
  )
  set(PROTOBUF_PREFIX "${CMAKE_CURRENT_BINARY_DIR}/protobuf_ep-install")
  set(PROTOBUF_INCLUDE_DIR "${PROTOBUF_PREFIX}/include")
  set(
    PROTOBUF_STATIC_LIB
    "${PROTOBUF_PREFIX}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}protobuf${CMAKE_STATIC_LIBRARY_SUFFIX}"
    )
  set(
    PROTOC_STATIC_LIB
    "${PROTOBUF_PREFIX}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}protoc${CMAKE_STATIC_LIBRARY_SUFFIX}"
    )
  set(
    PROTOC_BIN
    "${PROTOBUF_PREFIX}/bin/protoc"
    )
  set(
    PROTOBUF_INCLUDE
    "${PROTOBUF_PREFIX}/include"
    )
  set(PROTOBUF_COMPILER "${PROTOBUF_PREFIX}/bin/protoc")
  set(PROTOBUF_CONFIGURE_ARGS
      "AR=${CMAKE_AR}"
      "RANLIB=${CMAKE_RANLIB}"
      "CC=${CMAKE_C_COMPILER}"
      "CXX=${CMAKE_CXX_COMPILER}"
      "--disable-shared"
      "--prefix=${PROTOBUF_PREFIX}"
      "CFLAGS=-fPIC"
      "CXXFLAGS=-fPIC")
  set(PROTOBUF_BUILD_COMMAND ${MAKE} ${MAKE_BUILD_ARGS})
  ExternalProject_Add(protobuf_ep
                      PREFIX protobuf_ep
                      CONFIGURE_COMMAND "./configure" ${PROTOBUF_CONFIGURE_ARGS}
                      BUILD_COMMAND ${PROTOBUF_BUILD_COMMAND}
                      BUILD_IN_SOURCE 1
                      URL_MD5 cafa623d51361228c83c874d95f51992
                      URL ${PROTOBUF_SOURCE_URL}
  )

  file(MAKE_DIRECTORY "${PROTOBUF_INCLUDE_DIR}")
  add_library(protobuf::libprotobuf STATIC IMPORTED)
  set_target_properties(
    protobuf::libprotobuf
    PROPERTIES IMPORTED_LOCATION "${PROTOBUF_STATIC_LIB}" INTERFACE_INCLUDE_DIRECTORIES
               "${PROTOBUF_INCLUDE_DIR}")
  add_dependencies(protobuf::libprotobuf protobuf_ep)
endmacro()

macro(find_protobuf)
  # Find the existing Protobuf
  set(CMAKE_FIND_LIBRARY_SUFFIXES ".so")
  find_package(Protobuf)
  if ("${Protobuf_LIBRARY}" STREQUAL "Protobuf_LIBRARY-NOTFOUND")
    message(FATAL_ERROR "Protobuf Library Not Found")
  endif()
  set(PROTOC_BIN ${Protobuf_PROTOC_EXECUTABLE})
endmacro()

if(DEBUG)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O0 -DDEBUG -DDEBUG_LEVEL_1 -DDEBUG_LEVEL_2")
endif()

if(USE_AVX512)
  # Only enable additional instruction sets if they are supported
  message(STATUS "System processor: ${CMAKE_SYSTEM_PROCESSOR}")
  if (CMAKE_SYSTEM_PROCESSOR MATCHES "(x86)|(X86)|(amd64)|(AMD64)")
    set(AVX512_FLAG "-march=skylake-avx512")
    check_cxx_compiler_flag(${AVX512_FLAG} CXX_SUPPORTS_AVX512)
    if(NOT CXX_SUPPORTS_AVX512)
      message(FATAL_ERROR "AVX512 required but compiler doesn't support it.")
    endif()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${AVX512_FLAG}")
    add_definitions(-DCOLUMNAR_PLUGIN_USE_AVX512)
  endif ()
endif()

# Build Arrow macro
macro(build_arrow STATIC_ARROW)
  set(ARROW_PREFIX "${CMAKE_CURRENT_BINARY_DIR}/arrow_ep-install")
  message(STATUS "ARROW_PREFIX: ${ARROW_PREFIX}")
  set(ARROW_SOURCE_DIR "${CMAKE_CURRENT_BINARY_DIR}/arrow_ep")
  message(STATUS "ARROW_SOURCE_DIR: ${ARROW_SOURCE_DIR}")
  set(ARROW_INCLUDE_DIR "${ARROW_PREFIX}/include")
  set(BINARY_RELEASE_DIR "${root_directory}/releases")

  ExternalProject_Add(arrow_ep
                      GIT_REPOSITORY https://github.com/oap-project/arrow.git
                      SOURCE_DIR ${ARROW_SOURCE_DIR}
                      GIT_TAG arrow-4.0.0-oap
                      BUILD_IN_SOURCE 1
                      INSTALL_DIR ${ARROW_PREFIX}
                      INSTALL_COMMAND make install
                      SOURCE_SUBDIR cpp
                      CMAKE_ARGS 
                      -DARROW_BUILD_STATIC=OFF
                      -DARROW_BUILD_SHARED=ON
                      -DARROW_COMPUTE=ON
                      -DARROW_S3=ON
                      -DARROW_GANDIVA_JAVA=ON
                      -DARROW_GANDIVA=ON
                      -DARROW_PARQUET=ON
                      -DARROW_CSV=ON
                      -DARROW_HDFS=ON
                      -DARROW_BOOST_USE_SHARED=OFF
                      -DARROW_JNI=ON
                      -DARROW_DATASET=ON
                      -DARROW_WITH_PROTOBUF=ON
                      -DARROW_WITH_SNAPPY=ON
                      -DARROW_WITH_LZ4=ON
                      -DARROW_WITH_ZSTD=OFF
                      -DARROW_WITH_BROTLI=OFF
                      -DARROW_WITH_ZLIB=OFF
                      -DARROW_WITH_FASTPFOR=ON
                      -DARROW_FILESYSTEM=ON
                      -DARROW_JSON=ON
                      -DARROW_FLIGHT=OFF
                      -DARROW_JEMALLOC=ON
                      -DARROW_SIMD_LEVEL=AVX2
                      -DARROW_RUNTIME_SIMD_LEVEL=MAX
                      -DARROW_DEPENDENCY_SOURCE=BUNDLED
                      -DCMAKE_INSTALL_PREFIX=${ARROW_PREFIX}
                      -DCMAKE_INSTALL_LIBDIR=lib)

  ExternalProject_Add_Step(arrow_ep java_install
                      COMMAND mvn clean install -P arrow-jni -am -Darrow.cpp.build.dir=${ARROW_PREFIX}/lib -DskipTests -Dcheckstyle.skip
                      COMMENT "Arrow Java maven install after CPP make install"
                      DEPENDEES mkdir download update patch configure build install
                      WORKING_DIRECTORY "${ARROW_SOURCE_DIR}/java"
  )
  add_dependencies(arrow_ep jni_proto)

  file(MAKE_DIRECTORY "${ARROW_PREFIX}/include")

  if(STATIC_ARROW)
    # Load Static Arrow Library
    message(FATAL_ERROR "Not Support Static Arrow")

    set(THREADS_PREFER_PTHREAD_FLAG ON)
    find_package(Threads REQUIRED)

    set(ARROW_LIB_NAME arrow_bundled_dependencies)
    set(
      ARROW_STATIC_LIB
      "${ARROW_PREFIX}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}${ARROW_LIB_NAME}${CMAKE_STATIC_LIBRARY_SUFFIX}"
      )
    add_library(Arrow::arrow STATIC IMPORTED)
    set_target_properties(Arrow::arrow
                          PROPERTIES IMPORTED_LOCATION "${ARROW_STATIC_LIB}"
                                     INTERFACE_INCLUDE_DIRECTORIES
                                      "${ARROW_PREFIX}/include")
    add_dependencies(Arrow::arrow arrow_ep)

    # Load Static Gandiva Library
    set(
      GANDIVA_STATIC_LIB
      "${ARROW_PREFIX}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}${GANDIVA_LIB_NAME}${CMAKE_STATIC_LIBRARY_SUFFIX}"
      )
    add_library(Arrow::gandiva STATIC IMPORTED)
    set_target_properties(Arrow::gandiva
                          PROPERTIES IMPORTED_LOCATION "${GANDIVA_STATIC_LIB}"
                                     INTERFACE_INCLUDE_DIRECTORIES
                                      "${ARROW_PREFIX}/include")
    add_dependencies(Arrow::gandiva arrow_ep)
    target_link_libraries(spark_columnar_jni PRIVATE Arrow::arrow Arrow::gandiva Threads::Threads)

  else()

    # Copy Arrow Shared Library to releases directory for package jar
    ExternalProject_Add_Step(arrow_ep copy_arrow_binary
                      COMMAND cp -a ${ARROW_PREFIX}/lib/${CMAKE_SHARED_LIBRARY_PREFIX}${ARROW_LIB_NAME}${ARROW_SHARED_LIBRARY_SUFFIX} ${root_directory}/releases/
                      COMMENT "Copy libarrow.so to releases/"
                      DEPENDEES mkdir download update patch configure build install java_install
                      WORKING_DIRECTORY "${ARROW_PREFIX}/"
    )

    ExternalProject_Add_Step(arrow_ep copy_arrow_binary_300
                      COMMAND cp -a ${ARROW_PREFIX}/lib/${CMAKE_SHARED_LIBRARY_PREFIX}${ARROW_LIB_NAME}${ARROW_SHARED_LIBRARY_SUFFIX}.400 ${root_directory}/releases/
                      COMMENT "Copy libarrow.so.400 to releases/"
                      DEPENDEES mkdir download update patch configure build install java_install
                      WORKING_DIRECTORY "${ARROW_PREFIX}/"
    )

    ExternalProject_Add_Step(arrow_ep copy_arrow_binary_300_0_0
                      COMMAND cp -a ${ARROW_PREFIX}/lib/${CMAKE_SHARED_LIBRARY_PREFIX}${ARROW_LIB_NAME}${ARROW_SHARED_LIBRARY_SUFFIX}.400.0.0 ${root_directory}/releases/
                      COMMENT "Copy libarrow.so.400.0.0 to releases/"
                      DEPENDEES mkdir download update patch configure build install java_install
                      WORKING_DIRECTORY "${ARROW_PREFIX}/"
    )


    # Copy Gandiva Shared Library to releases directory for package jar
    ExternalProject_Add_Step(arrow_ep copy_gandiva_binary
                      COMMAND cp -a ${ARROW_PREFIX}/lib/${CMAKE_SHARED_LIBRARY_PREFIX}${GANDIVA_LIB_NAME}${ARROW_SHARED_LIBRARY_SUFFIX} ${root_directory}/releases/
                      COMMENT "Copy libgandiva.so to releases/"
                      DEPENDEES mkdir download update patch configure build install java_install
                      WORKING_DIRECTORY "${ARROW_PREFIX}/"
    )

    ExternalProject_Add_Step(arrow_ep copy_gandiva_binary_300
                      COMMAND cp -a ${ARROW_PREFIX}/lib/${CMAKE_SHARED_LIBRARY_PREFIX}${GANDIVA_LIB_NAME}${ARROW_SHARED_LIBRARY_SUFFIX}.400 ${root_directory}/releases/
                      COMMENT "Copy libgandiva.so.400 to releases/"
                      DEPENDEES mkdir download update patch configure build install java_install
                      WORKING_DIRECTORY "${ARROW_PREFIX}/"
    )

    ExternalProject_Add_Step(arrow_ep copy_gandiva_binary_300_0_0
                      COMMAND cp -a ${ARROW_PREFIX}/lib/${CMAKE_SHARED_LIBRARY_PREFIX}${GANDIVA_LIB_NAME}${ARROW_SHARED_LIBRARY_SUFFIX}.400.0.0 ${root_directory}/releases/
                      COMMENT "Copy libgandiva.so.400.0.0 to releases/"
                      DEPENDEES mkdir download update patch configure build install java_install
                      WORKING_DIRECTORY "${ARROW_PREFIX}/"
    )


    # Copy Arrow Headers to releases/include
    ExternalProject_Add_Step(arrow_ep copy_arrow_header
                      COMMAND cp -rf ${ARROW_PREFIX}/include/ ${root_directory}/releases/
                      COMMENT "Arrow Header to releases/include"
                      DEPENDEES mkdir download update patch configure build install java_install
                      WORKING_DIRECTORY "${ARROW_PREFIX}/"
    )

    # Set up Arrow Shared Library Directory
    set(
      ARROW_SHARED_LIB
      "${root_directory}/releases/${CMAKE_SHARED_LIBRARY_PREFIX}${ARROW_LIB_NAME}${ARROW_SHARED_LIBRARY_SUFFIX}"
      )
    add_library(Arrow::arrow SHARED IMPORTED)
    set_target_properties(Arrow::arrow
                          PROPERTIES IMPORTED_LOCATION "${ARROW_SHARED_LIB}"
                                     INTERFACE_INCLUDE_DIRECTORIES
                                      "${root_directory}/releases/include")
    add_dependencies(Arrow::arrow arrow_ep)

    # Set up Gandiva Shared Library Directory
    set(
      GANDIVA_SHARED_LIB
      "${root_directory}/releases/${CMAKE_SHARED_LIBRARY_PREFIX}${GANDIVA_LIB_NAME}${ARROW_SHARED_LIBRARY_SUFFIX}"
      )
    add_library(Arrow::gandiva SHARED IMPORTED)
    set_target_properties(Arrow::gandiva
                          PROPERTIES IMPORTED_LOCATION "${GANDIVA_SHARED_LIB}"
                                     INTERFACE_INCLUDE_DIRECTORIES
                                      "${root_directory}/releases/include")
    add_dependencies(Arrow::gandiva arrow_ep)

    target_link_libraries(spark_columnar_jni
                      LINK_PUBLIC Arrow::arrow Arrow::gandiva)
  endif()
endmacro()

# Find the existing Arrow library by using ARROW_RROT path
macro(find_arrow)
  set(ARROW_BFS_LIB_DIR "${ARROW_BFS_INSTALL_DIR}/lib")
  set(ARROW_LIB_DIR "${ARROW_ROOT}/lib")
  set(ARROW_LIB64_DIR "${ARROW_ROOT}/lib64")
  message(STATUS "Set Arrow Library Directory in ${ARROW_BFS_LIB_DIR} or ${ARROW_LIB_DIR} or ${ARROW_LIB64_DIR}")
  set(ARROW_BFS_INCLUDE_DIR "${ARROW_BFS_INSTALL_DIR}/include")
  set(ARROW_INCLUDE_DIR "${ARROW_ROOT}/include")
  message(STATUS "Set Arrow Include Directory in ${ARROW_BFS_INCLUDE_DIR} or ${ARROW_INCLUDE_DIR}")

  find_library(ARROW_LIB NAMES ${CMAKE_SHARED_LIBRARY_PREFIX}${ARROW_LIB_NAME}${ARROW_SHARED_LIBRARY_SUFFIX} PATHS ${ARROW_BFS_LIB_DIR} ${ARROW_LIB_DIR} ${ARROW_LIB64_DIR} NO_DEFAULT_PATH)
  if(NOT ARROW_LIB)
    message(FATAL_ERROR "Arrow Library Not Found")
  else()
    message(STATUS "Arrow Library Can Be Found in ${ARROW_LIB}")
  endif()

  find_library(GANDIVA_LIB NAMES ${CMAKE_SHARED_LIBRARY_PREFIX}${GANDIVA_LIB_NAME}${ARROW_SHARED_LIBRARY_SUFFIX} PATHS ${ARROW_BFS_LIB_DIR} ${ARROW_LIB_DIR} ${ARROW_LIB64_DIR} NO_DEFAULT_PATH)
  if(NOT GANDIVA_LIB)
    message(FATAL_ERROR "Gandiva Library Not Found")
  else()
    message(STATUS "Gandiva Library Can Be Found in ${GANDIVA_LIB}")
  endif()

  file(COPY ${ARROW_LIB} DESTINATION ${root_directory}/releases/ FOLLOW_SYMLINK_CHAIN)
  file(COPY ${GANDIVA_LIB} DESTINATION ${root_directory}/releases/ FOLLOW_SYMLINK_CHAIN)

  if(EXISTS ${ARROW_BFS_INCLUDE_DIR})
    message(STATUS "COPY and Set Arrow Header to: ${ARROW_BFS_INCLUDE_DIR}")
    file(COPY ${ARROW_BFS_INCLUDE_DIR}/arrow DESTINATION ${root_directory}/releases/include)
    file(COPY ${ARROW_BFS_INCLUDE_DIR}/gandiva DESTINATION ${root_directory}/releases/include)
    file(COPY ${ARROW_BFS_INCLUDE_DIR}/parquet DESTINATION ${root_directory}/releases/include)
  else()
    message(STATUS "COPY and Set Arrow Header to: ${ARROW_INCLUDE_DIR}")
    file(COPY ${ARROW_INCLUDE_DIR}/arrow DESTINATION ${root_directory}/releases/include)
    file(COPY ${ARROW_INCLUDE_DIR}/gandiva DESTINATION ${root_directory}/releases/include)
    file(COPY ${ARROW_INCLUDE_DIR}/parquet DESTINATION ${root_directory}/releases/include)
  endif()

  # Set up Arrow Shared Library Directory
  set(
    ARROW_SHARED_LIB
    "${root_directory}/releases/${CMAKE_SHARED_LIBRARY_PREFIX}${ARROW_LIB_NAME}${ARROW_SHARED_LIBRARY_SUFFIX}"
    )

  add_library(Arrow::arrow SHARED IMPORTED)
  set_target_properties(Arrow::arrow
                        PROPERTIES IMPORTED_LOCATION "${ARROW_SHARED_LIB}"
                                   INTERFACE_INCLUDE_DIRECTORIES
                                    "${root_directory}/releases/include")

  # Set up Gandiva Shared Library Directory
  set(
    GANDIVA_SHARED_LIB
    "${root_directory}/releases/${CMAKE_SHARED_LIBRARY_PREFIX}${GANDIVA_LIB_NAME}${ARROW_SHARED_LIBRARY_SUFFIX}"
    )
  add_library(Arrow::gandiva SHARED IMPORTED)
  set_target_properties(Arrow::gandiva
                        PROPERTIES IMPORTED_LOCATION "${GANDIVA_SHARED_LIB}"
                                   INTERFACE_INCLUDE_DIRECTORIES
                                    "${root_directory}/releases/include")

  target_link_libraries(spark_columnar_jni
                    LINK_PUBLIC Arrow::arrow Arrow::gandiva)

endmacro()

# Set up Proto
file(MAKE_DIRECTORY ${root_directory}/src/proto)
set(PROTO_OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/proto")
set(PROTO_OUTPUT_FILES "${PROTO_OUTPUT_DIR}/Exprs.pb.cc")
set(PROTO_OUTPUT_FILES ${PROTO_OUTPUT_FILES} "${PROTO_OUTPUT_DIR}/Exprs.pb.h")

set_source_files_properties(${PROTO_OUTPUT_FILES} PROPERTIES GENERATED TRUE)

get_filename_component(ABS_GANDIVA_PROTO ${CMAKE_CURRENT_SOURCE_DIR}/proto/Exprs.proto
                       ABSOLUTE)

set(PROTO_SRCS "${PROTO_OUTPUT_DIR}/Exprs.pb.cc")
set(PROTO_HDRS "${PROTO_OUTPUT_DIR}/Exprs.pb.h")

if(TESTS)
  find_package(GTest)
macro(package_add_test TESTNAME)
  add_executable(${TESTNAME} ${ARGN})
  target_link_libraries(${TESTNAME} gtest gtest_main spark_columnar_jni dl ${CMAKE_THREAD_LIBS_INIT})
  target_include_directories(${TESTNAME} PUBLIC ${source_root_directory})
  gtest_discover_tests(${TESTNAME}
    WORKING_DIRECTORY ${PROJECT_DIR}
    PROPERTIES VS_DEBUGGER_WORKING_DIRECTORY "${PROJECT_DIR}"
  )
  set_target_properties(${TESTNAME} PROPERTIES FOLDER tests)
endmacro()
  include(GoogleTest)
  ENABLE_TESTING()
  add_custom_target(test ${CMAKE_CTEST_COMMAND} -R TestArrowCompute --output-on-failure)
  add_subdirectory(tests)
endif()

if(BENCHMARKS)
  find_package(GTest)
  add_definitions(-DBENCHMARK_FILE_PATH="file://${CMAKE_CURRENT_SOURCE_DIR}/benchmarks/source_files/")
macro(package_add_benchmark TESTNAME)
  add_executable(${TESTNAME} ${ARGN})
  target_link_libraries(${TESTNAME} gtest gtest_main spark_columnar_jni parquet ${CMAKE_THREAD_LIBS_INIT})
  target_include_directories(${TESTNAME} PUBLIC ${source_root_directory})
  gtest_discover_tests(${TESTNAME}
    WORKING_DIRECTORY ${PROJECT_DIR}
    PROPERTIES VS_DEBUGGER_WORKING_DIRECTORY "${PROJECT_DIR}"
  )
  set_target_properties(${TESTNAME} PROPERTIES FOLDER tests)
endmacro()
  include(GoogleTest)
  ENABLE_TESTING()
  add_custom_target(benchmark ${CMAKE_CTEST_COMMAND} -R BenchmarkArrowCompute --output-on-failure)
  add_subdirectory(benchmarks)
endif()

set(CODEGEN_HEADERS
    third_party/
    )
set(THIRD_PARTY_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/third_party)
include_directories(${CMAKE_CURRENT_SOURCE_DIR} ${THIRD_PARTY_INCLUDE})
file(MAKE_DIRECTORY ${root_directory}/releases/include)
file(MAKE_DIRECTORY ${root_directory}/releases/include/codegen/common/)
file(MAKE_DIRECTORY ${root_directory}/releases/include/codegen/third_party/)
file(MAKE_DIRECTORY ${root_directory}/releases/include/codegen/precompile/)
file(MAKE_DIRECTORY ${root_directory}/releases/include/codegen/utils/)
file(MAKE_DIRECTORY ${root_directory}/releases/include/codegen/arrow_compute/ext/)
file(COPY third_party/ DESTINATION ${root_directory}/releases/include/)
file(COPY third_party/ DESTINATION ${root_directory}/releases/include/third_party/)
file(COPY precompile/ DESTINATION ${root_directory}/releases/include/precompile/)
file(COPY utils/ DESTINATION ${root_directory}/releases/include/utils/)
file(COPY codegen/arrow_compute/ext/array_item_index.h DESTINATION ${root_directory}/releases/include/codegen/arrow_compute/ext/)
file(COPY codegen/arrow_compute/ext/actions_impl.h DESTINATION ${root_directory}/releases/include/codegen/arrow_compute/ext/)
file(COPY codegen/arrow_compute/ext/code_generator_base.h DESTINATION ${root_directory}/releases/include/codegen/arrow_compute/ext/)
file(COPY codegen/arrow_compute/ext/kernels_ext.h DESTINATION ${root_directory}/releases/include/codegen/arrow_compute/ext/)
file(COPY codegen/common/result_iterator.h DESTINATION ${root_directory}/releases/include/codegen/common/)
file(COPY codegen/common/relation_column.h DESTINATION ${root_directory}/releases/include/codegen/common/)
file(COPY codegen/common/hash_relation.h DESTINATION ${root_directory}/releases/include/codegen/common/)
file(COPY codegen/common/sort_relation.h DESTINATION ${root_directory}/releases/include/codegen/common/)
file(COPY codegen/common/hash_relation_string.h DESTINATION ${root_directory}/releases/include/codegen/common/)
file(COPY codegen/common/hash_relation_number.h DESTINATION ${root_directory}/releases/include/codegen/common/)

add_definitions(-DNATIVESQL_SRC_PATH="${root_directory}/releases")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated-declarations -Wno-attributes")
set(SPARK_COLUMNAR_PLUGIN_SRCS
        jni/jni_wrapper.cc
        ${PROTO_SRCS}
        proto/protobuf_utils.cc
        codegen/common/relation.cc
        codegen/expr_visitor.cc
        codegen/arrow_compute/expr_visitor.cc
        codegen/arrow_compute/ext/hash_aggregate_kernel.cc
        codegen/arrow_compute/ext/probe_kernel.cc
        codegen/arrow_compute/ext/merge_join_kernel.cc
        codegen/arrow_compute/ext/codegen_common.cc
        codegen/arrow_compute/ext/codegen_node_visitor.cc
        codegen/arrow_compute/ext/codegen_register.cc
        codegen/arrow_compute/ext/actions_impl.cc
        codegen/arrow_compute/ext/whole_stage_codegen_kernel.cc
        codegen/arrow_compute/ext/hash_relation_kernel.cc
        codegen/arrow_compute/ext/conditioned_probe_kernel.cc
        codegen/arrow_compute/ext/conditioned_merge_join_kernel.cc
        codegen/arrow_compute/ext/basic_physical_kernels.cc
        codegen/arrow_compute/ext/expression_codegen_visitor.cc
        codegen/arrow_compute/ext/typed_node_visitor.cc
        codegen/arrow_compute/ext/window_kernel.cc
        codegen/arrow_compute/ext/sort_kernel.cc
        codegen/arrow_compute/ext/kernels_ext.cc
        shuffle/splitter.cc
        operators/unsafe_row_writer_and_reader.cc
        precompile/hash_map.cc
        precompile/sparse_hash_map.cc
        precompile/builder.cc
        precompile/array.cc
        precompile/type.cc
        precompile/sort.cc
        precompile/hash_arrays_kernel.cc
        precompile/unsafe_array.cc
        precompile/gandiva_projector.cc
        third_party/gandiva/decimal_ops.cc
        third_party/gandiva/time.cc
        )

add_subdirectory(third_party/gandiva)
message(STATUS "thirdparty is ${THIRDPARTY_GANDIVA_SRCS}")
file(MAKE_DIRECTORY ${root_directory}/releases)
add_library(spark_columnar_jni SHARED ${SPARK_COLUMNAR_PLUGIN_SRCS} ${THIRDPARTY_GANDIVA_SRCS})
add_dependencies(spark_columnar_jni jni_proto)

if(BUILD_PROTOBUF)
  build_protobuf()
  message(STATUS "Building ProtoBuf from Source: ${BUILD_PROTOBUF}")
  target_link_libraries(spark_columnar_jni
                        LINK_PRIVATE protobuf::libprotobuf)
else()
  find_protobuf()
  message(STATUS "Use existing ProtoBuf libraries: ${PROTOBUF_LIBRARY}")
  target_link_libraries(spark_columnar_jni
                        LINK_PUBLIC ${PROTOBUF_LIBRARY})
endif()

add_custom_command(OUTPUT ${PROTO_OUTPUT_FILES}
                   COMMAND ${PROTOC_BIN}
                           --proto_path
                           ${CMAKE_CURRENT_SOURCE_DIR}/proto
                           --cpp_out
                           ${PROTO_OUTPUT_DIR}
                           ${CMAKE_CURRENT_SOURCE_DIR}/proto/Exprs.proto
                   DEPENDS  ${ABS_GANDIVA_PROTO}
                   COMMENT "Running PROTO compiler on Exprs.proto"
                   VERBATIM)
add_custom_target(jni_proto ALL DEPENDS ${PROTO_OUTPUT_FILES})
add_dependencies(jni_proto protobuf::libprotobuf)
target_include_directories(spark_columnar_jni PUBLIC ${CMAKE_SYSTEM_INCLUDE_PATH} ${JNI_INCLUDE_DIRS} ${source_root_directory} ${PROTO_OUTPUT_DIR} ${PROTOBUF_INCLUDE})
set_target_properties(spark_columnar_jni PROPERTIES
                      LIBRARY_OUTPUT_DIRECTORY ${root_directory}/releases
)

# Build Arrow
#message(STATUS "Building ARROW from Source: ${BUILD_ARROW}")
if(BUILD_ARROW)
  build_arrow(${STATIC_ARROW})
  message(STATUS "Building Static ARROW: ${STATIC_ARROW}")
else() #
  find_arrow()
  message(STATUS "Use existing ARROW libraries")
endif()

if(DEFINED ENV{HADOOP_HOME})
  set(LIBHDFS3_DESTINATION $ENV{HADOOP_HOME}/lib/native)
else()
  set(LIBHDFS3_DESTINATION ${CMAKE_INSTALL_LIBDIR})
endif()

install(TARGETS spark_columnar_jni
        DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(FILES ${source_root_directory}/resources/libhdfs.so
        DESTINATION ${LIBHDFS3_DESTINATION})
install(FILES ${source_root_directory}/resources/libprotobuf.so.13
        DESTINATION ${CMAKE_INSTALL_LIBDIR})
