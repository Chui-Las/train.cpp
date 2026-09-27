# train.cpp - 用 ggml 本体验证导出的 GGUF（M1.6c）
#
# 独立构建（不参与主构建；在参考项目源码之外构建，不修改 ggml 源码）：
#   powershell -ExecutionPolicy Bypass -File scripts\ggml_crosscheck.ps1
#
# 或手动：
#   cmake -S tools/ggml_crosscheck -B build/ggml_crosscheck -G Ninja ^
#         -DTRC_GGML_SOURCE=<ggml 源码路径> -DTRC_TRAINCPP_SOURCE=<train.cpp 根目录>
cmake_minimum_required(VERSION 3.20)
project(trc_ggml_crosscheck C CXX ASM)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

if(NOT TRC_GGML_SOURCE)
    message(FATAL_ERROR "需要 -DTRC_GGML_SOURCE=<ggml 源码路径>（只读使用，不修改）")
endif()
if(NOT EXISTS ${TRC_GGML_SOURCE}/include/gguf.h)
    message(FATAL_ERROR "TRC_GGML_SOURCE 下未找到 include/gguf.h: ${TRC_GGML_SOURCE}")
endif()
if(NOT TRC_TRAINCPP_SOURCE)
    get_filename_component(TRC_TRAINCPP_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
endif()

# ggml：仅 CPU，关闭测试/示例/其它后端
set(GGML_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GGML_VULKAN         OFF CACHE BOOL "" FORCE)
set(GGML_CCACHE         OFF CACHE BOOL "" FORCE)
add_subdirectory(${TRC_GGML_SOURCE} ggml-src)

# train.cpp：仅核心（无 Vulkan/测试/示例）
set(TRC_CPU            ON  CACHE BOOL "" FORCE)
set(TRC_VULKAN         OFF CACHE BOOL "" FORCE)
set(TRC_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
set(TRC_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(TRC_INSTALL        OFF CACHE BOOL "" FORCE)
add_subdirectory(${TRC_TRAINCPP_SOURCE} traincpp-src)

add_executable(ggml_crosscheck main.cpp)
target_link_libraries(ggml_crosscheck PRIVATE ggml traincpp::traincpp)
if(MSVC)
    target_compile_options(ggml_crosscheck PRIVATE /utf-8)
endif()
