# traincpp - 通用 C++ 深度学习训练库
cmake_minimum_required(VERSION 3.20)

project(traincpp
    VERSION 0.1.0
    DESCRIPTION "通用 C++ 深度学习训练库（ggml 生态互操作）"
    LANGUAGES CXX)

# ------------------------------------------------------------------ 选项
option(TRC_CPU            "启用 CPU 参考后端"          ON)
option(TRC_VULKAN         "启用 Vulkan 后端"            ON)
option(TRC_BUILD_TESTS    "构建测试"                    ON)
option(TRC_BUILD_EXAMPLES "构建示例"                    ON)
option(TRC_INSTALL        "生成安装规则与 CMake 包配置（供其他程序 find_package）" ON)

include(GNUInstallDirs)

# ------------------------------------------------------------------ Vulkan 探测
# 找不到 SDK 时自动关闭（可显式 -DTRC_VULKAN=OFF 避免该探测）
if(TRC_VULKAN)
    find_package(Vulkan QUIET COMPONENTS glslc)
    if(NOT Vulkan_FOUND)
        message(WARNING "未找到 Vulkan SDK（VULKAN_SDK=${VULKAN_SDK}），TRC_VULKAN 自动关闭；"
                        "如需启用请安装 Vulkan SDK 或设置 VULKAN_SDK 环境变量")
        set(TRC_VULKAN OFF)
    else()
        message(STATUS "Vulkan SDK: ${Vulkan_INCLUDE_DIR}（glslc: ${Vulkan_GLSLC_EXECUTABLE}）")
    endif()
endif()

# ------------------------------------------------------------------ 全局设置
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(MSVC)
    add_compile_options(/utf-8 /W4)
else()
    add_compile_options(-Wall -Wextra)
endif()

# 统一输出目录（单配置生成器）
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
# 多配置生成器（Visual Studio）
foreach(cfg IN ITEMS DEBUG RELEASE RELWITHDEBINFO MINSIZEREL)
    string(TOUPPER "${cfg}" CFG)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_${CFG} ${CMAKE_BINARY_DIR}/bin)
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_${CFG} ${CMAKE_BINARY_DIR}/lib)
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_${CFG} ${CMAKE_BINARY_DIR}/lib)
endforeach()

# ------------------------------------------------------------------ 核心库
add_library(traincpp_core STATIC
    src/core/trc_impl.cpp
    src/core/trc_types.cpp
    src/core/trc_context.cpp
    src/core/trc_tensor.cpp
    src/core/trc_op.cpp
    src/core/trc_graph.cpp
    src/core/trc_backend_reg.cpp
    src/core/trc_autograd.cpp
    src/core/trc_rng.cpp
    src/ops/trc_ops.cpp
    src/nn/trc_nn.cpp
    src/loss/trc_loss.cpp
    src/optim/trc_optim.cpp
    src/io/trc_gguf.cpp
    src/io/trc_checkpoint.cpp
)
target_include_directories(traincpp_core
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src
)
# 公共接口要求 C++20：以 target 属性传播（而非全局 CMAKE_CXX_STANDARD），避免影响使用方其它目标
target_compile_features(traincpp_core PUBLIC cxx_std_20)
target_compile_definitions(traincpp_core PUBLIC
    TRC_USE_CPU=$<BOOL:${TRC_CPU}>
    TRC_USE_VULKAN=$<BOOL:${TRC_VULKAN}>)
if(MSVC)
    # 公共头包含中文注释：消费方（其他程序）也必须启用 /utf-8，否则可能出现解析问题
    target_compile_options(traincpp_core INTERFACE /utf-8)
endif()

# ------------------------------------------------------------------ 后端
set(TRC_BACKEND_TARGETS "")
if(TRC_CPU)
    add_subdirectory(src/backends/cpu)
    list(APPEND TRC_BACKEND_TARGETS traincpp_cpu)
endif()
if(TRC_VULKAN)
    add_subdirectory(src/backends/vulkan)
    list(APPEND TRC_BACKEND_TARGETS traincpp_vulkan)
endif()

# ------------------------------------------------------------------ 聚合目标
add_library(traincpp INTERFACE)
target_link_libraries(traincpp INTERFACE ${TRC_BACKEND_TARGETS})
# 便于 add_subdirectory 的消费者使用带命名空间的目标名
add_library(traincpp::core ALIAS traincpp_core)
add_library(traincpp::traincpp ALIAS traincpp)
foreach(backend_target IN LISTS TRC_BACKEND_TARGETS)
    string(REPLACE "traincpp_" "" backend_name "${backend_target}")
    add_library(traincpp::${backend_name} ALIAS ${backend_target})
endforeach()

# ------------------------------------------------------------------ 测试/示例
if(TRC_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
if(TRC_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

# ------------------------------------------------------------------ 安装与 CMake 包配置
# 目的：让本库能被其他程序以标准方式消费：
#   find_package(traincpp CONFIG REQUIRED)
#   target_link_libraries(app PRIVATE traincpp::traincpp)
if(TRC_INSTALL)
    include(CMakePackageConfigHelpers)

    set(TRC_EXPORT_TARGETS traincpp_core traincpp)
    list(APPEND TRC_EXPORT_TARGETS ${TRC_BACKEND_TARGETS})

    install(TARGETS ${TRC_EXPORT_TARGETS}
        EXPORT traincppTargets
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

    install(DIRECTORY include/traincpp DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

    set(TRC_CMAKE_INSTALL_DIR ${CMAKE_INSTALL_LIBDIR}/cmake/traincpp)
    install(EXPORT traincppTargets
        NAMESPACE traincpp::
        FILE traincppTargets.cmake
        DESTINATION ${TRC_CMAKE_INSTALL_DIR})

    configure_package_config_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/cmake/traincpp-config.cmake.in
        ${CMAKE_CURRENT_BINARY_DIR}/traincpp-config.cmake
        INSTALL_DESTINATION ${TRC_CMAKE_INSTALL_DIR})
    write_basic_package_version_file(
        ${CMAKE_CURRENT_BINARY_DIR}/traincpp-config-version.cmake
        VERSION ${PROJECT_VERSION}
        COMPATIBILITY SameMajorVersion)
    install(FILES
        ${CMAKE_CURRENT_BINARY_DIR}/traincpp-config.cmake
        ${CMAKE_CURRENT_BINARY_DIR}/traincpp-config-version.cmake
        DESTINATION ${TRC_CMAKE_INSTALL_DIR})
endif()

message(STATUS "traincpp 配置完成: CPU=${TRC_CPU} VULKAN=${TRC_VULKAN} TESTS=${TRC_BUILD_TESTS} INSTALL=${TRC_INSTALL}")
