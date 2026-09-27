# train.cpp - Vulkan 后端（M1.4）
add_library(traincpp_vulkan STATIC
    trc_vulkan.cpp
    trc_vulkan_ops.cpp
)
target_link_libraries(traincpp_vulkan PUBLIC traincpp_core Vulkan::Vulkan)
target_include_directories(traincpp_vulkan PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_BINARY_DIR}   # vulkan_shaders_gen.h
    ${PROJECT_SOURCE_DIR}/src
)

# ---------------------------------------------------------------- shader 编译与嵌入
set(TRC_VK_SHADER_DIR ${CMAKE_CURRENT_SOURCE_DIR}/shaders)
set(TRC_VK_SPV_DIR ${CMAKE_CURRENT_BINARY_DIR}/spv)
file(MAKE_DIRECTORY ${TRC_VK_SPV_DIR})

file(GLOB TRC_VK_SHADERS CONFIGURE_DEPENDS ${TRC_VK_SHADER_DIR}/*.comp)
set(TRC_VK_SPV_FILES "")
foreach(shader IN LISTS TRC_VK_SHADERS)
    get_filename_component(name ${shader} NAME_WE)
    set(spv ${TRC_VK_SPV_DIR}/${name}.spv)
    add_custom_command(
        OUTPUT ${spv}
        COMMAND ${Vulkan_GLSLC_EXECUTABLE} -fshader-stage=compute --target-env=vulkan1.2 -O
                "${shader}" -o "${spv}"
        DEPENDS "${shader}"
        COMMENT "glslc ${name}.comp"
        VERBATIM)
    list(APPEND TRC_VK_SPV_FILES ${spv})
endforeach()

set(TRC_VK_SHADER_HEADER ${CMAKE_CURRENT_BINARY_DIR}/vulkan_shaders_gen.h)
add_custom_command(
    OUTPUT ${TRC_VK_SHADER_HEADER}
    COMMAND ${CMAKE_COMMAND}
            -DSPV_DIR=${TRC_VK_SPV_DIR}
            -DOUT_HEADER=${TRC_VK_SHADER_HEADER}
            -P ${PROJECT_SOURCE_DIR}/cmake/embed_spirv.cmake
    DEPENDS ${TRC_VK_SPV_FILES} ${PROJECT_SOURCE_DIR}/cmake/embed_spirv.cmake
    COMMENT "嵌入 SPIR-V 到 vulkan_shaders_gen.h"
    VERBATIM)

target_sources(traincpp_vulkan PRIVATE ${TRC_VK_SHADER_HEADER})
