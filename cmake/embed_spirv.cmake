# 将目录下的 .spv 转换为 C++ 头（uint32_t 数组 + 名称表）
#
# 用法（脚本模式）：
#   cmake -DSPV_DIR=<spv 目录> -DOUT_HEADER=<输出头> -P cmake/embed_spirv.cmake
#
# 生成内容（命名空间 traincpp::vk_shaders）：
#   static const uint32_t spv_<name>[] = {...};
#   struct SpirvBinary { const char* name; const uint32_t* words; size_t word_count; };
#   static const SpirvBinary kAll[] = {...};
# 注意：SPIR-V 文件为小端字节序，这里按 4 字节组装成 uint32_t 字面量，避免对齐/端序问题。

file(GLOB spv_files "${SPV_DIR}/*.spv")
list(SORT spv_files)
if(NOT spv_files)
    message(FATAL_ERROR "embed_spirv: ${SPV_DIR} 下没有 .spv 文件")
endif()

set(body "// 由 cmake/embed_spirv.cmake 生成，请勿手改\n")
string(APPEND body "#pragma once\n")
string(APPEND body "#include <cstddef>\n#include <cstdint>\n\n")
string(APPEND body "namespace traincpp { namespace vk_shaders {\n\n")

set(entries "")
foreach(f IN LISTS spv_files)
    get_filename_component(name "${f}" NAME_WE)
    string(REPLACE "-" "_" sym "${name}")
    file(READ "${f}" hex HEX)
    string(LENGTH "${hex}" hex_len)
    math(EXPR n_words "${hex_len} / 8")

    string(APPEND body "alignas(4) static const uint32_t spv_${sym}[] = {\n")
    set(line "    ")
    set(written 0)
    math(EXPR last "${n_words} - 1")
    foreach(i RANGE ${last})
        math(EXPR off "${i} * 8")
        string(SUBSTRING "${hex}" ${off} 8 w)
        string(SUBSTRING "${w}" 0 2 b0)
        string(SUBSTRING "${w}" 2 2 b1)
        string(SUBSTRING "${w}" 4 2 b2)
        string(SUBSTRING "${w}" 6 2 b3)
        string(APPEND line "0x${b3}${b2}${b1}${b0}u, ")
        math(EXPR mod "${i} % 12")
        if(mod EQUAL 11)
            string(APPEND body "${line}\n")
            set(line "    ")
        endif()
    endforeach()
    string(APPEND body "${line}\n};\n\n")

    string(APPEND entries "    {\"${name}\", spv_${sym}, sizeof(spv_${sym}) / sizeof(uint32_t)},\n")
endforeach()

string(APPEND body "struct SpirvBinary {\n    const char* name;\n    const uint32_t* words;\n    size_t word_count;\n};\n\n")
string(APPEND body "static const SpirvBinary kAll[] = {\n${entries}};\n\n")
string(APPEND body "} } // namespace traincpp::vk_shaders\n")

file(WRITE "${OUT_HEADER}" "${body}")
message(STATUS "embed_spirv: 已生成 ${OUT_HEADER}")
