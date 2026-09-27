// train.cpp - 通用 C++ 深度学习训练库
// 数据类型定义：枚举值与 ggml v0.23.0 的 enum ggml_type 完全一致
// （参考 ggml v0.23.0 的 include/ggml.h），保证 GGUF 读写与 ggml 生态兼容。
#pragma once

#include <cstddef>
#include <cstdint>

namespace traincpp {

enum Type : int32_t {
    TYPE_F32     = 0,
    TYPE_F16     = 1,
    TYPE_Q4_0    = 2,
    TYPE_Q4_1    = 3,
    // 4/5 已被 ggml 移除，保留空位以对齐
    TYPE_Q5_0    = 6,
    TYPE_Q5_1    = 7,
    TYPE_Q8_0    = 8,
    TYPE_Q8_1    = 9,
    TYPE_Q2_K    = 10,
    TYPE_Q3_K    = 11,
    TYPE_Q4_K    = 12,
    TYPE_Q5_K    = 13,
    TYPE_Q6_K    = 14,
    TYPE_Q8_K    = 15,
    TYPE_IQ2_XXS = 16,
    TYPE_IQ2_XS  = 17,
    TYPE_IQ3_XXS = 18,
    TYPE_IQ1_S   = 19,
    TYPE_IQ4_NL  = 20,
    TYPE_IQ3_S   = 21,
    TYPE_IQ2_S   = 22,
    TYPE_IQ4_XS  = 23,
    TYPE_I8      = 24,
    TYPE_I16     = 25,
    TYPE_I32     = 26,
    TYPE_I64     = 27,
    TYPE_F64     = 28,
    TYPE_IQ1_M   = 29,
    TYPE_BF16    = 30,
    TYPE_TQ1_0   = 34,
    TYPE_TQ2_0   = 35,
    TYPE_MXFP4   = 39,
    TYPE_NVFP4   = 40,
    TYPE_Q1_0    = 41,
    TYPE_Q2_0    = 42,
    TYPE_COUNT   = 43,
};

// 块大小（每个 block 包含的元素个数）；未实现类型返回 0
int64_t type_blck_size(Type type);

// 单块字节数；未实现类型返回 0
size_t type_size(Type type);

// 每个元素的平均字节数（type_size / blck_size）
double type_sizef(Type type);

// 一行 ne 个元素占用的字节数；要求 ne 能被块大小整除
size_t type_row_size(Type type, int64_t ne);

// 类型名（与 ggml 的 ggml_type_name 输出一致）
const char* type_name(Type type);

// 是否为量化类型
bool type_is_quantized(Type type);

// 是否为可分配/可计算的基本类型（一期：F32/F16/BF16/F64/I8/I16/I32/I64）
bool type_is_supported(Type type);

} // namespace traincpp
