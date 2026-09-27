// train.cpp - 伪随机数（自研实现，无第三方依赖）
//
// 用途：权重初始化（M1.5b）与 Dropout。算法为 splitmix64：确定性、跨平台一致、
// 不依赖标准库 <random> 的实现差异；仅用于训练侧，不参与 ggml 模型兼容。
#pragma once

#include "trc_tensor.h"

#include <cstdint>

namespace traincpp {

struct Context;

struct Rng {
    uint64_t state = 0x9E3779B97F4A7C15ull;
};

void rng_seed(Rng* rng, uint64_t seed);

// splitmix64 下一个 64/32 位随机数
uint64_t rng_next_u64(Rng* rng);
uint32_t rng_next_u32(Rng* rng);

// [0,1) 均匀分布（24 位尾数精度，float 可精确表示）
float rng_uniform01(Rng* rng);

// [lo,hi) 均匀分布
float rng_uniform(Rng* rng, float lo, float hi);

// 正态分布（Box-Muller，单次调用生成一个样本）
float rng_normal(Rng* rng, float mean, float std);

// 填充连续 F32 张量。数据尚未分配（no_alloc 模式）时，登记为延迟填充：
// 由 buffer_alloc_ctx_tensors 在分配完成后执行；每个张量从 rng 派生独立种子，
// 因此结果与调用顺序、分配时机无关且可复现。
void rng_fill_constant(Context* ctx, Tensor* t, float value);
void rng_fill_uniform(Context* ctx, Rng* rng, Tensor* t, float lo, float hi);
void rng_fill_normal(Context* ctx, Rng* rng, Tensor* t, float mean, float std);

} // namespace traincpp
