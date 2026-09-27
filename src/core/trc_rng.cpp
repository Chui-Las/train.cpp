// train.cpp - 伪随机数实现（splitmix64）+ 张量填充（支持延迟填充）
#include "traincpp/trc_rng.h"

#include "core/trc_impl.h"
#include "traincpp/trc_context.h"

#include <cmath>
#include <vector>

namespace traincpp {

namespace {

bool fill_ready(const Tensor* t) {
    return t->data != nullptr;
}

void check_fill_target(const Tensor* t) {
    TRC_ASSERT(t != nullptr, "rng_fill: 张量为空");
    TRC_ASSERT(t->type == TYPE_F32, "rng_fill: 仅支持 F32（当前 %s）", type_name(t->type));
    TRC_ASSERT(tensor_is_contiguous(t), "rng_fill: 张量必须连续");
}

// 直接写入已分配数据的张量（经 tensor_set，兼容后端 buffer 抽象）
void fill_values(Tensor* t, const std::vector<float>& values) {
    TRC_ASSERT(values.size() == (size_t) tensor_nelements(t), "rng_fill: 元素数不一致");
    tensor_set(t, values.data(), 0, values.size() * sizeof(float));
}

} // namespace

void rng_seed(Rng* rng, uint64_t seed) {
    TRC_ASSERT(rng != nullptr, "rng_seed: rng 为空");
    rng->state = seed;
}

uint64_t rng_next_u64(Rng* rng) {
    TRC_ASSERT(rng != nullptr, "rng_next_u64: rng 为空");
    // splitmix64
    rng->state += 0x9E3779B97F4A7C15ull;
    uint64_t z = rng->state;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

uint32_t rng_next_u32(Rng* rng) {
    return (uint32_t) (rng_next_u64(rng) >> 32);
}

float rng_uniform01(Rng* rng) {
    return (float) (rng_next_u32(rng) >> 8) * (1.0f / 16777216.0f);
}

float rng_uniform(Rng* rng, float lo, float hi) {
    return lo + (hi - lo) * rng_uniform01(rng);
}

float rng_normal(Rng* rng, float mean, float std) {
    float u1 = rng_uniform01(rng);
    const float u2 = rng_uniform01(rng);
    if (u1 < 1e-7f) {
        u1 = 1e-7f;
    }
    const float r = std::sqrt(-2.0f * std::log(u1));
    return mean + std * r * std::cos(6.28318530717958647692f * u2);
}

void rng_fill_constant(Context* ctx, Tensor* t, float value) {
    check_fill_target(t);
    if (fill_ready(t)) {
        std::vector<float> v((size_t) tensor_nelements(t), value);
        fill_values(t, v);
    } else {
        TRC_ASSERT(ctx != nullptr, "rng_fill_constant: 张量未分配数据且 ctx 为空");
        Context::PendingFill fill;
        fill.tensor = t;
        fill.kind   = Context::PendingFill::CONST;
        fill.a      = value;
        ctx->pending_fills.push_back(fill);
    }
}

void rng_fill_uniform(Context* ctx, Rng* rng, Tensor* t, float lo, float hi) {
    check_fill_target(t);
    if (fill_ready(t)) {
        std::vector<float> v((size_t) tensor_nelements(t));
        for (float& x : v) {
            x = rng_uniform(rng, lo, hi);
        }
        fill_values(t, v);
    } else {
        TRC_ASSERT(ctx != nullptr, "rng_fill_uniform: 张量未分配数据且 ctx 为空");
        Context::PendingFill fill;
        fill.tensor = t;
        fill.kind   = Context::PendingFill::UNIFORM;
        fill.a      = lo;
        fill.b      = hi;
        fill.seed   = rng_next_u64(rng);  // 派生独立种子：与该张量的填充位置解耦
        ctx->pending_fills.push_back(fill);
    }
}

void rng_fill_normal(Context* ctx, Rng* rng, Tensor* t, float mean, float std) {
    check_fill_target(t);
    if (fill_ready(t)) {
        std::vector<float> v((size_t) tensor_nelements(t));
        for (float& x : v) {
            x = rng_normal(rng, mean, std);
        }
        fill_values(t, v);
    } else {
        TRC_ASSERT(ctx != nullptr, "rng_fill_normal: 张量未分配数据且 ctx 为空");
        Context::PendingFill fill;
        fill.tensor = t;
        fill.kind   = Context::PendingFill::NORMAL;
        fill.a      = mean;
        fill.b      = std;
        fill.seed   = rng_next_u64(rng);
        ctx->pending_fills.push_back(fill);
    }
}

} // namespace traincpp
