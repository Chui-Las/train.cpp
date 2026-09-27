// train.cpp - 浮点类型转换（内部使用）
// fp16 转换算法与 ggml-impl.h / ggml.c 保持一致，保证与 ggml 数值一致。
#pragma once

#include <cmath>
#include <cstdint>
#include <cstring>

namespace traincpp {

inline float fp32_from_bits(uint32_t w) {
    float f;
    std::memcpy(&f, &w, sizeof(f));
    return f;
}

inline uint32_t fp32_to_bits(float f) {
    uint32_t w;
    std::memcpy(&w, &f, sizeof(w));
    return w;
}

inline float fp16_to_fp32(uint16_t h) {
    const uint32_t w          = (uint32_t) h << 16;
    const uint32_t sign       = w & UINT32_C(0x80000000);
    const uint32_t two_w      = w + w;

    const uint32_t exp_offset = UINT32_C(0xE0) << 23;
    const float    exp_scale  = 0x1.0p-112f;
    const float    normalized_value = fp32_from_bits((two_w >> 4) + exp_offset) * exp_scale;

    const uint32_t magic_mask        = UINT32_C(126) << 23;  // 必须为 0x3F000000（=0.5f）；曾误写 <<22 使零/次正规数解码为 -0.5
    const float    magic_bias        = 0.5f;
    const float    denormalized_value = fp32_from_bits((two_w >> 17) | magic_mask) - magic_bias;

    const uint32_t denormalized_cutoff = UINT32_C(1) << 27;
    const uint32_t result = sign | (two_w < denormalized_cutoff ? fp32_to_bits(denormalized_value)
                                                                : fp32_to_bits(normalized_value));
    return fp32_from_bits(result);
}

inline uint16_t fp32_to_fp16(float f) {
    const float    scale_to_inf  = 0x1.0p+112f;
    const float    scale_to_zero = 0x1.0p-110f;
    float          base          = (fabsf(f) * scale_to_inf) * scale_to_zero;

    const uint32_t w      = fp32_to_bits(f);
    const uint32_t shl1_w = w + w;
    const uint32_t sign   = w & UINT32_C(0x80000000);
    uint32_t       bias   = shl1_w & UINT32_C(0xFF000000);
    if (bias < UINT32_C(0x71000000)) {
        bias = UINT32_C(0x71000000);
    }

    base = fp32_from_bits((bias >> 1) + UINT32_C(0x07800000)) + base;
    const uint32_t bits          = fp32_to_bits(base);
    const uint32_t exp_bits      = (bits >> 13) & UINT32_C(0x00007C00);
    const uint32_t mantissa_bits = bits & UINT32_C(0x00000FFF);
    const uint32_t nonsign       = exp_bits + mantissa_bits;
    return (uint16_t) ((sign >> 16) | (nonsign >= UINT32_C(0x7F800000)
                                            ? UINT32_C(0x7C00) | ((nonsign >> 16) & UINT32_C(0x03FF))
                                            : nonsign));
}

inline float bf16_to_fp32(uint16_t h) {
    return fp32_from_bits((uint32_t) h << 16);
}

inline uint16_t fp32_to_bf16(float f) {
    uint32_t w = fp32_to_bits(f);
    // 四舍五入到最近偶数
    w += 0x7FFF + ((w >> 16) & 1);
    return (uint16_t) (w >> 16);
}

} // namespace traincpp
