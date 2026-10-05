// train.cpp - CPU 参考算子实现
// 一期仅实现 f32 计算；所有 kernel 通用路径基于 nb 步长，支持视图/广播。
#include "trc_cpu.h"

#include "core/trc_float.h"
#include "core/trc_impl.h"

#include <cmath>
#include <cfloat>
#include <cstring>

namespace traincpp {

namespace {

// ---------------------------------------------------------------- 辅助
// 按目标形状 shape 的线性下标 i 计算张量 t 的字节偏移（ggml 广播语义：t 的维度为 1 或等于 shape）
size_t offset_broadcast(const Tensor* t, const Tensor* shape, int64_t i) {
    size_t offset = 0;
    for (int d = 0; d < MAX_DIMS; ++d) {
        const int64_t idx = i % shape->ne[d];
        i /= shape->ne[d];
        if (t->ne[d] != 1) {
            offset += (size_t) idx * t->nb[d];
        }
    }
    return offset;
}

inline float load_f32(const Tensor* t, size_t offset) {
    return *(const float*) ((const uint8_t*) t->data + offset);
}

inline void store_f32(Tensor* t, size_t offset, float value) {
    *(float*) ((uint8_t*) t->data + offset) = value;
}

// 四维下标 -> 字节偏移
inline size_t elem_offset(const Tensor* t, int64_t i0, int64_t i1, int64_t i2, int64_t i3) {
    return (size_t) i0 * t->nb[0] + (size_t) i1 * t->nb[1] + (size_t) i2 * t->nb[2] +
           (size_t) i3 * t->nb[3];
}

// ---------------------------------------------------------------- 二元
using BinaryFn = float (*)(float, float);

float fn_add(float a, float b) { return a + b; }
float fn_sub(float a, float b) { return a - b; }
float fn_mul(float a, float b) { return a * b; }
float fn_div(float a, float b) { return a / b; }

void compute_binary(Tensor* dst, BinaryFn fn) {
    const Tensor* a = dst->src[0];
    const Tensor* b = dst->src[1];
    TRC_ASSERT(dst->type == TYPE_F32 && a->type == TYPE_F32 && b->type == TYPE_F32,
               "CPU 二元算子一期仅支持 f32 (op=%s)", op_name(dst->op));

    const int64_t n = tensor_nelements(dst);

    // 快路径：形状相同且连续
    if (tensor_are_same_shape(a, b) && tensor_is_contiguous(a) && tensor_is_contiguous(b) &&
        tensor_is_contiguous(dst)) {
        const float* pa = (const float*) a->data;
        const float* pb = (const float*) b->data;
        float*       pd = (float*) dst->data;
        for (int64_t i = 0; i < n; ++i) {
            pd[i] = fn(pa[i], pb[i]);
        }
        return;
    }

    // 通用路径（广播 + 非连续/视图）
    for (int64_t i = 0; i < n; ++i) {
        const size_t oa = tensor_offset_linear(a, i);
        const size_t ob = offset_broadcast(b, dst, i);
        const size_t od = tensor_offset_linear(dst, i);
        store_f32(dst, od, fn(load_f32(a, oa), load_f32(b, ob)));
    }
}

// ---------------------------------------------------------------- 一元
using UnaryFn = float (*)(float);

float fn_neg(float x) { return -x; }
float fn_abs(float x) { return std::fabs(x); }
float fn_sqr(float x) { return x * x; }
float fn_sqrt(float x) { return std::sqrt(x); }
float fn_exp(float x) { return std::exp(x); }
float fn_log(float x) { return std::log(x); }
float fn_sin(float x) { return std::sin(x); }
float fn_cos(float x) { return std::cos(x); }
float fn_relu(float x) { return x > 0.0f ? x : 0.0f; }
float fn_sgn(float x) { return x > 0.0f ? 1.0f : (x < 0.0f ? -1.0f : 0.0f); }
float fn_step(float x) { return x > 0.0f ? 1.0f : 0.0f; }
float fn_sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }
float fn_tanh(float x) { return std::tanh(x); }
float fn_silu(float x) { return x / (1.0f + std::exp(-x)); }

// 与 ggml 一致：0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715*x^3)))
float fn_gelu(float x) {
    return 0.5f * x * (1.0f + std::tanh(0.7978845608028654f * (x + 0.044715f * x * x * x)));
}

float fn_gelu_erf(float x) {
    return 0.5f * x * (1.0f + std::erf(x / std::sqrt(2.0f)));
}

float fn_erf(float x) { return std::erf(x); }

// 与 ggml 一致：x > 20 时直接返回 x，避免 exp 溢出
float fn_softplus(float x) {
    return x > 20.0f ? x : std::log1p(std::exp(x));
}

float fn_hardswish(float x) {
    return x * std::min(std::max(x + 3.0f, 0.0f), 6.0f) / 6.0f;
}

void compute_unary(Tensor* dst, UnaryFn fn) {
    const Tensor* a = dst->src[0];
    TRC_ASSERT(dst->type == TYPE_F32 && a->type == TYPE_F32, "CPU 一元算子一期仅支持 f32 (op=%s)",
               op_name(dst->op));

    const int64_t n = tensor_nelements(dst);
    if (tensor_is_contiguous(a) && tensor_is_contiguous(dst)) {
        const float* pa = (const float*) a->data;
        float*       pd = (float*) dst->data;
        for (int64_t i = 0; i < n; ++i) {
            pd[i] = fn(pa[i]);
        }
        return;
    }
    for (int64_t i = 0; i < n; ++i) {
        store_f32(dst, tensor_offset_linear(dst, i), fn(load_f32(a, tensor_offset_linear(a, i))));
    }
}

// ---------------------------------------------------------------- 带参数一元
void compute_scale(Tensor* dst) {
    const Tensor* a = dst->src[0];
    TRC_ASSERT(dst->type == TYPE_F32 && a->type == TYPE_F32, "CPU scale 一期仅支持 f32");
    float s = 0.0f;
    std::memcpy(&s, dst->op_params, sizeof(float));

    const int64_t n = tensor_nelements(dst);
    for (int64_t i = 0; i < n; ++i) {
        store_f32(dst, tensor_offset_linear(dst, i), load_f32(a, tensor_offset_linear(a, i)) * s);
    }
}

void compute_leaky_relu(Tensor* dst) {
    const Tensor* a = dst->src[0];
    TRC_ASSERT(dst->type == TYPE_F32 && a->type == TYPE_F32, "CPU leaky_relu 一期仅支持 f32");
    float slope = 0.0f;
    std::memcpy(&slope, dst->op_params, sizeof(float));

    const int64_t n = tensor_nelements(dst);
    for (int64_t i = 0; i < n; ++i) {
        const float x = load_f32(a, tensor_offset_linear(a, i));
        store_f32(dst, tensor_offset_linear(dst, i), x > 0.0f ? x : slope * x);
    }
}

void compute_clamp(Tensor* dst) {
    const Tensor* a = dst->src[0];
    TRC_ASSERT(dst->type == TYPE_F32 && a->type == TYPE_F32, "CPU clamp 一期仅支持 f32");
    float min_val = 0.0f;
    float max_val = 0.0f;
    std::memcpy(&min_val, dst->op_params, sizeof(float));
    std::memcpy(&max_val, (const uint8_t*) dst->op_params + sizeof(float), sizeof(float));

    const int64_t n = tensor_nelements(dst);
    for (int64_t i = 0; i < n; ++i) {
        const float x = load_f32(a, tensor_offset_linear(a, i));
        store_f32(dst, tensor_offset_linear(dst, i), std::min(std::max(x, min_val), max_val));
    }
}

// ---------------------------------------------------------------- mul_mat
// ggml_mul_mat 语义：a [k, m, a2, a3] * b [k, n, b2, b3] -> [m, n, b2, b3]
// 平面 (i2, i3) 使用 a 的平面 (i2 % a2, i3 % a3)（a 的 batch 广播）
void compute_mul_mat(Tensor* dst) {
    const Tensor* a = dst->src[0];
    const Tensor* b = dst->src[1];
    TRC_ASSERT(dst->type == TYPE_F32 && a->type == TYPE_F32 && b->type == TYPE_F32,
               "CPU mul_mat 一期仅支持 f32");

    const int64_t k = a->ne[0];
    const int64_t m = a->ne[1];
    const int64_t n = b->ne[1];
    TRC_ASSERT(dst->ne[0] == m && dst->ne[1] == n && dst->ne[2] == b->ne[2] && dst->ne[3] == b->ne[3],
               "mul_mat: 输出形状非法");

    for (int64_t i3 = 0; i3 < b->ne[3]; ++i3) {
        const size_t oa3 = (size_t) (i3 % a->ne[3]) * a->nb[3];
        const size_t ob3 = (size_t) i3 * b->nb[3];
        const size_t od3 = (size_t) i3 * dst->nb[3];
        for (int64_t i2 = 0; i2 < b->ne[2]; ++i2) {
            const size_t oa = (size_t) (i2 % a->ne[2]) * a->nb[2] + oa3;
            const size_t ob = (size_t) i2 * b->nb[2] + ob3;
            const size_t od = (size_t) i2 * dst->nb[2] + od3;
            for (int64_t j = 0; j < n; ++j) {
                for (int64_t i = 0; i < m; ++i) {
                    float sum = 0.0f;
                    for (int64_t kk = 0; kk < k; ++kk) {
                        const size_t oa_k = oa + (size_t) i * a->nb[1] + (size_t) kk * a->nb[0];
                        const size_t ob_k = ob + (size_t) j * b->nb[1] + (size_t) kk * b->nb[0];
                        sum += load_f32(a, oa_k) * load_f32(b, ob_k);
                    }
                    const size_t od_ij = od + (size_t) i * dst->nb[0] + (size_t) j * dst->nb[1];
                    store_f32(dst, od_ij, sum);
                }
            }
        }
    }
}

// ---------------------------------------------------------------- dup / cont
void compute_dup(Tensor* dst) {
    const Tensor* a = dst->src[0];
    TRC_ASSERT(a->type == dst->type, "dup: 类型必须一致");
    TRC_ASSERT(tensor_are_same_shape(a, dst), "dup: 形状必须一致");

    if (tensor_is_contiguous(a) && tensor_is_contiguous(dst)) {
        std::memcpy(dst->data, a->data, tensor_nbytes(dst));
        return;
    }

    // 非连续：按元素字节宽度复制（一期支持基本类型）
    const size_t elem = type_size(dst->type);
    const int64_t n = tensor_nelements(dst);
    for (int64_t i = 0; i < n; ++i) {
        std::memcpy((uint8_t*) dst->data + tensor_offset_linear(dst, i),
                    (const uint8_t*) a->data + tensor_offset_linear(a, i), elem);
    }
}

// ---------------------------------------------------------------- cast
double load_as_double(const Tensor* t, size_t offset) {
    const uint8_t* p = (const uint8_t*) t->data + offset;
    switch (t->type) {
        case TYPE_F32:  return *(const float*) p;
        case TYPE_F16:  return fp16_to_fp32(*(const uint16_t*) p);
        case TYPE_BF16: return bf16_to_fp32(*(const uint16_t*) p);
        case TYPE_F64:  return *(const double*) p;
        case TYPE_I8:   return *(const int8_t*) p;
        case TYPE_I16:  return *(const int16_t*) p;
        case TYPE_I32:  return *(const int32_t*) p;
        case TYPE_I64:  return (double) *(const int64_t*) p;
        default:
            TRC_ABORT("cast: 源类型 %s 不支持", type_name(t->type));
    }
}

void store_from_double(Tensor* t, size_t offset, double value) {
    uint8_t* p = (uint8_t*) t->data + offset;
    switch (t->type) {
        case TYPE_F32:  *(float*) p = (float) value; break;
        case TYPE_F16:  *(uint16_t*) p = fp32_to_fp16((float) value); break;
        case TYPE_BF16: *(uint16_t*) p = fp32_to_bf16((float) value); break;
        case TYPE_F64:  *(double*) p = value; break;
        case TYPE_I8:   *(int8_t*) p = (int8_t) value; break;
        case TYPE_I16:  *(int16_t*) p = (int16_t) value; break;
        case TYPE_I32:  *(int32_t*) p = (int32_t) value; break;
        case TYPE_I64:  *(int64_t*) p = (int64_t) value; break;
        default:
            TRC_ABORT("cast: 目标类型 %s 不支持", type_name(t->type));
    }
}

void compute_cast(Tensor* dst) {
    const Tensor* a = dst->src[0];
    TRC_ASSERT(type_is_supported(a->type) && type_is_supported(dst->type), "cast: 类型不支持");

    const int64_t n = tensor_nelements(dst);
    for (int64_t i = 0; i < n; ++i) {
        store_from_double(dst, tensor_offset_linear(dst, i),
                          load_as_double(a, tensor_offset_linear(a, i)));
    }
}

// ---------------------------------------------------------------- 归约
// 行内（ne0 方向）求和，返回 double（与 ggml 的 ggml_float 累加一致）
double row_sum_f32(const Tensor* t, int64_t i1, int64_t i2, int64_t i3) {
    const uint8_t* base = (const uint8_t*) t->data + (size_t) i1 * t->nb[1] + (size_t) i2 * t->nb[2] +
                          (size_t) i3 * t->nb[3];
    const int64_t ne0 = t->ne[0];
    double sum = 0.0;
    if (t->nb[0] == sizeof(float)) {
        const float* p = (const float*) base;
        for (int64_t i0 = 0; i0 < ne0; ++i0) {
            sum += p[i0];
        }
    } else {
        for (int64_t i0 = 0; i0 < ne0; ++i0) {
            sum += *(const float*) (base + (size_t) i0 * t->nb[0]);
        }
    }
    return sum;
}

void compute_sum(Tensor* dst) {
    const Tensor* a = dst->src[0];
    TRC_ASSERT(a->type == TYPE_F32 && dst->type == TYPE_F32, "CPU sum 一期仅支持 f32");

    double sum = 0.0;
    for (int64_t i3 = 0; i3 < a->ne[3]; ++i3) {
        for (int64_t i2 = 0; i2 < a->ne[2]; ++i2) {
            for (int64_t i1 = 0; i1 < a->ne[1]; ++i1) {
                sum += row_sum_f32(a, i1, i2, i3);
            }
        }
    }
    *(float*) dst->data = (float) sum;
}

void compute_sum_rows_or_mean(Tensor* dst, bool take_mean) {
    const Tensor* a = dst->src[0];
    TRC_ASSERT(a->type == TYPE_F32 && dst->type == TYPE_F32, "CPU 归约一期仅支持 f32");

    for (int64_t i3 = 0; i3 < a->ne[3]; ++i3) {
        for (int64_t i2 = 0; i2 < a->ne[2]; ++i2) {
            for (int64_t i1 = 0; i1 < a->ne[1]; ++i1) {
                const double sum = row_sum_f32(a, i1, i2, i3);
                const float value = take_mean ? (float) (sum / a->ne[0]) : (float) sum;
                const size_t offset = (size_t) i1 * dst->nb[1] + (size_t) i2 * dst->nb[2] +
                                      (size_t) i3 * dst->nb[3];
                *(float*) ((uint8_t*) dst->data + offset) = value;
            }
        }
    }
}

// ---------------------------------------------------------------- 归一化
void compute_norm_impl(Tensor* dst, bool rms) {
    const Tensor* a = dst->src[0];
    TRC_ASSERT(a->type == TYPE_F32 && dst->type == TYPE_F32, "CPU 归一化一期仅支持 f32");
    TRC_ASSERT(tensor_are_same_shape(a, dst), "归一化要求输出与输入同形状");

    float eps = 0.0f;
    std::memcpy(&eps, dst->op_params, sizeof(float));

    const int64_t ne0 = a->ne[0];
    for (int64_t i3 = 0; i3 < a->ne[3]; ++i3) {
        for (int64_t i2 = 0; i2 < a->ne[2]; ++i2) {
            for (int64_t i1 = 0; i1 < a->ne[1]; ++i1) {
                const uint8_t* xbase = (const uint8_t*) a->data + (size_t) i1 * a->nb[1] +
                                       (size_t) i2 * a->nb[2] + (size_t) i3 * a->nb[3];
                uint8_t* ybase = (uint8_t*) dst->data + (size_t) i1 * dst->nb[1] +
                                 (size_t) i2 * dst->nb[2] + (size_t) i3 * dst->nb[3];

                // 快路径：行内连续
                if (a->nb[0] == sizeof(float) && dst->nb[0] == sizeof(float)) {
                    const float* x = (const float*) xbase;
                    float*       y = (float*) ybase;
                    if (rms) {
                        double ss = 0.0;
                        for (int64_t i0 = 0; i0 < ne0; ++i0) {
                            ss += (double) x[i0] * x[i0];
                        }
                        const float scale = 1.0f / std::sqrt((float) (ss / ne0) + eps);
                        for (int64_t i0 = 0; i0 < ne0; ++i0) {
                            y[i0] = x[i0] * scale;
                        }
                    } else {
                        double sum = 0.0;
                        for (int64_t i0 = 0; i0 < ne0; ++i0) {
                            sum += x[i0];
                        }
                        const float mean = (float) (sum / ne0);
                        double var = 0.0;
                        for (int64_t i0 = 0; i0 < ne0; ++i0) {
                            const double d = (double) x[i0] - mean;
                            var += d * d;
                        }
                        const float scale = 1.0f / std::sqrt((float) (var / ne0) + eps);
                        for (int64_t i0 = 0; i0 < ne0; ++i0) {
                            y[i0] = (float) (((double) x[i0] - mean) * scale);
                        }
                    }
                } else {
                    double sum = 0.0;
                    for (int64_t i0 = 0; i0 < ne0; ++i0) {
                        sum += load_f32(a, elem_offset(a, i0, i1, i2, i3));
                    }
                    const float mean = (float) (sum / ne0);
                    double var = 0.0;
                    for (int64_t i0 = 0; i0 < ne0; ++i0) {
                        const double x = load_f32(a, elem_offset(a, i0, i1, i2, i3));
                        const double d = rms ? x : (x - mean);
                        var += d * d;
                    }
                    const float scale = 1.0f / std::sqrt((float) (var / ne0) + eps);
                    for (int64_t i0 = 0; i0 < ne0; ++i0) {
                        const double x = load_f32(a, elem_offset(a, i0, i1, i2, i3));
                        store_f32(dst, elem_offset(dst, i0, i1, i2, i3),
                                  rms ? (float) (x * scale) : (float) ((x - mean) * scale));
                    }
                }
            }
        }
    }
}

void compute_group_norm(Tensor* dst) {
    const Tensor* a = dst->src[0];
    TRC_ASSERT(a->type == TYPE_F32 && dst->type == TYPE_F32, "CPU group_norm 一期仅支持 f32");
    TRC_ASSERT(tensor_are_same_shape(a, dst), "group_norm 要求输出与输入同形状");

    float eps = 0.0f;
    std::memcpy(&eps, (const uint8_t*) dst->op_params + sizeof(float), sizeof(float));

    const int n_channels = (int) a->ne[2];
    const int n_groups = dst->op_params[0];
    const int per_group = (n_channels + n_groups - 1) / n_groups;
    const double count = (double) a->ne[0] * a->ne[1];

    // 与 ggml 一致：通道维是 ne2，组在 ne2 上划分，对每组的 (i0, i1, i2) 统计
    for (int64_t i3 = 0; i3 < a->ne[3]; ++i3) {
        for (int g = 0; g < n_groups; ++g) {
            const int start = g * per_group;
            const int end = std::min(start + per_group, n_channels);
            const int step = end - start;
            if (step <= 0) {
                continue;
            }

            double sum = 0.0;
            for (int i2 = start; i2 < end; ++i2) {
                for (int64_t i1 = 0; i1 < a->ne[1]; ++i1) {
                    for (int64_t i0 = 0; i0 < a->ne[0]; ++i0) {
                        sum += load_f32(a, elem_offset(a, i0, i1, i2, i3));
                    }
                }
            }
            const float mean = (float) (sum / (count * step));

            double var = 0.0;
            for (int i2 = start; i2 < end; ++i2) {
                for (int64_t i1 = 0; i1 < a->ne[1]; ++i1) {
                    for (int64_t i0 = 0; i0 < a->ne[0]; ++i0) {
                        const double d = (double) load_f32(a, elem_offset(a, i0, i1, i2, i3)) - mean;
                        var += d * d;
                        store_f32(dst, elem_offset(dst, i0, i1, i2, i3), (float) d);
                    }
                }
            }
            const float scale = 1.0f / std::sqrt((float) (var / (count * step)) + eps);

            for (int i2 = start; i2 < end; ++i2) {
                for (int64_t i1 = 0; i1 < a->ne[1]; ++i1) {
                    for (int64_t i0 = 0; i0 < a->ne[0]; ++i0) {
                        const size_t od = elem_offset(dst, i0, i1, i2, i3);
                        store_f32(dst, od, load_f32(dst, od) * scale);
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------- 归一化反向
// LayerNorm 反向：dx = rstd*(dy - mean(dy) - xc*rstd^2*mean(dy*xc))，逐 (i1,i2,i3) 行
void compute_norm_back(Tensor* dst) {
    const Tensor* dy = dst->src[0];
    const Tensor* x  = dst->src[1];
    TRC_ASSERT(dy->type == TYPE_F32 && x->type == TYPE_F32 && dst->type == TYPE_F32,
               "CPU norm_back 一期仅支持 f32");
    TRC_ASSERT(tensor_are_same_shape(dy, x) && tensor_are_same_shape(dst, x),
               "norm_back: 形状必须一致");

    float eps = 0.0f;
    std::memcpy(&eps, dst->op_params, sizeof(float));

    const int64_t n = x->ne[0];
    for (int64_t i3 = 0; i3 < x->ne[3]; ++i3) {
        for (int64_t i2 = 0; i2 < x->ne[2]; ++i2) {
            for (int64_t i1 = 0; i1 < x->ne[1]; ++i1) {
                double sum = 0.0;
                for (int64_t i0 = 0; i0 < n; ++i0) {
                    sum += load_f32(x, elem_offset(x, i0, i1, i2, i3));
                }
                const double mean = sum / (double) n;

                double var = 0.0;
                for (int64_t i0 = 0; i0 < n; ++i0) {
                    const double d = (double) load_f32(x, elem_offset(x, i0, i1, i2, i3)) - mean;
                    var += d * d;
                }
                const double rstd = 1.0 / std::sqrt(var / (double) n + (double) eps);

                double sum_dy = 0.0;
                double sum_dyxc = 0.0;
                for (int64_t i0 = 0; i0 < n; ++i0) {
                    const double d  = (double) load_f32(x, elem_offset(x, i0, i1, i2, i3)) - mean;
                    const double g  = (double) load_f32(dy, elem_offset(dy, i0, i1, i2, i3));
                    sum_dy += g;
                    sum_dyxc += g * d;
                }

                for (int64_t i0 = 0; i0 < n; ++i0) {
                    const double d = (double) load_f32(x, elem_offset(x, i0, i1, i2, i3)) - mean;
                    const double g = (double) load_f32(dy, elem_offset(dy, i0, i1, i2, i3));
                    const double dx = rstd * (g - sum_dy / (double) n - d * rstd * rstd * sum_dyxc / (double) n);
                    store_f32(dst, elem_offset(dst, i0, i1, i2, i3), (float) dx);
                }
            }
        }
    }
}

// RMSNorm 反向（与 ggml_compute_forward_rms_norm_back_f32 的公式一致）：
//   dx = (dz + x * (-sum_xdz / (sum_xx + eps*n))) * (1/sqrt(sum_xx/n + eps))
void compute_rms_norm_back(Tensor* dst) {
    const Tensor* dz = dst->src[0];
    const Tensor* x  = dst->src[1];
    TRC_ASSERT(dz->type == TYPE_F32 && x->type == TYPE_F32 && dst->type == TYPE_F32,
               "CPU rms_norm_back 一期仅支持 f32");
    TRC_ASSERT(tensor_are_same_shape(dz, x) && tensor_are_same_shape(dst, x),
               "rms_norm_back: 形状必须一致");

    float eps = 0.0f;
    std::memcpy(&eps, dst->op_params, sizeof(float));

    const int64_t n = x->ne[0];
    for (int64_t i3 = 0; i3 < x->ne[3]; ++i3) {
        for (int64_t i2 = 0; i2 < x->ne[2]; ++i2) {
            for (int64_t i1 = 0; i1 < x->ne[1]; ++i1) {
                double sum_xx  = 0.0;
                double sum_xdz = 0.0;
                for (int64_t i0 = 0; i0 < n; ++i0) {
                    const double xv = load_f32(x, elem_offset(x, i0, i1, i2, i3));
                    const double g  = load_f32(dz, elem_offset(dz, i0, i1, i2, i3));
                    sum_xx += xv * xv;
                    sum_xdz += xv * g;
                }
                const double mean_eps = sum_xx / (double) n + (double) eps;
                const float  rrms     = (float) (1.0 / std::sqrt(mean_eps));
                const float  scale_x  = (float) (-sum_xdz / (sum_xx + (double) eps * (double) n));

                for (int64_t i0 = 0; i0 < n; ++i0) {
                    const float xv = load_f32(x, elem_offset(x, i0, i1, i2, i3));
                    const float g  = load_f32(dz, elem_offset(dz, i0, i1, i2, i3));
                    store_f32(dst, elem_offset(dst, i0, i1, i2, i3), (g + xv * scale_x) * rrms);
                }
            }
        }
    }
}

// GroupNorm 反向：对每个 (i3, 组) 在 (i0,i1) 上统计后套用 LayerNorm 公式
void compute_group_norm_back(Tensor* dst) {
    const Tensor* dy = dst->src[0];
    const Tensor* x  = dst->src[1];
    TRC_ASSERT(dy->type == TYPE_F32 && x->type == TYPE_F32 && dst->type == TYPE_F32,
               "CPU group_norm_back 一期仅支持 f32");
    TRC_ASSERT(tensor_are_same_shape(dy, x) && tensor_are_same_shape(dst, x),
               "group_norm_back: 形状必须一致");

    float eps = 0.0f;
    std::memcpy(&eps, (const uint8_t*) dst->op_params + sizeof(float), sizeof(float));

    const int n_channels = (int) x->ne[2];
    const int n_groups   = dst->op_params[0];
    const int per_group  = (n_channels + n_groups - 1) / n_groups;

    for (int64_t i3 = 0; i3 < x->ne[3]; ++i3) {
        for (int g = 0; g < n_groups; ++g) {
            const int c0 = g * per_group;
            const int c1 = std::min(c0 + per_group, n_channels);
            const int step = c1 - c0;
            if (step <= 0) {
                continue;
            }
            const double count = (double) x->ne[0] * x->ne[1] * step;

            double sum = 0.0;
            for (int i2 = c0; i2 < c1; ++i2) {
                for (int64_t i1 = 0; i1 < x->ne[1]; ++i1) {
                    for (int64_t i0 = 0; i0 < x->ne[0]; ++i0) {
                        sum += load_f32(x, elem_offset(x, i0, i1, i2, i3));
                    }
                }
            }
            const double mean = sum / count;

            double var = 0.0;
            for (int i2 = c0; i2 < c1; ++i2) {
                for (int64_t i1 = 0; i1 < x->ne[1]; ++i1) {
                    for (int64_t i0 = 0; i0 < x->ne[0]; ++i0) {
                        const double d = (double) load_f32(x, elem_offset(x, i0, i1, i2, i3)) - mean;
                        var += d * d;
                    }
                }
            }
            const double rstd = 1.0 / std::sqrt(var / count + (double) eps);

            double sum_dy = 0.0;
            double sum_dyxc = 0.0;
            for (int i2 = c0; i2 < c1; ++i2) {
                for (int64_t i1 = 0; i1 < x->ne[1]; ++i1) {
                    for (int64_t i0 = 0; i0 < x->ne[0]; ++i0) {
                        const double d = (double) load_f32(x, elem_offset(x, i0, i1, i2, i3)) - mean;
                        const double gv = (double) load_f32(dy, elem_offset(dy, i0, i1, i2, i3));
                        sum_dy += gv;
                        sum_dyxc += gv * d;
                    }
                }
            }

            for (int i2 = c0; i2 < c1; ++i2) {
                for (int64_t i1 = 0; i1 < x->ne[1]; ++i1) {
                    for (int64_t i0 = 0; i0 < x->ne[0]; ++i0) {
                        const double d = (double) load_f32(x, elem_offset(x, i0, i1, i2, i3)) - mean;
                        const double gv = (double) load_f32(dy, elem_offset(dy, i0, i1, i2, i3));
                        const double dx = rstd * (gv - sum_dy / count - d * rstd * rstd * sum_dyxc / count);
                        store_f32(dst, elem_offset(dst, i0, i1, i2, i3), (float) dx);
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------- soft_max
void compute_soft_max(Tensor* dst) {
    const Tensor* a = dst->src[0];
    const Tensor* mask = dst->src[1];
    TRC_ASSERT(a->type == TYPE_F32 && dst->type == TYPE_F32, "CPU soft_max 一期仅支持 f32");
    TRC_ASSERT(tensor_are_same_shape(a, dst), "soft_max 要求输出与输入同形状");
    TRC_ASSERT(tensor_is_contiguous(a) && tensor_is_contiguous(dst), "soft_max: 输入/输出必须连续");

    float scale = 1.0f;
    float max_bias = 0.0f;
    std::memcpy(&scale, (const float*) dst->op_params + 0, sizeof(float));
    std::memcpy(&max_bias, (const float*) dst->op_params + 1, sizeof(float));

    const int64_t ne0 = a->ne[0];
    const int64_t ne12 = mask ? mask->ne[2] : 1;
    const int64_t ne13 = mask ? mask->ne[3] : 1;

    // ALiBi 斜率参数（与 ggml 一致）
    const uint32_t n_head = (uint32_t) a->ne[2];
    const uint32_t n_head_log2 =
        n_head > 0 ? (1u << (uint32_t) std::floor(std::log2((double) n_head))) : 1u;
    const float m0 = std::pow(2.0f, -(max_bias) / (float) n_head_log2);
    const float m1 = std::pow(2.0f, -(max_bias / 2.0f) / (float) n_head_log2);

    for (int64_t i3 = 0; i3 < a->ne[3]; ++i3) {
        for (int64_t i2 = 0; i2 < a->ne[2]; ++i2) {
            for (int64_t i1 = 0; i1 < a->ne[1]; ++i1) {
                const float* x = (const float*) ((const uint8_t*) a->data + (size_t) i1 * a->nb[1] +
                                                 (size_t) i2 * a->nb[2] + (size_t) i3 * a->nb[3]);
                float* y = (float*) ((uint8_t*) dst->data + (size_t) i1 * dst->nb[1] +
                                     (size_t) i2 * dst->nb[2] + (size_t) i3 * dst->nb[3]);

                const uint8_t* mrow = nullptr;
                if (mask != nullptr) {
                    const int64_t i12 = i2 % ne12;
                    const int64_t i13 = i3 % ne13;
                    mrow = (const uint8_t*) mask->data + (size_t) i1 * mask->nb[1] +
                           (size_t) i12 * mask->nb[2] + (size_t) i13 * mask->nb[3];
                }

                float slope = 1.0f;
                if (max_bias > 0.0f) {
                    const uint32_t h = (uint32_t) i2;
                    slope = h < n_head_log2 ? std::pow(m0, (float) (h + 1))
                                            : std::pow(m1, (float) (2 * (h - n_head_log2) + 1));
                }

                // 合并 scale 与 mask（与 ggml 一致：x*scale + slope*mask）
                for (int64_t i0 = 0; i0 < ne0; ++i0) {
                    float v = x[i0] * scale;
                    if (mrow != nullptr) {
                        if (mask->type == TYPE_F32) {
                            v += slope * ((const float*) mrow)[i0];
                        } else {
                            v += slope * fp16_to_fp32(((const uint16_t*) mrow)[i0]);
                        }
                    }
                    y[i0] = v;
                }

                // 数值稳定 softmax
                float max_val = -INFINITY;
                for (int64_t i0 = 0; i0 < ne0; ++i0) {
                    max_val = y[i0] > max_val ? y[i0] : max_val;
                }
                double sum = 0.0;
                for (int64_t i0 = 0; i0 < ne0; ++i0) {
                    const float e = std::exp(y[i0] - max_val);
                    y[i0] = e;
                    sum += e;
                }
                const float inv = (float) (1.0 / sum);
                for (int64_t i0 = 0; i0 < ne0; ++i0) {
                    y[i0] *= inv;
                }
            }
        }
    }
}

// ---------------------------------------------------------------- 交叉熵损失
void compute_cross_entropy_loss(Tensor* dst) {
    const Tensor* logits = dst->src[0];
    const Tensor* target = dst->src[1];
    TRC_ASSERT(logits->type == TYPE_F32 && target->type == TYPE_F32 && dst->type == TYPE_F32,
               "CPU cross_entropy_loss 一期仅支持 f32");
    TRC_ASSERT(tensor_are_same_shape(logits, target), "cross_entropy_loss: 两个输入必须同形状");
    TRC_ASSERT(logits->nb[0] == sizeof(float) && target->nb[0] == sizeof(float),
               "cross_entropy_loss: 最内维必须连续（与 ggml 一致）");

    const int64_t nc = logits->ne[0];
    const int64_t nr = logits->ne[1] * logits->ne[2] * logits->ne[3];
    TRC_ASSERT(nc > 0 && nr > 0, "cross_entropy_loss: 输入不能为空");

    // 与 ggml 相同的逐行数值稳定 log_softmax（CPU 沿用 double 累加）
    double total = 0.0;
    for (int64_t i3 = 0; i3 < logits->ne[3]; ++i3) {
        for (int64_t i2 = 0; i2 < logits->ne[2]; ++i2) {
            for (int64_t i1 = 0; i1 < logits->ne[1]; ++i1) {
                const float* s0 =
                    (const float*) ((const uint8_t*) logits->data + elem_offset(logits, 0, i1, i2, i3));
                const float* s1 =
                    (const float*) ((const uint8_t*) target->data + elem_offset(target, 0, i1, i2, i3));

                float max_val = -INFINITY;
                for (int64_t i0 = 0; i0 < nc; ++i0) {
                    max_val = s0[i0] > max_val ? s0[i0] : max_val;
                }
                double sum_exp = 0.0;
                for (int64_t i0 = 0; i0 < nc; ++i0) {
                    sum_exp += std::exp((double) s0[i0] - (double) max_val);
                }
                const double lse = (double) max_val + std::log(sum_exp);
                double row = 0.0;
                for (int64_t i0 = 0; i0 < nc; ++i0) {
                    row += (double) s1[i0] * ((double) s0[i0] - lse);
                }
                total += row;
            }
        }
    }
    *(float*) dst->data = (float) (-total / (double) nr);
}

void compute_cross_entropy_loss_back(Tensor* dst) {
    const Tensor* grad = dst->src[0];
    const Tensor* logits = dst->src[1];
    const Tensor* target = dst->src[2];
    TRC_ASSERT(logits->type == TYPE_F32 && target->type == TYPE_F32 && grad->type == TYPE_F32 &&
                   dst->type == TYPE_F32,
               "CPU cross_entropy_loss_back 一期仅支持 f32");
    TRC_ASSERT(tensor_nelements(grad) == 1, "cross_entropy_loss_back: grad 必须是标量");
    TRC_ASSERT(tensor_are_same_shape(logits, target), "cross_entropy_loss_back: 两个输入必须同形状");
    TRC_ASSERT(logits->nb[0] == sizeof(float) && target->nb[0] == sizeof(float) &&
                   dst->nb[0] == sizeof(float),
               "cross_entropy_loss_back: 最内维必须连续（与 ggml 一致）");

    const int64_t nc = logits->ne[0];
    const int64_t nr = logits->ne[1] * logits->ne[2] * logits->ne[3];
    TRC_ASSERT(nc > 0 && nr > 0, "cross_entropy_loss_back: 输入不能为空");
    const float d_by_nr = ((const float*) grad->data)[0] / (float) nr;

    // grad(logits) = (softmax(logits) - target) * dloss / nr（与 ggml 公式一致）
    for (int64_t i3 = 0; i3 < logits->ne[3]; ++i3) {
        for (int64_t i2 = 0; i2 < logits->ne[2]; ++i2) {
            for (int64_t i1 = 0; i1 < logits->ne[1]; ++i1) {
                const float* s0 =
                    (const float*) ((const uint8_t*) logits->data + elem_offset(logits, 0, i1, i2, i3));
                const float* s1 =
                    (const float*) ((const uint8_t*) target->data + elem_offset(target, 0, i1, i2, i3));
                float* ds0 = (float*) ((uint8_t*) dst->data + elem_offset(dst, 0, i1, i2, i3));

                float max_val = -INFINITY;
                for (int64_t i0 = 0; i0 < nc; ++i0) {
                    max_val = s0[i0] > max_val ? s0[i0] : max_val;
                }
                double sum_exp = 0.0;
                for (int64_t i0 = 0; i0 < nc; ++i0) {
                    sum_exp += std::exp((double) s0[i0] - (double) max_val);
                }
                const float inv = (float) (1.0 / sum_exp);
                for (int64_t i0 = 0; i0 < nc; ++i0) {
                    const float soft = (float) (std::exp((double) s0[i0] - (double) max_val) * inv);
                    ds0[i0] = (soft - s1[i0]) * d_by_nr;
                }
            }
        }
    }
}

// ---------------------------------------------------------------- 索引
int64_t load_index(const Tensor* t, size_t offset) {
    if (t->type == TYPE_I32) {
        return (int64_t) * (const int32_t*) ((const uint8_t*) t->data + offset);
    }
    if (t->type == TYPE_I64) {
        return *(const int64_t*) ((const uint8_t*) t->data + offset);
    }
    TRC_ABORT("内部错误：索引类型 %s（应为 I32/I64）", type_name(t->type));
}

void compute_get_rows(Tensor* dst) {
    const Tensor* table = dst->src[0];
    const Tensor* idx = dst->src[1];
    TRC_ASSERT(table->type == TYPE_F32 || table->type == TYPE_F16, "CPU get_rows 一期表仅支持 f32/f16");
    TRC_ASSERT(dst->type == TYPE_F32, "CPU get_rows 一期输出仅支持 f32");
    TRC_ASSERT(idx->ne[3] == 1, "get_rows: 索引 ne[3] 必须为 1");

    const int64_t nr = tensor_nelements(idx);
    const int64_t plane = idx->ne[1] * idx->ne[0];

    for (int64_t i = 0; i < nr; ++i) {
        const int64_t i12 = i / plane;
        const int64_t i11 = (i - i12 * plane) / idx->ne[0];
        const int64_t i10 = i - i12 * plane - i11 * idx->ne[0];

        const int64_t row = load_index(idx, elem_offset(idx, i10, i11, i12, 0));
        TRC_ASSERT(row >= 0 && row < table->ne[1], "get_rows: 索引越界 %lld（词表大小 %lld）",
                   (long long) row, (long long) table->ne[1]);

        const uint8_t* srow = (const uint8_t*) table->data + (size_t) row * table->nb[1] +
                              (size_t) i11 * table->nb[2] + (size_t) i12 * table->nb[3];
        uint8_t* drow = (uint8_t*) dst->data + (size_t) i10 * dst->nb[1] + (size_t) i11 * dst->nb[2] +
                        (size_t) i12 * dst->nb[3];

        // 表行内必须按 nb[0] 读取（表可能是转置/非连续视图；输出按 dst->nb[0] 写）
        for (int64_t i0 = 0; i0 < table->ne[0]; ++i0) {
            const uint8_t* sp = srow + (size_t) i0 * table->nb[0];
            uint8_t*       dp = drow + (size_t) i0 * dst->nb[0];
            float          v  = 0.0f;
            if (table->type == TYPE_F16) {
                uint16_t h = 0;
                std::memcpy(&h, sp, sizeof(uint16_t));
                v = fp16_to_fp32(h);
            } else {
                std::memcpy(&v, sp, sizeof(float));
            }
            std::memcpy(dp, &v, sizeof(float));
        }
    }
}

// get_rows 的反向：先清零，再按索引把梯度行 scatter-add 回表
void compute_get_rows_back(Tensor* dst) {
    const Tensor* grad = dst->src[0];
    const Tensor* idx  = dst->src[1];
    TRC_ASSERT(grad->type == TYPE_F32 && dst->type == TYPE_F32 && idx->type == TYPE_I32,
               "CPU get_rows_back 一期仅支持 f32 + i32");
    TRC_ASSERT(tensor_is_contiguous(dst), "get_rows_back: 输出必须连续");
    TRC_ASSERT(idx->ne[3] == 1, "get_rows_back: 索引 ne[3] 必须为 1");

    std::memset(dst->data, 0, tensor_nbytes(dst));

    const int64_t nr    = tensor_nelements(idx);
    const int64_t plane = idx->ne[1] * idx->ne[0];
    for (int64_t i = 0; i < nr; ++i) {
        const int64_t i12 = i / plane;
        const int64_t i11 = (i - i12 * plane) / idx->ne[0];
        const int64_t i10 = i - i12 * plane - i11 * idx->ne[0];

        const int64_t row = load_index(idx, elem_offset(idx, i10, i11, i12, 0));
        TRC_ASSERT(row >= 0 && row < dst->ne[1], "get_rows_back: 索引越界 %lld（词表大小 %lld）",
                   (long long) row, (long long) dst->ne[1]);

        const size_t doff = (size_t) row * dst->nb[1] + (size_t) i11 * dst->nb[2] +
                            (size_t) i12 * dst->nb[3];
        const size_t goff = (size_t) i10 * grad->nb[1] + (size_t) i11 * grad->nb[2] +
                            (size_t) i12 * grad->nb[3];
        for (int64_t i0 = 0; i0 < grad->ne[0]; ++i0) {
            const size_t od = doff + (size_t) i0 * dst->nb[0];
            store_f32(dst, od, load_f32(dst, od) + load_f32(grad, goff + (size_t) i0 * grad->nb[0]));
        }
    }
}

void compute_set_rows(Tensor* dst) {
    const Tensor* values  = dst->src[0];
    const Tensor* indices = dst->src[1];
    Tensor*       target  = dst->src[2];  // 被写入的张量（与 dst 共享数据）

    TRC_ASSERT(values->type == TYPE_F32 || values->type == TYPE_F16, "CPU set_rows 源仅支持 f32/f16");
    TRC_ASSERT(indices->type == TYPE_I32 || indices->type == TYPE_I64, "set_rows: 索引类型非法");

    const int64_t nr = values->ne[1];
    for (int64_t i3 = 0; i3 < values->ne[3]; ++i3) {
        for (int64_t i2 = 0; i2 < values->ne[2]; ++i2) {
            const int64_t i11 = i2 % indices->ne[1];
            const int64_t i12 = i3 % indices->ne[2];
            for (int64_t i = 0; i < nr; ++i) {
                const int64_t row = load_index(indices, elem_offset(indices, i, i11, i12, 0));
                TRC_ASSERT(row >= 0 && row < target->ne[1], "set_rows: 索引越界 %lld（目标行数 %lld）",
                           (long long) row, (long long) target->ne[1]);

                const size_t soff = (size_t) i * values->nb[1] + (size_t) i2 * values->nb[2] +
                                    (size_t) i3 * values->nb[3];
                const size_t doff = (size_t) row * target->nb[1] + (size_t) i2 * target->nb[2] +
                                    (size_t) i3 * target->nb[3];

                for (int64_t i0 = 0; i0 < values->ne[0]; ++i0) {
                    store_from_double(target, doff + (size_t) i0 * target->nb[0],
                                      load_as_double(values, soff + (size_t) i0 * values->nb[0]));
                }
            }
        }
    }
}

// ---------------------------------------------------------------- 形状扩展
void compute_repeat(Tensor* dst) {
    const Tensor* a = dst->src[0];
    const int64_t n = tensor_nelements(dst);

    for (int64_t i = 0; i < n; ++i) {
        int64_t rem = i;
        size_t  oa  = 0;
        for (int d = 0; d < MAX_DIMS; ++d) {
            const int64_t idx = rem % dst->ne[d];
            rem /= dst->ne[d];
            const int64_t ia = a->ne[d] == 1 ? 0 : idx % a->ne[d];
            oa += (size_t) ia * a->nb[d];
        }
        store_from_double(dst, tensor_offset_linear(dst, i), load_as_double(a, oa));
    }
}

// repeat 的反向：沿被重复的维度求和回 b 的形状（a 可重复到 dst 的形状）
// 实现为 scatter-add：遍历 a 的每个元素，按 a->ne % dst->ne 归并到 dst
void compute_repeat_back(Tensor* dst) {
    const Tensor* a = dst->src[0];
    TRC_ASSERT(dst->type == TYPE_F32 && a->type == TYPE_F32, "CPU repeat_back 一期仅支持 f32");
    TRC_ASSERT(tensor_is_contiguous(dst), "repeat_back: 输出必须连续");

    std::memset(dst->data, 0, tensor_nbytes(dst));

    const int64_t n = tensor_nelements(a);
    for (int64_t i = 0; i < n; ++i) {
        int64_t rem = i;
        size_t  od  = 0;
        for (int d = 0; d < MAX_DIMS; ++d) {
            const int64_t idx = rem % a->ne[d];
            rem /= a->ne[d];
            od += (size_t) (idx % dst->ne[d]) * dst->nb[d];
        }
        const size_t oa = tensor_offset_linear(a, i);
        store_f32(dst, od, load_f32(dst, od) + load_f32(a, oa));
    }
}

// acc：把 b 累加到 a 的指定步长区域；inplace=false 时先整体拷贝 a
void compute_acc(Tensor* dst) {
    const Tensor* a = dst->src[0];
    const Tensor* b = dst->src[1];
    TRC_ASSERT(dst->type == TYPE_F32 && a->type == TYPE_F32 && b->type == TYPE_F32,
               "CPU acc 一期仅支持 f32");
    TRC_ASSERT(tensor_are_same_shape(a, dst), "acc: dst 必须与 a 同形状");
    TRC_ASSERT(tensor_is_contiguous(a) && tensor_is_contiguous(dst), "acc: a/dst 必须连续");

    int32_t nb1i = 0;
    int32_t nb2i = 0;
    int32_t nb3i = 0;
    int32_t offseti = 0;
    int32_t inplace_i = 0;
    std::memcpy(&nb1i, (const uint8_t*) dst->op_params + 0, sizeof(int32_t));
    std::memcpy(&nb2i, (const uint8_t*) dst->op_params + 4, sizeof(int32_t));
    std::memcpy(&nb3i, (const uint8_t*) dst->op_params + 8, sizeof(int32_t));
    std::memcpy(&offseti, (const uint8_t*) dst->op_params + 12, sizeof(int32_t));
    std::memcpy(&inplace_i, (const uint8_t*) dst->op_params + 16, sizeof(int32_t));
    const size_t nb1 = (size_t) nb1i;
    const size_t nb2 = (size_t) nb2i;
    const size_t nb3 = (size_t) nb3i;
    const size_t offset = (size_t) offseti;
    const bool inplace = inplace_i != 0;

    if (!inplace) {
        std::memcpy(dst->data, a->data, tensor_nbytes(dst));
    }

    // inplace 时 dst 是 a 的完整视图（data = a->data，offset=0），区域偏移按 op_params 累加
    const size_t base = offset;
    const size_t nb0 = sizeof(float);
    const int64_t n = tensor_nelements(b);
    for (int64_t i = 0; i < n; ++i) {
        int64_t rem = i;
        int64_t idx[MAX_DIMS];
        for (int d = 0; d < MAX_DIMS; ++d) {
            idx[d] = rem % b->ne[d];
            rem /= b->ne[d];
        }
        const size_t od = base + (size_t) idx[0] * nb0 + (size_t) idx[1] * nb1 +
                          (size_t) idx[2] * nb2 + (size_t) idx[3] * nb3;
        const size_t ob = tensor_offset_linear(b, i);
        store_f32(dst, od, load_f32(dst, od) + load_f32(b, ob));
    }
}

void compute_concat(Tensor* dst) {
    const Tensor* a = dst->src[0];
    const Tensor* b = dst->src[1];
    const int dim = dst->op_params[0];
    const int64_t n = tensor_nelements(dst);

    for (int64_t i = 0; i < n; ++i) {
        int64_t rem = i;
        int64_t idx[MAX_DIMS];
        for (int d = 0; d < MAX_DIMS; ++d) {
            idx[d] = rem % dst->ne[d];
            rem /= dst->ne[d];
        }

        const bool    from_b = idx[dim] >= a->ne[dim];
        const Tensor* src = from_b ? b : a;

        size_t os = 0;
        for (int d = 0; d < MAX_DIMS; ++d) {
            const int64_t sidx = (from_b && d == dim) ? idx[d] - a->ne[dim] : idx[d];
            os += (size_t) sidx * src->nb[d];
        }
        store_from_double(dst, tensor_offset_linear(dst, i), load_as_double(src, os));
    }
}

void compute_pad(Tensor* dst) {
    const Tensor* a = dst->src[0];
    const int32_t* p = dst->op_params;  // lp0,rp0,lp1,rp1,lp2,rp2,lp3,rp3；p[8]=mode
    const int     mode = p[8];
    const int64_t n = tensor_nelements(dst);

    for (int64_t i = 0; i < n; ++i) {
        int64_t rem = i;
        size_t  oa  = 0;
        bool    inside = true;
        for (int d = 0; d < MAX_DIMS; ++d) {
            const int64_t idx = rem % dst->ne[d];
            rem /= dst->ne[d];
            int64_t ia = idx - p[2 * d];
            if (mode == PAD_REFLECT) {
                // 镜像反射（不重复边缘；pads < ne，映射后必落在 [0, ne)）
                if (ia < 0) {
                    ia = -ia;
                } else if (ia >= a->ne[d]) {
                    ia = 2 * a->ne[d] - 2 - ia;
                }
            } else if (ia < 0 || ia >= a->ne[d]) {
                inside = false;
                break;
            }
            oa += (size_t) ia * a->nb[d];
        }
        store_from_double(dst, tensor_offset_linear(dst, i), inside ? load_as_double(a, oa) : 0.0);
    }
}

// reflect 反向（M2.3c）：dX[j] = Σ_{i: reflect(i-lp)=j} dY[i]
// 每维候选输出位置：中心 i=j+lp（恒有效）；左折叠 i=lp-j（1<=j<=lp）；
// 右折叠 i=lp+2*n-2-j（n-1-rp<=j<=n-2）；各路笛卡尔积（至多 3^4=81 项）
void compute_pad_back(Tensor* dst) {
    const Tensor* g     = dst->src[0];  // 前向输出的梯度
    const int32_t* p    = dst->op_params;
    TRC_ASSERT(dst->type == TYPE_F32 && g->type == TYPE_F32, "CPU pad_back 仅支持 f32");

    std::memset(dst->data, 0, tensor_nbytes(dst));

    const int64_t n = tensor_nelements(dst);
    for (int64_t i = 0; i < n; ++i) {
        int64_t rem = i;
        size_t  cand[MAX_DIMS][3];
        int     cnt[MAX_DIMS] = {0, 0, 0, 0};
        for (int d = 0; d < MAX_DIMS; ++d) {
            const int64_t j  = rem % dst->ne[d];
            rem /= dst->ne[d];
            const int64_t lp = p[2 * d];
            const int64_t rp = p[2 * d + 1];
            const int64_t nd = dst->ne[d];  // 原输入长度（右折叠条件的基准）
            cand[d][cnt[d]++] = (size_t) (j + lp) * g->nb[d];
            if (j >= 1 && j <= lp) {
                cand[d][cnt[d]++] = (size_t) (lp - j) * g->nb[d];
            }
            if (j >= nd - 1 - rp && j <= nd - 2) {
                cand[d][cnt[d]++] = (size_t) (lp + 2 * nd - 2 - j) * g->nb[d];
            }
        }
        double acc = 0.0;
        for (int c0 = 0; c0 < cnt[0]; ++c0) {
            for (int c1 = 0; c1 < cnt[1]; ++c1) {
                for (int c2 = 0; c2 < cnt[2]; ++c2) {
                    for (int c3 = 0; c3 < cnt[3]; ++c3) {
                        acc += (double) load_f32(g, cand[0][c0] + cand[1][c1] + cand[2][c2] +
                                                        cand[3][c3]);
                    }
                }
            }
        }
        store_f32(dst, tensor_offset_linear(dst, i), (float) acc);
    }
}

// ---------------------------------------------------------------- 卷积
void compute_im2col(Tensor* dst) {
    const Tensor* kernel = dst->src[0];
    const Tensor* img = dst->src[1];
    TRC_ASSERT(dst->type == TYPE_F32, "CPU im2col 一期输出仅支持 f32");
    TRC_ASSERT(img->type == TYPE_F32 || img->type == TYPE_F16, "im2col: 输入类型不支持");

    const int32_t* p = dst->op_params;
    const int s0 = p[0];
    const int s1 = p[1];
    const int p0 = p[2];
    const int p1 = p[3];
    const int d0 = p[4];
    const int d1 = p[5];
    const bool is_2d = p[6] == 1;

    const int64_t N  = is_2d ? img->ne[3] : img->ne[2];
    const int64_t IC = is_2d ? img->ne[2] : img->ne[1];
    const int64_t IH = is_2d ? img->ne[1] : 1;
    const int64_t IW = img->ne[0];
    const int64_t KH = is_2d ? kernel->ne[1] : 1;
    const int64_t KW = kernel->ne[0];
    const int64_t OH = is_2d ? dst->ne[2] : 1;
    const int64_t OW = dst->ne[1];
    const size_t  ofs_n = is_2d ? img->nb[3] : img->nb[2];
    const size_t  ofs_c = is_2d ? img->nb[2] : img->nb[1];

    const int64_t feat = IC * KH * KW;
    float* dst_data = (float*) dst->data;

    for (int64_t in = 0; in < N; ++in) {
        for (int64_t ioh = 0; ioh < OH; ++ioh) {
            for (int64_t iow = 0; iow < OW; ++iow) {
                float* drow = dst_data + (size_t) (in * OH * OW + ioh * OW + iow) * feat;
                for (int64_t iic = 0; iic < IC; ++iic) {
                    for (int64_t ikh = 0; ikh < KH; ++ikh) {
                        for (int64_t ikw = 0; ikw < KW; ++ikw) {
                            const int64_t iiw = iow * s0 + ikw * d0 - p0;
                            const int64_t iih = ioh * s1 + ikh * d1 - p1;
                            float v = 0.0f;
                            if (iih >= 0 && iih < IH && iiw >= 0 && iiw < IW) {
                                const size_t off = (size_t) in * ofs_n + (size_t) iic * ofs_c +
                                                   (size_t) iih * img->nb[1] + (size_t) iiw * img->nb[0];
                                v = (float) load_as_double(img, off);
                            }
                            drow[iic * (KH * KW) + ikh * KW + ikw] = v;
                        }
                    }
                }
            }
        }
    }
}

// im2col 的反向：把输出梯度 scatter 回输入图像（对输入图像的梯度）
void compute_im2col_back(Tensor* dst) {
    const Tensor* grad   = dst->src[0];
    const Tensor* kernel = dst->src[1];
    TRC_ASSERT(grad->type == TYPE_F32 && dst->type == TYPE_F32 && kernel->type == TYPE_F32,
               "CPU im2col_back 一期仅支持 f32");
    TRC_ASSERT(tensor_is_contiguous(grad) && tensor_is_contiguous(dst),
               "im2col_back: grad/dst 必须连续");

    const int32_t* p = dst->op_params;
    const int s0 = p[0];
    const int s1 = p[1];
    const int p0 = p[2];
    const int p1 = p[3];
    const int d0 = p[4];
    const int d1 = p[5];
    const bool is_2d = p[6] == 1;

    const int64_t N  = is_2d ? dst->ne[3] : dst->ne[2];
    const int64_t IC = is_2d ? dst->ne[2] : dst->ne[1];
    const int64_t IH = is_2d ? dst->ne[1] : 1;
    const int64_t IW = dst->ne[0];
    const int64_t KH = is_2d ? kernel->ne[1] : 1;
    const int64_t KW = kernel->ne[0];
    const int64_t OH = is_2d ? grad->ne[2] : 1;
    const int64_t OW = grad->ne[1];
    const int64_t feat = IC * KH * KW;

    // 与 ggml 一致：N/IC 的步长在 2D 时是 nb3/nb2，1D 时是 nb2/nb1
    // （不能统一用 elem_offset(dst, iiw, iih, iic, in)——1D 时会把 IC 写到 dim2）
    const size_t ofs_n = is_2d ? dst->nb[3] : dst->nb[2];
    const size_t ofs_c = is_2d ? dst->nb[2] : dst->nb[1];

    TRC_ASSERT(grad->ne[0] == feat, "im2col_back: 梯度特征维与卷积核不匹配");

    for (int64_t in = 0; in < N; ++in) {
        for (int64_t iic = 0; iic < IC; ++iic) {
            for (int64_t iih = 0; iih < IH; ++iih) {
                for (int64_t iiw = 0; iiw < IW; ++iiw) {
                    double sum = 0.0;
                    for (int64_t ikh = 0; ikh < KH; ++ikh) {
                        for (int64_t ikw = 0; ikw < KW; ++ikw) {
                            int64_t ioh = 0;
                            if (is_2d) {
                                const int64_t tmph = iih + p1 - ikh * d1;
                                if (tmph < 0 || tmph % s1 != 0) {
                                    continue;
                                }
                                ioh = tmph / s1;
                                if (ioh >= OH) {
                                    continue;
                                }
                            }
                            const int64_t tmpw = iiw + p0 - ikw * d0;
                            if (tmpw < 0 || tmpw % s0 != 0) {
                                continue;
                            }
                            const int64_t iow = tmpw / s0;
                            if (iow >= OW) {
                                continue;
                            }
                            const size_t goff = (size_t) (((in * OH + ioh) * OW + iow) * feat +
                                                          iic * (KH * KW) + ikh * KW + ikw) *
                                                sizeof(float);
                            sum += (double) load_f32(grad, goff);
                        }
                    }
                    const size_t od = (size_t) in * ofs_n + (size_t) iic * ofs_c +
                                      (size_t) iih * dst->nb[1] + (size_t) iiw * dst->nb[0];
                    store_f32(dst, od, (float) sum);
                }
            }
        }
    }
}

void compute_col2im_1d(Tensor* dst) {
    const Tensor* src = dst->src[0];
    TRC_ASSERT(src->type == TYPE_F32 && dst->type == TYPE_F32, "CPU col2im_1d 一期仅支持 f32");

    const int32_t s0 = dst->op_params[0];
    const int32_t oc = dst->op_params[1];
    const int32_t p0 = dst->op_params[2];

    const int64_t K_OC = src->ne[0];
    const int64_t T_in = src->ne[1];
    const int64_t K = K_OC / oc;
    const int64_t T_out = dst->ne[0];

    // 与 ggml 相同：按输出位置 gather（col 布局 [K*OC, T_in]，dst 布局 [T_out, OC]）
    for (int64_t o = 0; o < oc; ++o) {
        for (int64_t t_out = 0; t_out < T_out; ++t_out) {
            const int64_t t_abs = t_out + p0;
            int64_t t_min = (t_abs - K + 1 + s0 - 1) / s0;
            if (t_min < 0) {
                t_min = 0;
            }
            int64_t t_max = t_abs / s0;
            if (t_max >= T_in) {
                t_max = T_in - 1;
            }

            double sum = 0.0;
            for (int64_t t_in = t_min; t_in <= t_max; ++t_in) {
                const int64_t k = t_abs - t_in * s0;
                if (k >= 0 && k < K) {
                    sum += load_f32(src, (size_t) (o * K + k) * src->nb[0] + (size_t) t_in * src->nb[1]);
                }
            }
            store_f32(dst, (size_t) t_out * dst->nb[0] + (size_t) o * dst->nb[1], (float) sum);
        }
    }
}

void compute_conv_transpose_1d(Tensor* dst) {
    const Tensor* kernel = dst->src[0];  // [K, Cout, Cin]
    const Tensor* input  = dst->src[1];  // [T_in, Cin]
    TRC_ASSERT(kernel->type == TYPE_F32 && input->type == TYPE_F32 && dst->type == TYPE_F32,
               "CPU conv_transpose_1d 一期仅支持 f32");

    const int32_t s0 = dst->op_params[0];
    const int64_t K = kernel->ne[0];
    const int64_t Cout = kernel->ne[1];
    const int64_t Cin = kernel->ne[2];
    const int64_t T_in = input->ne[0];
    TRC_ASSERT(Cin == input->ne[1], "conv_transpose_1d: 通道数不一致");
    TRC_ASSERT(Cout == dst->ne[1], "conv_transpose_1d: 输出通道数不一致");

    std::memset(dst->data, 0, tensor_nbytes(dst));

    for (int64_t co = 0; co < Cout; ++co) {
        for (int64_t t = 0; t < T_in; ++t) {
            for (int64_t k = 0; k < K; ++k) {
                double acc = 0.0;
                for (int64_t ci = 0; ci < Cin; ++ci) {
                    acc += (double) load_f32(kernel, (size_t) k * kernel->nb[0] +
                                                         (size_t) co * kernel->nb[1] +
                                                         (size_t) ci * kernel->nb[2]) *
                           (double) load_f32(input, (size_t) t * input->nb[0] +
                                                        (size_t) ci * input->nb[1]);
                }
                const size_t od = (size_t) (t * s0 + k) * dst->nb[0] + (size_t) co * dst->nb[1];
                store_f32(dst, od, (float) ((double) load_f32(dst, od) + acc));
            }
        }
    }
}

// 2D 池化（M2.3g；语义与 ggml_pool_2d 一致）：
//   AVG 分母恒为 k0*k1（越界元素按 0 参与累加）、MAX 跳过越界元素（全越界输出 -FLT_MAX）
void compute_pool_2d(Tensor* dst) {
    const Tensor* a = dst->src[0];  // [W, H, C, N]
    TRC_ASSERT(dst->type == TYPE_F32 && a->type == TYPE_F32, "CPU pool_2d 一期仅支持 f32");

    const int32_t* p = dst->op_params;
    const int mode = p[0];
    const int k0 = p[1];
    const int k1 = p[2];
    const int s0 = p[3];
    const int s1 = p[4];
    const int p0 = p[5];
    const int p1 = p[6];

    const int64_t OW = dst->ne[0];
    const int64_t OH = dst->ne[1];
    const int64_t C  = dst->ne[2];
    const int64_t N  = dst->ne[3];
    const int64_t W  = a->ne[0];
    const int64_t H  = a->ne[1];
    const double  ka = (double) (k0 * k1);

    for (int64_t n = 0; n < N; ++n) {
        for (int64_t c = 0; c < C; ++c) {
            for (int64_t oh = 0; oh < OH; ++oh) {
                for (int64_t ow = 0; ow < OW; ++ow) {
                    double acc  = 0.0;
                    float  best = -FLT_MAX;
                    for (int kh = 0; kh < k1; ++kh) {
                        const int64_t ih = oh * s1 + kh - p1;
                        if (ih < 0 || ih >= H) {
                            continue;
                        }
                        for (int kw = 0; kw < k0; ++kw) {
                            const int64_t iw = ow * s0 + kw - p0;
                            if (iw < 0 || iw >= W) {
                                continue;
                            }
                            const float v = load_f32(a, elem_offset(a, iw, ih, c, n));
                            if (mode == (int) POOL_AVG) {
                                acc += (double) v;
                            } else if (v > best) {
                                best = v;
                            }
                        }
                    }
                    const float out = (mode == (int) POOL_AVG) ? (float) (acc / ka) : best;
                    store_f32(dst, elem_offset(dst, ow, oh, c, n), out);
                }
            }
        }
    }
}

// 2D 池化反向（专用 kernel；a = 前向输出梯度，b = 前向输入取形状）
void compute_pool_2d_back(Tensor* dst) {
    const Tensor* grad  = dst->src[0];
    const Tensor* input = dst->src[1];
    TRC_ASSERT(grad->type == TYPE_F32 && input->type == TYPE_F32 && dst->type == TYPE_F32,
               "CPU pool_2d_back 一期仅支持 f32");

    const int32_t* p = dst->op_params;
    const int mode = p[0];
    const int k0 = p[1];
    const int k1 = p[2];
    const int s0 = p[3];
    const int s1 = p[4];
    const int p0 = p[5];
    const int p1 = p[6];

    const int64_t W  = input->ne[0];
    const int64_t H  = input->ne[1];
    const int64_t C  = input->ne[2];
    const int64_t N  = input->ne[3];
    const int64_t OW = grad->ne[0];
    const int64_t OH = grad->ne[1];
    const double  ka = (double) (k0 * k1);

    std::memset(dst->data, 0, tensor_nbytes(dst));

    for (int64_t n = 0; n < N; ++n) {
        for (int64_t c = 0; c < C; ++c) {
            for (int64_t oh = 0; oh < OH; ++oh) {
                for (int64_t ow = 0; ow < OW; ++ow) {
                    const float g = load_f32(grad, elem_offset(grad, ow, oh, c, n));
                    if (mode == (int) POOL_AVG) {
                        const float gv = (float) ((double) g / ka);
                        for (int kh = 0; kh < k1; ++kh) {
                            const int64_t ih = oh * s1 + kh - p1;
                            if (ih < 0 || ih >= H) {
                                continue;
                            }
                            for (int kw = 0; kw < k0; ++kw) {
                                const int64_t iw = ow * s0 + kw - p0;
                                if (iw < 0 || iw >= W) {
                                    continue;
                                }
                                const size_t off = elem_offset(dst, iw, ih, c, n);
                                store_f32(dst, off, load_f32(dst, off) + gv);
                            }
                        }
                    } else {
                        // MAX：累加到窗口内首个严格最大值（扫描 kh 外/kw 内，与 ggml 一致）
                        float   best  = -FLT_MAX;
                        int64_t best_iw = -1;
                        int64_t best_ih = -1;
                        for (int kh = 0; kh < k1; ++kh) {
                            const int64_t ih = oh * s1 + kh - p1;
                            if (ih < 0 || ih >= H) {
                                continue;
                            }
                            for (int kw = 0; kw < k0; ++kw) {
                                const int64_t iw = ow * s0 + kw - p0;
                                if (iw < 0 || iw >= W) {
                                    continue;
                                }
                                const float v = load_f32(input, elem_offset(input, iw, ih, c, n));
                                if (v > best) {
                                    best    = v;
                                    best_iw = iw;
                                    best_ih = ih;
                                }
                            }
                        }
                        if (best_iw >= 0) {
                            const size_t off = elem_offset(dst, best_iw, best_ih, c, n);
                            store_f32(dst, off, load_f32(dst, off) + g);
                        }
                    }
                }
            }
        }
    }
}

// 2D 转置卷积（M2.3g；语义与 ggml_conv_transpose_2d_p0 一致）
//   kernel=[KW, KH, Cout, Cin]、input=[W, H, Cin, N] -> dst=[OW, OH, Cout, N]
//   dst[ow, oh, co, n] = Σ_{kh, kw, ci, (ow-kw)%stride==0, (oh-kh)%stride==0} W[kw,kh,co,ci]*X[iw,ih,ci,n]
// 采用逐输出 gather（每个输出元素恰好写一次，无需清零；与 Vulkan shader 同构）
void compute_conv_transpose_2d(Tensor* dst) {
    const Tensor* kernel = dst->src[0];  // [KW, KH, Cout, Cin]
    const Tensor* input  = dst->src[1];  // [W, H, Cin, N]
    TRC_ASSERT(kernel->type == TYPE_F32 && input->type == TYPE_F32 && dst->type == TYPE_F32,
               "CPU conv_transpose_2d 一期仅支持 f32");

    const int32_t stride = dst->op_params[0];
    const int64_t KW   = kernel->ne[0];
    const int64_t KH   = kernel->ne[1];
    const int64_t Cout = kernel->ne[2];
    const int64_t Cin  = kernel->ne[3];
    const int64_t W    = input->ne[0];
    const int64_t H    = input->ne[1];
    const int64_t N    = input->ne[3];
    const int64_t OW   = dst->ne[0];
    const int64_t OH   = dst->ne[1];
    TRC_ASSERT(Cin == input->ne[2] && Cout == dst->ne[2] && N == dst->ne[3],
               "conv_transpose_2d: 通道/批量数不一致");

    for (int64_t n = 0; n < N; ++n) {
        for (int64_t co = 0; co < Cout; ++co) {
            for (int64_t oh = 0; oh < OH; ++oh) {
                for (int64_t ow = 0; ow < OW; ++ow) {
                    double acc = 0.0;
                    for (int64_t kh = 0; kh < KH; ++kh) {
                        const int64_t nh = oh - kh;
                        if (nh < 0 || nh % stride != 0) {
                            continue;
                        }
                        const int64_t ih = nh / stride;
                        if (ih >= H) {
                            continue;
                        }
                        for (int64_t kw = 0; kw < KW; ++kw) {
                            const int64_t nw = ow - kw;
                            if (nw < 0 || nw % stride != 0) {
                                continue;
                            }
                            const int64_t iw = nw / stride;
                            if (iw >= W) {
                                continue;
                            }
                            for (int64_t ci = 0; ci < Cin; ++ci) {
                                acc += (double) load_f32(kernel, elem_offset(kernel, kw, kh, co, ci)) *
                                       (double) load_f32(input, elem_offset(input, iw, ih, ci, n));
                            }
                        }
                    }
                    store_f32(dst, elem_offset(dst, ow, oh, co, n), (float) acc);
                }
            }
        }
    }
}

// ---------------------------------------------------------------- 设备端优化器步（M4.1）

// op_params float[6] 读取（与 trc_ops.cpp 的构造一致）
void read_op_params_f32(const Tensor* t, float* out6) {
    std::memcpy(out6, t->op_params, 6 * sizeof(float));
}

// node 是 param 的完整视图（node->data == param->data）；逐元素运算顺序与主机优化器一致
void compute_opt_step_adamw(Tensor* node) {
    const Tensor* param = node->src[0];
    const Tensor* grad  = node->src[1];
    const Tensor* m     = node->src[2];
    const Tensor* v     = node->src[3];
    TRC_ASSERT(node->type == TYPE_F32 && param->type == TYPE_F32 && grad->type == TYPE_F32 &&
                   m->type == TYPE_F32 && v->type == TYPE_F32,
               "CPU opt_step_adamw 仅支持 F32");

    float ps[6];
    read_op_params_f32(node, ps);
    const float step_sz  = ps[0];
    const float bc2_sqrt = ps[1];
    const float decay    = ps[2];
    const float beta1    = ps[3];
    const float beta2    = ps[4];
    const float eps      = ps[5];

    const int64_t n = tensor_nelements(param);
    float*        p = (float*) node->data;
    const float*  g = (const float*) grad->data;
    float*        mm = (float*) m->data;
    float*        vv = (float*) v->data;
    for (int64_t i = 0; i < n; ++i) {
        const float gi = g[i];
        mm[i] += (gi - mm[i]) * (1.0f - beta1);
        vv[i] = vv[i] * beta2 + gi * gi * (1.0f - beta2);

        const float pv    = p[i] * decay;
        const float denom = std::sqrt(vv[i]) / bc2_sqrt + eps;
        p[i]              = pv - step_sz * (mm[i] / denom);
    }
}

void compute_opt_step_sgd(Tensor* node) {
    const Tensor* param = node->src[0];
    const Tensor* grad  = node->src[1];
    const Tensor* mom   = node->src[2];
    TRC_ASSERT(node->type == TYPE_F32 && param->type == TYPE_F32 && grad->type == TYPE_F32 &&
                   mom->type == TYPE_F32,
               "CPU opt_step_sgd 仅支持 F32");

    float ps[6];
    read_op_params_f32(node, ps);
    const float lr            = ps[0];
    const float momentum_coef = ps[1];
    const float dampening     = ps[2];
    const float weight_decay  = ps[3];
    const float nesterov      = ps[4];
    const float is_first      = ps[5];

    const int64_t n = tensor_nelements(param);
    float*        p = (float*) node->data;
    const float*  g = (const float*) grad->data;
    float*        mm = (float*) mom->data;
    for (int64_t i = 0; i < n; ++i) {
        float d = g[i] + weight_decay * p[i];
        if (momentum_coef != 0.0f) {
            if (is_first != 0.0f) {
                mm[i] = d;
            } else {
                mm[i] = momentum_coef * mm[i] + (1.0f - dampening) * d;
            }
            d = nesterov != 0.0f ? d + momentum_coef * mm[i] : mm[i];
        }
        p[i] -= lr * d;
    }
}

// M4.5：设备端梯度裁剪原语 / WeightNorm 同步（CPU 参考实现）

// acc[0] += Σ a²（node 是 acc 的完整视图）；与主机 clip 的 double 累加一致
void compute_sum_sqr_acc(Tensor* node) {
    const Tensor* acc = node->src[0];
    const Tensor* a   = node->src[1];
    TRC_ASSERT(node->type == TYPE_F32 && acc->type == TYPE_F32 && a->type == TYPE_F32,
               "CPU sum_sqr_acc 仅支持 F32");
    const int64_t n = tensor_nelements(a);
    const float*  x = (const float*) a->data;
    double        s = 0.0;
    for (int64_t i = 0; i < n; ++i) {
        s += (double) x[i] * (double) x[i];
    }
    float* dst = (float*) node->data;  // == acc->data
    dst[0]     = dst[0] + (float) s;
}

// a *= min(1, max_norm / (sqrt(norm[0]) + eps))；与主机 optim_clip_grad_norm 的缩放一致
void compute_clip_scale_inplace(Tensor* node) {
    const Tensor* a    = node->src[0];
    const Tensor* norm = node->src[1];
    TRC_ASSERT(node->type == TYPE_F32 && a->type == TYPE_F32 && norm->type == TYPE_F32,
               "CPU clip_scale_inplace 仅支持 F32");
    float params[2];
    std::memcpy(params, node->op_params, sizeof(params));
    const float total = std::sqrt(*(const float*) norm->data);
    float       scale = params[0] / (total + params[1]);
    if (scale > 1.0f) {
        scale = 1.0f;
    }
    const int64_t n = tensor_nelements(a);
    float*        p = (float*) node->data;  // == a->data
    for (int64_t i = 0; i < n; ++i) {
        p[i] *= scale;
    }
}

// g[oc] = sqrt(Σ_rest v²)（逐输出通道；v 连续，oc 为最后一维）
void compute_weightnorm_sync(Tensor* node) {
    const Tensor* v = node->src[0];
    Tensor*       g = node->src[1];
    TRC_ASSERT(node->type == TYPE_F32 && v->type == TYPE_F32 && g->type == TYPE_F32,
               "CPU weightnorm_sync 仅支持 F32");
    const int     nd    = tensor_n_dims(v);
    const int64_t oc    = v->ne[nd - 1];
    int64_t       n_rest = 1;
    for (int d = 0; d + 1 < nd; ++d) {
        n_rest *= v->ne[d];
    }
    const float* vp = (const float*) v->data;
    float*       gp = (float*) node->data;  // == g->data
    for (int64_t o = 0; o < oc; ++o) {
        double s = 0.0;
        for (int64_t r = 0; r < n_rest; ++r) {
            const float x = vp[r + o * n_rest];
            s += (double) x * (double) x;
        }
        gp[o] = (float) std::sqrt(s);
    }
}

// ---------------------------------------------------------------- 能力声明

bool sup_f32(const Tensor* t) {
    return t != nullptr && t->type == TYPE_F32;
}

bool sup_srcs_f32(const Tensor* t, int n_src) {
    if (!sup_f32(t)) {
        return false;
    }
    for (int i = 0; i < n_src; ++i) {
        if (!sup_f32(t->src[i])) {
            return false;
        }
    }
    return true;
}

} // namespace

// 如实声明 CPU 后端支持范围（M2.1f / R9）：与 kernel 内的真实类型/布局约束一致，
// 避免"声明支持、compute 时才炸"或调用方无法预检（graph_first_unsupported）。
bool cpu_supports_op(const Tensor* t) {
    switch (t->op) {
        // 叶子与视图不产生计算：数据由主机/缓冲/源张量提供
        case OP_NONE:
        case OP_RESHAPE:
        case OP_VIEW:
        case OP_PERMUTE:
        case OP_TRANSPOSE:
            return true;

        // dup/cont/cpy：kernel 按元素字节拷贝，要求输出与输入同类型
        case OP_DUP:
        case OP_CPY:
        case OP_CONT:
            return t->src[0] != nullptr && t->src[0]->type == t->type;

        // cast：CPU 经 double 桥接，支持所有基本类型互转
        case OP_CAST:
            return t->src[0] != nullptr && type_is_supported(t->src[0]->type) &&
                   type_is_supported(t->type);

        // 逐元素二元（F32）
        case OP_ADD:
        case OP_ADD1:
        case OP_SUB:
        case OP_MUL:
        case OP_DIV:
            return sup_srcs_f32(t, 2);

        // 逐元素一元 / 激活（F32）
        case OP_NEG:
        case OP_ABS:
        case OP_SQR:
        case OP_SQRT:
        case OP_EXP:
        case OP_LOG:
        case OP_SIN:
        case OP_COS:
        case OP_CLAMP:
        case OP_RELU:
        case OP_LEAKY_RELU:
        case OP_GELU:
        case OP_GELU_ERF:
        case OP_SILU:
        case OP_SIGMOID:
        case OP_TANH:
        case OP_SOFTPLUS:
        case OP_HARDSWISH:
        case OP_SCALE:
        case OP_SGN:
        case OP_STEP:
        case OP_ERF:
            return sup_srcs_f32(t, 1);

        // 线代（F32；形状条件与 ggml_can_mul_mat 一致）
        case OP_MUL_MAT: {
            if (!sup_srcs_f32(t, 2)) {
                return false;
            }
            const Tensor* a = t->src[0];
            const Tensor* b = t->src[1];
            return a->ne[0] == b->ne[0] && b->ne[2] % a->ne[2] == 0 && b->ne[3] % a->ne[3] == 0;
        }

        // 归约 / 归一化（F32）
        case OP_SUM:
        case OP_SUM_ROWS:
        case OP_MEAN:
        case OP_NORM:
        case OP_RMS_NORM:
        case OP_GROUP_NORM:
            return sup_srcs_f32(t, 1);
        case OP_NORM_BACK:
        case OP_RMS_NORM_BACK:
        case OP_GROUP_NORM_BACK:
            return sup_srcs_f32(t, 2);

        // soft_max：F32 且输入/输出连续（核心构造也要求连续）；mask 可选 F32/F16
        case OP_SOFT_MAX: {
            if (!sup_srcs_f32(t, 1) || !tensor_is_contiguous(t->src[0]) ||
                !tensor_is_contiguous(t)) {
                return false;
            }
            const Tensor* mask = t->src[1];
            return mask == nullptr || mask->type == TYPE_F32 || mask->type == TYPE_F16;
        }

        // 交叉熵（F32、同形状、最内维连续）
        case OP_CROSS_ENTROPY_LOSS: {
            if (!sup_srcs_f32(t, 2)) {
                return false;
            }
            const Tensor* a = t->src[0];
            const Tensor* b = t->src[1];
            return tensor_are_same_shape(a, b) && a->nb[0] == sizeof(float) &&
                   b->nb[0] == sizeof(float);
        }
        case OP_CROSS_ENTROPY_LOSS_BACK: {
            if (!sup_srcs_f32(t, 3)) {
                return false;
            }
            const Tensor* g = t->src[0];
            const Tensor* a = t->src[1];
            const Tensor* b = t->src[2];
            return tensor_nelements(g) == 1 && tensor_are_same_shape(a, b) &&
                   a->nb[0] == sizeof(float) && b->nb[0] == sizeof(float) &&
                   t->nb[0] == sizeof(float);
        }

        // 索引：表 F32/F16、索引 I32/I64（核心构造目前只产生 I32）
        case OP_GET_ROWS: {
            const Tensor* table = t->src[0];
            const Tensor* idx   = t->src[1];
            if (table == nullptr || idx == nullptr || !sup_f32(t) || idx->ne[3] != 1) {
                return false;
            }
            const bool idx_ok   = idx->type == TYPE_I32 || idx->type == TYPE_I64;
            const bool table_ok = table->type == TYPE_F32 || table->type == TYPE_F16;
            return idx_ok && table_ok;
        }
        case OP_GET_ROWS_BACK: {
            const Tensor* grad = t->src[0];
            const Tensor* idx  = t->src[1];
            return sup_f32(t) && sup_f32(grad) && idx != nullptr && idx->type == TYPE_I32 &&
                   idx->ne[3] == 1 && tensor_is_contiguous(t);
        }
        case OP_SET_ROWS: {
            const Tensor* values  = t->src[0];
            const Tensor* indices = t->src[1];
            if (values == nullptr || indices == nullptr) {
                return false;
            }
            const bool values_ok = values->type == TYPE_F32 || values->type == TYPE_F16;
            const bool idx_ok = indices->type == TYPE_I32 || indices->type == TYPE_I64;
            return values_ok && idx_ok;
        }

        // 形状扩展：repeat/concat/pad 经 double 桥接，支持基本类型；repeat_back 仅 F32 连续
        case OP_REPEAT:
            return t->src[0] != nullptr && type_is_supported(t->src[0]->type) &&
                   type_is_supported(t->type);
        case OP_CONCAT:
            return t->src[0] != nullptr && t->src[1] != nullptr &&
                   type_is_supported(t->src[0]->type) && type_is_supported(t->src[1]->type) &&
                   type_is_supported(t->type);
        case OP_PAD:
            return t->src[0] != nullptr && type_is_supported(t->src[0]->type) &&
                   type_is_supported(t->type);
        case OP_PAD_BACK:
            // reflect 反向：一期 F32（与 Vulkan 一致）
            return sup_srcs_f32(t, 2);
        case OP_REPEAT_BACK:
            return sup_srcs_f32(t, 1) && tensor_is_contiguous(t);

        // acc：F32；a/dst 连续
        case OP_ACC:
            return sup_srcs_f32(t, 2) && tensor_is_contiguous(t->src[0]) &&
                   tensor_is_contiguous(t);

        // 卷积：im2col 输出 F32、图像 F32/F16（核仅用于形状）；其余 F32 且连续要求同 kernel
        case OP_IM2COL: {
            const Tensor* kernel = t->src[0];
            const Tensor* img    = t->src[1];
            if (kernel == nullptr || img == nullptr || !sup_f32(t)) {
                return false;
            }
            return img->type == TYPE_F32 || img->type == TYPE_F16;
        }
        case OP_IM2COL_BACK:
            return sup_srcs_f32(t, 2) && tensor_is_contiguous(t->src[0]) &&
                   tensor_is_contiguous(t);
        case OP_COL2IM:
            return sup_srcs_f32(t, 1);
        case OP_CONV_TRANSPOSE_1D:
            return sup_srcs_f32(t, 2);
        case OP_POOL_2D:
            return sup_srcs_f32(t, 1);
        case OP_POOL_2D_BACK:
            return sup_srcs_f32(t, 2);
        case OP_CONV_TRANSPOSE_2D:
            return sup_srcs_f32(t, 2);

        // 设备端优化器步：F32 且 param/grad/状态/结果均连续（主机实现要求参数连续）
        case OP_OPT_STEP_ADAMW:
            return sup_srcs_f32(t, 4) && tensor_is_contiguous(t->src[0]) &&
                   tensor_is_contiguous(t->src[1]) && tensor_is_contiguous(t->src[2]) &&
                   tensor_is_contiguous(t->src[3]) && tensor_is_contiguous(t);
        case OP_OPT_STEP_SGD:
            return sup_srcs_f32(t, 3) && tensor_is_contiguous(t->src[0]) &&
                   tensor_is_contiguous(t->src[1]) && tensor_is_contiguous(t->src[2]) &&
                   tensor_is_contiguous(t);

        // M4.5：梯度裁剪原语 / WeightNorm 同步（F32，连续）
        case OP_SUM_SQR_ACC:
            return sup_srcs_f32(t, 2) && tensor_nelements(t->src[0]) == 1 &&
                   tensor_is_contiguous(t->src[1]);
        case OP_CLIP_SCALE_INPLACE:
            return sup_srcs_f32(t, 2) && tensor_nelements(t->src[1]) == 1 &&
                   tensor_is_contiguous(t->src[0]);
        case OP_WEIGHTNORM_SYNC:
            return sup_srcs_f32(t, 2) && tensor_is_contiguous(t->src[0]) &&
                   tensor_is_contiguous(t->src[1]);

        default:
            return false;
    }
}

bool cpu_compute_node(Tensor* node) {
    switch (node->op) {
        case OP_NONE:
            return true;  // 叶子节点：数据已由主机或 buffer 提供
        // 视图类算子不产生计算：data 指针在创建/缓冲分配时已指向源数据
        case OP_RESHAPE:
        case OP_VIEW:
        case OP_PERMUTE:
        case OP_TRANSPOSE:
            return true;
        case OP_DUP:
            compute_dup(node);
            return true;
        case OP_CPY:
        case OP_CONT:
            compute_dup(node);
            return true;
        case OP_CAST:
            compute_cast(node);
            return true;
        case OP_ADD:
            compute_binary(node, fn_add);
            return true;
        case OP_ADD1:
            compute_binary(node, fn_add);
            return true;
        case OP_SUB:
            compute_binary(node, fn_sub);
            return true;
        case OP_MUL:
            compute_binary(node, fn_mul);
            return true;
        case OP_DIV:
            compute_binary(node, fn_div);
            return true;
        case OP_NEG:
            compute_unary(node, fn_neg);
            return true;
        case OP_SGN:
            compute_unary(node, fn_sgn);
            return true;
        case OP_STEP:
            compute_unary(node, fn_step);
            return true;
        case OP_ABS:
            compute_unary(node, fn_abs);
            return true;
        case OP_SQR:
            compute_unary(node, fn_sqr);
            return true;
        case OP_SQRT:
            compute_unary(node, fn_sqrt);
            return true;
        case OP_EXP:
            compute_unary(node, fn_exp);
            return true;
        case OP_LOG:
            compute_unary(node, fn_log);
            return true;
        case OP_SIN:
            compute_unary(node, fn_sin);
            return true;
        case OP_COS:
            compute_unary(node, fn_cos);
            return true;
        case OP_CLAMP:
            compute_clamp(node);
            return true;
        case OP_RELU:
            compute_unary(node, fn_relu);
            return true;
        case OP_LEAKY_RELU:
            compute_leaky_relu(node);
            return true;
        case OP_GELU:
            compute_unary(node, fn_gelu);
            return true;
        case OP_GELU_ERF:
            compute_unary(node, fn_gelu_erf);
            return true;
        case OP_ERF:
            compute_unary(node, fn_erf);
            return true;
        case OP_SILU:
            compute_unary(node, fn_silu);
            return true;
        case OP_SIGMOID:
            compute_unary(node, fn_sigmoid);
            return true;
        case OP_TANH:
            compute_unary(node, fn_tanh);
            return true;
        case OP_SOFTPLUS:
            compute_unary(node, fn_softplus);
            return true;
        case OP_HARDSWISH:
            compute_unary(node, fn_hardswish);
            return true;
        case OP_SCALE:
            compute_scale(node);
            return true;
        case OP_MUL_MAT:
            compute_mul_mat(node);
            return true;
        case OP_SUM:
            compute_sum(node);
            return true;
        case OP_SUM_ROWS:
            compute_sum_rows_or_mean(node, false);
            return true;
        case OP_MEAN:
            compute_sum_rows_or_mean(node, true);
            return true;
        case OP_NORM:
            compute_norm_impl(node, false);
            return true;
        case OP_NORM_BACK:
            compute_norm_back(node);
            return true;
        case OP_RMS_NORM:
            compute_norm_impl(node, true);
            return true;
        case OP_RMS_NORM_BACK:
            compute_rms_norm_back(node);
            return true;
        case OP_GROUP_NORM:
            compute_group_norm(node);
            return true;
        case OP_GROUP_NORM_BACK:
            compute_group_norm_back(node);
            return true;
        case OP_SOFT_MAX:
            compute_soft_max(node);
            return true;
        case OP_CROSS_ENTROPY_LOSS:
            compute_cross_entropy_loss(node);
            return true;
        case OP_CROSS_ENTROPY_LOSS_BACK:
            compute_cross_entropy_loss_back(node);
            return true;
        case OP_GET_ROWS:
            compute_get_rows(node);
            return true;
        case OP_GET_ROWS_BACK:
            compute_get_rows_back(node);
            return true;
        case OP_SET_ROWS:
            compute_set_rows(node);
            return true;
        case OP_REPEAT:
            compute_repeat(node);
            return true;
        case OP_REPEAT_BACK:
            compute_repeat_back(node);
            return true;
        case OP_ACC:
            compute_acc(node);
            return true;
        case OP_CONCAT:
            compute_concat(node);
            return true;
        case OP_PAD:
            compute_pad(node);
            return true;
        case OP_PAD_BACK:
            compute_pad_back(node);
            return true;
        case OP_IM2COL:
            compute_im2col(node);
            return true;
        case OP_IM2COL_BACK:
            compute_im2col_back(node);
            return true;
        case OP_COL2IM:
            compute_col2im_1d(node);
            return true;
        case OP_CONV_TRANSPOSE_1D:
            compute_conv_transpose_1d(node);
            return true;
        case OP_POOL_2D:
            compute_pool_2d(node);
            return true;
        case OP_POOL_2D_BACK:
            compute_pool_2d_back(node);
            return true;
        case OP_CONV_TRANSPOSE_2D:
            compute_conv_transpose_2d(node);
            return true;
        case OP_OPT_STEP_ADAMW:
            compute_opt_step_adamw(node);
            return true;
        case OP_OPT_STEP_SGD:
            compute_opt_step_sgd(node);
            return true;
        case OP_SUM_SQR_ACC:
            compute_sum_sqr_acc(node);
            return true;
        case OP_CLIP_SCALE_INPLACE:
            compute_clip_scale_inplace(node);
            return true;
        case OP_WEIGHTNORM_SYNC:
            compute_weightnorm_sync(node);
            return true;
        default:
            TRC_ABORT("CPU 后端不支持算子 %s (op=%d)", op_name(node->op), (int) node->op);
    }
}

} // namespace traincpp
