// train.cpp - 算子构造函数
#include "traincpp/trc_ops.h"

#include "core/trc_impl.h"
#include "traincpp/trc_context.h"

#include <cstring>

namespace traincpp {

Tensor* dup(Context* ctx, const Tensor* a) {
    TRC_ASSERT(a != nullptr, "dup: 输入为空");
    return tensor_dup_meta(ctx, a);
}

namespace {

// 二元算子通用构造：结果形状取 a，b 必须可重复到 a（ggml 广播规则）
Tensor* new_binary(Context* ctx, Op op, Tensor* a, Tensor* b, const char* name) {
    TRC_ASSERT(a != nullptr && b != nullptr, "%s: 输入为空", name);
    TRC_ASSERT(a->type == b->type, "%s: 输入类型不一致 (%s vs %s)", name, type_name(a->type),
               type_name(b->type));
    TRC_ASSERT(tensor_are_same_shape(a, b) || tensor_can_repeat(b, a),
               "%s: 形状不兼容，b 无法广播到 a", name);

    Tensor* result = new_tensor_nd(ctx, a->type, MAX_DIMS, a->ne);
    result->op = op;
    result->src[0] = a;
    result->src[1] = b;
    result->nsrc = 2;
    return result;
}

// 一元算子通用构造
Tensor* new_unary(Context* ctx, Op op, Tensor* a) {
    TRC_ASSERT(a != nullptr, "%s: 输入为空", op_name(op));
    Tensor* result = new_tensor_nd(ctx, a->type, MAX_DIMS, a->ne);
    result->op = op;
    result->src[0] = a;
    result->nsrc = 1;
    return result;
}

} // namespace

Tensor* add(Context* ctx, Tensor* a, Tensor* b) {
    return new_binary(ctx, OP_ADD, a, b, "add");
}

Tensor* add1(Context* ctx, Tensor* a, Tensor* b) {
    TRC_ASSERT(tensor_nelements(b) == 1, "add1: 第二个输入必须是标量（1 个元素）");
    return new_binary(ctx, OP_ADD1, a, b, "add1");
}

Tensor* sub(Context* ctx, Tensor* a, Tensor* b) {
    return new_binary(ctx, OP_SUB, a, b, "sub");
}

Tensor* mul(Context* ctx, Tensor* a, Tensor* b) {
    return new_binary(ctx, OP_MUL, a, b, "mul");
}

Tensor* div(Context* ctx, Tensor* a, Tensor* b) {
    return new_binary(ctx, OP_DIV, a, b, "div");
}

Tensor* neg(Context* ctx, Tensor* a) {
    return new_unary(ctx, OP_NEG, a);
}

Tensor* abs_op(Context* ctx, Tensor* a) {
    return new_unary(ctx, OP_ABS, a);
}

Tensor* sqr(Context* ctx, Tensor* a) {
    return new_unary(ctx, OP_SQR, a);
}

Tensor* sqrt_op(Context* ctx, Tensor* a) {
    return new_unary(ctx, OP_SQRT, a);
}

Tensor* exp_op(Context* ctx, Tensor* a) {
    return new_unary(ctx, OP_EXP, a);
}

Tensor* log_op(Context* ctx, Tensor* a) {
    return new_unary(ctx, OP_LOG, a);
}

Tensor* relu(Context* ctx, Tensor* a) {
    return new_unary(ctx, OP_RELU, a);
}

Tensor* sigmoid(Context* ctx, Tensor* a) {
    return new_unary(ctx, OP_SIGMOID, a);
}

Tensor* tanh_op(Context* ctx, Tensor* a) {
    return new_unary(ctx, OP_TANH, a);
}

Tensor* silu(Context* ctx, Tensor* a) {
    return new_unary(ctx, OP_SILU, a);
}

Tensor* scale(Context* ctx, Tensor* a, float s) {
    TRC_ASSERT(a != nullptr, "scale: 输入为空");
    Tensor* result = new_tensor_nd(ctx, a->type, MAX_DIMS, a->ne);
    result->op = OP_SCALE;
    result->src[0] = a;
    result->nsrc = 1;
    std::memcpy(result->op_params, &s, sizeof(float));
    return result;
}

Tensor* scale_bias(Context* ctx, Tensor* a, float s, float b) {
    // 与 ggml_scale_bias 语义一致：y = s*a + b（标量 b）。组合实现：scale + add1（常量标量）。
    // 反任由 OP_SCALE（dA = s*dY）与 OP_ADD1（常量 b 不产生参数梯度）自动组合。
    TRC_ASSERT(a != nullptr, "scale_bias: 输入为空");
    return add1(ctx, scale(ctx, a, s), new_scalar_const(ctx, b));
}

Tensor* cont(Context* ctx, Tensor* a) {
    TRC_ASSERT(a != nullptr, "cont: 输入为空");
    if (tensor_is_contiguous(a)) {
        return a;  // 与 ggml 一致：已连续则直接返回源
    }
    Tensor* result = new_tensor_nd(ctx, a->type, MAX_DIMS, a->ne);
    result->op = OP_CONT;
    result->src[0] = a;
    result->nsrc = 1;
    return result;
}

Tensor* cast(Context* ctx, Tensor* a, Type type) {
    TRC_ASSERT(a != nullptr, "cast: 输入为空");
    if (a->type == type) {
        return a;  // 与 ggml 一致：类型相同直接返回源
    }
    TRC_ASSERT(type_is_supported(type), "cast: 目标类型 %s 不支持", type_name(type));
    Tensor* result = new_tensor_nd(ctx, type, MAX_DIMS, a->ne);
    result->op = OP_CAST;
    result->src[0] = a;
    result->nsrc = 1;
    return result;
}

Tensor* leaky_relu(Context* ctx, Tensor* a, float slope) {
    Tensor* result = new_unary(ctx, OP_LEAKY_RELU, a);
    std::memcpy(result->op_params, &slope, sizeof(float));
    return result;
}

Tensor* clamp(Context* ctx, Tensor* a, float min_val, float max_val) {
    Tensor* result = new_unary(ctx, OP_CLAMP, a);
    std::memcpy(result->op_params, &min_val, sizeof(float));
    std::memcpy((uint8_t*) result->op_params + sizeof(float), &max_val, sizeof(float));
    return result;
}

Tensor* sin_op(Context* ctx, Tensor* a) {
    return new_unary(ctx, OP_SIN, a);
}

Tensor* cos_op(Context* ctx, Tensor* a) {
    return new_unary(ctx, OP_COS, a);
}

Tensor* sgn(Context* ctx, Tensor* a) {
    return new_unary(ctx, OP_SGN, a);
}

Tensor* step(Context* ctx, Tensor* a) {
    return new_unary(ctx, OP_STEP, a);
}

Tensor* gelu(Context* ctx, Tensor* a) {
    return new_unary(ctx, OP_GELU, a);
}

Tensor* gelu_erf(Context* ctx, Tensor* a) {
    return new_unary(ctx, OP_GELU_ERF, a);
}

Tensor* erf_op(Context* ctx, Tensor* a) {
    return new_unary(ctx, OP_ERF, a);
}

Tensor* softplus(Context* ctx, Tensor* a) {
    return new_unary(ctx, OP_SOFTPLUS, a);
}

Tensor* hardswish(Context* ctx, Tensor* a) {
    return new_unary(ctx, OP_HARDSWISH, a);
}

// ---------------- 归约 ----------------

namespace {

// 复制元数据并设置为一元算子结果（与 ggml_dup_tensor 的使用方式一致）
Tensor* new_dup_unary(Context* ctx, Op op, Tensor* a) {
    TRC_ASSERT(a != nullptr, "%s: 输入为空", op_name(op));
    Tensor* result = tensor_dup_meta(ctx, a);
    result->op = op;
    result->src[0] = a;
    result->nsrc = 1;
    return result;
}

} // namespace

Tensor* sum(Context* ctx, Tensor* a) {
    TRC_ASSERT(a != nullptr, "sum: 输入为空");
    Tensor* result = new_tensor_1d(ctx, a->type, 1);
    result->op = OP_SUM;
    result->src[0] = a;
    result->nsrc = 1;
    return result;
}

Tensor* sum_rows(Context* ctx, Tensor* a) {
    TRC_ASSERT(a != nullptr, "sum_rows: 输入为空");
    const int64_t ne[MAX_DIMS] = {1, a->ne[1], a->ne[2], a->ne[3]};
    Tensor* result = new_tensor_nd(ctx, a->type, MAX_DIMS, ne);
    result->op = OP_SUM_ROWS;
    result->src[0] = a;
    result->nsrc = 1;
    return result;
}

Tensor* mean(Context* ctx, Tensor* a) {
    TRC_ASSERT(a != nullptr, "mean: 输入为空");
    const int64_t ne[MAX_DIMS] = {1, a->ne[1], a->ne[2], a->ne[3]};
    Tensor* result = new_tensor_nd(ctx, TYPE_F32, MAX_DIMS, ne);
    result->op = OP_MEAN;
    result->src[0] = a;
    result->nsrc = 1;
    return result;
}

// ---------------- 归一化 ----------------

Tensor* norm(Context* ctx, Tensor* a, float eps) {
    Tensor* result = new_dup_unary(ctx, OP_NORM, a);
    std::memcpy(result->op_params, &eps, sizeof(float));
    return result;
}

Tensor* rms_norm(Context* ctx, Tensor* a, float eps) {
    Tensor* result = new_dup_unary(ctx, OP_RMS_NORM, a);
    std::memcpy(result->op_params, &eps, sizeof(float));
    return result;
}

Tensor* group_norm(Context* ctx, Tensor* a, int n_groups, float eps) {
    TRC_ASSERT(n_groups > 0, "group_norm: n_groups 必须大于 0");
    Tensor* result = new_dup_unary(ctx, OP_GROUP_NORM, a);
    result->op_params[0] = n_groups;                                       // int32
    std::memcpy((uint8_t*) result->op_params + sizeof(float), &eps, sizeof(float));  // 偏移 4 字节
    return result;
}

// ---------------- 归一化反向 ----------------

namespace {

// 归一化反向的公共构造：a = dy，b = x，结果形状与 b 相同
Tensor* new_norm_back(Context* ctx, Op op, Tensor* a, Tensor* b, float eps) {
    TRC_ASSERT(a != nullptr && b != nullptr, "%s: 输入为空", op_name(op));
    TRC_ASSERT(a->type == TYPE_F32 && b->type == TYPE_F32, "%s: 一期仅支持 f32", op_name(op));
    TRC_ASSERT(tensor_are_same_shape(a, b), "%s: dy 与 x 形状必须一致", op_name(op));

    Tensor* result = tensor_dup_meta(ctx, a);
    result->op = op;
    result->src[0] = a;
    result->src[1] = b;
    result->nsrc = 2;
    std::memcpy(result->op_params, &eps, sizeof(float));
    return result;
}

} // namespace

Tensor* norm_back(Context* ctx, Tensor* a, Tensor* b, float eps) {
    return new_norm_back(ctx, OP_NORM_BACK, a, b, eps);
}

Tensor* rms_norm_back(Context* ctx, Tensor* a, Tensor* b, float eps) {
    return new_norm_back(ctx, OP_RMS_NORM_BACK, a, b, eps);
}

Tensor* group_norm_back(Context* ctx, Tensor* a, Tensor* b, int n_groups, float eps) {
    TRC_ASSERT(n_groups > 0, "group_norm_back: n_groups 必须大于 0");
    Tensor* result = new_norm_back(ctx, OP_GROUP_NORM_BACK, a, b, eps);
    result->op_params[0] = n_groups;
    std::memcpy((uint8_t*) result->op_params + sizeof(float), &eps, sizeof(float));
    return result;
}

// ---------------- 注意力 ----------------

Tensor* soft_max(Context* ctx, Tensor* a) {
    return soft_max_ext(ctx, a, nullptr, 1.0f, 0.0f);
}

Tensor* soft_max_ext(Context* ctx, Tensor* a, Tensor* mask, float scale, float max_bias) {
    TRC_ASSERT(a != nullptr, "soft_max: 输入为空");
    TRC_ASSERT(tensor_is_contiguous(a), "soft_max: 输入必须连续（与 ggml 一致）");
    if (mask != nullptr) {
        TRC_ASSERT(mask->type == TYPE_F32 || mask->type == TYPE_F16, "soft_max: mask 类型不支持");
        TRC_ASSERT(tensor_is_contiguous(mask), "soft_max: mask 必须连续（与 ggml 一致）");
        TRC_ASSERT(mask->ne[0] == a->ne[0], "soft_max: mask.ne[0] 必须等于输入的 ne[0]");
        TRC_ASSERT(mask->ne[1] >= a->ne[1], "soft_max: mask.ne[1] 必须不小于输入的 ne[1]（与 ggml 一致）");
        TRC_ASSERT(a->ne[2] % mask->ne[2] == 0 && a->ne[3] % mask->ne[3] == 0,
                   "soft_max: mask 的 ne2/ne3 必须整除输入");
        TRC_ASSERT(max_bias <= 0.0f || mask != nullptr, "soft_max: max_bias > 0 时必须提供 mask");
    }
    Tensor* result = new_dup_unary(ctx, OP_SOFT_MAX, a);
    result->src[1] = mask;
    result->nsrc = mask != nullptr ? 2 : 1;

    const float params[2] = {scale, max_bias};
    std::memcpy(result->op_params, params, sizeof(params));
    return result;
}

// ---------------- 损失 ----------------

Tensor* cross_entropy_loss(Context* ctx, Tensor* a, Tensor* b) {
    TRC_ASSERT(a != nullptr && b != nullptr, "cross_entropy_loss: 输入为空");
    TRC_ASSERT(a->type == TYPE_F32 && b->type == TYPE_F32, "cross_entropy_loss: 一期仅支持 f32");
    TRC_ASSERT(tensor_are_same_shape(a, b), "cross_entropy_loss: logits 与 target 必须同形状（与 ggml 一致）");
    TRC_ASSERT(a->ne[0] > 0, "cross_entropy_loss: ne0 必须大于 0");
    TRC_ASSERT(a->nb[0] == (size_t) type_size(a->type) && b->nb[0] == (size_t) type_size(b->type),
               "cross_entropy_loss: 最内维必须连续（与 ggml 一致）");

    Tensor* result = new_tensor_1d(ctx, TYPE_F32, 1);
    result->op = OP_CROSS_ENTROPY_LOSS;
    result->src[0] = a;
    result->src[1] = b;
    result->nsrc = 2;
    return result;
}

Tensor* cross_entropy_loss_back(Context* ctx, Tensor* grad, Tensor* a, Tensor* b) {
    TRC_ASSERT(grad != nullptr && a != nullptr && b != nullptr, "cross_entropy_loss_back: 输入为空");
    TRC_ASSERT(grad->type == TYPE_F32 && a->type == TYPE_F32 && b->type == TYPE_F32,
               "cross_entropy_loss_back: 一期仅支持 f32");
    TRC_ASSERT(tensor_nelements(grad) == 1, "cross_entropy_loss_back: grad 必须是标量（与 ggml 一致）");
    TRC_ASSERT(tensor_are_same_shape(a, b),
               "cross_entropy_loss_back: logits 与 target 必须同形状（与 ggml 一致）");
    TRC_ASSERT(a->nb[0] == (size_t) type_size(a->type) && b->nb[0] == (size_t) type_size(b->type),
               "cross_entropy_loss_back: 最内维必须连续（与 ggml 一致）");

    Tensor* result = tensor_dup_meta(ctx, a);
    result->op = OP_CROSS_ENTROPY_LOSS_BACK;
    result->src[0] = grad;
    result->src[1] = a;
    result->src[2] = b;
    result->nsrc = 3;
    return result;
}

Tensor* mul_mat(Context* ctx, Tensor* a, Tensor* b) {
    TRC_ASSERT(a != nullptr && b != nullptr, "mul_mat: 输入为空");
    TRC_ASSERT(a->type == b->type, "mul_mat: 输入类型不一致");
    TRC_ASSERT(a->ne[0] == b->ne[0], "mul_mat: 内维不一致 (a.ne[0]=%lld, b.ne[0]=%lld)",
               (long long) a->ne[0], (long long) b->ne[0]);
    // ggml_can_mul_mat：b 的 batch 平面数必须是 a 的整数倍（a 按取模广播到 b 的 batch）
    TRC_ASSERT(b->ne[2] % a->ne[2] == 0 && b->ne[3] % a->ne[3] == 0,
               "mul_mat: a 的 batch [%lld,%lld] 无法广播到 b 的 batch [%lld,%lld]"
               "（要求 b.ne %% a.ne == 0，与 ggml_can_mul_mat 一致）",
               (long long) a->ne[2], (long long) a->ne[3], (long long) b->ne[2], (long long) b->ne[3]);

    // ggml_mul_mat 语义：a [k, m, a2, a3] * b [k, n, b2, b3] -> [m, n, b2, b3]
    // 平面 (i2, i3) 使用 a 的平面 (i2 % a2, i3 % a3)
    Tensor* result = new_tensor_4d(ctx, a->type, a->ne[1], b->ne[1], b->ne[2], b->ne[3]);
    result->op = OP_MUL_MAT;
    result->src[0] = a;
    result->src[1] = b;
    result->nsrc = 2;
    return result;
}

// ---------------- 索引 ----------------

Tensor* get_rows(Context* ctx, Tensor* a, Tensor* b) {
    TRC_ASSERT(a != nullptr && b != nullptr, "get_rows: 输入为空");
    TRC_ASSERT(b->type == TYPE_I32, "get_rows: 索引类型必须是 I32（与 ggml 一致）");
    TRC_ASSERT(a->ne[2] == b->ne[1] && a->ne[3] == b->ne[2], "get_rows: 表与索引的形状不匹配（与 ggml 一致）");
    TRC_ASSERT(b->ne[3] == 1, "get_rows: 索引 b->ne[3] 必须为 1（与 ggml 一致）");
    TRC_ASSERT(a->type == TYPE_F32 || a->type == TYPE_F16, "get_rows: 表类型一期仅支持 f32/f16");

    // ggml 语义：结果 [a->ne[0], b->ne[0], b->ne[1], b->ne[2]]，F32
    const int64_t ne[MAX_DIMS] = {a->ne[0], b->ne[0], b->ne[1], b->ne[2]};
    Tensor* result = new_tensor_nd(ctx, TYPE_F32, MAX_DIMS, ne);
    result->op = OP_GET_ROWS;
    result->src[0] = a;
    result->src[1] = b;
    result->nsrc = 2;
    return result;
}

Tensor* get_rows_back(Context* ctx, Tensor* a, Tensor* b, Tensor* c) {
    TRC_ASSERT(a != nullptr && b != nullptr && c != nullptr, "get_rows_back: 输入为空");
    TRC_ASSERT(b->type == TYPE_I32, "get_rows_back: 索引类型必须是 I32（与 ggml 一致）");
    TRC_ASSERT(a->type == TYPE_F32 && c->type == TYPE_F32, "get_rows_back: 一期仅支持 f32");
    TRC_ASSERT(b->ne[3] == 1, "get_rows_back: 索引 b->ne[3] 必须为 1");
    TRC_ASSERT(a->ne[0] == c->ne[0], "get_rows_back: 梯度与表的行宽不一致");
    TRC_ASSERT(a->ne[1] == b->ne[0] && a->ne[2] == b->ne[1] && a->ne[3] == b->ne[2],
               "get_rows_back: 梯度与索引的形状不匹配");

    Tensor* result = new_tensor_nd(ctx, c->type, MAX_DIMS, c->ne);
    result->op = OP_GET_ROWS_BACK;
    result->src[0] = a;
    result->src[1] = b;
    result->src[2] = c;  // 仅用于输出形状（内核不读取其数据）；登记为 src 以便图覆盖/分配（M2.2d）
    result->nsrc = 3;
    return result;
}

Tensor* set_rows(Context* ctx, Tensor* a, Tensor* b, Tensor* c) {
    TRC_ASSERT(a != nullptr && b != nullptr && c != nullptr, "set_rows: 输入为空");
    TRC_ASSERT(a->ne[0] == b->ne[0], "set_rows: a->ne[0] 必须等于 b->ne[0]（与 ggml 一致）");
    TRC_ASSERT(a->ne[2] == b->ne[2] && a->ne[3] == b->ne[3], "set_rows: a/b 的 ne2/ne3 必须一致");
    TRC_ASSERT(b->ne[1] == c->ne[0], "set_rows: b->ne[1] 必须等于 c->ne[0]");
    TRC_ASSERT(b->ne[2] % c->ne[1] == 0 && b->ne[3] % c->ne[2] == 0, "set_rows: b/c 的形状不匹配");
    TRC_ASSERT(c->ne[3] == 1, "set_rows: c->ne[3] 必须为 1");
    TRC_ASSERT(b->type == TYPE_F32 || b->type == TYPE_F16, "set_rows: b 类型一期仅支持 f32/f16");
    TRC_ASSERT(c->type == TYPE_I32 || c->type == TYPE_I64, "set_rows: 索引类型必须是 I32/I64");
    TRC_ASSERT(type_is_supported(a->type), "set_rows: 目标类型 %s 不支持", type_name(a->type));

    // 结果与 a 共享内存（写入 a），与 ggml_set_rows 一致；src 顺序照搬 ggml（b, c, a）
    Tensor* result = view_4d(ctx, a, a->ne[0], a->ne[1], a->ne[2], a->ne[3], 0);
    result->op = OP_SET_ROWS;
    result->src[0] = b;  // 源行数据
    result->src[1] = c;  // 行索引
    result->src[2] = a;  // 目标张量
    result->nsrc = 3;
    return result;
}

// ---------------- 形状扩展 ----------------

Tensor* repeat(Context* ctx, Tensor* a, Tensor* target) {
    TRC_ASSERT(a != nullptr && target != nullptr, "repeat: 输入为空");
    TRC_ASSERT(tensor_can_repeat(a, target), "repeat: a 无法重复到 target 的形状（与 ggml 一致）");

    Tensor* result = new_tensor_nd(ctx, a->type, MAX_DIMS, target->ne);
    result->op = OP_REPEAT;
    result->src[0] = a;
    result->nsrc = 1;
    return result;
}

Tensor* repeat_back(Context* ctx, Tensor* a, Tensor* b) {
    TRC_ASSERT(a != nullptr && b != nullptr, "repeat_back: 输入为空");
    // ggml_can_repeat(b, a)：a 的每一维必须是 b 的整数倍（沿被重复维求和回 b）
    TRC_ASSERT(a->ne[0] % b->ne[0] == 0 && a->ne[1] % b->ne[1] == 0 && a->ne[2] % b->ne[2] == 0 &&
                   a->ne[3] % b->ne[3] == 0,
               "repeat_back: a 的形状无法按 b 归约（要求 a.ne %% b.ne == 0，与 ggml_repeat_back 一致）");

    Tensor* result = new_tensor_nd(ctx, a->type, MAX_DIMS, b->ne);
    result->op = OP_REPEAT_BACK;
    result->src[0] = a;
    result->nsrc = 1;
    return result;
}

Tensor* concat(Context* ctx, Tensor* a, Tensor* b, int dim) {
    TRC_ASSERT(a != nullptr && b != nullptr, "concat: 输入为空");
    TRC_ASSERT(dim >= 0 && dim < MAX_DIMS, "concat: dim 非法 %d", dim);
    TRC_ASSERT(a->type == b->type, "concat: 输入类型不一致");

    int64_t ne[MAX_DIMS];
    for (int d = 0; d < MAX_DIMS; ++d) {
        if (d == dim) {
            ne[d] = a->ne[d] + b->ne[d];
        } else {
            TRC_ASSERT(a->ne[d] == b->ne[d], "concat: 第 %d 维形状不一致 (%lld vs %lld)", d,
                       (long long) a->ne[d], (long long) b->ne[d]);
            ne[d] = a->ne[d];
        }
    }

    Tensor* result = new_tensor_nd(ctx, a->type, MAX_DIMS, ne);
    result->op = OP_CONCAT;
    result->src[0] = a;
    result->src[1] = b;
    result->nsrc = 2;
    result->op_params[0] = dim;
    return result;
}

Tensor* pad_ext(Context* ctx, Tensor* a, int lp0, int rp0, int lp1, int rp1, int lp2, int rp2,
                int lp3, int rp3, PadMode mode) {
    TRC_ASSERT(a != nullptr, "pad_ext: 输入为空");
    const int lp[MAX_DIMS] = {lp0, lp1, lp2, lp3};
    const int rp[MAX_DIMS] = {rp0, rp1, rp2, rp3};
    for (int d = 0; d < MAX_DIMS; ++d) {
        TRC_ASSERT(lp[d] >= 0 && rp[d] >= 0, "pad_ext: 补齐量必须非负（第 %d 维）", d);
        if (mode == PAD_REFLECT) {
            TRC_ASSERT(lp[d] < a->ne[d] && rp[d] < a->ne[d],
                       "pad_ext(reflect): 每侧填充量必须小于对应维长度（第 %d 维：ne=%lld, lp=%d, rp=%d）",
                       d, (long long) a->ne[d], lp[d], rp[d]);
        }
    }
    TRC_ASSERT(mode == PAD_ZERO || mode == PAD_REFLECT, "pad_ext: 未知填充模式 %d", (int) mode);

    const int64_t ne[MAX_DIMS] = {a->ne[0] + lp0 + rp0, a->ne[1] + lp1 + rp1, a->ne[2] + lp2 + rp2,
                                  a->ne[3] + lp3 + rp3};
    Tensor* result = new_tensor_nd(ctx, a->type, MAX_DIMS, ne);
    result->op = OP_PAD;
    result->src[0] = a;
    result->nsrc = 1;

    // op_params 与 ggml_pad_ext 一致：lp0,rp0,lp1,rp1,lp2,rp2,lp3,rp3；
    // op_params[8] 存 mode（ggml 不使用该槽位）
    int32_t params[16] = {0};
    const int32_t pads[8] = {lp0, rp0, lp1, rp1, lp2, rp2, lp3, rp3};
    std::memcpy(params, pads, sizeof(pads));
    params[8] = (int32_t) mode;
    std::memcpy(result->op_params, params, sizeof(params));
    return result;
}

Tensor* pad(Context* ctx, Tensor* a, int p0, int p1, int p2, int p3) {
    return pad_ext(ctx, a, 0, p0, 0, p1, 0, p2, 0, p3, PAD_ZERO);
}

Tensor* pad_back(Context* ctx, Tensor* a, Tensor* b, int lp0, int rp0, int lp1, int rp1, int lp2,
                 int rp2, int lp3, int rp3) {
    TRC_ASSERT(a != nullptr && b != nullptr, "pad_back: 输入为空");
    TRC_ASSERT(a->type == b->type, "pad_back: 输入类型不一致");

    Tensor* result = new_tensor_nd(ctx, b->type, MAX_DIMS, b->ne);
    result->op = OP_PAD_BACK;
    result->src[0] = a;  // 前向输出的梯度
    result->src[1] = b;  // 前向输入（仅取形状；登记为 src 以便图覆盖/分配）
    result->nsrc = 2;

    int32_t params[16] = {0};
    const int32_t pads[8] = {lp0, rp0, lp1, rp1, lp2, rp2, lp3, rp3};
    std::memcpy(params, pads, sizeof(pads));
    params[8] = (int32_t) PAD_REFLECT;
    std::memcpy(result->op_params, params, sizeof(params));
    return result;
}

// ---------------- 卷积 ----------------

namespace {

// 卷积输出尺寸（与 ggml_calc_conv_output_size 一致）
int64_t conv_output_size(int64_t in, int64_t k, int s, int p, int d) {
    return (in + 2 * p - d * (k - 1) - 1) / s + 1;
}

} // namespace

Tensor* im2col(Context* ctx, Tensor* a, Tensor* b, int s0, int s1, int p0, int p1, int d0, int d1,
               bool is_2d, Type dst_type) {
    TRC_ASSERT(a != nullptr && b != nullptr, "im2col: 输入为空");
    if (is_2d) {
        TRC_ASSERT(a->ne[2] == b->ne[2], "im2col(2D): a->ne[2] 必须等于 b->ne[2]（IC）");
    } else {
        TRC_ASSERT(b->ne[1] == a->ne[1], "im2col(1D): b->ne[1] 必须等于 a->ne[1]（IC）");
        TRC_ASSERT(b->ne[3] == 1, "im2col(1D): b->ne[3] 必须为 1");
    }

    const int64_t OH = is_2d ? conv_output_size(b->ne[1], a->ne[1], s1, p1, d1) : 0;
    const int64_t OW = conv_output_size(b->ne[0], a->ne[0], s0, p0, d0);
    TRC_ASSERT((!is_2d || OH > 0) && OW > 0, "im2col: 输入相对卷积核太小（输出尺寸非正）");

    const int64_t ne[MAX_DIMS] = {
        is_2d ? (a->ne[2] * a->ne[1] * a->ne[0]) : (a->ne[1] * a->ne[0]),
        OW,
        is_2d ? OH : b->ne[2],
        is_2d ? b->ne[3] : 1,
    };

    Tensor* result = new_tensor_nd(ctx, dst_type, MAX_DIMS, ne);
    result->op = OP_IM2COL;
    result->src[0] = a;
    result->src[1] = b;
    result->nsrc = 2;

    const int32_t params[7] = {s0, s1, p0, p1, d0, d1, is_2d ? 1 : 0};
    std::memcpy(result->op_params, params, sizeof(params));
    return result;
}

Tensor* im2col_back(Context* ctx, Tensor* a, Tensor* b, const int64_t* ne, int s0, int s1, int p0,
                    int p1, int d0, int d1, bool is_2d) {
    TRC_ASSERT(a != nullptr && b != nullptr && ne != nullptr, "im2col_back: 输入为空");
    TRC_ASSERT(a->type == TYPE_F32, "im2col_back: 一期仅支持 f32");
    TRC_ASSERT(b->type == TYPE_F32, "im2col_back: 卷积核一期仅支持 f32");

    Tensor* result = new_tensor_nd(ctx, TYPE_F32, MAX_DIMS, ne);
    result->op = OP_IM2COL_BACK;
    result->src[0] = a;  // im2col 输出的梯度
    result->src[1] = b;  // 卷积核（形状用途）
    result->nsrc = 2;

    const int32_t params[7] = {s0, s1, p0, p1, d0, d1, is_2d ? 1 : 0};
    std::memcpy(result->op_params, params, sizeof(params));
    return result;
}

Tensor* conv_1d(Context* ctx, Tensor* a, Tensor* b, int s0, int p0, int d0) {
    TRC_ASSERT(a != nullptr && b != nullptr, "conv_1d: 输入为空");
    TRC_ASSERT(a->type == TYPE_F32 && b->type == TYPE_F32, "conv_1d: 一期仅支持 f32");

    // 与 ggml_conv_1d 相同的组合（一期 im2col 输出 F32）
    Tensor* im = im2col(ctx, a, b, s0, 0, p0, 0, d0, 0, false, TYPE_F32);  // [IC*KW, OW, N, 1]
    Tensor* im2 = reshape_2d(ctx, im, im->ne[0], im->ne[2] * im->ne[1]);   // [IC*KW, OW*N]
    Tensor* w = reshape_2d(ctx, a, a->ne[0] * a->ne[1], a->ne[2]);         // [IC*KW, OC]
    Tensor* r = mul_mat(ctx, im2, w);                                      // [OW*N, OC]

    if (im->ne[2] == 1) {
        return reshape_3d(ctx, r, im->ne[1], a->ne[2], im->ne[2]);  // [OW, OC, N]（与 ggml 一致）
    }
    // N>1：ggml v0.23.0 的 reshape_3d(r, OW, OC, N) 与 r 的实际内存布局不符
    // （mul_mat 行序为 in*OW+iow，而 [OW,OC,N] 要求 in 步长为 OW*OC），只有 N=1 时正确。
    // 正确解释：r 的内存布局按 [OW, N, OC] reshape，再 permute 成 [OW, OC, N]。
    // 差异已记入 docs/兼容性.md §4。
    Tensor* r3 = reshape_3d(ctx, r, im->ne[1], im->ne[2], a->ne[2]);  // [OW, N, OC]
    return cont(ctx, permute(ctx, r3, 0, 2, 1, 3));                    // [OW, OC, N, 1]
}

Tensor* conv_2d(Context* ctx, Tensor* a, Tensor* b, int s0, int s1, int p0, int p1, int d0, int d1) {
    TRC_ASSERT(a != nullptr && b != nullptr, "conv_2d: 输入为空");
    TRC_ASSERT(a->type == TYPE_F32 && b->type == TYPE_F32, "conv_2d: 一期仅支持 f32");

    // 与 ggml_conv_2d 相同的组合
    Tensor* im = im2col(ctx, a, b, s0, s1, p0, p1, d0, d1, true, TYPE_F32);  // [IC*KH*KW, OW, OH, N]
    Tensor* im2 = reshape_2d(ctx, im, im->ne[0], im->ne[3] * im->ne[2] * im->ne[1]);  // [IC*KH*KW, N*OH*OW]
    Tensor* w = reshape_2d(ctx, a, a->ne[0] * a->ne[1] * a->ne[2], a->ne[3]);         // [IC*KH*KW, OC]
    Tensor* r = mul_mat(ctx, im2, w);                                                 // [N*OH*OW, OC]
    Tensor* r4 = reshape_4d(ctx, r, im->ne[1], im->ne[2], im->ne[3], a->ne[3]);       // [OW, OH, N, OC]
    return cont(ctx, permute(ctx, r4, 0, 1, 3, 2));                                   // [OW, OH, OC, N]
}

Tensor* col2im_1d(Context* ctx, Tensor* a, int s0, int oc, int p0) {
    TRC_ASSERT(a != nullptr, "col2im_1d: 输入为空");
    TRC_ASSERT(a->ne[2] == 1 && a->ne[3] == 1, "col2im_1d: 输入必须是 2D 矩阵（与 ggml 一致）");
    TRC_ASSERT(tensor_is_contiguous(a), "col2im_1d: 输入必须连续（与 ggml 一致）");
    TRC_ASSERT(a->type == TYPE_F32, "col2im_1d: 一期仅支持 f32");
    TRC_ASSERT(s0 > 0 && oc > 0 && p0 >= 0, "col2im_1d: 参数非法");

    const int64_t K_OC = a->ne[0];
    const int64_t T_in = a->ne[1];
    const int64_t K = K_OC / oc;
    const int64_t T_out = (T_in - 1) * s0 + K - 2 * p0;
    TRC_ASSERT(K_OC == K * oc, "col2im_1d: a->ne[0] 必须能被 oc 整除");
    TRC_ASSERT(K > 0 && T_out > 0, "col2im_1d: 输出尺寸非正");

    Tensor* result = new_tensor_2d(ctx, a->type, T_out, oc);
    result->op = OP_COL2IM;
    result->src[0] = a;
    result->nsrc = 1;

    const int32_t params[3] = {s0, oc, p0};
    std::memcpy(result->op_params, params, sizeof(params));
    return result;
}

Tensor* conv_transpose_1d(Context* ctx, Tensor* a, Tensor* b, int s0, int p0, int d0) {
    TRC_ASSERT(a != nullptr && b != nullptr, "conv_transpose_1d: 输入为空");
    TRC_ASSERT(b->ne[2] == 1 && b->ne[3] == 1, "conv_transpose_1d: b 必须是 2D 矩阵（与 ggml 一致）");
    TRC_ASSERT(a->ne[2] == b->ne[1], "conv_transpose_1d: a->ne[2] 必须等于 b->ne[1]（Cin）");
    TRC_ASSERT(a->ne[3] == 1, "conv_transpose_1d: a->ne[3] 必须为 1（与 ggml 一致）");
    TRC_ASSERT(p0 == 0, "conv_transpose_1d: 一期仅支持 p0==0（与 ggml 一致）");
    TRC_ASSERT(d0 == 1, "conv_transpose_1d: 一期仅支持 d0==1（与 ggml 一致）");
    TRC_ASSERT(a->type == TYPE_F32 && b->type == TYPE_F32, "conv_transpose_1d: 一期仅支持 f32");

    const int64_t T_out = (b->ne[0] - 1) * s0 + a->ne[0];
    const int64_t ne[MAX_DIMS] = {T_out, a->ne[1], b->ne[2], 1};
    Tensor* result = new_tensor_nd(ctx, TYPE_F32, MAX_DIMS, ne);
    result->op = OP_CONV_TRANSPOSE_1D;
    result->src[0] = a;
    result->src[1] = b;
    result->nsrc = 2;

    const int32_t params[3] = {s0, p0, d0};
    std::memcpy(result->op_params, params, sizeof(params));
    return result;
}

// ---------------- 池化 / 2D 转置卷积（M2.3g）----------------

namespace {

// 池化输出尺寸（与 ggml_calc_pool_output_size 一致：正数整除 = floor）
int64_t pool_output_size(int64_t in, int k, int s, int p) {
    return (in + 2 * p - k) / s + 1;
}

} // namespace

Tensor* pool_2d(Context* ctx, Tensor* a, PoolMode mode, int k0, int k1, int s0, int s1, int p0,
                int p1) {
    TRC_ASSERT(a != nullptr, "pool_2d: 输入为空");
    TRC_ASSERT(mode == POOL_MAX || mode == POOL_AVG, "pool_2d: 未知池化模式 %d", (int) mode);
    TRC_ASSERT(k0 > 0 && k1 > 0 && s0 > 0 && s1 > 0 && p0 >= 0 && p1 >= 0,
               "pool_2d: 核/步长/填充参数非法（k>0、s>0、p>=0）");
    TRC_ASSERT(a->type == TYPE_F32, "pool_2d: 一期仅支持 f32");

    const int64_t OW = pool_output_size(a->ne[0], k0, s0, p0);
    const int64_t OH = pool_output_size(a->ne[1], k1, s1, p1);
    TRC_ASSERT(OW > 0 && OH > 0, "pool_2d: 输入相对核/填充太小（输出尺寸非正）");

    Tensor* result = new_tensor_4d(ctx, TYPE_F32, OW, OH, a->ne[2], a->ne[3]);
    result->op = OP_POOL_2D;
    result->src[0] = a;
    result->nsrc = 1;

    // op_params 槽位与 ggml_pool_2d 一致：{op, k0, k1, s0, s1, p0, p1}
    const int32_t params[7] = {(int32_t) mode, k0, k1, s0, s1, p0, p1};
    std::memcpy(result->op_params, params, sizeof(params));
    return result;
}

Tensor* pool_2d_back(Context* ctx, Tensor* a, Tensor* b, PoolMode mode, int k0, int k1, int s0,
                     int s1, int p0, int p1) {
    TRC_ASSERT(a != nullptr && b != nullptr, "pool_2d_back: 输入为空");
    TRC_ASSERT(mode == POOL_MAX || mode == POOL_AVG, "pool_2d_back: 未知池化模式 %d", (int) mode);
    TRC_ASSERT(a->type == TYPE_F32 && b->type == TYPE_F32, "pool_2d_back: 一期仅支持 f32");

    const int64_t OW = pool_output_size(b->ne[0], k0, s0, p0);
    const int64_t OH = pool_output_size(b->ne[1], k1, s1, p1);
    TRC_ASSERT(OW == a->ne[0] && OH == a->ne[1] && a->ne[2] == b->ne[2] && a->ne[3] == b->ne[3],
               "pool_2d_back: 梯度形状必须是前向输出形状（[%lld,%lld,%lld,%lld] vs [%lld,%lld,%lld,%lld]）",
               (long long) OW, (long long) OH, (long long) b->ne[2], (long long) b->ne[3],
               (long long) a->ne[0], (long long) a->ne[1], (long long) a->ne[2], (long long) a->ne[3]);

    Tensor* result = new_tensor_4d(ctx, TYPE_F32, b->ne[0], b->ne[1], b->ne[2], b->ne[3]);
    result->op = OP_POOL_2D_BACK;
    result->src[0] = a;  // 前向输出梯度
    result->src[1] = b;  // 前向输入（仅取形状；同时保证图级分配覆盖）
    result->nsrc = 2;

    const int32_t params[7] = {(int32_t) mode, k0, k1, s0, s1, p0, p1};
    std::memcpy(result->op_params, params, sizeof(params));
    return result;
}

Tensor* conv_transpose_2d(Context* ctx, Tensor* a, Tensor* b, int stride) {
    TRC_ASSERT(a != nullptr && b != nullptr, "conv_transpose_2d: 输入为空");
    TRC_ASSERT(stride > 0, "conv_transpose_2d: stride 必须为正");
    TRC_ASSERT(a->ne[3] == b->ne[2], "conv_transpose_2d: a->ne[3] 必须等于 b->ne[2]（Cin）");
    TRC_ASSERT(a->type == TYPE_F32 && b->type == TYPE_F32, "conv_transpose_2d: 一期仅支持 f32");

    const int64_t OW = (b->ne[0] - 1) * stride + a->ne[0];
    const int64_t OH = (b->ne[1] - 1) * stride + a->ne[1];
    Tensor* result = new_tensor_4d(ctx, TYPE_F32, OW, OH, a->ne[2], b->ne[3]);
    result->op = OP_CONV_TRANSPOSE_2D;
    result->src[0] = a;
    result->src[1] = b;
    result->nsrc = 2;

    const int32_t params[1] = {stride};
    std::memcpy(result->op_params, params, sizeof(params));
    return result;
}

// ---------------- 设备端优化器步（M4.1）----------------

namespace {

// 公共构造：结果 = param 的完整视图（就地更新），src 顺序固定
Tensor* new_opt_step(Context* ctx, Op op, Tensor* param, Tensor* grad, Tensor* m, Tensor* v,
                     int n_extra_src) {
    TRC_ASSERT(param != nullptr && grad != nullptr && m != nullptr, "%s: 输入为空", op_name(op));
    TRC_ASSERT(param->type == TYPE_F32 && grad->type == TYPE_F32 && m->type == TYPE_F32,
               "%s: 仅支持 F32", op_name(op));
    Tensor* result =
        view_4d(ctx, param, param->ne[0], param->ne[1], param->ne[2], param->ne[3], 0);
    result->op     = op;
    result->src[0] = param;
    result->src[1] = grad;
    result->src[2] = m;
    result->src[3] = v;
    result->nsrc   = 3 + n_extra_src;  // SGD: 3，AdamW: 4
    return result;
}

} // namespace

Tensor* opt_step_adamw(Context* ctx, Tensor* param, Tensor* grad, Tensor* m, Tensor* v,
                       float step_sz, float bc2_sqrt, float decay, float beta1, float beta2,
                       float eps) {
    TRC_ASSERT(v != nullptr, "opt_step_adamw: v 为空");
    TRC_ASSERT(v->type == TYPE_F32, "opt_step_adamw: v 仅支持 F32");
    Tensor* result = new_opt_step(ctx, OP_OPT_STEP_ADAMW, param, grad, m, v, 1);
    const float params[6] = {step_sz, bc2_sqrt, decay, beta1, beta2, eps};
    std::memcpy(result->op_params, params, sizeof(params));
    return result;
}

Tensor* opt_step_sgd(Context* ctx, Tensor* param, Tensor* grad, Tensor* momentum, float lr,
                     float momentum_coef, float dampening, float weight_decay, bool nesterov,
                     bool is_first) {
    Tensor* result = new_opt_step(ctx, OP_OPT_STEP_SGD, param, grad, momentum, nullptr, 0);
    const float params[6] = {lr, momentum_coef, dampening, weight_decay, nesterov ? 1.0f : 0.0f,
                             is_first ? 1.0f : 0.0f};
    std::memcpy(result->op_params, params, sizeof(params));
    return result;
}

// ---------------- 设备端梯度裁剪原语（M4.5）----------------

Tensor* sum_sqr_acc(Context* ctx, Tensor* acc, Tensor* a) {
    TRC_ASSERT(acc != nullptr && a != nullptr, "sum_sqr_acc: 输入为空");
    TRC_ASSERT(acc->type == TYPE_F32 && a->type == TYPE_F32, "sum_sqr_acc: 仅支持 F32");
    TRC_ASSERT(tensor_nelements(acc) == 1, "sum_sqr_acc: acc 必须是 1 元素");
    TRC_ASSERT(tensor_is_contiguous(a), "sum_sqr_acc: a 必须连续");
    // 结果 = acc 的完整视图（就地累加）
    Tensor* result =
        view_4d(ctx, acc, acc->ne[0], acc->ne[1], acc->ne[2], acc->ne[3], 0);
    result->op     = OP_SUM_SQR_ACC;
    result->src[0] = acc;
    result->src[1] = a;
    result->nsrc   = 2;
    return result;
}

Tensor* clip_scale_inplace(Context* ctx, Tensor* a, Tensor* norm, float max_norm, float eps) {
    TRC_ASSERT(a != nullptr && norm != nullptr, "clip_scale_inplace: 输入为空");
    TRC_ASSERT(a->type == TYPE_F32 && norm->type == TYPE_F32, "clip_scale_inplace: 仅支持 F32");
    TRC_ASSERT(tensor_nelements(norm) == 1, "clip_scale_inplace: norm 必须是 1 元素");
    TRC_ASSERT(tensor_is_contiguous(a), "clip_scale_inplace: a 必须连续");
    // 结果 = a 的完整视图（就地缩放）
    Tensor* result = view_4d(ctx, a, a->ne[0], a->ne[1], a->ne[2], a->ne[3], 0);
    result->op     = OP_CLIP_SCALE_INPLACE;
    result->src[0] = a;
    result->src[1] = norm;
    result->nsrc   = 2;
    const float params[2] = {max_norm, eps};
    std::memcpy(result->op_params, params, sizeof(params));
    return result;
}

Tensor* weightnorm_sync(Context* ctx, Tensor* v, Tensor* g) {
    TRC_ASSERT(v != nullptr && g != nullptr, "weightnorm_sync: 输入为空");
    TRC_ASSERT(v->type == TYPE_F32 && g->type == TYPE_F32, "weightnorm_sync: 仅支持 F32");
    TRC_ASSERT(tensor_is_contiguous(v) && tensor_is_contiguous(g),
               "weightnorm_sync: v/g 必须连续");
    // 结果 = g 的完整视图（就地写回各输出通道范数）
    Tensor* result = view_4d(ctx, g, g->ne[0], g->ne[1], g->ne[2], g->ne[3], 0);
    result->op     = OP_WEIGHTNORM_SYNC;
    result->src[0] = v;
    result->src[1] = g;
    result->nsrc   = 2;
    return result;
}

} // namespace traincpp
