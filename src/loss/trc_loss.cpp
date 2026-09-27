// train.cpp - 损失函数实现（M1.5c）
//
// MSE/L1 由已有算子组合：scale(sum(逐元素误差), 1/N)（全元素均值，与 PyTorch 默认一致）；
// 反向由 autograd 自动得到，无专用 kernel。
#include "traincpp/trc_loss.h"

#include "core/trc_impl.h"
#include "traincpp/trc_ops.h"

namespace traincpp {

namespace {

// 全元素均值：scale(sum(elemwise), 1/nelements)
Tensor* mean_all(Context* ctx, Tensor* elemwise, const char* name) {
    const int64_t n = tensor_nelements(elemwise);
    TRC_ASSERT(n > 0, "%s: 输入元素数为 0", name);
    return scale(ctx, sum(ctx, elemwise), 1.0f / (float) n);
}

} // namespace

Tensor* mse_loss(Context* ctx, Tensor* pred, Tensor* target) {
    TRC_ASSERT(pred != nullptr && target != nullptr, "mse_loss: 输入为空");
    TRC_ASSERT(pred->type == TYPE_F32 && target->type == TYPE_F32, "mse_loss: 一期仅支持 f32");
    TRC_ASSERT(tensor_are_same_shape(pred, target), "mse_loss: pred 与 target 必须同形状");
    return mean_all(ctx, sqr(ctx, sub(ctx, pred, target)), "mse_loss");
}

Tensor* l1_loss(Context* ctx, Tensor* pred, Tensor* target) {
    TRC_ASSERT(pred != nullptr && target != nullptr, "l1_loss: 输入为空");
    TRC_ASSERT(pred->type == TYPE_F32 && target->type == TYPE_F32, "l1_loss: 一期仅支持 f32");
    TRC_ASSERT(tensor_are_same_shape(pred, target), "l1_loss: pred 与 target 必须同形状");
    return mean_all(ctx, abs_op(ctx, sub(ctx, pred, target)), "l1_loss");
}

} // namespace traincpp
