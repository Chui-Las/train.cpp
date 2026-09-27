// train.cpp - 损失函数（训练扩展，M1.5c）
//
// 设计：
//   - 交叉熵为独立算子 `cross_entropy_loss` / `cross_entropy_loss_back`（声明见 trc_ops.h），
//     数学语义与 ggml 的 `ggml_cross_entropy_loss(_back)` 对齐（F32、同形状概率/one-hot 目标）
//   - MSE/L1 由已有算子组合实现，不新增 kernel；reduction 固定为全元素均值，
//     与 PyTorch `nn.MSELoss()/nn.L1Loss()` 的默认 reduction='mean' 一致
//   - 所有损失输出 F32 标量 [1]，可直接 `tensor_set_loss` + `graph_build_backward_expand`
//
// 反向：全部由 autograd 组合/kernel 自动给出，无需额外 API。
#pragma once

#include "trc_context.h"
#include "trc_tensor.h"

namespace traincpp {

// mean((pred - target)^2)：全元素均值（PyTorch MSELoss(reduction='mean')）
Tensor* mse_loss(Context* ctx, Tensor* pred, Tensor* target);

// mean(|pred - target|)：全元素均值（PyTorch L1Loss(reduction='mean')）
Tensor* l1_loss(Context* ctx, Tensor* pred, Tensor* target);

} // namespace traincpp
