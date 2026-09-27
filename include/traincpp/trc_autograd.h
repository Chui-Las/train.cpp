// train.cpp - 自动求导（autograd）
//
// 设计原则（与 ggml 的 build_backward_expand 语义对齐）：
//   - 反向 = 前向算子组合；反向节点追加到同一 Graph，一次 compute 同时完成前向与反向
//   - 梯度存于 Tensor::grad / Tensor::grad_acc（PARAM/LOSS 自动创建 F32 累加器）
//   - 梯度按"先累加后使用"处理，同一张量被多处使用时自动求和
//
// 典型训练循环：
//   Tensor* w = ...; tensor_set_param(w);
//   Tensor* loss = ...; tensor_set_loss(loss);
//   Graph* g = graph_new(ctx);
//   graph_build_forward_expand(ctx, g, loss);
//   graph_build_backward_expand(g);                       // 必须在分配内存之前
//   Buffer* buf = buffer_alloc_ctx_tensors(ctx, dev->default_buffer_type());
//   for (每个训练步) {
//       graph_reset(g);                                   // 清零累加器 + LOSS 播种 1.0
//       tensor_set(输入张量, ...);                          // 刷新本步数据
//       dev->graph_compute(g);                            // 前向 + 反向一次完成
//       // 用 graph_get_grad(g, w) 读梯度做优化器更新（M1.5）
//   }
//
// 梯度语义（M1.5d）：
//   PARAM/LOSS 的梯度在持久累加器 `grad_acc` 中 **inplace 累加**（与 ggml 的 inplace add 一致），
//   同一累加窗口内多次 compute 会累积梯度（用于 micro-batch 梯度累积）。因此每个训练步必须
//   用 graph_reset 清零；累积窗口内用 graph_reset_accumulate 只播种不清零。
#pragma once

#include "trc_graph.h"
#include "trc_tensor.h"

namespace traincpp {

// 标记可训练参数：张量必须是无输入的叶子（OP_NONE）；仅打标志
void tensor_set_param(Tensor* t);

// 取消可训练参数标记（冻结；M2.3e 双图训练协议的"冻结/解冻"基础）。
// 只影响**下一次** graph_build_backward_expand：已构建反向的图不会撤销既有 grad_acc。
void tensor_clear_param(Tensor* t);

// 标记损失：必须是 F32 标量；多个 LOSS 的梯度在反向图中自然累加
void tensor_set_loss(Tensor* t);

// 构建反向图：
//   - 逆拓扑遍历前向节点，按各算子反向规则组装前向算子并追加到同一 Graph
//   - 为 PARAM / LOSS 节点自动创建 F32 梯度累加器（t->grad_acc）并置 t->grad
//   - 必须在 buffer_alloc_ctx_tensors 之前调用（反向张量也要参与分配）
void graph_build_backward_expand(Graph* graph);

// 取张量梯度 / 梯度累加器（等价于 t->grad / t->grad_acc；保留函数形态便于对齐 ggml）
Tensor* graph_get_grad(const Graph* graph, const Tensor* t);
Tensor* graph_get_grad_acc(const Graph* graph, const Tensor* t);

// 清零所有梯度累加器，并把 LOSS 的累加器置 1.0（与 ggml_graph_reset 一致）
// 每个训练步 compute 之前调用；要求已完成 buffer 分配
void graph_reset(Graph* graph);

// 梯度累积：只把 LOSS 的累加器置 1.0，**不清零**其它梯度累加器（M1.5d）
//
// 用途：micro-batch 梯度累积。多个 micro-batch 的反向会把梯度叠加进同一批累加器，
// 累积窗口结束后再做优化器更新。标准循环：
//
//   graph_reset(g);                        // 窗口开始：清零 + 播种
//   for (k = 0; k < K; ++k) {
//       graph_reset_accumulate(g);         // 只播种（重复调用无副作用）
//       tensor_set(输入张量, micro_batch[k]);
//       dev->graph_compute(g);             // 本 micro-batch 的前向 + 反向
//   }
//   optim_clip_grad_norm(opt, max_norm);   // 可选
//   optim_step(opt);
//
// 语义：累积结果是 K 个 micro-batch 梯度之**和**。若损失为批均值，要让累积等价于
// 全批量均值，需要把每个 micro-batch 的损失缩放 1/K（`scale(loss, 1/K)`）。
// 要求已完成 buffer 分配。
void graph_reset_accumulate(Graph* graph);

} // namespace traincpp
