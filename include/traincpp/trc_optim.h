// train.cpp - 优化器与学习率调度（主机端实现，后端无关）
//
// 设计要点（M1.5a）：
//   - 优化器在主机侧读写参数与梯度（tensor_get/tensor_set），CPU/Vulkan 等后端同样可用；
//     设备端融合算子（OP_OPT_STEP_SGD/ADAMW）留待后续性能阶段。
//   - 超参与更新公式与 PyTorch（torch.optim.SGD / torch.optim.AdamW）逐条对齐，
//     便于黄金数据对拍与从 PyTorch 迁移训练脚本。
//   - 状态（SGD momentum、AdamW m/v）以 F32 张量形式暴露，供 checkpoint 序列化（M1.6）。
//
// 典型用法：
//   Optimizer* opt = optim_adamw_new(ctx, params, n_params, AdamwOptions{});
//   ... 反向 + graph_reset + compute ...
//   optim_step(opt);        // 读取各参数 t->grad 更新参数
//   optim_zero_grad(opt);   // 或每步前 graph_reset 清梯度
//
// 梯度累积（micro-batch，M1.5d；详见 trc_autograd.h）：
//   graph_reset(g);
//   for (k = 0; k < K; ++k) { graph_reset_accumulate(g); 填 micro-batch; compute; }
//   optim_clip_grad_norm(opt, max_norm);   // 可选
//   optim_step(opt);
#pragma once

#include "trc_tensor.h"

namespace traincpp {

struct Context;
struct Optimizer;  // 不透明句柄

// ---------------- SGD ----------------
// 与 torch.optim.SGD 对齐：
//   d_p = g + weight_decay·p
//   momentum != 0 时：buf = momentum·buf + (1-dampening)·d_p（首步直接取 d_p，不乘 dampening）
//                     nesterov ? d_p = d_p + momentum·buf : d_p = buf
//   p -= lr·d_p
struct SgdOptions {
    float lr           = 1e-2f;
    float momentum     = 0.0f;
    float dampening    = 0.0f;
    float weight_decay = 0.0f;
    bool  nesterov     = false;
};

// ---------------- AdamW ----------------
// 与 torch.optim.AdamW 对齐（解耦权重衰减）：
//   p *= 1 - lr·weight_decay
//   m = b1·m + (1-b1)·g；v = b2·v + (1-b2)·g²
//   p -= lr·(m/(1-b1^t)) / (sqrt(v/(1-b2^t)) + eps)
struct AdamwOptions {
    float lr           = 1e-3f;
    float beta1        = 0.9f;
    float beta2        = 0.999f;
    float eps          = 1e-8f;
    float weight_decay = 1e-2f;
};

// 创建优化器：params 为可训练参数数组（通常来自各层的 ParamList）
Optimizer* optim_sgd_new(Context* ctx, Tensor* const* params, int64_t n_params,
                         const SgdOptions& opts);
Optimizer* optim_adamw_new(Context* ctx, Tensor* const* params, int64_t n_params,
                           const AdamwOptions& opts);
void optim_free(Optimizer* opt);

// 参数分组：为已有优化器追加一组参数（组间超参独立；步数由优化器统一计数）
void optim_add_param_group_sgd(Optimizer* opt, Tensor* const* params, int64_t n_params,
                               const SgdOptions& opts);
void optim_add_param_group_adamw(Optimizer* opt, Tensor* const* params, int64_t n_params,
                                 const AdamwOptions& opts);

int64_t optim_group_count(const Optimizer* opt);
int64_t optim_param_count(const Optimizer* opt);           // 全部组的参数总数
Tensor* optim_param(const Optimizer* opt, int64_t index);  // 按 组→参数 顺序
// 状态张量（SGD：每参数 1 个 momentum；AdamW：每参数 2 个 m、v），按 组→参数 顺序展开
int64_t optim_state_count(const Optimizer* opt);
Tensor* optim_state(const Optimizer* opt, int64_t index);

// ---- 分组信息与断点续训（M1.6b；checkpoint 序列化使用）----

enum class OptimType : int32_t {
    SGD   = 0,
    ADAMW = 1,
};

OptimType optim_group_type(const Optimizer* opt, int64_t group);        // 该组类型
int64_t   optim_group_param_count(const Optimizer* opt, int64_t group); // 该组参数数
int64_t   optim_group_state_count(const Optimizer* opt, int64_t group); // 该组状态张量数
float     optim_get_lr_group(const Optimizer* opt, int64_t group);      // 该组学习率
// 该组超参（类型不符时报错）；checkpoint 保存/校验使用
SgdOptions   optim_group_sgd_options(const Optimizer* opt, int64_t group);
AdamwOptions optim_group_adamw_options(const Optimizer* opt, int64_t group);

// 分配优化器状态张量的数据（断点续训在 checkpoint_load 之前调用；正常训练无需调用，
// 首次 optim_step 会自动分配）。参数已由 buffer 分配时使用参数所在后端的 BufferType。
void optim_alloc_state(Optimizer* opt);

// 恢复 step 计数（断点续训；正常训练不要调用；AdamW 的 bias correction 依赖它）
void optim_set_step_count(Optimizer* opt, int64_t step);

// 每个参数各自的优化器状态步数（AdamW bias correction / SGD 首步判断；断点续训必须一并恢复）。
// param_index 为 组→参数 顺序的参数下标（同 optim_param）
int64_t optim_state_step(const Optimizer* opt, int64_t param_index);
void    optim_set_state_step(Optimizer* opt, int64_t param_index, int64_t step);

float   optim_get_lr(const Optimizer* opt);      // 第 0 组学习率
void    optim_set_lr(Optimizer* opt, float lr);  // 覆盖全部组
void    optim_set_lr_group(Optimizer* opt, int64_t group, float lr);
int64_t optim_step_count(const Optimizer* opt);  // 已执行的 optim_step 次数

// 清零全部参数梯度（t->grad_acc；为空时无操作）
void optim_zero_grad(Optimizer* opt);

// 执行一步参数更新（读取 t->grad / t->grad_acc；要求已完成反向与 buffer 分配）
void optim_step(Optimizer* opt);

// 全局 L2 梯度裁剪（同 torch.nn.utils.clip_grad_norm_）：
// 返回裁剪前的总范数；总范数超过 max_norm 时原地缩放全部梯度
float optim_clip_grad_norm(Optimizer* opt, float max_norm);

// ---------------- 学习率调度 ----------------
struct LrScheduler;  // 不透明句柄

// StepLR：每 step_size 次 step 后 lr *= gamma（与 torch 一致：第 step_size 次 step 生效）
LrScheduler* lr_scheduler_step_new(Optimizer* opt, int64_t step_size, float gamma);
// ExponentialLR：每次 step 后 lr *= gamma
LrScheduler* lr_scheduler_exponential_new(Optimizer* opt, float gamma);
// CosineAnnealingLR：lr = eta_min + (base-eta_min)·(1+cos(pi·t/T_max))/2（t 为已推进次数）
LrScheduler* lr_scheduler_cosine_new(Optimizer* opt, int64_t t_max, float eta_min);
// 线性 warmup + 余弦退火（本库扩展）：
//   t <= warmup_steps：lr = base·t/warmup_steps（t=0 时为 0）
//   t >  warmup_steps：在剩余 total_steps-warmup_steps 步内从 base 余弦退火到 eta_min
LrScheduler* lr_scheduler_warmup_cosine_new(Optimizer* opt, int64_t warmup_steps, int64_t total_steps,
                                            float eta_min);
void lr_scheduler_free(LrScheduler* sched);

// 推进一步并写回优化器；返回新的学习率
float   lr_scheduler_step(LrScheduler* sched);
float   lr_scheduler_get_lr(const LrScheduler* sched);  // 当前（未再推进）学习率
int64_t lr_scheduler_epoch(const LrScheduler* sched);   // 已推进次数

} // namespace traincpp
