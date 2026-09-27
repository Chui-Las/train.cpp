// train.cpp - 训练检查点（参数 + 优化器状态 + RNG + 调度器；M1.6b；schema v2 多优化器 M2.3f）
//
// 容器：复用 GGUF（见 trc_gguf.h），以 general.architecture = "traincpp.checkpoint" 与推理模型区分；
// schema 版本存于 traincpp.checkpoint.version（**恒写 2；读兼容 v1**）。
//
// 内容：
//   - 全部参数张量（名字取 Tensor::name，必须非空且唯一；优化器要求 F32 连续非视图）
//   - 优化器状态张量（SGD: <param>.momentum；AdamW: <param>.m / <param>.v）
//   - 元数据：按优化器分区（traincpp.optimizer.{i}.*）：分区名/类型/分组/step/每组 lr 与超参、
//     参数名与状态名列表、每参数状态步数；可选每分区调度器 epoch
//   - 可选：RNG 状态（Rng::state，多优化器共享一份）
//
// 用法（断点续训，单优化器）：
//   checkpoint_save("ckpt.gguf", {opt, &rng, sched}, "my.model");
//   // 新进程：构建同结构模型 → buffer_alloc_ctx_tensors → 创建同类型优化器/调度器 →
//   // optim_alloc_state(opt);
//   checkpoint_load("ckpt.gguf", {opt, &rng, sched});
//
// 用法（多优化器，如 GAN 的 G/D 各一个优化器与调度器；schema v2）：
//   CheckpointOptimizer parts[2] = {{"G", opt_g, sched_g}, {"D", opt_d, sched_d}};
//   CheckpointItems items{};
//   items.rng = &rng; items.optimizers = parts; items.optimizer_count = 2;
//   checkpoint_save("ckpt.gguf", items, "gan");
//   // 载入：按分区名匹配（顺序无关），每个优化器先 optim_alloc_state
//
// 契约：按参数名严格匹配（缺名/改名/结构或数量不符/优化器类型或超参不符 → 返回 false 并打印原因）；
// 多优化器按分区名匹配（名字非空且唯一）；分区张量名跨分区不得重复（GGUF 要求全局唯一）；
// 调度器 epoch 只能前进（load 时用 lr_scheduler_step 回放到存档 epoch）。
#pragma once

#include "trc_optim.h"
#include "trc_rng.h"
#include "trc_tensor.h"

namespace traincpp {

// 单个优化器分区（schema v2 多优化器，如 GAN 的 "G"/"D"）：
//   - name 必填、非空、同一文件内唯一（作为分区的匹配键）；
//   - scheduler 可选，与该优化器配套（如两个 CosineAnnealingLR）。
struct CheckpointOptimizer {
    const char*  name      = nullptr;  // 分区名（非空、唯一）
    Optimizer*   optimizer = nullptr;  // 必填
    LrScheduler* scheduler = nullptr;  // 可选
};

struct CheckpointItems {
    Optimizer*   optimizer = nullptr;  // 单优化器快捷字段（旧调用不变；多优化器时忽略）
    Rng*         rng       = nullptr;  // 可选：保存/恢复随机数状态（多优化器只存一份）
    LrScheduler* scheduler = nullptr;  // 可选：单优化器配套调度器（多优化器时忽略）
    // 多优化器（schema v2）：非空时以上 optimizer/scheduler 被忽略；rng 仍共享
    const CheckpointOptimizer* optimizers = nullptr;
    int64_t                    optimizer_count = 0;
};

struct CheckpointInfo {
    int64_t version         = 0;  // schema 版本（1 或 2）
    int64_t step            = 0;  // optim_step_count（第 0 个分区）
    int64_t param_count     = 0;  // 第 0 个分区的参数数
    int64_t group_count     = 0;  // 第 0 个分区的组数
    int64_t scheduler_epoch = 0;  // 第 0 个分区调度器 epoch（无调度器为 0）
    bool    has_rng         = false;
    bool    has_scheduler   = false;  // 第 0 个分区是否有调度器
    char    optimizer_type[16]   = {0};  // "sgd" / "adamw"（第 0 个分区）
    char    model_arch[MAX_NAME] = {0};  // checkpoint_save 传入的模型标记（可为空）
    int64_t optimizer_count = 0;  // 优化器分区数（v1 恒 1）
};

// 保存（覆盖写，恒为 schema v2）：单优化器用 items.optimizer；多优化器用 items.optimizers
// + optimizer_count（分区名非空且唯一；跨分区张量名不得重复）。model_arch 为可选的模型结构标记。
// 参数/状态张量的数据必须已分配（参数经 buffer_alloc_ctx_tensors；状态经 optim_alloc_state 或 step）。
// step_count > 0 但状态未分配时报错（避免写出无法续训的检查点）。
bool checkpoint_save(const char* path, const CheckpointItems& items, const char* model_arch = nullptr);

// 载入（兼容 v1 与 v2）：要求参数张量已分配；文件中含状态时要求状态张量已分配（先 optim_alloc_state）。
// v2 多分区按名字匹配（顺序无关；单优化器快捷字段仅接受单分区文件）；v1 仅支持单优化器。
// 顺序：校验 arch/版本/分区/类型/分组/参数名与状态名 → 载入张量 → 恢复 step → 恢复每组 lr
//       → （可选）恢复 RNG 状态、按分区回放调度器到存档 epoch。
bool checkpoint_load(const char* path, const CheckpointItems& items);

// 只读元数据探测（不加载张量数据）；文件不存在/不是 checkpoint 时返回 false。
// step/param_count/group_count/optimizer_type/scheduler_epoch 取第 0 个分区（v1 语义）。
bool checkpoint_probe(const char* path, CheckpointInfo* out);

// 导出自检（只读，不修改内存状态）：重新打开文件并与当前优化器逐项比对——
// 参数/状态张量数据（逐元素）、step、每组 lr、超参、RNG state、调度器 epoch（多优化器逐分区）。
// 全部一致返回 true 并打印摘要；任一不符打印原因并返回 false。
bool checkpoint_verify(const char* path, const CheckpointItems& items);

} // namespace traincpp
