// train.cpp - nn 层系统（POD struct + 自由函数，M1.5b）
//
// 设计（用户确认）：
//   - 层是纯数据结构：持有参数张量指针与超参；前向用已有算子组合实现（自由函数）
//   - 权重布局与 PyTorch state_dict 一致（张量 ne = torch 形状反序）——可直接导出 GGUF
//   - 参数遍历用 ParamList，供优化器（optim_*）与 M1.6 checkpoint/GGUF 使用
//   - 复合模型由使用方自定义结构体 + 自由函数组装（本库不含具体模型层）
//
// 初始化默认值与 PyTorch 对齐：
//   Linear/Conv：kaiming_uniform(a=√5)（等价 bound=1/√fan_in），bias 为 ±1/√fan_in 均匀分布
//   Embedding：N(0,1)；LayerNorm/GroupNorm：weight=1、bias=0
#pragma once

#include "trc_autograd.h"
#include "trc_context.h"
#include "trc_ops.h"
#include "trc_rng.h"
#include "trc_tensor.h"

#include <vector>

namespace traincpp {

// ---------------------------------------------------------------- 参数列表

struct ParamList {
    std::vector<Tensor*> items;
};

void    param_list_add(ParamList* list, Tensor* t);
int64_t param_list_count(const ParamList* list);
Tensor* param_list_get(const ParamList* list, int64_t index);

// 批量标记/取消标记整个参数列表（is_param=true 可训练、false 冻结）。
// 冻结/解冻只影响**下一次** graph_build_backward_expand（已构建的图不会撤销既有 grad_acc）。
void    param_list_set_param(ParamList* list, bool is_param);

// ---------------------------------------------------------------- 初始化工具

// fan_in/fan_out：与 PyTorch `_calculate_fan_in_and_fan_out` 一致（torch 形状 = ne 反序）。
// fan_in = ne 中除最后一维（输出通道）之外所有维的乘积。
int64_t nn_fan_in(const Tensor* t);
int64_t nn_fan_out(const Tensor* t);

// 数据未分配（no_alloc 模式）时自动登记为延迟填充，由 buffer_alloc_ctx_tensors 执行
void nn_init_constant(Context* ctx, Tensor* t, float value);
void nn_init_uniform(Context* ctx, Tensor* t, float lo, float hi, Rng* rng);
void nn_init_normal(Context* ctx, Tensor* t, float mean, float std, Rng* rng);
// PyTorch 的 kaiming_uniform（Linear/Conv 默认）：bound = sqrt(6/((1+a²)·fan_in))
void nn_init_kaiming_uniform(Context* ctx, Tensor* t, float a, Rng* rng);

// ---------------------------------------------------------------- Linear

// y = x·W + b；W=[in,out]（torch [out,in]），b=[out]，x=[in,batch] → [out,batch]
struct Linear {
    Tensor* weight = nullptr;
    Tensor* bias   = nullptr;
    int64_t in_features  = 0;
    int64_t out_features = 0;
    bool    has_bias     = true;
    char    name[MAX_NAME] = {0};
};

void    linear_init(Context* ctx, Linear* layer, int64_t in_features, int64_t out_features,
                    bool use_bias, Rng* rng, const char* name = nullptr);
// 用外部权重（checkpoint/GGUF）创建；weight 布局 [in,out]，bias 可为 nullptr
void    linear_init_weight(Context* ctx, Linear* layer, Tensor* weight, Tensor* bias);
Tensor* linear_forward(Context* ctx, const Linear* layer, Tensor* x);
// 以显式 weight 前向（bias 仍取自 layer）：用于 WeightNorm 重参数化权重等场景
Tensor* linear_forward_weight(Context* ctx, const Linear* layer, Tensor* weight, Tensor* x);
void    linear_params(const Linear* layer, ParamList* out);

// ---------------------------------------------------------------- Embedding

// y = W[idx]；W=[dim, n_vocab]（torch [n_vocab,dim]）；indices 为 I32
struct Embedding {
    Tensor* weight = nullptr;
    int64_t num_embeddings = 0;
    int64_t embedding_dim  = 0;
    char    name[MAX_NAME] = {0};
};

void    embedding_init(Context* ctx, Embedding* layer, int64_t num_embeddings, int64_t embedding_dim,
                       Rng* rng, const char* name = nullptr);
void    embedding_init_weight(Context* ctx, Embedding* layer, Tensor* weight);
Tensor* embedding_forward(Context* ctx, const Embedding* layer, Tensor* indices);
void    embedding_params(const Embedding* layer, ParamList* out);

// ---------------------------------------------------------------- Conv1d

// y = conv1d(x, W) + b；W=[KW,IC/groups,OC]（torch [OC,IC/groups,KW]），b=[OC]，
// x=[W,IC,N] → [OW,OC,N]；groups=1 时即普通卷积（默认）
struct Conv1d {
    Tensor* weight = nullptr;
    Tensor* bias   = nullptr;
    int64_t in_channels  = 0;
    int64_t out_channels = 0;
    int64_t kernel_size  = 0;
    int64_t stride       = 1;
    int64_t padding      = 0;
    int64_t dilation     = 1;
    int64_t groups       = 1;
    bool    has_bias     = true;
    char    name[MAX_NAME] = {0};
};

void    conv1d_init(Context* ctx, Conv1d* layer, int64_t in_channels, int64_t out_channels,
                    int64_t kernel_size, int64_t stride, int64_t padding, int64_t dilation,
                    bool use_bias, Rng* rng, const char* name = nullptr);
// 分组卷积（groups>1）：in/out 通道都必须被 groups 整除；权重 [KW, IC/groups, OC]。
// 前向按组切片 + conv_1d + 沿 ne1 concat 组合（自动反传）。
void    conv1d_init_groups(Context* ctx, Conv1d* layer, int64_t in_channels, int64_t out_channels,
                           int64_t kernel_size, int64_t stride, int64_t padding, int64_t dilation,
                           int64_t groups, bool use_bias, Rng* rng, const char* name = nullptr);
void    conv1d_init_weight(Context* ctx, Conv1d* layer, Tensor* weight, Tensor* bias);
Tensor* conv1d_forward(Context* ctx, const Conv1d* layer, Tensor* x);
// 以显式 weight 前向（bias 仍取自 layer）：用于 WeightNorm 重参数化权重等场景；
// groups>1 时按组切片（weight=[KW,IC/g,OC]、x=[W,IC,N]）→ conv_1d → 沿 ne1 concat
Tensor* conv1d_forward_weight(Context* ctx, const Conv1d* layer, Tensor* weight, Tensor* x);
void    conv1d_params(const Conv1d* layer, ParamList* out);

// ---------------------------------------------------------------- Conv2d

// y = conv2d(x, W) + b；W=[KW,KH,IC,OC]（torch [OC,IC,KH,KW]），b=[OC]，x=[W,H,IC,N] → [OW,OH,OC,N]
struct Conv2d {
    Tensor* weight = nullptr;
    Tensor* bias   = nullptr;
    int64_t in_channels  = 0;
    int64_t out_channels = 0;
    int64_t kernel_w     = 0;
    int64_t kernel_h     = 0;
    int64_t stride_w     = 1;
    int64_t stride_h     = 1;
    int64_t padding_w    = 0;
    int64_t padding_h    = 0;
    int64_t dilation_w   = 1;
    int64_t dilation_h   = 1;
    bool    has_bias     = true;
    char    name[MAX_NAME] = {0};
};

void    conv2d_init(Context* ctx, Conv2d* layer, int64_t in_channels, int64_t out_channels,
                    int64_t kernel_w, int64_t kernel_h, int64_t stride_w, int64_t stride_h,
                    int64_t padding_w, int64_t padding_h, int64_t dilation_w, int64_t dilation_h,
                    bool use_bias, Rng* rng, const char* name = nullptr);
void    conv2d_init_weight(Context* ctx, Conv2d* layer, Tensor* weight, Tensor* bias);
Tensor* conv2d_forward(Context* ctx, const Conv2d* layer, Tensor* x);
// 以显式 weight 前向（bias 仍取自 layer）：用于 WeightNorm 重参数化权重等场景
Tensor* conv2d_forward_weight(Context* ctx, const Conv2d* layer, Tensor* weight, Tensor* x);
void    conv2d_params(const Conv2d* layer, ParamList* out);

// ---------------------------------------------------------------- ConvTranspose1d

// y = conv_transpose1d(x, W) + b；W=[K, Cout, Cin]（= torch [Cin, Cout, K] 反序），b=[OC]，
// x=[T, IC, N] → [T_out, OC, N]，T_out = (T-1)*stride - 2*padding + K（output_padding=0）
// 一期仅支持 dilation=1（核心与 ggml 一致）；padding 通过输出的中心裁剪（view）实现。
// 核心 conv_transpose_1d 仅支持单样本，N>1 由层内按样本切片 view + 沿 ne2 concat 组合实现。
struct ConvTranspose1d {
    Tensor* weight = nullptr;
    Tensor* bias   = nullptr;
    int64_t in_channels  = 0;
    int64_t out_channels = 0;
    int64_t kernel_size  = 0;
    int64_t stride       = 1;
    int64_t padding      = 0;
    int64_t dilation     = 1;
    bool    has_bias     = true;
    char    name[MAX_NAME] = {0};
};

void    convtranspose1d_init(Context* ctx, ConvTranspose1d* layer, int64_t in_channels,
                             int64_t out_channels, int64_t kernel_size, int64_t stride,
                             int64_t padding, int64_t dilation, bool use_bias, Rng* rng,
                             const char* name = nullptr);
void    convtranspose1d_init_weight(Context* ctx, ConvTranspose1d* layer, Tensor* weight,
                                    Tensor* bias);
Tensor* convtranspose1d_forward(Context* ctx, const ConvTranspose1d* layer, Tensor* x);
// 以显式 weight 前向（bias 仍取自 layer；weight 形状必须与 layer->weight 一致 [K,Cout,Cin]）：
// 用于 WeightNorm 重参数化权重、GGUF/checkpoint 权重装载等场景
Tensor* convtranspose1d_forward_weight(Context* ctx, const ConvTranspose1d* layer, Tensor* weight,
                                       Tensor* x);
void    convtranspose1d_params(const ConvTranspose1d* layer, ParamList* out);

// ---------------------------------------------------------------- WeightNorm

// 权重归一化（torch.nn.utils.weight_norm(dim=0) 等价语义）：w = g · v / ||v||。
//   v 与被包装权重同形；g 在输出通道维（**ne 的最后一维**）非 1、其余维为 1；
//   ||v|| 对**除输出通道维外的所有维**求范数（即逐输出通道归一化，与 torch 一致）。
// 用法：v/g 由外部创建（v 可用 kaiming 等初始化），weightnorm_init 自动标参；
//       buffer 分配后调用 weightnorm_sync_g 令 g=||v||（与 torch 初始化对齐，可选）。
struct WeightNorm {
    Tensor* g = nullptr;
    Tensor* v = nullptr;
    char    name[MAX_NAME] = {0};
};

void    weightnorm_init(Context* ctx, WeightNorm* wn, Tensor* v, Tensor* g,
                        const char* name = nullptr);
// 重参数化前向：返回 w = g·v/sqrt(Σ_nonOC(v²)+eps)（与 v 同形，连续）
Tensor* weightnorm_forward(Context* ctx, const WeightNorm* wn);
void    weightnorm_params(const WeightNorm* wn, ParamList* out);
// 主机端 g = ||v||（逐输出通道；要求 buffer 已分配）——与 torch weight_norm 初始化对齐
void    weightnorm_sync_g(const WeightNorm* wn);

// ---------------------------------------------------------------- 归一化（仿射由层内施加）

// LayerNorm：对每行（ne0）归一化后乘 weight、加 bias；weight/bias 形状 [n]（torch [n]）
struct LayerNorm {
    Tensor* weight = nullptr;
    Tensor* bias   = nullptr;
    int64_t normalized_size = 0;
    float   eps      = 1e-5f;
    bool    has_bias = true;
    char    name[MAX_NAME] = {0};
};

void    layernorm_init(Context* ctx, LayerNorm* layer, int64_t normalized_size, float eps,
                       bool use_bias, const char* name = nullptr);
void    layernorm_init_weight(Context* ctx, LayerNorm* layer, Tensor* weight, Tensor* bias);
Tensor* layernorm_forward(Context* ctx, const LayerNorm* layer, Tensor* x);
void    layernorm_params(const LayerNorm* layer, ParamList* out);

// RMSNorm：x / sqrt(mean(x²)+eps) * weight；无 bias
struct RmsNorm {
    Tensor* weight = nullptr;
    int64_t normalized_size = 0;
    float   eps = 1e-5f;
    char    name[MAX_NAME] = {0};
};

void    rmsnorm_init(Context* ctx, RmsNorm* layer, int64_t normalized_size, float eps,
                     const char* name = nullptr);
void    rmsnorm_init_weight(Context* ctx, RmsNorm* layer, Tensor* weight);
Tensor* rmsnorm_forward(Context* ctx, const RmsNorm* layer, Tensor* x);
void    rmsnorm_params(const RmsNorm* layer, ParamList* out);

// GroupNorm：沿 ne2（通道维）分组归一化后乘 weight、加 bias；weight/bias 形状 [C]
struct GroupNorm {
    Tensor* weight = nullptr;
    Tensor* bias   = nullptr;
    int64_t num_groups   = 1;
    int64_t num_channels = 0;
    float   eps      = 1e-5f;
    bool    has_bias = true;
    char    name[MAX_NAME] = {0};
};

void    groupnorm_init(Context* ctx, GroupNorm* layer, int64_t num_groups, int64_t num_channels,
                       float eps, bool use_bias, const char* name = nullptr);
void    groupnorm_init_weight(Context* ctx, GroupNorm* layer, Tensor* weight, Tensor* bias);
Tensor* groupnorm_forward(Context* ctx, const GroupNorm* layer, Tensor* x);
void    groupnorm_params(const GroupNorm* layer, ParamList* out);

// ---------------------------------------------------------------- Dropout

// Inverted dropout：训练时 y = x * mask（mask 为 0 或 1/(1-p)）；eval 时 y = x。
// mask 与输入同形状、延迟创建；每个训练步 compute 之前需调用 dropout_refresh 重新采样。
struct Dropout {
    Tensor* mask = nullptr;
    float   p        = 0.5f;
    bool    training = true;
    char    name[MAX_NAME] = {0};
};

void    dropout_init(Dropout* layer, float p, const char* name = nullptr);
void    dropout_set_training(Dropout* layer, bool training);
Tensor* dropout_forward(Context* ctx, Dropout* layer, Tensor* x);
// 重新采样 mask；要求 dropout_forward 已经调用且张量已分配内存
void    dropout_refresh(Dropout* layer, Rng* rng);
void    dropout_params(const Dropout* layer, ParamList* out);

// ---------------------------------------------------------------- 复合模型辅助

// 收集多个 ParamList 到 out（组装模型时使用）
void param_list_merge(ParamList* out, const ParamList* src);

} // namespace traincpp
