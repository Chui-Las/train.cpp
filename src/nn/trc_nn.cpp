// train.cpp - nn 层系统实现（POD struct + 自由函数）
//
// 全部层都基于已有算子组合实现（不新增 kernel）；权重布局与 PyTorch state_dict 一致
// （张量 ne = torch 形状反序），便于 GGUF 导出与从 PyTorch 迁移权重。
#include "traincpp/trc_nn.h"

#include "core/trc_impl.h"

#include <cmath>
#include <cstdio>
#include <vector>

namespace traincpp {

namespace {

void set_layer_name(char* dst, const char* name, const char* fallback) {
    std::snprintf(dst, MAX_NAME, "%s", (name != nullptr && name[0] != '\0') ? name : fallback);
}

void set_param_name(Tensor* t, const char* layer_name, const char* suffix) {
    tensor_set_name(t, "%s.%s", layer_name, suffix);
}

float bias_bound_for_fan_in(int64_t fan_in) {
    return fan_in > 0 ? 1.0f / std::sqrt((float) fan_in) : 0.0f;
}

void check_param(const Tensor* t, const char* what) {
    TRC_ASSERT(t != nullptr, "%s 为空", what);
    TRC_ASSERT(t->type == TYPE_F32, "%s 必须是 F32（当前 %s）", what, type_name(t->type));
    TRC_ASSERT(tensor_is_contiguous(t), "%s 必须连续", what);
}

// 权重布局校验：rank 为 API 约定的有效维数，尾部更高维必须为 1。
// 不能用 tensor_n_dims（ggml 语义会折叠尾部 1 维：如 [K,Cout,1] 被算作 2 维），否则
// Cin/OC=1 的合法权重会被误拒或误推布局（M2.4 验收发现，见 docs 风险登记册）。
void check_weight_tail_ones(const Tensor* t, int rank, const char* what) {
    for (int i = rank; i < MAX_DIMS; ++i) {
        TRC_ASSERT(t->ne[i] == 1, "%s: 有效维数为 %d，第 %d 维必须为 1（当前 %lld）", what, rank, i,
                   (long long) t->ne[i]);
    }
}

// 以显式 fan_in 做 kaiming_uniform（层初始化时布局已知，避免 nn_fan_in 的尾部 1 维折叠）
void init_kaiming_uniform_fan(Context* ctx, Tensor* t, float a, int64_t fan_in, Rng* rng) {
    TRC_ASSERT(t != nullptr && rng != nullptr, "init_kaiming_uniform_fan: 参数为空");
    TRC_ASSERT(fan_in > 0, "init_kaiming_uniform_fan: fan_in 非法");
    const double bound = std::sqrt(6.0 / ((1.0 + (double) a * (double) a) * (double) fan_in));
    rng_fill_uniform(ctx, rng, t, (float) -bound, (float) bound);
}

} // namespace

// ---------------------------------------------------------------- 参数列表

void param_list_add(ParamList* list, Tensor* t) {
    TRC_ASSERT(list != nullptr && t != nullptr, "param_list_add: 参数为空");
    list->items.push_back(t);
}

int64_t param_list_count(const ParamList* list) {
    return (int64_t) list->items.size();
}

Tensor* param_list_get(const ParamList* list, int64_t index) {
    TRC_ASSERT(index >= 0 && index < (int64_t) list->items.size(), "param_list_get: 下标越界");
    return list->items[(size_t) index];
}

void param_list_merge(ParamList* out, const ParamList* src) {
    TRC_ASSERT(out != nullptr && src != nullptr, "param_list_merge: 参数为空");
    out->items.insert(out->items.end(), src->items.begin(), src->items.end());
}

void param_list_set_param(ParamList* list, bool is_param) {
    TRC_ASSERT(list != nullptr, "param_list_set_param: 列表为空");
    for (Tensor* t : list->items) {
        if (is_param) {
            tensor_set_param(t);
        } else {
            tensor_clear_param(t);
        }
    }
}

// ---------------------------------------------------------------- 初始化工具

int64_t nn_fan_in(const Tensor* t) {
    TRC_ASSERT(t != nullptr, "nn_fan_in: 张量为空");
    const int n_dims = tensor_n_dims(t);
    if (n_dims <= 1) {
        return t->ne[0];
    }
    int64_t fan = 1;
    for (int i = 0; i < n_dims - 1; ++i) {
        fan *= t->ne[i];
    }
    return fan;
}

int64_t nn_fan_out(const Tensor* t) {
    TRC_ASSERT(t != nullptr, "nn_fan_out: 张量为空");
    const int n_dims = tensor_n_dims(t);
    if (n_dims <= 1) {
        return t->ne[0];
    }
    // torch.size(0) = ne[n_dims-1]；torch.shape[2:] = ne[0..n_dims-3]
    int64_t fan = t->ne[n_dims - 1];
    for (int i = 0; i < n_dims - 2; ++i) {
        fan *= t->ne[i];
    }
    return fan;
}

void nn_init_constant(Context* ctx, Tensor* t, float value) {
    rng_fill_constant(ctx, t, value);
}

void nn_init_uniform(Context* ctx, Tensor* t, float lo, float hi, Rng* rng) {
    TRC_ASSERT(rng != nullptr, "nn_init_uniform: rng 为空");
    rng_fill_uniform(ctx, rng, t, lo, hi);
}

void nn_init_normal(Context* ctx, Tensor* t, float mean, float std, Rng* rng) {
    TRC_ASSERT(rng != nullptr, "nn_init_normal: rng 为空");
    rng_fill_normal(ctx, rng, t, mean, std);
}

void nn_init_kaiming_uniform(Context* ctx, Tensor* t, float a, Rng* rng) {
    const int64_t fan_in = nn_fan_in(t);
    TRC_ASSERT(fan_in > 0, "nn_init_kaiming_uniform: fan_in 非法");
    const double bound = std::sqrt(6.0 / ((1.0 + (double) a * (double) a) * (double) fan_in));
    rng_fill_uniform(ctx, rng, t, (float) -bound, (float) bound);
}

// ---------------------------------------------------------------- Linear

void linear_init(Context* ctx, Linear* layer, int64_t in_features, int64_t out_features,
                 bool use_bias, Rng* rng, const char* name) {
    TRC_ASSERT(ctx != nullptr && layer != nullptr, "linear_init: 参数为空");
    TRC_ASSERT(in_features > 0 && out_features > 0, "linear_init: 维度非法");
    TRC_ASSERT(rng != nullptr, "linear_init: rng 为空（初始化需要确定性随机源）");

    set_layer_name(layer->name, name, "linear");
    layer->in_features  = in_features;
    layer->out_features = out_features;
    layer->has_bias     = use_bias;

    layer->weight = new_tensor_2d(ctx, TYPE_F32, in_features, out_features);
    set_param_name(layer->weight, layer->name, "weight");
    init_kaiming_uniform_fan(ctx, layer->weight, std::sqrt(5.0f), in_features, rng);

    if (use_bias) {
        layer->bias = new_tensor_1d(ctx, TYPE_F32, out_features);
        set_param_name(layer->bias, layer->name, "bias");
        const float bound = bias_bound_for_fan_in(in_features);
        nn_init_uniform(ctx, layer->bias, -bound, bound, rng);
    } else {
        layer->bias = nullptr;
    }
    // 自动标记可训练参数（可用 param_list_set_param 冻结/解冻）
    tensor_set_param(layer->weight);
    if (layer->bias != nullptr) {
        tensor_set_param(layer->bias);
    }
}

void linear_init_weight(Context* ctx, Linear* layer, Tensor* weight, Tensor* bias) {
    TRC_ASSERT(ctx != nullptr && layer != nullptr && weight != nullptr,
               "linear_init_weight: 参数为空");
    check_param(weight, "linear_init_weight: weight");
    check_weight_tail_ones(weight, 2, "linear_init_weight: weight");
    if (bias != nullptr) {
        check_param(bias, "linear_init_weight: bias");
        TRC_ASSERT(bias->ne[0] == weight->ne[1], "linear_init_weight: bias 维度不匹配");
    }

    if (layer->name[0] == '\0') {
        set_layer_name(layer->name, nullptr, "linear");
    }
    layer->weight      = weight;
    layer->bias        = bias;
    layer->in_features  = weight->ne[0];
    layer->out_features = weight->ne[1];
    layer->has_bias     = bias != nullptr;
    set_param_name(layer->weight, layer->name, "weight");
    if (bias != nullptr) {
        set_param_name(layer->bias, layer->name, "bias");
    }
    tensor_set_param(layer->weight);
    if (bias != nullptr) {
        tensor_set_param(layer->bias);
    }
}

Tensor* linear_forward(Context* ctx, const Linear* layer, Tensor* x) {
    TRC_ASSERT(ctx != nullptr && layer != nullptr && layer->weight != nullptr && x != nullptr,
               "linear_forward: 参数为空");
    TRC_ASSERT(x->ne[0] == layer->in_features, "linear_forward: 输入维度 %lld != in_features %lld",
               (long long) x->ne[0], (long long) layer->in_features);
    Tensor* y = mul_mat(ctx, layer->weight, x);
    if (layer->has_bias && layer->bias != nullptr) {
        y = add(ctx, y, layer->bias);
    }
    return y;
}

Tensor* linear_forward_weight(Context* ctx, const Linear* layer, Tensor* weight, Tensor* x) {
    TRC_ASSERT(ctx != nullptr && layer != nullptr && weight != nullptr && x != nullptr,
               "linear_forward_weight: 参数为空");
    TRC_ASSERT(weight->ne[0] == x->ne[0],
               "linear_forward_weight: 输入维度 %lld != weight 输入维 %lld", (long long) x->ne[0],
               (long long) weight->ne[0]);
    Tensor* y = mul_mat(ctx, weight, x);
    if (layer->has_bias && layer->bias != nullptr) {
        y = add(ctx, y, layer->bias);
    }
    return y;
}

void linear_params(const Linear* layer, ParamList* out) {
    param_list_add(out, layer->weight);
    if (layer->has_bias && layer->bias != nullptr) {
        param_list_add(out, layer->bias);
    }
}

// ---------------------------------------------------------------- Embedding

void embedding_init(Context* ctx, Embedding* layer, int64_t num_embeddings, int64_t embedding_dim,
                    Rng* rng, const char* name) {
    TRC_ASSERT(ctx != nullptr && layer != nullptr, "embedding_init: 参数为空");
    TRC_ASSERT(num_embeddings > 0 && embedding_dim > 0, "embedding_init: 维度非法");
    TRC_ASSERT(rng != nullptr, "embedding_init: rng 为空");

    set_layer_name(layer->name, name, "embedding");
    layer->num_embeddings = num_embeddings;
    layer->embedding_dim  = embedding_dim;

    // 与 torch.nn.Embedding.reset_parameters 一致：N(0,1)
    layer->weight = new_tensor_2d(ctx, TYPE_F32, embedding_dim, num_embeddings);
    set_param_name(layer->weight, layer->name, "weight");
    nn_init_normal(ctx, layer->weight, 0.0f, 1.0f, rng);
    tensor_set_param(layer->weight);
}

void embedding_init_weight(Context* ctx, Embedding* layer, Tensor* weight) {
    TRC_ASSERT(ctx != nullptr && layer != nullptr && weight != nullptr,
               "embedding_init_weight: 参数为空");
    check_param(weight, "embedding_init_weight: weight");
    check_weight_tail_ones(weight, 2, "embedding_init_weight: weight");
    if (layer->name[0] == '\0') {
        set_layer_name(layer->name, nullptr, "embedding");
    }
    layer->weight         = weight;
    layer->embedding_dim  = weight->ne[0];
    layer->num_embeddings = weight->ne[1];
    set_param_name(layer->weight, layer->name, "weight");
    tensor_set_param(layer->weight);
}

Tensor* embedding_forward(Context* ctx, const Embedding* layer, Tensor* indices) {
    TRC_ASSERT(ctx != nullptr && layer != nullptr && layer->weight != nullptr && indices != nullptr,
               "embedding_forward: 参数为空");
    TRC_ASSERT(indices->type == TYPE_I32, "embedding_forward: 索引必须是 I32（当前 %s）",
               type_name(indices->type));
    return get_rows(ctx, layer->weight, indices);
}

void embedding_params(const Embedding* layer, ParamList* out) {
    param_list_add(out, layer->weight);
}

// ---------------------------------------------------------------- Conv1d

void conv1d_init(Context* ctx, Conv1d* layer, int64_t in_channels, int64_t out_channels,
                 int64_t kernel_size, int64_t stride, int64_t padding, int64_t dilation,
                 bool use_bias, Rng* rng, const char* name) {
    conv1d_init_groups(ctx, layer, in_channels, out_channels, kernel_size, stride, padding,
                       dilation, 1, use_bias, rng, name);
}

void conv1d_init_groups(Context* ctx, Conv1d* layer, int64_t in_channels, int64_t out_channels,
                        int64_t kernel_size, int64_t stride, int64_t padding, int64_t dilation,
                        int64_t groups, bool use_bias, Rng* rng, const char* name) {
    TRC_ASSERT(ctx != nullptr && layer != nullptr, "conv1d_init_groups: 参数为空");
    TRC_ASSERT(in_channels > 0 && out_channels > 0 && kernel_size > 0,
               "conv1d_init_groups: 维度非法");
    TRC_ASSERT(stride > 0 && padding >= 0 && dilation > 0, "conv1d_init_groups: 卷积参数非法");
    TRC_ASSERT(groups > 0, "conv1d_init_groups: groups 必须 > 0（当前 %lld）", (long long) groups);
    TRC_ASSERT(in_channels % groups == 0 && out_channels % groups == 0,
               "conv1d_init_groups: 输入/输出通道必须被 groups 整除（IC=%lld, OC=%lld, groups=%lld）",
               (long long) in_channels, (long long) out_channels, (long long) groups);
    TRC_ASSERT(rng != nullptr, "conv1d_init_groups: rng 为空");

    set_layer_name(layer->name, name, "conv1d");
    layer->in_channels  = in_channels;
    layer->out_channels = out_channels;
    layer->kernel_size  = kernel_size;
    layer->stride       = stride;
    layer->padding      = padding;
    layer->dilation     = dilation;
    layer->groups       = groups;
    layer->has_bias     = use_bias;

    // 权重 [KW, IC/groups, OC]（与 torch grouped conv 的 (OC, IC/g, KW) 反序一致）
    layer->weight = new_tensor_3d(ctx, TYPE_F32, kernel_size, in_channels / groups, out_channels);
    set_param_name(layer->weight, layer->name, "weight");
    init_kaiming_uniform_fan(ctx, layer->weight, std::sqrt(5.0f),
                             kernel_size * (in_channels / groups), rng);

    if (use_bias) {
        layer->bias = new_tensor_1d(ctx, TYPE_F32, out_channels);
        set_param_name(layer->bias, layer->name, "bias");
        const float bound = bias_bound_for_fan_in(kernel_size * (in_channels / groups));
        nn_init_uniform(ctx, layer->bias, -bound, bound, rng);
    } else {
        layer->bias = nullptr;
    }
    tensor_set_param(layer->weight);
    if (layer->bias != nullptr) {
        tensor_set_param(layer->bias);
    }
}

void conv1d_init_weight(Context* ctx, Conv1d* layer, Tensor* weight, Tensor* bias) {
    TRC_ASSERT(ctx != nullptr && layer != nullptr && weight != nullptr,
               "conv1d_init_weight: 参数为空");
    check_param(weight, "conv1d_init_weight: weight");
    check_weight_tail_ones(weight, 3, "conv1d_init_weight: weight");
    if (layer->name[0] == '\0') {
        set_layer_name(layer->name, nullptr, "conv1d");
    }
    TRC_ASSERT(layer->groups > 0, "conv1d_init_weight: groups 必须 > 0（当前 %lld）",
               (long long) layer->groups);
    TRC_ASSERT(weight->ne[2] % layer->groups == 0,
               "conv1d_init_weight: 输出通道 %lld 必须被 groups %lld 整除",
               (long long) weight->ne[2], (long long) layer->groups);
    layer->weight       = weight;
    layer->bias         = bias;
    layer->kernel_size  = weight->ne[0];
    layer->in_channels  = weight->ne[1] * layer->groups;
    layer->out_channels = weight->ne[2];
    layer->has_bias     = bias != nullptr;
    set_param_name(layer->weight, layer->name, "weight");
    if (bias != nullptr) {
        check_param(bias, "conv1d_init_weight: bias");
        TRC_ASSERT(bias->ne[0] == layer->out_channels, "conv1d_init_weight: bias 维度不匹配");
        set_param_name(layer->bias, layer->name, "bias");
    }
    tensor_set_param(layer->weight);
    if (bias != nullptr) {
        tensor_set_param(layer->bias);
    }
}

Tensor* conv1d_forward(Context* ctx, const Conv1d* layer, Tensor* x) {
    TRC_ASSERT(ctx != nullptr && layer != nullptr && layer->weight != nullptr && x != nullptr,
               "conv1d_forward: 参数为空");
    TRC_ASSERT(x->ne[1] == layer->in_channels, "conv1d_forward: 输入通道 %lld != in_channels %lld",
               (long long) x->ne[1], (long long) layer->in_channels);

    Tensor* y = nullptr;
    if (layer->groups <= 1) {
        y = conv_1d(ctx, layer->weight, x, (int) layer->stride, (int) layer->padding,
                    (int) layer->dilation);
    } else {
        // 分组卷积：逐组切片（x 切 ne1 输入通道、w 切 ne2 输出通道）→ conv_1d → 沿 ne1 concat
        const int64_t g    = layer->groups;
        const int64_t ic_g = layer->in_channels / g;
        const int64_t oc_g = layer->out_channels / g;
        TRC_ASSERT(ic_g * g == layer->in_channels && oc_g * g == layer->out_channels,
                   "conv1d_forward: groups %lld 与通道数不整除", (long long) g);
        for (int64_t j = 0; j < g; ++j) {
            Tensor* xj = view_3d(ctx, x, x->ne[0], ic_g, x->ne[2], (size_t) j * ic_g * x->nb[1]);
            Tensor* wj = view_3d(ctx, layer->weight, layer->kernel_size, ic_g, oc_g,
                                 (size_t) j * oc_g * layer->weight->nb[2]);
            // conv_1d 内部对权重做 reshape_2d（要求连续）：跨组切片是跨步视图，必须先 cont
            Tensor* yj = conv_1d(ctx, cont(ctx, wj), xj, (int) layer->stride, (int) layer->padding,
                                 (int) layer->dilation);
            y = (y == nullptr) ? yj : concat(ctx, y, yj, 1);
        }
    }

    if (layer->has_bias && layer->bias != nullptr) {
        // 输出 [OW,OC,N,1]：bias [OC] reshape 成 [1,OC,1,1] 后广播
        Tensor* b = reshape_4d(ctx, layer->bias, 1, layer->out_channels, 1, 1);
        y = add(ctx, y, b);
    }
    return y;
}

Tensor* conv1d_forward_weight(Context* ctx, const Conv1d* layer, Tensor* weight, Tensor* x) {
    TRC_ASSERT(ctx != nullptr && layer != nullptr && weight != nullptr && x != nullptr,
               "conv1d_forward_weight: 参数为空");
    const int64_t g = layer->groups > 1 ? layer->groups : 1;
    TRC_ASSERT(x->ne[1] == weight->ne[1] * g,
               "conv1d_forward_weight: 输入通道 %lld != weight 输入通道 %lld × groups %lld",
               (long long) x->ne[1], (long long) weight->ne[1], (long long) g);
    if (layer->out_channels > 0) {
        TRC_ASSERT(layer->out_channels == weight->ne[2],
                   "conv1d_forward_weight: weight 输出通道 %lld != out_channels %lld",
                   (long long) weight->ne[2], (long long) layer->out_channels);
    }

    Tensor* y = nullptr;
    if (g <= 1) {
        y = conv_1d(ctx, weight, x, (int) layer->stride, (int) layer->padding,
                    (int) layer->dilation);
    } else {
        // 分组卷积：逐组切片（x 切 ne1 输入通道、weight 切 ne2 输出通道）→ conv_1d → 沿 ne1 concat
        const int64_t ic_g = weight->ne[1];
        const int64_t oc_g = weight->ne[2] / g;
        TRC_ASSERT(oc_g * g == weight->ne[2],
                   "conv1d_forward_weight: 输出通道 %lld 必须被 groups %lld 整除",
                   (long long) weight->ne[2], (long long) g);
        for (int64_t j = 0; j < g; ++j) {
            Tensor* xj = view_3d(ctx, x, x->ne[0], ic_g, x->ne[2], (size_t) j * ic_g * x->nb[1]);
            Tensor* wj = view_3d(ctx, weight, weight->ne[0], ic_g, oc_g,
                                 (size_t) j * oc_g * weight->nb[2]);
            // conv_1d 内部对权重做 reshape_2d（要求连续）：跨组切片是跨步视图，必须先 cont
            Tensor* yj = conv_1d(ctx, cont(ctx, wj), xj, (int) layer->stride, (int) layer->padding,
                                 (int) layer->dilation);
            y = (y == nullptr) ? yj : concat(ctx, y, yj, 1);
        }
    }

    if (layer->has_bias && layer->bias != nullptr) {
        // 输出 [OW,OC,N,1]：bias [OC] reshape 成 [1,OC,1,1] 后广播（分组时也只在 concat 后加一次）
        Tensor* b = reshape_4d(ctx, layer->bias, 1, weight->ne[2], 1, 1);
        y = add(ctx, y, b);
    }
    return y;
}

void conv1d_params(const Conv1d* layer, ParamList* out) {
    param_list_add(out, layer->weight);
    if (layer->has_bias && layer->bias != nullptr) {
        param_list_add(out, layer->bias);
    }
}

// ---------------------------------------------------------------- Conv2d

void conv2d_init(Context* ctx, Conv2d* layer, int64_t in_channels, int64_t out_channels,
                 int64_t kernel_w, int64_t kernel_h, int64_t stride_w, int64_t stride_h,
                 int64_t padding_w, int64_t padding_h, int64_t dilation_w, int64_t dilation_h,
                 bool use_bias, Rng* rng, const char* name) {
    TRC_ASSERT(ctx != nullptr && layer != nullptr, "conv2d_init: 参数为空");
    TRC_ASSERT(in_channels > 0 && out_channels > 0 && kernel_w > 0 && kernel_h > 0,
               "conv2d_init: 维度非法");
    TRC_ASSERT(stride_w > 0 && stride_h > 0 && padding_w >= 0 && padding_h >= 0 &&
                   dilation_w > 0 && dilation_h > 0,
               "conv2d_init: 卷积参数非法");
    TRC_ASSERT(rng != nullptr, "conv2d_init: rng 为空");

    set_layer_name(layer->name, name, "conv2d");
    layer->in_channels  = in_channels;
    layer->out_channels = out_channels;
    layer->kernel_w     = kernel_w;
    layer->kernel_h     = kernel_h;
    layer->stride_w     = stride_w;
    layer->stride_h     = stride_h;
    layer->padding_w    = padding_w;
    layer->padding_h    = padding_h;
    layer->dilation_w   = dilation_w;
    layer->dilation_h   = dilation_h;
    layer->has_bias     = use_bias;

    layer->weight = new_tensor_4d(ctx, TYPE_F32, kernel_w, kernel_h, in_channels, out_channels);
    set_param_name(layer->weight, layer->name, "weight");
    init_kaiming_uniform_fan(ctx, layer->weight, std::sqrt(5.0f),
                             kernel_w * kernel_h * in_channels, rng);

    if (use_bias) {
        layer->bias = new_tensor_1d(ctx, TYPE_F32, out_channels);
        set_param_name(layer->bias, layer->name, "bias");
        const float bound = bias_bound_for_fan_in(kernel_w * kernel_h * in_channels);
        nn_init_uniform(ctx, layer->bias, -bound, bound, rng);
    } else {
        layer->bias = nullptr;
    }
    tensor_set_param(layer->weight);
    if (layer->bias != nullptr) {
        tensor_set_param(layer->bias);
    }
}

void conv2d_init_weight(Context* ctx, Conv2d* layer, Tensor* weight, Tensor* bias) {
    TRC_ASSERT(ctx != nullptr && layer != nullptr && weight != nullptr,
               "conv2d_init_weight: 参数为空");
    check_param(weight, "conv2d_init_weight: weight");
    // 4 维布局 [KW,KH,IC,OC]：尾部维允许为 1（无更高维可折叠），不再用 tensor_n_dims 判维
    if (layer->name[0] == '\0') {
        set_layer_name(layer->name, nullptr, "conv2d");
    }
    layer->weight       = weight;
    layer->bias         = bias;
    layer->kernel_w     = weight->ne[0];
    layer->kernel_h     = weight->ne[1];
    layer->in_channels  = weight->ne[2];
    layer->out_channels = weight->ne[3];
    layer->has_bias     = bias != nullptr;
    set_param_name(layer->weight, layer->name, "weight");
    if (bias != nullptr) {
        check_param(bias, "conv2d_init_weight: bias");
        TRC_ASSERT(bias->ne[0] == layer->out_channels, "conv2d_init_weight: bias 维度不匹配");
        set_param_name(layer->bias, layer->name, "bias");
    }
    tensor_set_param(layer->weight);
    if (bias != nullptr) {
        tensor_set_param(layer->bias);
    }
}

Tensor* conv2d_forward(Context* ctx, const Conv2d* layer, Tensor* x) {
    TRC_ASSERT(ctx != nullptr && layer != nullptr && layer->weight != nullptr && x != nullptr,
               "conv2d_forward: 参数为空");
    TRC_ASSERT(x->ne[2] == layer->in_channels, "conv2d_forward: 输入通道 %lld != in_channels %lld",
               (long long) x->ne[2], (long long) layer->in_channels);
    Tensor* y = conv_2d(ctx, layer->weight, x, (int) layer->stride_w, (int) layer->stride_h,
                        (int) layer->padding_w, (int) layer->padding_h, (int) layer->dilation_w,
                        (int) layer->dilation_h);
    if (layer->has_bias && layer->bias != nullptr) {
        // 输出 [OW,OH,OC,N]：bias [OC] reshape 成 [1,1,OC,1] 后广播
        Tensor* b = reshape_4d(ctx, layer->bias, 1, 1, layer->out_channels, 1);
        y = add(ctx, y, b);
    }
    return y;
}

Tensor* conv2d_forward_weight(Context* ctx, const Conv2d* layer, Tensor* weight, Tensor* x) {
    TRC_ASSERT(ctx != nullptr && layer != nullptr && weight != nullptr && x != nullptr,
               "conv2d_forward_weight: 参数为空");
    TRC_ASSERT(x->ne[2] == weight->ne[2],
               "conv2d_forward_weight: 输入通道 %lld != weight 输入通道 %lld",
               (long long) x->ne[2], (long long) weight->ne[2]);
    Tensor* y = conv_2d(ctx, weight, x, (int) layer->stride_w, (int) layer->stride_h,
                        (int) layer->padding_w, (int) layer->padding_h, (int) layer->dilation_w,
                        (int) layer->dilation_h);
    if (layer->has_bias && layer->bias != nullptr) {
        Tensor* b = reshape_4d(ctx, layer->bias, 1, 1, layer->out_channels, 1);
        y = add(ctx, y, b);
    }
    return y;
}

void conv2d_params(const Conv2d* layer, ParamList* out) {
    param_list_add(out, layer->weight);
    if (layer->has_bias && layer->bias != nullptr) {
        param_list_add(out, layer->bias);
    }
}

// ---------------------------------------------------------------- ConvTranspose1d

void convtranspose1d_init(Context* ctx, ConvTranspose1d* layer, int64_t in_channels,
                          int64_t out_channels, int64_t kernel_size, int64_t stride,
                          int64_t padding, int64_t dilation, bool use_bias, Rng* rng,
                          const char* name) {
    TRC_ASSERT(ctx != nullptr && layer != nullptr, "convtranspose1d_init: 参数为空");
    TRC_ASSERT(in_channels > 0 && out_channels > 0 && kernel_size > 0,
               "convtranspose1d_init: 维度非法");
    TRC_ASSERT(stride > 0 && padding >= 0 && dilation > 0, "convtranspose1d_init: 卷积参数非法");
    TRC_ASSERT(dilation == 1, "convtranspose1d_init: 一期仅支持 dilation=1（核心与 ggml 一致）");
    TRC_ASSERT(rng != nullptr, "convtranspose1d_init: rng 为空");

    set_layer_name(layer->name, name, "convtranspose1d");
    layer->in_channels  = in_channels;
    layer->out_channels = out_channels;
    layer->kernel_size  = kernel_size;
    layer->stride       = stride;
    layer->padding      = padding;
    layer->dilation     = dilation;
    layer->has_bias     = use_bias;

    // 布局 [K, Cout, Cin]（= torch ConvTranspose1d.weight (Cin, Cout, K) 的 ne 反序）
    layer->weight = new_tensor_3d(ctx, TYPE_F32, kernel_size, out_channels, in_channels);
    set_param_name(layer->weight, layer->name, "weight");
    // torch ConvTranspose1d 的 fan_in = out_channels * K
    init_kaiming_uniform_fan(ctx, layer->weight, std::sqrt(5.0f), kernel_size * out_channels, rng);

    if (use_bias) {
        layer->bias = new_tensor_1d(ctx, TYPE_F32, out_channels);
        set_param_name(layer->bias, layer->name, "bias");
        // torch ConvTranspose1d 的 fan_in = out_channels * K
        const float bound = bias_bound_for_fan_in(kernel_size * out_channels);
        nn_init_uniform(ctx, layer->bias, -bound, bound, rng);
    } else {
        layer->bias = nullptr;
    }
    tensor_set_param(layer->weight);
    if (layer->bias != nullptr) {
        tensor_set_param(layer->bias);
    }
}

void convtranspose1d_init_weight(Context* ctx, ConvTranspose1d* layer, Tensor* weight,
                                 Tensor* bias) {
    TRC_ASSERT(ctx != nullptr && layer != nullptr && weight != nullptr,
               "convtranspose1d_init_weight: 参数为空");
    check_param(weight, "convtranspose1d_init_weight: weight");
    check_weight_tail_ones(weight, 3, "convtranspose1d_init_weight: weight");
    if (layer->name[0] == '\0') {
        set_layer_name(layer->name, nullptr, "convtranspose1d");
    }
    layer->weight       = weight;
    layer->bias         = bias;
    layer->kernel_size  = weight->ne[0];
    layer->out_channels = weight->ne[1];
    layer->in_channels  = weight->ne[2];
    layer->has_bias     = bias != nullptr;
    set_param_name(layer->weight, layer->name, "weight");
    if (bias != nullptr) {
        check_param(bias, "convtranspose1d_init_weight: bias");
        TRC_ASSERT(bias->ne[0] == layer->out_channels,
                   "convtranspose1d_init_weight: bias 维度不匹配");
        set_param_name(layer->bias, layer->name, "bias");
    }
    tensor_set_param(layer->weight);
    if (bias != nullptr) {
        tensor_set_param(layer->bias);
    }
}

namespace {

// 转置卷积前向公共实现：weight 可直接取自 layer，也可由外部传入
// （WeightNorm 重参数化权重、GGUF/checkpoint 权重装载）；bias 始终取自 layer。
Tensor* convtranspose1d_forward_impl(Context* ctx, const ConvTranspose1d* layer, Tensor* weight,
                                     Tensor* x) {
    TRC_ASSERT(ctx != nullptr && layer != nullptr && weight != nullptr && x != nullptr,
               "convtranspose1d_forward: 参数为空");
    TRC_ASSERT(layer->dilation == 1, "convtranspose1d_forward: 一期仅支持 dilation=1");
    check_weight_tail_ones(weight, 3, "convtranspose1d_forward: weight");
    TRC_ASSERT(weight->ne[0] == layer->kernel_size && weight->ne[1] == layer->out_channels &&
                   weight->ne[2] == layer->in_channels,
               "convtranspose1d_forward: weight 形状 [%lld,%lld,%lld] 与层 [%lld,%lld,%lld] 不匹配",
               (long long) weight->ne[0], (long long) weight->ne[1], (long long) weight->ne[2],
               (long long) layer->kernel_size, (long long) layer->out_channels,
               (long long) layer->in_channels);
    TRC_ASSERT(x->ne[1] == layer->in_channels,
               "convtranspose1d_forward: 输入通道 %lld != in_channels %lld",
               (long long) x->ne[1], (long long) layer->in_channels);
    TRC_ASSERT(x->ne[3] == 1, "convtranspose1d_forward: 输入第 4 维必须为 1");

    const int64_t T    = x->ne[0];
    const int64_t N    = x->ne[2];
    const int64_t K    = layer->kernel_size;
    const int64_t Cout = layer->out_channels;
    const int     s0   = (int) layer->stride;
    const int64_t p    = layer->padding;

    Tensor* y = nullptr;
    if (N == 1) {
        // 核心要求 b 为 2D（ne2/ne3=1）；x 为 [T,IC,1] 时同样满足
        y = conv_transpose_1d(ctx, weight, x, s0, 0, 1);
    } else {
        // 核心仅支持单样本：逐样本切片（view_2d 保持 nb 步长）→ 转置卷积 → 沿 ne2 拼接
        for (int64_t n = 0; n < N; ++n) {
            Tensor* xn = view_2d(ctx, x, T, layer->in_channels, (size_t) n * x->nb[2]);
            Tensor* yn = conv_transpose_1d(ctx, weight, xn, s0, 0, 1);
            y = (y == nullptr) ? yn : concat(ctx, y, yn, 2);
        }
    }

    const int64_t T_full = (T - 1) * (int64_t) s0 + K;
    if (p > 0) {
        // 核心输出 [T_full, Cout, N, 1]，取中心区域 [T_full-2p, Cout, N, 1]（view 可反传）
        TRC_ASSERT(T_full - 2 * p > 0,
                   "convtranspose1d_forward: padding %lld 过大（核心输出 T_full=%lld）",
                   (long long) p, (long long) T_full);
        y = view_4d(ctx, y, T_full - 2 * p, Cout, N, 1, (size_t) p * y->nb[0]);
    }

    if (layer->has_bias && layer->bias != nullptr) {
        // 输出 [T_out,OC,N,1]：bias [OC] reshape 成 [1,OC,1,1] 后广播
        y = add(ctx, y, reshape_4d(ctx, layer->bias, 1, Cout, 1, 1));
    }
    return y;
}

} // namespace

Tensor* convtranspose1d_forward(Context* ctx, const ConvTranspose1d* layer, Tensor* x) {
    TRC_ASSERT(layer != nullptr && layer->weight != nullptr,
               "convtranspose1d_forward: 层或权重为空");
    return convtranspose1d_forward_impl(ctx, layer, layer->weight, x);
}

Tensor* convtranspose1d_forward_weight(Context* ctx, const ConvTranspose1d* layer, Tensor* weight,
                                       Tensor* x) {
    TRC_ASSERT(layer != nullptr && weight != nullptr, "convtranspose1d_forward_weight: 层或权重为空");
    return convtranspose1d_forward_impl(ctx, layer, weight, x);
}

void convtranspose1d_params(const ConvTranspose1d* layer, ParamList* out) {
    param_list_add(out, layer->weight);
    if (layer->has_bias && layer->bias != nullptr) {
        param_list_add(out, layer->bias);
    }
}

// ---------------------------------------------------------------- WeightNorm

namespace {

// 归一化分母的数值保护（torch 无 eps；1e-12 对典型范数相对误差 < 1e-8，可忽略）
constexpr float kWeightNormEps = 1e-12f;

void check_weightnorm_shapes(const Tensor* v, const Tensor* g, const char* what) {
    const int nd = tensor_n_dims(v);
    TRC_ASSERT(nd >= 2 && nd <= 4, "%s: v 必须是 2~4 维（当前 %d）", what, nd);
    TRC_ASSERT(tensor_n_dims(g) == nd, "%s: v/g 维数不一致（%d vs %d）", what, nd,
               tensor_n_dims(g));
    TRC_ASSERT(g->ne[nd - 1] == v->ne[nd - 1],
               "%s: g 的输出通道维 %lld != v 的输出通道维 %lld", what, (long long) g->ne[nd - 1],
               (long long) v->ne[nd - 1]);
    for (int d = 0; d + 1 < nd; ++d) {
        TRC_ASSERT(g->ne[d] == 1, "%s: g 除输出通道维外必须为 1（第 %d 维 = %lld）", what, d,
                   (long long) g->ne[d]);
    }
}

} // namespace

void weightnorm_init(Context* ctx, WeightNorm* wn, Tensor* v, Tensor* g, const char* name) {
    TRC_ASSERT(ctx != nullptr && wn != nullptr && v != nullptr && g != nullptr,
               "weightnorm_init: 参数为空");
    check_param(v, "weightnorm_init: v");
    check_param(g, "weightnorm_init: g");
    check_weightnorm_shapes(v, g, "weightnorm_init");

    set_layer_name(wn->name, name, "weightnorm");
    wn->v = v;
    wn->g = g;
    set_param_name(v, wn->name, "weight_v");
    set_param_name(g, wn->name, "weight_g");
    tensor_set_param(v);
    tensor_set_param(g);
}

Tensor* weightnorm_forward(Context* ctx, const WeightNorm* wn) {
    TRC_ASSERT(ctx != nullptr && wn != nullptr && wn->v != nullptr && wn->g != nullptr,
               "weightnorm_forward: 参数为空");
    Tensor* v = wn->v;
    Tensor* g = wn->g;
    check_weightnorm_shapes(v, g, "weightnorm_forward");

    const int     nd = tensor_n_dims(v);
    const int64_t oc = v->ne[nd - 1];
    int64_t       n_rest = 1;
    for (int d = 0; d + 1 < nd; ++d) {
        n_rest *= v->ne[d];
    }

    // w = g · v / sqrt(Σ_非OC(v²) + eps)：逐输出通道归一化（torch weight_norm(dim=0) 语义）。
    // 先把非输出通道维展平成 ne0（v 连续 → reshape 合法），sum_rows 得到 [1, oc]，
    // 再 reshape 成 "除最后一维外全 1" 的形状以符合广播规则（每维相等或 1）。
    Tensor* vf  = reshape_2d(ctx, v, n_rest, oc);
    Tensor* n2  = sum_rows(ctx, mul(ctx, vf, vf));  // [1, oc, 1, 1]
    int64_t ne4[4] = {1, 1, 1, 1};
    ne4[nd - 1] = oc;
    Tensor* n2b = reshape_4d(ctx, n2, ne4[0], ne4[1], ne4[2], ne4[3]);

    Tensor* e = new_tensor_1d(ctx, TYPE_F32, 1);
    nn_init_constant(ctx, e, kWeightNormEps);
    Tensor* den = sqrt_op(ctx, add1(ctx, n2b, e));
    return mul(ctx, div(ctx, v, den), g);
}

void weightnorm_params(const WeightNorm* wn, ParamList* out) {
    TRC_ASSERT(wn != nullptr && wn->v != nullptr && wn->g != nullptr, "weightnorm_params: 参数为空");
    param_list_add(out, wn->g);
    param_list_add(out, wn->v);
}

void weightnorm_sync_g(const WeightNorm* wn) {
    TRC_ASSERT(wn != nullptr && wn->v != nullptr && wn->g != nullptr,
               "weightnorm_sync_g: 参数为空");
    const Tensor* v = wn->v;
    Tensor*       g = wn->g;
    check_param(v, "weightnorm_sync_g: v");
    check_param(g, "weightnorm_sync_g: g");
    check_weightnorm_shapes(v, g, "weightnorm_sync_g");
    TRC_ASSERT(v->data != nullptr && g->data != nullptr,
               "weightnorm_sync_g: v/g 无数据（需先 buffer_alloc_ctx_tensors）");

    const int     nd = tensor_n_dims(v);
    const int64_t oc = v->ne[nd - 1];
    int64_t       n_rest = 1;
    for (int d = 0; d + 1 < nd; ++d) {
        n_rest *= v->ne[d];
    }

    float* gd = (float*) g->data;
    for (int64_t o = 0; o < oc; ++o) {
        double acc = 0.0;
        for (int64_t idx = 0; idx < n_rest; ++idx) {
            size_t  off = 0;
            int64_t rem = idx;
            for (int d = 0; d + 1 < nd; ++d) {
                off += (size_t) (rem % v->ne[d]) * v->nb[d];
                rem /= v->ne[d];
            }
            const float x = *(const float*) ((const uint8_t*) v->data + off +
                                             (size_t) o * v->nb[nd - 1]);
            acc += (double) x * (double) x;
        }
        gd[o] = (float) std::sqrt(acc);
    }
}

// ---------------------------------------------------------------- LayerNorm

void layernorm_init(Context* ctx, LayerNorm* layer, int64_t normalized_size, float eps,
                    bool use_bias, const char* name) {
    TRC_ASSERT(ctx != nullptr && layer != nullptr, "layernorm_init: 参数为空");
    TRC_ASSERT(normalized_size > 0, "layernorm_init: normalized_size 非法");

    set_layer_name(layer->name, name, "layernorm");
    layer->normalized_size = normalized_size;
    layer->eps              = eps;
    layer->has_bias         = use_bias;

    layer->weight = new_tensor_1d(ctx, TYPE_F32, normalized_size);
    set_param_name(layer->weight, layer->name, "weight");
    nn_init_constant(ctx, layer->weight, 1.0f);

    if (use_bias) {
        layer->bias = new_tensor_1d(ctx, TYPE_F32, normalized_size);
        set_param_name(layer->bias, layer->name, "bias");
        nn_init_constant(ctx, layer->bias, 0.0f);
    } else {
        layer->bias = nullptr;
    }
    tensor_set_param(layer->weight);
    if (layer->bias != nullptr) {
        tensor_set_param(layer->bias);
    }
}

void layernorm_init_weight(Context* ctx, LayerNorm* layer, Tensor* weight, Tensor* bias) {
    TRC_ASSERT(ctx != nullptr && layer != nullptr && weight != nullptr,
               "layernorm_init_weight: 参数为空");
    check_param(weight, "layernorm_init_weight: weight");
    TRC_ASSERT(tensor_n_dims(weight) == 1, "layernorm_init_weight: weight 必须是 1 维");
    if (layer->name[0] == '\0') {
        set_layer_name(layer->name, nullptr, "layernorm");
    }
    layer->weight          = weight;
    layer->bias            = bias;
    layer->normalized_size = weight->ne[0];
    layer->has_bias        = bias != nullptr;
    set_param_name(layer->weight, layer->name, "weight");
    if (bias != nullptr) {
        check_param(bias, "layernorm_init_weight: bias");
        TRC_ASSERT(bias->ne[0] == layer->normalized_size, "layernorm_init_weight: bias 维度不匹配");
        set_param_name(layer->bias, layer->name, "bias");
    }
    tensor_set_param(layer->weight);
    if (bias != nullptr) {
        tensor_set_param(layer->bias);
    }
}

Tensor* layernorm_forward(Context* ctx, const LayerNorm* layer, Tensor* x) {
    TRC_ASSERT(ctx != nullptr && layer != nullptr && layer->weight != nullptr && x != nullptr,
               "layernorm_forward: 参数为空");
    TRC_ASSERT(x->ne[0] == layer->normalized_size,
               "layernorm_forward: 归一化维 %lld != normalized_size %lld", (long long) x->ne[0],
               (long long) layer->normalized_size);
    Tensor* y = norm(ctx, x, layer->eps);
    y = mul(ctx, y, layer->weight);
    if (layer->has_bias && layer->bias != nullptr) {
        y = add(ctx, y, layer->bias);
    }
    return y;
}

void layernorm_params(const LayerNorm* layer, ParamList* out) {
    param_list_add(out, layer->weight);
    if (layer->has_bias && layer->bias != nullptr) {
        param_list_add(out, layer->bias);
    }
}

// ---------------------------------------------------------------- RMSNorm

void rmsnorm_init(Context* ctx, RmsNorm* layer, int64_t normalized_size, float eps,
                  const char* name) {
    TRC_ASSERT(ctx != nullptr && layer != nullptr, "rmsnorm_init: 参数为空");
    TRC_ASSERT(normalized_size > 0, "rmsnorm_init: normalized_size 非法");

    set_layer_name(layer->name, name, "rmsnorm");
    layer->normalized_size = normalized_size;
    layer->eps = eps;

    layer->weight = new_tensor_1d(ctx, TYPE_F32, normalized_size);
    set_param_name(layer->weight, layer->name, "weight");
    nn_init_constant(ctx, layer->weight, 1.0f);
    tensor_set_param(layer->weight);
}

void rmsnorm_init_weight(Context* ctx, RmsNorm* layer, Tensor* weight) {
    TRC_ASSERT(ctx != nullptr && layer != nullptr && weight != nullptr,
               "rmsnorm_init_weight: 参数为空");
    check_param(weight, "rmsnorm_init_weight: weight");
    TRC_ASSERT(tensor_n_dims(weight) == 1, "rmsnorm_init_weight: weight 必须是 1 维");
    if (layer->name[0] == '\0') {
        set_layer_name(layer->name, nullptr, "rmsnorm");
    }
    layer->weight          = weight;
    layer->normalized_size = weight->ne[0];
    set_param_name(layer->weight, layer->name, "weight");
    tensor_set_param(layer->weight);
}

Tensor* rmsnorm_forward(Context* ctx, const RmsNorm* layer, Tensor* x) {
    TRC_ASSERT(ctx != nullptr && layer != nullptr && layer->weight != nullptr && x != nullptr,
               "rmsnorm_forward: 参数为空");
    TRC_ASSERT(x->ne[0] == layer->normalized_size,
               "rmsnorm_forward: 归一化维 %lld != normalized_size %lld", (long long) x->ne[0],
               (long long) layer->normalized_size);
    return mul(ctx, rms_norm(ctx, x, layer->eps), layer->weight);
}

void rmsnorm_params(const RmsNorm* layer, ParamList* out) {
    param_list_add(out, layer->weight);
}

// ---------------------------------------------------------------- GroupNorm

void groupnorm_init(Context* ctx, GroupNorm* layer, int64_t num_groups, int64_t num_channels,
                    float eps, bool use_bias, const char* name) {
    TRC_ASSERT(ctx != nullptr && layer != nullptr, "groupnorm_init: 参数为空");
    TRC_ASSERT(num_groups > 0 && num_channels > 0, "groupnorm_init: 维度非法");

    set_layer_name(layer->name, name, "groupnorm");
    layer->num_groups   = num_groups;
    layer->num_channels = num_channels;
    layer->eps          = eps;
    layer->has_bias     = use_bias;

    layer->weight = new_tensor_1d(ctx, TYPE_F32, num_channels);
    set_param_name(layer->weight, layer->name, "weight");
    nn_init_constant(ctx, layer->weight, 1.0f);

    if (use_bias) {
        layer->bias = new_tensor_1d(ctx, TYPE_F32, num_channels);
        set_param_name(layer->bias, layer->name, "bias");
        nn_init_constant(ctx, layer->bias, 0.0f);
    } else {
        layer->bias = nullptr;
    }
    tensor_set_param(layer->weight);
    if (layer->bias != nullptr) {
        tensor_set_param(layer->bias);
    }
}

void groupnorm_init_weight(Context* ctx, GroupNorm* layer, Tensor* weight, Tensor* bias) {
    TRC_ASSERT(ctx != nullptr && layer != nullptr && weight != nullptr,
               "groupnorm_init_weight: 参数为空");
    check_param(weight, "groupnorm_init_weight: weight");
    TRC_ASSERT(tensor_n_dims(weight) == 1, "groupnorm_init_weight: weight 必须是 1 维");
    if (layer->name[0] == '\0') {
        set_layer_name(layer->name, nullptr, "groupnorm");
    }
    layer->weight       = weight;
    layer->bias         = bias;
    layer->num_channels = weight->ne[0];
    layer->has_bias     = bias != nullptr;
    set_param_name(layer->weight, layer->name, "weight");
    if (bias != nullptr) {
        check_param(bias, "groupnorm_init_weight: bias");
        TRC_ASSERT(bias->ne[0] == layer->num_channels, "groupnorm_init_weight: bias 维度不匹配");
        set_param_name(layer->bias, layer->name, "bias");
    }
    tensor_set_param(layer->weight);
    if (bias != nullptr) {
        tensor_set_param(layer->bias);
    }
}

Tensor* groupnorm_forward(Context* ctx, const GroupNorm* layer, Tensor* x) {
    TRC_ASSERT(ctx != nullptr && layer != nullptr && layer->weight != nullptr && x != nullptr,
               "groupnorm_forward: 参数为空");
    TRC_ASSERT(x->ne[2] == layer->num_channels,
               "groupnorm_forward: 通道维 %lld != num_channels %lld", (long long) x->ne[2],
               (long long) layer->num_channels);
    Tensor* y = group_norm(ctx, x, (int) layer->num_groups, layer->eps);
    // 仿射参数 [C] reshape 成 [1,1,C,1] 后按通道广播
    y = mul(ctx, y, reshape_4d(ctx, layer->weight, 1, 1, layer->num_channels, 1));
    if (layer->has_bias && layer->bias != nullptr) {
        y = add(ctx, y, reshape_4d(ctx, layer->bias, 1, 1, layer->num_channels, 1));
    }
    return y;
}

void groupnorm_params(const GroupNorm* layer, ParamList* out) {
    param_list_add(out, layer->weight);
    if (layer->has_bias && layer->bias != nullptr) {
        param_list_add(out, layer->bias);
    }
}

// ---------------------------------------------------------------- Dropout

void dropout_init(Dropout* layer, float p, const char* name) {
    TRC_ASSERT(layer != nullptr, "dropout_init: layer 为空");
    TRC_ASSERT(p >= 0.0f && p < 1.0f, "dropout_init: p 必须在 [0,1) 内");
    set_layer_name(layer->name, name, "dropout");
    layer->p        = p;
    layer->training = true;
    layer->mask     = nullptr;
}

void dropout_set_training(Dropout* layer, bool training) {
    TRC_ASSERT(layer != nullptr, "dropout_set_training: layer 为空");
    layer->training = training;
}

Tensor* dropout_forward(Context* ctx, Dropout* layer, Tensor* x) {
    TRC_ASSERT(ctx != nullptr && layer != nullptr && x != nullptr, "dropout_forward: 参数为空");
    if (!layer->training || layer->p <= 0.0f) {
        return x;  // 推理/关闭：恒等
    }
    if (layer->mask == nullptr || !tensor_are_same_shape(layer->mask, x)) {
        layer->mask = new_tensor_nd(ctx, TYPE_F32, MAX_DIMS, x->ne);
        tensor_set_name(layer->mask, "%s.mask", layer->name);
    }
    return mul(ctx, x, layer->mask);
}

void dropout_refresh(Dropout* layer, Rng* rng) {
    TRC_ASSERT(layer != nullptr, "dropout_refresh: layer 为空");
    TRC_ASSERT(layer->mask != nullptr, "dropout_refresh: 请先调用 dropout_forward");
    TRC_ASSERT(rng != nullptr, "dropout_refresh: rng 为空");
    if (!layer->training || layer->p <= 0.0f) {
        return;
    }

    const float keep = 1.0f / (1.0f - layer->p);
    const size_t n = (size_t) tensor_nelements(layer->mask);
    std::vector<float> mask(n);
    for (float& m : mask) {
        m = (rng_uniform01(rng) < layer->p) ? 0.0f : keep;
    }
    tensor_set(layer->mask, mask.data(), 0, mask.size() * sizeof(float));
}

void dropout_params(const Dropout* layer, ParamList* out) {
    // Dropout 无参数；保留接口便于统一遍历
    (void) layer;
    (void) out;
}

} // namespace traincpp
