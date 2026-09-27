// train.cpp - 张量定义
// 内存布局语义与 ggml 完全一致：
//   nb[0] = type_size(type)
//   nb[1] = nb[0] * (ne[0] / blck_size(type))
//   nb[i] = nb[i-1] * ne[i-1]
#pragma once

#include "trc_op.h"
#include "trc_types.h"

#include <cstddef>
#include <cstdint>

namespace traincpp {

constexpr int    MAX_DIMS      = 4;   // 对应 GGML_MAX_DIMS
constexpr int    MAX_SRC       = 10;  // 对应 GGML_MAX_SRC（图输入最多 10 个）
constexpr int    MAX_NAME      = 64;  // 对应 GGML_MAX_NAME
constexpr size_t MAX_OP_PARAMS = 64;  // 对应 GGML_MAX_OP_PARAMS（字节）
constexpr size_t MEM_ALIGN     = 16;  // 对应 GGML_MEM_ALIGN

struct Context;
class Buffer;
struct Tensor;
struct Graph;

// 张量标志，取值与 ggml_tensor_flag 一致（ggml.h:662-668）
enum TensorFlags : int32_t {
    TENSOR_FLAG_NONE    = 0,
    TENSOR_FLAG_INPUT   = 1,   // 计算图输入
    TENSOR_FLAG_OUTPUT  = 2,   // 计算图输出
    TENSOR_FLAG_PARAM   = 4,   // 可训练参数
    TENSOR_FLAG_LOSS    = 8,   // 损失（多个损失相加）
    TENSOR_FLAG_COMPUTE = 16,  // 必须参与计算
};

struct Tensor {
    Type type = TYPE_F32;

    Buffer* buffer = nullptr;  // 所属后端 buffer（由 buffer_alloc_ctx_tensors 设置）

    int64_t ne[MAX_DIMS] = {1, 1, 1, 1};  // 各维度元素数
    size_t  nb[MAX_DIMS] = {0, 0, 0, 0};  // 各维度字节步长

    Op op = OP_NONE;

    int32_t op_params[MAX_OP_PARAMS / sizeof(int32_t)] = {0};  // 算子参数

    int32_t flags = 0;

    Tensor* src[MAX_SRC] = {nullptr};  // 输入张量
    int32_t nsrc = 0;

    Tensor* view_src  = nullptr;  // 视图的源张量
    size_t  view_offs = 0;        // 视图在源张量中的字节偏移

    void* data = nullptr;  // 张量数据（可能为 nullptr：no_alloc 模式）

    char name[MAX_NAME] = {0};

    void* extra = nullptr;  // 后端扩展数据（如 Vulkan 设备指针）

    // ---- 训练扩展字段（ggml 无此字段，不参与兼容）----
    Tensor* grad     = nullptr;  // 反向图中本张量的梯度
    Tensor* grad_acc = nullptr;  // 梯度累加器（优化器使用）
    // 已构建反向所属图（M2.1e / R4：禁止多图共享参数与梯度路径中间量；
    // graph_free/graph_clear 会释放归属，之后可重新用于新图）
    Graph* backward_graph = nullptr;
};

// 张量形状/大小工具（与 ggml 同名函数语义一致）
int     tensor_n_dims(const Tensor* t);       // 实际维度数（ne>1 的最大维度 + 1，最少 1）
int64_t tensor_nelements(const Tensor* t);
int64_t tensor_nrows(const Tensor* t);
size_t  tensor_nbytes(const Tensor* t);
size_t  tensor_element_size(const Tensor* t);
size_t  tensor_row_size(const Tensor* t);     // 第 0 维整行字节数
bool    tensor_is_contiguous(const Tensor* t);
bool    tensor_are_same_shape(const Tensor* a, const Tensor* b);
// ggml_can_repeat 语义：b 的每个维度等于 1 或等于 a 的对应维度
bool    tensor_can_repeat(const Tensor* b, const Tensor* a);
bool    tensor_is_view(const Tensor* t);

// 线性索引（基于 ne 行主序）到字节偏移
size_t tensor_offset_linear(const Tensor* t, int64_t i);

// 创建张量：从 ctx 的 arena 分配元数据；no_alloc=false 时同时分配数据
Tensor* new_tensor_nd(Context* ctx, Type type, int n_dims, const int64_t* ne);
Tensor* new_tensor_1d(Context* ctx, Type type, int64_t ne0);
Tensor* new_tensor_2d(Context* ctx, Type type, int64_t ne0, int64_t ne1);
Tensor* new_tensor_3d(Context* ctx, Type type, int64_t ne0, int64_t ne1, int64_t ne2);
Tensor* new_tensor_4d(Context* ctx, Type type, int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3);

// 仅复制张量元数据（不复制数据、不带 op/src），对应 ggml_dup_tensor
Tensor* tensor_dup_meta(Context* ctx, const Tensor* src);

// 数据访问辅助（要求 ctx 为 no_alloc=false 或已由后端 buffer 分配）
void tensor_set_name(Tensor* t, const char* fmt, ...);

// 便捷数据指针
float*   tensor_data_f32(Tensor* t);
const float* tensor_data_f32(const Tensor* t);

} // namespace traincpp
