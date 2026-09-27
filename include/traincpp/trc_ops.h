// train.cpp - 算子构造函数
// 命名与语义尽量贴近 ggml：例如 add 对应 ggml_add、mul_mat 对应 ggml_mul_mat。
#pragma once

#include "trc_tensor.h"

namespace traincpp {

struct Context;

// ---------------- 元数据复制 ----------------
// 对应 ggml_dup_tensor：只复制形状/类型/名称，不复制数据与 op/src
Tensor* dup(Context* ctx, const Tensor* a);

// ---------------- 视图与形状 ----------------
// 视图语义与 ggml 一致：沿用源张量的 nb 步长，data 指向源张量数据 + offset。
// 注意：视图不拥有内存；缓冲分配（buffer_alloc_ctx_tensors）时会自动解析 data 指针。
Tensor* view_1d(Context* ctx, Tensor* a, int64_t ne0, size_t offset);
Tensor* view_2d(Context* ctx, Tensor* a, int64_t ne0, int64_t ne1, size_t offset);
Tensor* view_3d(Context* ctx, Tensor* a, int64_t ne0, int64_t ne1, int64_t ne2, size_t offset);
Tensor* view_4d(Context* ctx, Tensor* a, int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3, size_t offset);

// reshape 要求源张量连续（与 ggml 一致）；结果是与源共享数据的连续视图
Tensor* reshape_1d(Context* ctx, Tensor* a, int64_t ne0);
Tensor* reshape_2d(Context* ctx, Tensor* a, int64_t ne0, int64_t ne1);
Tensor* reshape_3d(Context* ctx, Tensor* a, int64_t ne0, int64_t ne1, int64_t ne2);
Tensor* reshape_4d(Context* ctx, Tensor* a, int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3);

// 维度置换（axes 必须是 0..3 的排列）
Tensor* permute(Context* ctx, Tensor* a, int axis0, int axis1, int axis2, int axis3);
// 等价于 permute(a, 1, 0, 2, 3)
Tensor* transpose(Context* ctx, Tensor* a);

// 保证连续：源已连续时直接返回源（与 ggml 一致），否则生成 OP_CONT 拷贝节点
Tensor* cont(Context* ctx, Tensor* a);
// 类型转换：类型相同时直接返回源（与 ggml 一致）
Tensor* cast(Context* ctx, Tensor* a, Type type);

// ---------------- 逐元素二元 ----------------
// 结果形状取 a 的形状，b 必须可重复到 a（ggml 广播规则）
Tensor* add(Context* ctx, Tensor* a, Tensor* b);
Tensor* add1(Context* ctx, Tensor* a, Tensor* b);  // b 为标量（1 元素）
Tensor* sub(Context* ctx, Tensor* a, Tensor* b);
Tensor* mul(Context* ctx, Tensor* a, Tensor* b);
Tensor* div(Context* ctx, Tensor* a, Tensor* b);

// ---------------- 逐元素一元 ----------------
Tensor* neg(Context* ctx, Tensor* a);
Tensor* abs_op(Context* ctx, Tensor* a);
Tensor* sqr(Context* ctx, Tensor* a);
Tensor* sqrt_op(Context* ctx, Tensor* a);
Tensor* exp_op(Context* ctx, Tensor* a);
Tensor* log_op(Context* ctx, Tensor* a);
Tensor* sin_op(Context* ctx, Tensor* a);
Tensor* cos_op(Context* ctx, Tensor* a);
Tensor* clamp(Context* ctx, Tensor* a, float min_val, float max_val);
// 符号函数（x>0→1，x<0→-1，x=0→0）与阶跃（x>0→1，否则 0）；分段常量，用于反向组合
Tensor* sgn(Context* ctx, Tensor* a);
Tensor* step(Context* ctx, Tensor* a);

// ---------------- 激活 ----------------
Tensor* relu(Context* ctx, Tensor* a);
Tensor* leaky_relu(Context* ctx, Tensor* a, float slope);
Tensor* sigmoid(Context* ctx, Tensor* a);
Tensor* tanh_op(Context* ctx, Tensor* a);
Tensor* silu(Context* ctx, Tensor* a);
Tensor* gelu(Context* ctx, Tensor* a);      // tanh 近似
Tensor* gelu_erf(Context* ctx, Tensor* a);  // erf 精确
Tensor* erf_op(Context* ctx, Tensor* a);    // erf（ggml 无此独立算子，本库扩展，用于 gelu_erf 反向）
Tensor* softplus(Context* ctx, Tensor* a);
Tensor* hardswish(Context* ctx, Tensor* a);

// ---------------- 缩放 ----------------
// dst = a * s
Tensor* scale(Context* ctx, Tensor* a, float s);
// 标量仿射（与 ggml_scale_bias 语义一致）：dst = s * a + b（逐元素，无张量广播）。
// 组合实现（scale + add1 + 标量常量），自动反传：dA = s * dY（b 为常量）
Tensor* scale_bias(Context* ctx, Tensor* a, float s, float b);

// ---------------- 归约（形状规则与 ggml 一致）----------------
// 全部元素求和：结果形状 [1]，类型与输入相同
Tensor* sum(Context* ctx, Tensor* a);
// 沿 ne0 求和：结果形状 [1, ne1, ne2, ne3]，类型与输入相同
Tensor* sum_rows(Context* ctx, Tensor* a);
// 沿 ne0 求均值：结果形状 [1, ne1, ne2, ne3]，类型 F32
Tensor* mean(Context* ctx, Tensor* a);

// ---------------- 归一化（语义与 ggml 一致）----------------
// 行内 LayerNorm：对每个 (i1,i2,i3) 行（长度 ne0）做 (x-mean)/sqrt(var+eps)
Tensor* norm(Context* ctx, Tensor* a, float eps);
// 行内 RMSNorm：x / sqrt(mean(x²)+eps)
Tensor* rms_norm(Context* ctx, Tensor* a, float eps);
// 分组归一化：沿 ne2（通道维）分组，对每组做 (x-mean)/sqrt(var+eps)
Tensor* group_norm(Context* ctx, Tensor* a, int n_groups, float eps);

// ---------------- 归一化反向（专用 kernel）----------------
// 入参统一为：a = 前向输出的梯度 dy，b = 前向输入 x；结果形状与 b 相同
Tensor* norm_back(Context* ctx, Tensor* a, Tensor* b, float eps);
Tensor* rms_norm_back(Context* ctx, Tensor* a, Tensor* b, float eps);
Tensor* group_norm_back(Context* ctx, Tensor* a, Tensor* b, int n_groups, float eps);

// ---------------- 注意力 ----------------
// softmax 沿 ne0；等价 soft_max_ext(a, NULL, 1.0f, 0.0f)
Tensor* soft_max(Context* ctx, Tensor* a);
// y = softmax(scale * a + slope * mask)；mask 形状 [ne0, >=ne1, 除数, 除数]（与 ggml 相同）
// max_bias > 0 时启用 ALiBi（与 ggml 相同）
Tensor* soft_max_ext(Context* ctx, Tensor* a, Tensor* mask, float scale, float max_bias);

// ---------------- 损失（语义与 ggml 一致）----------------
// 交叉熵：a = logits（F32），b = 目标概率分布（与 a 同形状、最内维连续，one-hot 或软标签）
//   输出 F32 标量 [1]：-1/nr * Σ_row Σ_c b * log_softmax(a)，nr = ne1*ne2*ne3（与 ggml_nrows 一致）
Tensor* cross_entropy_loss(Context* ctx, Tensor* a, Tensor* b);
// 交叉熵反向（与 ggml_cross_entropy_loss_back 一致）：
//   grad = 标量 [1]（损失的梯度），a = 前向 logits，b = 前向目标
//   输出与 a 同形状：dA = (softmax(a) - b) * grad / nr；target 视为常量（无梯度）
Tensor* cross_entropy_loss_back(Context* ctx, Tensor* grad, Tensor* a, Tensor* b);

// ---------------- 索引（形状规则与 ggml 一致）----------------
// 按行查表：b 为 I32 索引 [ni0, ni1, ni2]；a 表 [ne0, n_vocab, ni1, ni2]
// 结果形状 [a->ne[0], b->ne[0], b->ne[1], b->ne[2]]，类型 F32（I32 表暂不支持）
Tensor* get_rows(Context* ctx, Tensor* a, Tensor* b);
// get_rows 的反向（scatter-add，与 ggml_get_rows_back 一致）：
//   a 为输出梯度 [ne0, nr, ni1, ni2]，b 为索引 [nr, ni1, ni2]（I32），c 为查表 [ne0, n_vocab, ni1, ni2]
//   结果形状与 c 相同（F32，先清零再按行累加）；c 仅用于形状，但登记为 src[2]
//   （图覆盖/内存分配需要，内核不读取其数据）
Tensor* get_rows_back(Context* ctx, Tensor* a, Tensor* b, Tensor* c);
// 按行写入（Embedding 反向的 scatter 基础）：
//   a 目标张量，b 源行数据 [ne0, n_rows, b.ne2, b.ne3]，c 为 I32/I64 行索引 [n_rows, c.ne1, c.ne2]
// 结果与 a 共享内存（写入 a），与 ggml_set_rows 一致
Tensor* set_rows(Context* ctx, Tensor* a, Tensor* b, Tensor* c);

// ---------------- 形状扩展 ----------------
// 将 a 重复到 target 的形状（要求 a 可重复到 target，与 ggml_repeat 一致）
Tensor* repeat(Context* ctx, Tensor* a, Tensor* target);
// repeat 的反向：沿被重复的维度求和回 b 的形状（要求 b 可重复到 a，与 ggml_repeat_back 一致）
Tensor* repeat_back(Context* ctx, Tensor* a, Tensor* b);
// 沿 dim(0..3) 拼接；其余维度必须相同（与 ggml_concat 一致）
Tensor* concat(Context* ctx, Tensor* a, Tensor* b, int dim);
// 在每一维末尾补零：结果 ne[i] = a->ne[i] + p[i]（与 ggml_pad 一致）
Tensor* pad(Context* ctx, Tensor* a, int p0, int p1, int p2, int p3);
// 填充模式：ZERO = 补零（ggml_pad_ext）；REFLECT = 镜像反射（torch F.pad(mode='reflect')，
// 不重复边缘；要求每侧填充量 < 对应维长度，且至少为 2 才能产生非零反射）
enum PadMode : int32_t {
    PAD_ZERO    = 0,
    PAD_REFLECT = 1,
};
// 扩展填充（M2.3c）：每维左右两侧分别填充；mode=PAD_REFLECT 时镜像反射。
// op_params 布局与 ggml_pad_ext 一致：int32[8]={lp0,rp0,lp1,rp1,lp2,rp2,lp3,rp3}，
// 第 9 个槽位 op_params[8] 存 mode（ggml 不使用该槽位）
Tensor* pad_ext(Context* ctx, Tensor* a, int lp0, int rp0, int lp1, int rp1, int lp2, int rp2,
                int lp3, int rp3, PadMode mode = PAD_ZERO);
// reflect 填充的反向（组合内核）：输入 grad 形状 [a.ne+lp+rp]、b 为前向输入（取形状）。
//   仅对 PAD_REFLECT 需要（PAD_ZERO 反向走视图裁剪，见 autograd）
Tensor* pad_back(Context* ctx, Tensor* a, Tensor* b, int lp0, int rp0, int lp1, int rp1, int lp2,
                 int rp2, int lp3, int rp3);
// 将 b 累加到 a 的指定区域（视图反向的 scatter-add 基础，与 ggml_acc 一致）：
//   nb1..nb3 为视图第 1..3 维字节步长，offset 为相对 a 起点的字节偏移；b 的形状即视图形状
//   inplace=true 时结果与 a 共享内存；否则结果为 a 的完整拷贝再加上 b
Tensor* acc(Context* ctx, Tensor* a, Tensor* b, size_t nb1, size_t nb2, size_t nb3, size_t offset,
            bool inplace);

// ---------------- 卷积 ----------------
// im2col：a 为卷积核（1D: [KW, IC, OC]；2D: [KW, KH, IC, OC]），b 为输入
//   1D: b=[W, IC, N]（b.ne3 必须为 1）
//   2D: b=[W, H, IC, N]（a.ne2 == b.ne2）
// 结果（与 ggml 相同）：
//   1D: [IC*KW, OW, N, 1]
//   2D: [IC*KH*KW, OW, OH, N]
Tensor* im2col(Context* ctx, Tensor* a, Tensor* b, int s0, int s1, int p0, int p1, int d0, int d1,
               bool is_2d, Type dst_type);

// im2col 的反向（对输入图像的梯度，与 ggml_im2col_back 一致）：
//   a 为 im2col 输出的梯度 [IC*KW, OW, N, 1] 或 [IC*KH*KW, OW, OH, N]（F32，需连续）
//   b 为卷积核（仅用于取形状），ne 为输入图像形状（4 维），结果形状 = ne（F32）
Tensor* im2col_back(Context* ctx, Tensor* a, Tensor* b, const int64_t* ne, int s0, int s1, int p0,
                    int p1, int d0, int d1, bool is_2d);

// 1D 卷积：a=[KW, IC, OC]，b=[W, IC, N]，结果 [OW, OC, N]（ggml_conv_1d 组合语义）
Tensor* conv_1d(Context* ctx, Tensor* a, Tensor* b, int s0, int p0, int d0);
// 2D 卷积：a=[KW, KH, IC, OC]，b=[W, H, IC, N]，结果 [OW, OH, OC, N]（ggml_conv_2d 组合语义）
Tensor* conv_2d(Context* ctx, Tensor* a, Tensor* b, int s0, int s1, int p0, int p1, int d0, int d1);

// col2im_1d：a=[K*OC, T_in]，结果 [T_out, OC]，T_out=(T_in-1)*s0+K-2*p0（与 ggml_col2im_1d 一致）
Tensor* col2im_1d(Context* ctx, Tensor* a, int s0, int oc, int p0);
// 1D 转置卷积：a=[K, Cout, Cin]，b=[T_in, Cin]（2D），结果 [T_out, Cout, 1, 1]
// T_out=(T_in-1)*s0+K（ggml 仅支持 p0=0、d0=1，本库同）
Tensor* conv_transpose_1d(Context* ctx, Tensor* a, Tensor* b, int s0, int p0, int d0);

// ---------------- 池化 / 2D 转置卷积（M2.3g，对齐 ggml）----------------
// 池化模式（数值与 enum ggml_op_pool 一致：0=MAX、1=AVG）
enum PoolMode : int32_t {
    POOL_MAX = 0,
    POOL_AVG = 1,
};

// 2D 池化（语义与 ggml_pool_2d 一致；p0/p1 取整——ggml 形参为 float 但写入 op_params 时截断）：
//   a=[W, H, C, N]；输出 [OW, OH, C, N]，OW=(W+2*p0-k0)/s0+1、OH=(H+2*p1-k1)/s1+1（floor）
//   AVG 分母恒为 k0*k1（越界元素按 0 计入），MAX 跳过越界元素（窗口全越界输出 -FLT_MAX）
//   op_params int32[7] = {mode, k0, k1, s0, s1, p0, p1}（与 ggml 槽位一致）
Tensor* pool_2d(Context* ctx, Tensor* a, PoolMode mode, int k0, int k1, int s0, int s1, int p0,
                int p1);
// pool_2d 的反向（专用 kernel；a = 前向输出梯度，b = 前向输入，仅取形状；结果形状 = b）：
//   AVG：每个界内元素 += grad/(k0*k1)；MAX：grad 累加到窗口内首个严格最大值（与 ggml 一致）
Tensor* pool_2d_back(Context* ctx, Tensor* a, Tensor* b, PoolMode mode, int k0, int k1, int s0,
                     int s1, int p0, int p1);

// 2D 转置卷积（语义与 ggml_conv_transpose_2d_p0 一致；仅 stride、p=0、d=1）：
//   a=[KW, KH, Cout, Cin]（要求 a->ne[3] == b->ne[2]），b=[W, H, Cin, N]
//   结果 [OW, OH, Cout, N]，OW=(W-1)*stride+KW、OH=(H-1)*stride+KH；op_params[0]=stride
Tensor* conv_transpose_2d(Context* ctx, Tensor* a, Tensor* b, int stride);

// ---------------- 线性代数 ----------------
// dst = a * b（ggml_mul_mat 语义：a 形状 [k, m]，b 形状 [k, n]，结果 [m, n]）
Tensor* mul_mat(Context* ctx, Tensor* a, Tensor* b);

} // namespace traincpp
