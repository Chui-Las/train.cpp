// train.cpp - Vulkan 后端算子 dispatch
//
// M1.4b：逐元素/一元/激活
// M1.4c：拷贝类（cont/cpy/dup/pad/concat）、归约（sum/sum_rows/mean）、
//        归一化（norm/rms_norm/group_norm）、soft_max/soft_max_ext
// 寻址：ne/nb（元素单位）由 push constant 传给 shader，天然支持视图；二元/softmax 支持广播。
#include "trc_vulkan.h"

#include "core/trc_impl.h"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace traincpp {

namespace {

bool is_f32(const Tensor* t) { return t != nullptr && t->type == TYPE_F32; }

bool srcs_f32(const Tensor* t, int n_src) {
    if (!is_f32(t)) {
        return false;
    }
    for (int i = 0; i < n_src; ++i) {
        if (!is_f32(t->src[i])) {
            return false;
        }
    }
    return true;
}

void fill_ne(uint32_t dst[4], const Tensor* t) {
    for (int d = 0; d < 4; ++d) {
        TRC_ASSERT(t->ne[d] > 0 && t->ne[d] <= (int64_t) UINT32_MAX,
                   "Vulkan 后端：维度越界（%s, ne[%d]=%lld）", t->name, d, (long long) t->ne[d]);
        dst[d] = (uint32_t) t->ne[d];
    }
}

void fill_nb(uint32_t dst[4], const Tensor* t) {
    for (int d = 0; d < 4; ++d) {
        TRC_ASSERT(t->nb[d] % sizeof(float) == 0,
                   "Vulkan 后端要求 F32 步长 4 字节对齐（%s, nb[%d]=%zu）", t->name, d, t->nb[d]);
        const size_t n = t->nb[d] / sizeof(float);
        TRC_ASSERT(n <= (size_t) UINT32_MAX, "Vulkan 后端：步长越界（%s, nb[%d]=%zu 元素 > UINT32_MAX）",
                   t->name, d, t->nb[d]);  // R8
        dst[d] = (uint32_t) n;
    }
}

uint32_t nb_elem(const Tensor* t, int d) {
    TRC_ASSERT(t->nb[d] % sizeof(float) == 0,
               "Vulkan 后端要求 F32 步长 4 字节对齐（%s, nb[%d]=%zu）", t->name, d, t->nb[d]);
    const size_t n = t->nb[d] / sizeof(float);
    TRC_ASSERT(n <= (size_t) UINT32_MAX, "Vulkan 后端：步长越界（%s, nb[%d]=%zu 元素 > UINT32_MAX）",
               t->name, d, t->nb[d]);  // R8
    return (uint32_t) n;
}

// 按类型宽度换算步长（F16=2、F32/I32=4）
void fill_nb_typed(uint32_t dst[4], const Tensor* t) {
    const size_t elem = type_size(t->type);
    for (int d = 0; d < 4; ++d) {
        TRC_ASSERT(t->nb[d] % elem == 0, "Vulkan 后端：步长必须按类型宽度对齐（%s, nb[%d]=%zu）",
                   t->name, d, t->nb[d]);
        const size_t n = t->nb[d] / elem;
        TRC_ASSERT(n <= (size_t) UINT32_MAX, "Vulkan 后端：步长越界（%s, nb[%d]=%zu 元素 > UINT32_MAX）",
                   t->name, d, t->nb[d]);  // R8
        dst[d] = (uint32_t) n;
    }
}

uint32_t count_u32(const Tensor* t) {
    const int64_t n = tensor_nelements(t);
    TRC_ASSERT(n > 0 && n <= (int64_t) UINT32_MAX, "Vulkan 后端：元素数越界（%s）", t->name);
    return (uint32_t) n;
}

// uint32 乘积守卫（M2.1h / R8）：push constant 字段都是 32 位，乘积静默截断会算错
uint32_t mul_u32_checked(uint64_t a, uint64_t b, const char* what) {
    const uint64_t p = a * b;
    TRC_ASSERT(p <= (uint64_t) UINT32_MAX,
               "Vulkan 后端：%s 乘积越界（%llu * %llu = %llu > UINT32_MAX）", what,
               (unsigned long long) a, (unsigned long long) b, (unsigned long long) p);
    return (uint32_t) p;
}

uint32_t mul_u32_checked(uint64_t a, uint64_t b, uint64_t c, const char* what) {
    return mul_u32_checked(mul_u32_checked(a, b, what), c, what);
}

// 覆盖 numel 个元素所需的 workgroup 数（local_size=256）；uint64 防 numel+255 溢出
uint32_t group_count(uint32_t numel) {
    return (uint32_t) (((uint64_t) numel + 255u) / 256u);
}

// workgroup 上限：所有使用 group_count 的 shader 都用 gl_NumWorkGroups 做 stride 循环，
// 钳制后仍覆盖全部元素（TRC_VK_MAX_GROUPS 可在测试中强制更小上限）。
uint32_t limit_groups(uint32_t wanted) {
    return std::max(1u, std::min(wanted, vulkan_max_grid_x()));
}

// ---------------------------------------------------------------- 逐元素

void dispatch_binary(VulkanGraphContext& gc, Tensor* node, VulkanBinaryOp op) {
    const Tensor* a = node->src[0];
    const Tensor* b = node->src[1];

    VulkanBinaryPC pc{};
    pc.op = (uint32_t) op;
    fill_ne(pc.ne_a, node);
    fill_ne(pc.ne_b, b);
    fill_nb(pc.nb_a, a);
    fill_nb(pc.nb_b, b);
    fill_nb(pc.nb_d, node);

    const VulkanTensorBinding binds[3] = { vulkan_bind_tensor(a), vulkan_bind_tensor(b),
                                           vulkan_bind_tensor(node) };
    pc.base_a = binds[0].base_elem;
    pc.base_b = binds[1].base_elem;
    pc.base_d = binds[2].base_elem;
    pc.numel  = count_u32(node);

    vulkan_dispatch(gc, vulkan_pipeline("binary", sizeof(VulkanBinaryPC)), &pc, sizeof(pc), binds, 3,
                    limit_groups(group_count(pc.numel)));
}

void dispatch_unary(VulkanGraphContext& gc, Tensor* node, VulkanUnaryOp op, float p0 = 0.0f,
                    float p1 = 0.0f) {
    const Tensor* a = node->src[0];

    VulkanUnaryPC pc{};
    pc.op = (uint32_t) op;
    fill_ne(pc.ne, node);
    fill_nb(pc.nb_a, a);
    fill_nb(pc.nb_d, node);

    const VulkanTensorBinding binds[2] = { vulkan_bind_tensor(a), vulkan_bind_tensor(node) };
    pc.base_a = binds[0].base_elem;
    pc.base_d = binds[1].base_elem;
    pc.numel  = count_u32(node);
    pc.p0     = p0;
    pc.p1     = p1;

    vulkan_dispatch(gc, vulkan_pipeline("unary", sizeof(VulkanUnaryPC)), &pc, sizeof(pc), binds, 2,
                    limit_groups(group_count(pc.numel)));
}

// ---------------------------------------------------------------- 类型转换

// F32 -> F16 / F16 -> F32（与 CPU `cast` 相同语义；需要 16 位存储特性）
void dispatch_cast(VulkanGraphContext& gc, Tensor* node) {
    const Tensor* a      = node->src[0];
    const bool    to_f16 = node->type == TYPE_F16;
    // R6：只支持 F32↔F16；其他组合（如 F32→I32）必须在分派前中止，不得按 F16 路径错读
    TRC_ASSERT((to_f16 && a->type == TYPE_F32) ||
                   (!to_f16 && node->type == TYPE_F32 && a->type == TYPE_F16),
               "Vulkan cast 仅支持 F32↔F16（dst=%s, src=%s）", type_name(node->type),
               type_name(a->type));

    VulkanCastPC pc{};
    fill_ne(pc.ne, node);
    fill_nb_typed(pc.nb_s, a);
    fill_nb_typed(pc.nb_d, node);

    const VulkanTensorBinding binds[2] = { vulkan_bind_tensor(a), vulkan_bind_tensor(node) };
    pc.base_s = binds[0].base_elem;
    pc.base_d = binds[1].base_elem;
    pc.numel  = count_u32(node);

    vulkan_dispatch(gc,
                    vulkan_pipeline(to_f16 ? "cast_f32_to_f16" : "cast_f16_to_f32",
                                    sizeof(VulkanCastPC)),
                    &pc, sizeof(pc), binds, 2, limit_groups(group_count(pc.numel)));
}

// ---------------------------------------------------------------- 线代

// ggml_mul_mat 语义：a=[k,m,a2,a3] * b=[k,n,b2,b3] -> d=[m,n,b2,b3]
// 平面 (i2,i3) 使用 a 的平面 (i2 % a2, i3 % a3)（batch 广播）；平面放 grid z
// 分块与 shader 内常量一致：BM=BN=64、BK=16
void dispatch_mul_mat(VulkanGraphContext& gc, Tensor* node) {
    const Tensor* a = node->src[0];
    const Tensor* b = node->src[1];

    VulkanMatMulPC pc{};
    pc.k  = (uint32_t) a->ne[0];
    pc.m  = (uint32_t) a->ne[1];
    pc.n  = (uint32_t) b->ne[1];
    pc.n_planes = mul_u32_checked((uint64_t) b->ne[2], (uint64_t) b->ne[3], "mul_mat n_planes");
    pc.b_ne2 = (uint32_t) b->ne[2];
    pc.a_ne2 = (uint32_t) a->ne[2];
    pc.a_ne3 = (uint32_t) a->ne[3];
    pc.nb_a0 = nb_elem(a, 0);
    pc.nb_a1 = nb_elem(a, 1);
    pc.nb_a2 = nb_elem(a, 2);
    pc.nb_a3 = nb_elem(a, 3);
    pc.nb_b0 = nb_elem(b, 0);
    pc.nb_b1 = nb_elem(b, 1);
    pc.nb_b2 = nb_elem(b, 2);
    pc.nb_b3 = nb_elem(b, 3);
    pc.nb_d0 = nb_elem(node, 0);
    pc.nb_d1 = nb_elem(node, 1);
    pc.nb_d2 = nb_elem(node, 2);
    pc.nb_d3 = nb_elem(node, 3);

    const VulkanTensorBinding binds[3] = { vulkan_bind_tensor(a), vulkan_bind_tensor(b),
                                           vulkan_bind_tensor(node) };
    pc.base_a = binds[0].base_elem;
    pc.base_b = binds[1].base_elem;
    pc.base_d = binds[2].base_elem;

    const uint32_t gx = (pc.n + 63u) / 64u;
    const uint32_t gy = (pc.m + 63u) / 64u;
    const uint32_t max_x = vulkan_state().props.limits.maxComputeWorkGroupCount[0];
    const uint32_t max_y = vulkan_state().props.limits.maxComputeWorkGroupCount[1];
    TRC_ASSERT(gx <= max_x && gy <= max_y,
               "Vulkan mul_mat: 分块网格 (%u,%u) 超设备上限 (%u,%u)", gx, gy, max_x, max_y);
    // z（batch 平面）钳制：shader 用 gl_NumWorkGroups.z 跨步覆盖全部平面
    const uint32_t gz = std::max(1u, std::min(pc.n_planes, vulkan_max_grid_z()));

    vulkan_dispatch(gc, vulkan_pipeline("matmul", sizeof(VulkanMatMulPC)), &pc, sizeof(pc), binds, 3,
                    gx, gy, gz);
}

// ---------------------------------------------------------------- 卷积

// im2col（1D/2D；形状与 op_params 布局同 ggml/CPU）
void dispatch_im2col(VulkanGraphContext& gc, Tensor* node) {
    const Tensor*  kernel = node->src[0];
    const Tensor*  img    = node->src[1];
    const int32_t* p      = node->op_params;
    const bool     is_2d  = p[6] == 1;

    const int64_t N  = is_2d ? img->ne[3] : img->ne[2];
    const int64_t IC = is_2d ? img->ne[2] : img->ne[1];
    const int64_t IH = is_2d ? img->ne[1] : 1;
    const int64_t IW = img->ne[0];
    const int64_t KH = is_2d ? kernel->ne[1] : 1;
    const int64_t KW = kernel->ne[0];
    const int64_t OH = is_2d ? node->ne[2] : 1;
    const int64_t OW = node->ne[1];

    VulkanIm2ColPC pc{};
    pc.feat  = mul_u32_checked(IC, KH, KW, "im2col feat");
    pc.OW    = (uint32_t) OW;
    pc.OH    = (uint32_t) OH;
    pc.N     = (uint32_t) N;
    pc.IC    = (uint32_t) IC;
    pc.IH    = (uint32_t) IH;
    pc.IW    = (uint32_t) IW;
    pc.KH    = (uint32_t) KH;
    pc.KW    = (uint32_t) KW;
    pc.s0    = p[0];
    pc.s1    = p[1];
    pc.p0    = p[2];
    pc.p1    = p[3];
    pc.d0    = p[4];
    pc.d1    = p[5];
    pc.is_2d = is_2d ? 1u : 0u;
    fill_nb(pc.nb_img, img);
    fill_nb(pc.nb_d, node);

    const VulkanTensorBinding binds[2] = { vulkan_bind_tensor(img), vulkan_bind_tensor(node) };
    pc.base_img = binds[0].base_elem;
    pc.base_d   = binds[1].base_elem;
    pc.numel    = count_u32(node);

    vulkan_dispatch(gc, vulkan_pipeline("im2col", sizeof(VulkanIm2ColPC)), &pc, sizeof(pc), binds, 2,
                    limit_groups(group_count(pc.numel)));
}

// im2col 反向（对输入图像的梯度；grad/dst 步长均支持视图）
void dispatch_im2col_back(VulkanGraphContext& gc, Tensor* node) {
    const Tensor*  grad   = node->src[0];
    const Tensor*  kernel = node->src[1];
    const int32_t* p      = node->op_params;
    const bool     is_2d  = p[6] == 1;

    const int64_t N  = is_2d ? node->ne[3] : node->ne[2];
    const int64_t IC = is_2d ? node->ne[2] : node->ne[1];
    const int64_t IH = is_2d ? node->ne[1] : 1;
    const int64_t IW = node->ne[0];
    const int64_t KH = is_2d ? kernel->ne[1] : 1;
    const int64_t KW = kernel->ne[0];
    const int64_t OH = is_2d ? grad->ne[2] : 1;
    const int64_t OW = grad->ne[1];

    VulkanIm2ColPC pc{};
    pc.feat  = mul_u32_checked(IC, KH, KW, "im2col feat");
    pc.OW    = (uint32_t) OW;
    pc.OH    = (uint32_t) OH;
    pc.N     = (uint32_t) N;
    pc.IC    = (uint32_t) IC;
    pc.IH    = (uint32_t) IH;
    pc.IW    = (uint32_t) IW;
    pc.KH    = (uint32_t) KH;
    pc.KW    = (uint32_t) KW;
    pc.s0    = p[0];
    pc.s1    = p[1];
    pc.p0    = p[2];
    pc.p1    = p[3];
    pc.d0    = p[4];
    pc.d1    = p[5];
    pc.is_2d = is_2d ? 1u : 0u;
    fill_nb(pc.nb_img, grad);
    fill_nb(pc.nb_d, node);

    const VulkanTensorBinding binds[2] = { vulkan_bind_tensor(grad), vulkan_bind_tensor(node) };
    pc.base_img = binds[0].base_elem;
    pc.base_d   = binds[1].base_elem;
    pc.numel    = count_u32(node);

    vulkan_dispatch(gc, vulkan_pipeline("im2col_back", sizeof(VulkanIm2ColPC)), &pc, sizeof(pc),
                    binds, 2, limit_groups(group_count(pc.numel)));
}

// col2im_1d：src=[K*OC,T_in] -> dst=[T_out,OC]（gather 语义与 ggml 一致）
void dispatch_col2im(VulkanGraphContext& gc, Tensor* node) {
    const Tensor* src = node->src[0];
    const int32_t s0  = node->op_params[0];
    const int32_t oc  = node->op_params[1];
    const int32_t p0  = node->op_params[2];
    TRC_ASSERT(oc > 0 && s0 > 0 && p0 >= 0, "Vulkan col2im: 参数非法");

    VulkanCol2ImPC pc{};
    pc.K     = (uint32_t) (src->ne[0] / oc);
    pc.OC    = (uint32_t) oc;
    pc.T_in  = (uint32_t) src->ne[1];
    pc.s0    = s0;
    pc.p0    = p0;
    pc.nb_s0 = nb_elem(src, 0);
    pc.nb_s1 = nb_elem(src, 1);
    pc.nb_d0 = nb_elem(node, 0);
    pc.nb_d1 = nb_elem(node, 1);

    const VulkanTensorBinding binds[2] = { vulkan_bind_tensor(src), vulkan_bind_tensor(node) };
    pc.base_s = binds[0].base_elem;
    pc.base_d = binds[1].base_elem;
    pc.numel  = mul_u32_checked((uint64_t) oc, (uint64_t) node->ne[0], "col2im numel");

    vulkan_dispatch(gc, vulkan_pipeline("col2im", sizeof(VulkanCol2ImPC)), &pc, sizeof(pc), binds, 2,
                    limit_groups(group_count(pc.numel)));
}

// conv_transpose_1d：kernel=[K,Cout,Cin]、input=[T_in,Cin] -> dst=[T_out,Cout,1,1]
void dispatch_conv_transpose_1d(VulkanGraphContext& gc, Tensor* node) {
    const Tensor* kernel = node->src[0];
    const Tensor* input  = node->src[1];

    VulkanConvTranspose1DPC pc{};
    pc.K     = (uint32_t) kernel->ne[0];
    pc.Cout  = (uint32_t) kernel->ne[1];
    pc.Cin   = (uint32_t) kernel->ne[2];
    pc.T_in  = (uint32_t) input->ne[0];
    pc.s0    = node->op_params[0];
    pc.nb_k0 = nb_elem(kernel, 0);
    pc.nb_k1 = nb_elem(kernel, 1);
    pc.nb_k2 = nb_elem(kernel, 2);
    pc.nb_x0 = nb_elem(input, 0);
    pc.nb_x1 = nb_elem(input, 1);
    pc.nb_d0 = nb_elem(node, 0);
    pc.nb_d1 = nb_elem(node, 1);

    const VulkanTensorBinding binds[3] = { vulkan_bind_tensor(kernel), vulkan_bind_tensor(input),
                                           vulkan_bind_tensor(node) };
    pc.base_k = binds[0].base_elem;
    pc.base_x = binds[1].base_elem;
    pc.base_d = binds[2].base_elem;
    pc.numel  = mul_u32_checked(kernel->ne[1], node->ne[0], "conv_transpose_1d numel");

    vulkan_dispatch(gc, vulkan_pipeline("conv_transpose_1d", sizeof(VulkanConvTranspose1DPC)), &pc,
                    sizeof(pc), binds, 3, limit_groups(group_count(pc.numel)));
}

// pool_2d：a=[W,H,C,N] -> dst=[OW,OH,C,N]（AVG 分母 k0*k1；MAX 跳过越界）
void dispatch_pool_2d(VulkanGraphContext& gc, Tensor* node) {
    const Tensor* a = node->src[0];

    VulkanPool2DPC pc{};
    pc.mode = node->op_params[0];
    pc.k0   = node->op_params[1];
    pc.k1   = node->op_params[2];
    pc.s0   = node->op_params[3];
    pc.s1   = node->op_params[4];
    pc.p0   = node->op_params[5];
    pc.p1   = node->op_params[6];
    pc.OW   = (uint32_t) node->ne[0];
    pc.OH   = (uint32_t) node->ne[1];
    pc.C    = (uint32_t) node->ne[2];
    pc.N    = (uint32_t) node->ne[3];
    pc.W    = (uint32_t) a->ne[0];
    pc.H    = (uint32_t) a->ne[1];
    fill_nb(pc.nb_a, a);
    fill_nb(pc.nb_d, node);

    const VulkanTensorBinding binds[2] = { vulkan_bind_tensor(a), vulkan_bind_tensor(node) };
    pc.base_a = binds[0].base_elem;
    pc.base_d = binds[1].base_elem;
    pc.numel  = mul_u32_checked(node->ne[0], node->ne[1], node->ne[2] * node->ne[3],
                                "pool_2d numel");

    vulkan_dispatch(gc, vulkan_pipeline("pool_2d", sizeof(VulkanPool2DPC)), &pc, sizeof(pc), binds,
                    2, limit_groups(group_count(pc.numel)));
}

// conv_transpose_2d：kernel=[KW,KH,Cout,Cin]、input=[W,H,Cin,N] -> dst=[OW,OH,Cout,N]
void dispatch_conv_transpose_2d(VulkanGraphContext& gc, Tensor* node) {
    const Tensor* kernel = node->src[0];
    const Tensor* input  = node->src[1];

    VulkanConvTranspose2DPC pc{};
    pc.KW    = (uint32_t) kernel->ne[0];
    pc.KH    = (uint32_t) kernel->ne[1];
    pc.Cout  = (uint32_t) kernel->ne[2];
    pc.Cin   = (uint32_t) kernel->ne[3];
    pc.W     = (uint32_t) input->ne[0];
    pc.H     = (uint32_t) input->ne[1];
    pc.N     = (uint32_t) input->ne[3];
    pc.OW    = (uint32_t) node->ne[0];
    pc.OH    = (uint32_t) node->ne[1];
    pc.stride = node->op_params[0];
    fill_nb(pc.nb_w, kernel);
    fill_nb(pc.nb_x, input);
    fill_nb(pc.nb_d, node);

    const VulkanTensorBinding binds[3] = { vulkan_bind_tensor(kernel), vulkan_bind_tensor(input),
                                           vulkan_bind_tensor(node) };
    pc.base_w = binds[0].base_elem;
    pc.base_x = binds[1].base_elem;
    pc.base_d = binds[2].base_elem;
    pc.numel  = mul_u32_checked(node->ne[0], node->ne[1], node->ne[2] * node->ne[3],
                                "conv_transpose_2d numel");

    vulkan_dispatch(gc, vulkan_pipeline("conv_transpose_2d", sizeof(VulkanConvTranspose2DPC)), &pc,
                    sizeof(pc), binds, 3, limit_groups(group_count(pc.numel)));
}

// ---------------------------------------------------------------- 拷贝类

// 把 src 拷到 dst（范围取 src 形状；base_d_extra 为相对 dst 起点的元素偏移，供 concat 用）
void dispatch_copy(VulkanGraphContext& gc, const Tensor* src, const Tensor* dst,
                   uint32_t base_d_extra = 0) {
    VulkanCopyPC pc{};
    fill_ne(pc.ne, src);
    fill_nb(pc.nb_a, src);
    fill_nb(pc.nb_d, dst);

    const VulkanTensorBinding binds[2] = { vulkan_bind_tensor(src), vulkan_bind_tensor(dst) };
    pc.base_a = binds[0].base_elem;
    pc.base_d = binds[1].base_elem + base_d_extra;
    pc.numel  = count_u32(src);

    vulkan_dispatch(gc, vulkan_pipeline("copy", sizeof(VulkanCopyPC)), &pc, sizeof(pc), binds, 2,
                    limit_groups(group_count(pc.numel)));
}

void dispatch_concat(VulkanGraphContext& gc, Tensor* node) {
    const Tensor* a   = node->src[0];
    const Tensor* b   = node->src[1];
    const int     dim = (int) node->op_params[0];
    TRC_ASSERT(dim >= 0 && dim < 4, "Vulkan concat: dim 非法 %d", dim);
    dispatch_copy(gc, a, node);
    const uint32_t extra = mul_u32_checked(a->ne[dim], node->nb[dim] / sizeof(float), "concat offset");
    dispatch_copy(gc, b, node, extra);
}

// pad：用输出元素数独立计算（不依赖"分配时恰好为 0"）
//   - mode=ZERO：输出整块先显式清零，再把源写进内部区域（含左填充偏移；R16）
//   - mode=REFLECT（M2.3c）：pad_reflect.comp 每个输出元素一次 gather（全量写出，无需清零）
void dispatch_pad(VulkanGraphContext& gc, Tensor* node) {
    const Tensor*  a    = node->src[0];
    const int32_t* p    = node->op_params;
    const size_t   elem = type_size(node->type);
    TRC_ASSERT(elem == sizeof(float), "Vulkan pad 仅支持 F32（当前 %s）", type_name(node->type));

    if (p[8] == (int32_t) PAD_REFLECT) {
        VulkanPadReflectPC pc{};
        fill_ne(pc.ne_d, node);
        fill_ne(pc.ne_a, a);
        fill_nb(pc.nb_a, a);
        for (int d = 0; d < 4; ++d) {
            pc.lp[d] = (uint32_t) p[2 * d];
        }
        const VulkanTensorBinding binds[2] = { vulkan_bind_tensor(a), vulkan_bind_tensor(node) };
        pc.base_a = binds[0].base_elem;
        pc.base_d = binds[1].base_elem;
        pc.numel  = count_u32(node);

        vulkan_dispatch(gc, vulkan_pipeline("pad_reflect", sizeof(VulkanPadReflectPC)), &pc, sizeof(pc),
                        binds, 2, limit_groups(group_count(pc.numel)));
        return;
    }

    const VulkanTensorBinding dst = vulkan_bind_tensor(node);
    const VkDeviceSize        off = dst.info.offset + (VkDeviceSize) dst.base_elem * elem;
    const VkDeviceSize        len = dst.info.range - (VkDeviceSize) dst.base_elem * elem;
    TRC_ASSERT(len % 4 == 0, "Vulkan pad: 填充长度必须 4 字节对齐（%llu）",
               (unsigned long long) len);

    vkCmdFillBuffer(gc.cmd, dst.info.buffer, off, len, 0u);

    // transfer 写 -> 后续 compute 可见（dispatch_copy 之后的保守屏障会再次覆盖）
    VkMemoryBarrier mb{};
    mb.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    mb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    vkCmdPipelineBarrier(gc.cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         0, 1, &mb, 0, nullptr, 0, nullptr);

    // 左填充偏移（元素单位；pad 输出为新建连续张量，nb 即元素步长）
    uint32_t extra = 0;
    for (int d = 0; d < 4; ++d) {
        extra += (uint32_t) p[2 * d] * (uint32_t) (node->nb[d] / elem);
    }
    dispatch_copy(gc, a, node, extra);
}

// pad reflect 反向（M2.3c）：dX[j] = Σ_{i: reflect(i-lp)=j} dY[i]（每维至多 3 路，无原子）
void dispatch_pad_back(VulkanGraphContext& gc, Tensor* node) {
    const Tensor*  g = node->src[0];
    const Tensor*  x = node->src[1];
    const int32_t* p = node->op_params;

    VulkanPadBackPC pc{};
    fill_ne(pc.ne_x, x);
    fill_ne(pc.ne_g, g);
    fill_nb(pc.nb_g, g);
    for (int d = 0; d < 4; ++d) {
        pc.lp[d] = (uint32_t) p[2 * d];
    }
    const VulkanTensorBinding binds[2] = { vulkan_bind_tensor(g), vulkan_bind_tensor(node) };
    pc.base_g = binds[0].base_elem;
    pc.base_x = binds[1].base_elem;
    pc.numel  = count_u32(node);

    vulkan_dispatch(gc, vulkan_pipeline("pad_back", sizeof(VulkanPadBackPC)), &pc, sizeof(pc), binds,
                    2, limit_groups(group_count(pc.numel)));
}

// ---------------------------------------------------------------- 归约

void dispatch_reduce_rows(VulkanGraphContext& gc, Tensor* node, uint32_t mode) {
    const Tensor* a = node->src[0];
    TRC_ASSERT(a->ne[3] * a->ne[2] * a->ne[1] > 0, "Vulkan 归约：行数为 0");

    VulkanReduceRowsPC pc{};
    pc.mode   = mode;  // 0=sum_rows, 1=mean
    pc.ne0    = (uint32_t) a->ne[0];
    pc.ne1    = (uint32_t) a->ne[1];
    pc.ne2    = (uint32_t) a->ne[2];
    pc.n_rows = mul_u32_checked(a->ne[1], a->ne[2], a->ne[3], "reduce_rows n_rows");
    fill_nb(pc.nb_a, a);

    const VulkanTensorBinding binds[2] = { vulkan_bind_tensor(a), vulkan_bind_tensor(node) };
    pc.base_a = binds[0].base_elem;
    pc.base_d = binds[1].base_elem;
    pc.inv_ne0 = 1.0f / (float) a->ne[0];

    vulkan_dispatch(gc, vulkan_pipeline("reduce_rows", sizeof(VulkanReduceRowsPC)), &pc, sizeof(pc),
                    binds, 2, limit_groups(pc.n_rows));
}

void dispatch_sum_all(VulkanGraphContext& gc, Tensor* node) {
    const Tensor* a = node->src[0];

    VulkanSumAllPC pc{};
    fill_ne(pc.ne, a);
    fill_nb(pc.nb_a, a);

    const VulkanTensorBinding binds[2] = { vulkan_bind_tensor(a), vulkan_bind_tensor(node) };
    pc.base_a = binds[0].base_elem;
    pc.base_d = binds[1].base_elem;
    pc.numel  = count_u32(a);

    vulkan_dispatch(gc, vulkan_pipeline("sum_all", sizeof(VulkanSumAllPC)), &pc, sizeof(pc), binds, 2,
                    1);
}

// ---------------------------------------------------------------- 归一化

void dispatch_norm(VulkanGraphContext& gc, Tensor* node, uint32_t mode) {
    const Tensor* a = node->src[0];

    VulkanNormPC pc{};
    pc.mode   = mode;  // 0=norm, 1=rms_norm
    pc.ne0    = (uint32_t) a->ne[0];
    pc.ne1    = (uint32_t) a->ne[1];
    pc.ne2    = (uint32_t) a->ne[2];
    pc.n_rows = mul_u32_checked(a->ne[1], a->ne[2], a->ne[3], "norm n_rows");
    fill_nb(pc.nb_a, a);
    fill_nb(pc.nb_d, node);
    std::memcpy(&pc.eps, node->op_params, sizeof(float));

    const VulkanTensorBinding binds[2] = { vulkan_bind_tensor(a), vulkan_bind_tensor(node) };
    pc.base_a = binds[0].base_elem;
    pc.base_d = binds[1].base_elem;

    vulkan_dispatch(gc, vulkan_pipeline("norm", sizeof(VulkanNormPC)), &pc, sizeof(pc), binds, 2,
                    limit_groups(pc.n_rows));
}

void dispatch_group_norm(VulkanGraphContext& gc, Tensor* node) {
    const Tensor* a = node->src[0];

    VulkanGroupNormPC pc{};
    pc.ne0       = (uint32_t) a->ne[0];
    pc.ne1       = (uint32_t) a->ne[1];
    pc.ne2       = (uint32_t) a->ne[2];
    pc.ne3       = (uint32_t) a->ne[3];
    pc.n_groups  = (uint32_t) node->op_params[0];
    TRC_ASSERT(pc.n_groups > 0, "Vulkan group_norm: n_groups 非法");
    pc.per_group = (pc.ne2 + pc.n_groups - 1) / pc.n_groups;
    fill_nb(pc.nb_a, a);
    fill_nb(pc.nb_d, node);
    std::memcpy(&pc.eps, (const uint8_t*) node->op_params + sizeof(float), sizeof(float));

    const VulkanTensorBinding binds[2] = { vulkan_bind_tensor(a), vulkan_bind_tensor(node) };
    pc.base_a = binds[0].base_elem;
    pc.base_d = binds[1].base_elem;

    const uint32_t total = mul_u32_checked(pc.n_groups, pc.ne3, "group_norm total");
    vulkan_dispatch(gc, vulkan_pipeline("group_norm", sizeof(VulkanGroupNormPC)), &pc, sizeof(pc),
                    binds, 2, limit_groups(total));
}

void dispatch_soft_max(VulkanGraphContext& gc, Tensor* node) {
    const Tensor* a    = node->src[0];
    const Tensor* mask = node->src[1];

    VulkanSoftMaxPC pc{};
    pc.ne0 = (uint32_t) a->ne[0];
    pc.ne1 = (uint32_t) a->ne[1];
    pc.ne2 = (uint32_t) a->ne[2];
    pc.ne3 = (uint32_t) a->ne[3];
    fill_nb(pc.nb_a, a);
    fill_nb(pc.nb_d, node);

    float scale    = 1.0f;
    float max_bias = 0.0f;
    std::memcpy(&scale, node->op_params, sizeof(float));
    std::memcpy(&max_bias, (const uint8_t*) node->op_params + sizeof(float), sizeof(float));
    pc.scale = scale;

    const VulkanTensorBinding bind_a = vulkan_bind_tensor(a);
    const VulkanTensorBinding bind_d = vulkan_bind_tensor(node);
    pc.base_a = bind_a.base_elem;
    pc.base_d = bind_d.base_elem;

    VulkanTensorBinding binds[3] = { bind_a, bind_d, {} };
    uint32_t           n_binds   = 2;
    if (mask != nullptr) {
        fill_nb(pc.nb_m, mask);
        binds[2]      = vulkan_bind_tensor(mask);
        pc.base_m     = binds[2].base_elem;
        pc.has_mask   = 1;
        pc.ne12       = (uint32_t) mask->ne[2];
        pc.ne13       = (uint32_t) mask->ne[3];
        n_binds       = 3;
    } else {
        pc.has_mask = 0;
        pc.ne12     = 1;
        pc.ne13     = 1;
    }

    // ALiBi（与 CPU/ggml 一致）
    if (max_bias > 0.0f) {
        const uint32_t n_head = (uint32_t) a->ne[2];
        uint32_t       n_head_log2 = 1;
        while (n_head_log2 * 2u <= n_head) {
            n_head_log2 *= 2u;
        }
        pc.use_alibi    = 1;
        pc.n_head_log2  = n_head_log2;
        pc.m0           = std::pow(2.0f, -max_bias / (float) n_head_log2);
        pc.m1           = std::pow(2.0f, -(max_bias / 2.0f) / (float) n_head_log2);
    } else {
        pc.use_alibi   = 0;
        pc.n_head_log2 = 1;
        pc.m0          = 1.0f;
        pc.m1          = 1.0f;
    }

    const uint32_t n_rows = mul_u32_checked(pc.ne1, pc.ne2, pc.ne3, "soft_max n_rows");
    vulkan_dispatch(gc, vulkan_pipeline("softmax", sizeof(VulkanSoftMaxPC)), &pc, sizeof(pc), binds,
                    n_binds, limit_groups(n_rows));
}

// ---------------------------------------------------------------- 损失

void dispatch_cross_entropy_loss(VulkanGraphContext& gc, Tensor* node) {
    const Tensor* logits = node->src[0];
    const Tensor* target = node->src[1];

    VulkanCrossEntropyPC pc{};
    pc.nc  = (uint32_t) logits->ne[0];
    pc.nr  = mul_u32_checked(logits->ne[1], logits->ne[2], logits->ne[3], "cross_entropy nr");
    pc.ne1 = (uint32_t) logits->ne[1];
    pc.ne2 = (uint32_t) logits->ne[2];
    TRC_ASSERT(pc.nc > 0 && pc.nr > 0, "Vulkan cross_entropy_loss: 输入不能为空");
    fill_nb(pc.nb_a, logits);
    fill_nb(pc.nb_b, target);

    const VulkanTensorBinding binds[3] = { vulkan_bind_tensor(logits), vulkan_bind_tensor(target),
                                           vulkan_bind_tensor(node) };
    pc.base_a = binds[0].base_elem;
    pc.base_b = binds[1].base_elem;
    pc.base_d = binds[2].base_elem;
    pc.inv_nr = 1.0f / (float) pc.nr;

    vulkan_dispatch(gc, vulkan_pipeline("cross_entropy_loss", sizeof(VulkanCrossEntropyPC)), &pc,
                    sizeof(pc), binds, 3, 1);
}

void dispatch_cross_entropy_loss_back(VulkanGraphContext& gc, Tensor* node) {
    const Tensor* grad   = node->src[0];
    const Tensor* logits = node->src[1];
    const Tensor* target = node->src[2];

    VulkanCrossEntropyBackPC pc{};
    pc.nc  = (uint32_t) logits->ne[0];
    pc.nr  = mul_u32_checked(logits->ne[1], logits->ne[2], logits->ne[3], "cross_entropy nr");
    pc.ne1 = (uint32_t) logits->ne[1];
    pc.ne2 = (uint32_t) logits->ne[2];
    TRC_ASSERT(pc.nc > 0 && pc.nr > 0, "Vulkan cross_entropy_loss_back: 输入不能为空");
    fill_nb(pc.nb_a, logits);
    fill_nb(pc.nb_b, target);
    fill_nb(pc.nb_d, node);
    pc.inv_nr = 1.0f / (float) pc.nr;

    // shader 绑定：0=A logits、1=B target、2=D 输出、3=G grad 标量
    const VulkanTensorBinding binds[4] = { vulkan_bind_tensor(logits), vulkan_bind_tensor(target),
                                           vulkan_bind_tensor(node), vulkan_bind_tensor(grad) };
    pc.base_a = binds[0].base_elem;
    pc.base_b = binds[1].base_elem;
    pc.base_d = binds[2].base_elem;
    pc.base_g = binds[3].base_elem;

    vulkan_dispatch(gc, vulkan_pipeline("cross_entropy_loss_back", sizeof(VulkanCrossEntropyBackPC)),
                    &pc, sizeof(pc), binds, 4, limit_groups(pc.nr));
}

// ---------------------------------------------------------------- 归一化反向

void dispatch_norm_back(VulkanGraphContext& gc, Tensor* node, uint32_t mode) {
    const Tensor* dy = node->src[0];
    const Tensor* x  = node->src[1];

    VulkanNormBackPC pc{};
    pc.mode   = mode;  // 0=norm_back, 1=rms_norm_back
    pc.ne0    = (uint32_t) x->ne[0];
    pc.ne1    = (uint32_t) x->ne[1];
    pc.ne2    = (uint32_t) x->ne[2];
    pc.n_rows = mul_u32_checked(x->ne[1], x->ne[2], x->ne[3], "norm_back n_rows");
    fill_nb(pc.nb_dy, dy);
    fill_nb(pc.nb_x, x);
    fill_nb(pc.nb_d, node);
    std::memcpy(&pc.eps, node->op_params, sizeof(float));

    const VulkanTensorBinding binds[3] = { vulkan_bind_tensor(dy), vulkan_bind_tensor(x),
                                           vulkan_bind_tensor(node) };
    pc.base_dy = binds[0].base_elem;
    pc.base_x  = binds[1].base_elem;
    pc.base_d  = binds[2].base_elem;

    vulkan_dispatch(gc, vulkan_pipeline("norm_back", sizeof(VulkanNormBackPC)), &pc, sizeof(pc), binds,
                    3, limit_groups(pc.n_rows));
}

void dispatch_group_norm_back(VulkanGraphContext& gc, Tensor* node) {
    const Tensor* dy = node->src[0];
    const Tensor* x  = node->src[1];

    VulkanGroupNormBackPC pc{};
    pc.ne0      = (uint32_t) x->ne[0];
    pc.ne1      = (uint32_t) x->ne[1];
    pc.ne2      = (uint32_t) x->ne[2];
    pc.ne3      = (uint32_t) x->ne[3];
    pc.n_groups = (uint32_t) node->op_params[0];
    TRC_ASSERT(pc.n_groups > 0, "Vulkan group_norm_back: n_groups 非法");
    pc.per_group = (pc.ne2 + pc.n_groups - 1) / pc.n_groups;
    fill_nb(pc.nb_dy, dy);
    fill_nb(pc.nb_x, x);
    fill_nb(pc.nb_d, node);
    std::memcpy(&pc.eps, (const uint8_t*) node->op_params + sizeof(float), sizeof(float));

    const VulkanTensorBinding binds[3] = { vulkan_bind_tensor(dy), vulkan_bind_tensor(x),
                                           vulkan_bind_tensor(node) };
    pc.base_dy = binds[0].base_elem;
    pc.base_x  = binds[1].base_elem;
    pc.base_d  = binds[2].base_elem;

    const uint32_t total = mul_u32_checked(pc.n_groups, pc.ne3, "group_norm total");
    vulkan_dispatch(gc, vulkan_pipeline("group_norm_back", sizeof(VulkanGroupNormBackPC)), &pc,
                    sizeof(pc), binds, 3, limit_groups(total));
}

// ---------------------------------------------------------------- 形状扩展

void dispatch_broadcast_copy(VulkanGraphContext& gc, Tensor* node) {
    const Tensor* a = node->src[0];

    VulkanBroadcastPC pc{};
    fill_ne(pc.ne_s, a);
    fill_ne(pc.ne_d, node);
    fill_nb(pc.nb_s, a);
    fill_nb(pc.nb_d, node);

    const VulkanTensorBinding binds[2] = { vulkan_bind_tensor(a), vulkan_bind_tensor(node) };
    pc.base_s = binds[0].base_elem;
    pc.base_d = binds[1].base_elem;
    pc.numel  = count_u32(node);

    vulkan_dispatch(gc, vulkan_pipeline("broadcast_copy", sizeof(VulkanBroadcastPC)), &pc, sizeof(pc),
                    binds, 2, limit_groups(group_count(pc.numel)));
}

void dispatch_repeat_back(VulkanGraphContext& gc, Tensor* node) {
    const Tensor* a = node->src[0];  // 大张量（可重复到它的形状是 node 的形状）

    VulkanRepeatBackPC pc{};
    fill_ne(pc.ne_d, node);
    fill_nb(pc.nb_a, a);
    fill_nb(pc.nb_d, node);
    for (int d = 0; d < 4; ++d) {
        TRC_ASSERT(node->ne[d] > 0 && a->ne[d] % node->ne[d] == 0,
                   "Vulkan repeat_back: 维度不可整除（ne[%d]=%lld vs %lld）", d, (long long) a->ne[d],
                   (long long) node->ne[d]);
        pc.reps[d] = (uint32_t) (a->ne[d] / node->ne[d]);
    }

    const VulkanTensorBinding binds[2] = { vulkan_bind_tensor(a), vulkan_bind_tensor(node) };
    pc.base_a  = binds[0].base_elem;
    pc.base_d  = binds[1].base_elem;
    pc.numel_d = count_u32(node);

    vulkan_dispatch(gc, vulkan_pipeline("repeat_back", sizeof(VulkanRepeatBackPC)), &pc, sizeof(pc),
                    binds, 2, limit_groups(group_count(pc.numel_d)));
}

void dispatch_acc(VulkanGraphContext& gc, Tensor* node) {
    const Tensor* a = node->src[0];
    const Tensor* b = node->src[1];

    int32_t nb1i = 0;
    int32_t nb2i = 0;
    int32_t nb3i = 0;
    int32_t offseti = 0;
    int32_t inplace_i = 0;
    std::memcpy(&nb1i, (const uint8_t*) node->op_params + 0, sizeof(int32_t));
    std::memcpy(&nb2i, (const uint8_t*) node->op_params + 4, sizeof(int32_t));
    std::memcpy(&nb3i, (const uint8_t*) node->op_params + 8, sizeof(int32_t));
    std::memcpy(&offseti, (const uint8_t*) node->op_params + 12, sizeof(int32_t));
    std::memcpy(&inplace_i, (const uint8_t*) node->op_params + 16, sizeof(int32_t));
    const size_t nb1 = (size_t) nb1i;
    const size_t nb2 = (size_t) nb2i;
    const size_t nb3 = (size_t) nb3i;
    const size_t offset = (size_t) offseti;
    const bool inplace = inplace_i != 0;

    if (!inplace) {
        dispatch_copy(gc, a, node);  // 先整体拷贝 a
    }

    TRC_ASSERT(nb1 % sizeof(float) == 0 && nb2 % sizeof(float) == 0 && nb3 % sizeof(float) == 0 &&
                   offset % sizeof(float) == 0,
               "Vulkan acc: 步长/偏移必须 4 字节对齐");

    VulkanAccAddPC pc{};
    fill_ne(pc.ne, b);
    fill_nb(pc.nb_b, b);
    pc.nb_d[0] = 1;
    pc.nb_d[1] = (uint32_t) (nb1 / sizeof(float));
    pc.nb_d[2] = (uint32_t) (nb2 / sizeof(float));
    pc.nb_d[3] = (uint32_t) (nb3 / sizeof(float));

    const VulkanTensorBinding binds[2] = { vulkan_bind_tensor(b), vulkan_bind_tensor(node) };
    pc.base_b = binds[0].base_elem;
    // inplace 时 node 是 a 的完整视图（base_elem = a 的基址），区域偏移统一按 op_params 加
    pc.base_d = binds[1].base_elem + (uint32_t) (offset / sizeof(float));
    pc.numel  = count_u32(b);

    vulkan_dispatch(gc, vulkan_pipeline("acc_add", sizeof(VulkanAccAddPC)), &pc, sizeof(pc), binds, 2,
                    limit_groups(group_count(pc.numel)));
}

// ---------------------------------------------------------------- 索引

bool is_i32(const Tensor* t) { return t != nullptr && t->type == TYPE_I32; }

void fill_nb_i32(uint32_t dst[3], const Tensor* t) {
    for (int d = 0; d < 3; ++d) {
        TRC_ASSERT(t->nb[d] % sizeof(int32_t) == 0, "Vulkan 索引步长必须 4 字节对齐（%s）", t->name);
        const size_t n = t->nb[d] / sizeof(int32_t);
        TRC_ASSERT(n <= (size_t) UINT32_MAX, "Vulkan 后端：索引步长越界（%s, nb[%d]=%zu）", t->name, d,
                   t->nb[d]);  // R8
        dst[d] = (uint32_t) n;
    }
}

void dispatch_get_rows(VulkanGraphContext& gc, Tensor* node) {
    const Tensor* table = node->src[0];
    const Tensor* idx   = node->src[1];

    VulkanGetRowsPC pc{};
    pc.ne0     = (uint32_t) table->ne[0];
    pc.n_vocab = (uint32_t) table->ne[1];
    pc.nr      = (uint32_t) idx->ne[0];
    pc.ni1     = (uint32_t) idx->ne[1];
    pc.ni2     = (uint32_t) idx->ne[2];
    fill_nb(pc.nb_t, table);
    fill_nb(pc.nb_o, node);
    fill_nb_i32(pc.nb_i, idx);

    const VulkanTensorBinding binds[4] = { vulkan_bind_tensor(table), vulkan_bind_tensor(idx),
                                           vulkan_bind_tensor(node), vulkan_bind_error_buffer() };
    pc.base_t = binds[0].base_elem;
    pc.base_i = binds[1].base_elem;
    pc.base_o = binds[2].base_elem;
    pc.numel  = count_u32(node);

    vulkan_dispatch(gc, vulkan_pipeline("get_rows", sizeof(VulkanGetRowsPC)), &pc, sizeof(pc), binds,
                    4, limit_groups(group_count(pc.numel)));
}

void dispatch_set_rows(VulkanGraphContext& gc, Tensor* node) {
    const Tensor* values  = node->src[0];
    const Tensor* indices = node->src[1];
    const Tensor* target  = node->src[2];  // 与 node 共享内存
    // R6：显式类型断言（支持范围与 vulkan_supports_op 一致）
    TRC_ASSERT(values->type == TYPE_F32, "Vulkan set_rows 值仅支持 F32（当前 %s）",
               type_name(values->type));
    TRC_ASSERT(indices->type == TYPE_I32, "Vulkan set_rows 索引仅支持 I32（当前 %s）",
               type_name(indices->type));

    VulkanSetRowsPC pc{};
    pc.ne0     = (uint32_t) values->ne[0];
    pc.nr      = (uint32_t) values->ne[1];
    pc.ni2     = (uint32_t) values->ne[2];
    pc.ni3     = (uint32_t) values->ne[3];
    pc.idx_ne1 = (uint32_t) indices->ne[1];
    pc.idx_ne2 = (uint32_t) indices->ne[2];
    pc.n_vocab = (uint32_t) target->ne[1];
    fill_nb(pc.nb_v, values);
    fill_nb(pc.nb_t, target);
    fill_nb_i32(pc.nb_i, indices);

    const VulkanTensorBinding binds[4] = { vulkan_bind_tensor(values), vulkan_bind_tensor(indices),
                                           vulkan_bind_tensor(target), vulkan_bind_error_buffer() };
    pc.base_v = binds[0].base_elem;
    pc.base_i = binds[1].base_elem;
    pc.base_t = binds[2].base_elem;
    pc.numel  = count_u32(values);

    vulkan_dispatch(gc, vulkan_pipeline("set_rows", sizeof(VulkanSetRowsPC)), &pc, sizeof(pc), binds,
                    4, limit_groups(group_count(pc.numel)));
}

void dispatch_get_rows_back(VulkanGraphContext& gc, Tensor* node) {
    const Tensor* grad = node->src[0];
    const Tensor* idx  = node->src[1];

    VulkanGetRowsBackPC pc{};
    pc.ne0     = (uint32_t) node->ne[0];
    pc.n_vocab = (uint32_t) node->ne[1];
    pc.nr      = (uint32_t) idx->ne[0];
    pc.ni1     = (uint32_t) idx->ne[1];
    pc.ni2     = (uint32_t) idx->ne[2];
    fill_nb(pc.nb_g, grad);
    fill_nb(pc.nb_d, node);
    fill_nb_i32(pc.nb_i, idx);

    const VulkanTensorBinding binds[4] = { vulkan_bind_tensor(grad), vulkan_bind_tensor(node),
                                           vulkan_bind_tensor(idx), vulkan_bind_error_buffer() };
    pc.base_g = binds[0].base_elem;
    pc.base_d = binds[1].base_elem;
    pc.base_i = binds[2].base_elem;

    const uint32_t total = mul_u32_checked(pc.ni1, pc.ni2, pc.n_vocab, "get_rows_back total");
    vulkan_dispatch(gc, vulkan_pipeline("get_rows_back", sizeof(VulkanGetRowsBackPC)), &pc, sizeof(pc),
                    binds, 4, limit_groups(total));
}

} // namespace

bool vulkan_supports_op(const Tensor* t) {
    switch (t->op) {
        // 叶子与视图不产生 GPU 计算：数据由缓冲/源张量提供
        case OP_NONE:
        case OP_RESHAPE:
        case OP_VIEW:
        case OP_PERMUTE:
        case OP_TRANSPOSE:
            return true;

        // 逐元素（F32）
        case OP_ADD:
        case OP_ADD1:
        case OP_SUB:
        case OP_MUL:
        case OP_DIV:
            return srcs_f32(t, 2);
        case OP_NEG:
        case OP_ABS:
        case OP_SQR:
        case OP_SQRT:
        case OP_EXP:
        case OP_LOG:
        case OP_SIN:
        case OP_COS:
        case OP_RELU:
        case OP_LEAKY_RELU:
        case OP_GELU:
        case OP_GELU_ERF:
        case OP_ERF:
        case OP_SILU:
        case OP_SIGMOID:
        case OP_TANH:
        case OP_SOFTPLUS:
        case OP_HARDSWISH:
        case OP_SCALE:
        case OP_CLAMP:
        case OP_SGN:
        case OP_STEP:
            return srcs_f32(t, 1);

        // 线代（F32；形状条件与 ggml_can_mul_mat 一致）
        case OP_MUL_MAT: {
            if (!srcs_f32(t, 2)) {
                return false;
            }
            const Tensor* a = t->src[0];
            const Tensor* b = t->src[1];
            return a->ne[0] == b->ne[0] && b->ne[2] % a->ne[2] == 0 && b->ne[3] % a->ne[3] == 0;
        }

        // 卷积（F32；conv_1d/conv_2d 为核心组合图）
        case OP_IM2COL:
        case OP_IM2COL_BACK:
        case OP_CONV_TRANSPOSE_1D:
        case OP_CONV_TRANSPOSE_2D:
            return srcs_f32(t, 2);
        case OP_COL2IM:
        case OP_POOL_2D:
            return srcs_f32(t, 1);
        // 注意：pool_2d_back（scatter-add 有跨窗口写竞争）无 Vulkan 实现——不列入即返回 false，
        // 训练图含池化反向时 graph_first_unsupported 会明确报告（GPU 上池化训练回退 CPU）

        // 类型转换（仅 F32↔F16；需要设备支持 16 位存储）
        case OP_CAST:
            if (!vulkan_state().storage_16bit) {
                return false;
            }
            return (t->type == TYPE_F16 && t->src[0]->type == TYPE_F32) ||
                   (t->type == TYPE_F32 && t->src[0]->type == TYPE_F16);

        // 拷贝类 / 形状
        case OP_DUP:
        case OP_CPY:
        case OP_CONT:
        case OP_PAD:
            return srcs_f32(t, 1);
        case OP_PAD_BACK:
            return srcs_f32(t, 2);
        case OP_CONCAT:
        case OP_ACC:
            return srcs_f32(t, 2);
        case OP_REPEAT:
        case OP_REPEAT_BACK:
            return srcs_f32(t, 1);

        // 索引（表/值 F32，索引 I32；一期）
        case OP_GET_ROWS:
        case OP_GET_ROWS_BACK:
        case OP_SET_ROWS:
            return is_f32(t) && is_f32(t->src[0]) && is_i32(t->src[1]);

        // 归约 / 归一化
        case OP_SUM:
        case OP_SUM_ROWS:
        case OP_MEAN:
        case OP_NORM:
        case OP_RMS_NORM:
        case OP_GROUP_NORM:
            return srcs_f32(t, 1);

        // 归一化反向（dy, x 均为 F32）
        case OP_NORM_BACK:
        case OP_RMS_NORM_BACK:
        case OP_GROUP_NORM_BACK:
            return srcs_f32(t, 2);

        // softmax（mask 可选，F32）
        case OP_SOFT_MAX: {
            if (!srcs_f32(t, 1)) {
                return false;
            }
            const Tensor* mask = t->src[1];
            return mask == nullptr || mask->type == TYPE_F32;
        }

        // 损失（F32、同形状、最内维连续；与 core 构造约束一致）
        case OP_CROSS_ENTROPY_LOSS: {
            if (!srcs_f32(t, 2)) {
                return false;
            }
            const Tensor* a = t->src[0];
            const Tensor* b = t->src[1];
            return tensor_are_same_shape(a, b) && a->nb[0] == sizeof(float) &&
                   b->nb[0] == sizeof(float);
        }
        case OP_CROSS_ENTROPY_LOSS_BACK: {
            if (!srcs_f32(t, 3)) {
                return false;
            }
            const Tensor* g = t->src[0];
            const Tensor* a = t->src[1];
            const Tensor* b = t->src[2];
            return tensor_nelements(g) == 1 && tensor_are_same_shape(a, b) &&
                   a->nb[0] == sizeof(float) && b->nb[0] == sizeof(float);
        }

        default:
            // 其余算子随 M1.4c/M1.4d 逐个补齐；未支持时由调度器回退
            return false;
    }
}

bool vulkan_compute_node(VulkanGraphContext& gc, Tensor* node) {
    // R6：分派前如实校验（与 vulkan_supports_op 同源）；不支持的类型/布局必须响亮中止，
    // 不得静默走错 shader 路径。调用方也可用 graph_first_unsupported 预检整图。
    if (!vulkan_supports_op(node)) {
        TRC_ABORT("Vulkan 后端不支持算子 %s（op=%d，类型/布局不满足支持声明）", op_name(node->op),
                  (int) node->op);
    }

    switch (node->op) {
        case OP_NONE:
        case OP_RESHAPE:
        case OP_VIEW:
        case OP_PERMUTE:
        case OP_TRANSPOSE:
            return true;

        case OP_ADD:  dispatch_binary(gc, node, VulkanBinaryOp::ADD); return true;
        case OP_ADD1: dispatch_binary(gc, node, VulkanBinaryOp::ADD); return true;
        case OP_SUB:  dispatch_binary(gc, node, VulkanBinaryOp::SUB); return true;
        case OP_MUL:  dispatch_binary(gc, node, VulkanBinaryOp::MUL); return true;
        case OP_DIV:  dispatch_binary(gc, node, VulkanBinaryOp::DIV); return true;

        case OP_NEG:       dispatch_unary(gc, node, VulkanUnaryOp::NEG); return true;
        case OP_ABS:       dispatch_unary(gc, node, VulkanUnaryOp::ABS); return true;
        case OP_SQR:       dispatch_unary(gc, node, VulkanUnaryOp::SQR); return true;
        case OP_SQRT:      dispatch_unary(gc, node, VulkanUnaryOp::SQRT); return true;
        case OP_EXP:       dispatch_unary(gc, node, VulkanUnaryOp::EXP); return true;
        case OP_LOG:       dispatch_unary(gc, node, VulkanUnaryOp::LOG); return true;
        case OP_SIN:       dispatch_unary(gc, node, VulkanUnaryOp::SIN); return true;
        case OP_COS:       dispatch_unary(gc, node, VulkanUnaryOp::COS); return true;
        case OP_RELU:      dispatch_unary(gc, node, VulkanUnaryOp::RELU); return true;
        case OP_SIGMOID:   dispatch_unary(gc, node, VulkanUnaryOp::SIGMOID); return true;
        case OP_TANH:      dispatch_unary(gc, node, VulkanUnaryOp::TANH); return true;
        case OP_SILU:      dispatch_unary(gc, node, VulkanUnaryOp::SILU); return true;
        case OP_GELU:      dispatch_unary(gc, node, VulkanUnaryOp::GELU); return true;
        case OP_GELU_ERF:  dispatch_unary(gc, node, VulkanUnaryOp::GELU_ERF); return true;
        case OP_ERF:       dispatch_unary(gc, node, VulkanUnaryOp::ERF); return true;
        case OP_SOFTPLUS:  dispatch_unary(gc, node, VulkanUnaryOp::SOFTPLUS); return true;
        case OP_HARDSWISH: dispatch_unary(gc, node, VulkanUnaryOp::HARDSWISH); return true;
        case OP_SGN:       dispatch_unary(gc, node, VulkanUnaryOp::SGN); return true;
        case OP_STEP:      dispatch_unary(gc, node, VulkanUnaryOp::STEP); return true;

        case OP_LEAKY_RELU: {
            float slope = 0.0f;
            std::memcpy(&slope, node->op_params, sizeof(float));
            dispatch_unary(gc, node, VulkanUnaryOp::LEAKY_RELU, slope);
            return true;
        }
        case OP_CLAMP: {
            float min_val = 0.0f;
            float max_val = 0.0f;
            std::memcpy(&min_val, node->op_params, sizeof(float));
            std::memcpy(&max_val, (const uint8_t*) node->op_params + sizeof(float), sizeof(float));
            dispatch_unary(gc, node, VulkanUnaryOp::CLAMP, min_val, max_val);
            return true;
        }
        case OP_SCALE: {
            float s = 0.0f;
            std::memcpy(&s, node->op_params, sizeof(float));
            dispatch_unary(gc, node, VulkanUnaryOp::SCALE, s);
            return true;
        }

        // 线代
        case OP_MUL_MAT:
            dispatch_mul_mat(gc, node);
            return true;

        // 卷积
        case OP_IM2COL:            dispatch_im2col(gc, node); return true;
        case OP_IM2COL_BACK:       dispatch_im2col_back(gc, node); return true;
        case OP_COL2IM:            dispatch_col2im(gc, node); return true;
        case OP_CONV_TRANSPOSE_1D: dispatch_conv_transpose_1d(gc, node); return true;
        case OP_POOL_2D:           dispatch_pool_2d(gc, node); return true;
        case OP_CONV_TRANSPOSE_2D: dispatch_conv_transpose_2d(gc, node); return true;

        // 类型转换
        case OP_CAST: dispatch_cast(gc, node); return true;

        // 拷贝类
        case OP_DUP:
        case OP_CPY:
        case OP_CONT:
            dispatch_copy(gc, node->src[0], node);
            return true;
        case OP_PAD:
            dispatch_pad(gc, node);
            return true;
        case OP_PAD_BACK:
            dispatch_pad_back(gc, node);
            return true;
        case OP_CONCAT:
            dispatch_concat(gc, node);
            return true;
        case OP_REPEAT:
            dispatch_broadcast_copy(gc, node);
            return true;
        case OP_REPEAT_BACK:
            dispatch_repeat_back(gc, node);
            return true;
        case OP_ACC:
            dispatch_acc(gc, node);
            return true;

        // 索引
        case OP_GET_ROWS:      dispatch_get_rows(gc, node); return true;
        case OP_SET_ROWS:      dispatch_set_rows(gc, node); return true;
        case OP_GET_ROWS_BACK: dispatch_get_rows_back(gc, node); return true;

        // 归约
        case OP_SUM:      dispatch_sum_all(gc, node); return true;
        case OP_SUM_ROWS: dispatch_reduce_rows(gc, node, 0); return true;
        case OP_MEAN:     dispatch_reduce_rows(gc, node, 1); return true;

        // 归一化
        case OP_NORM:            dispatch_norm(gc, node, 0); return true;
        case OP_RMS_NORM:        dispatch_norm(gc, node, 1); return true;
        case OP_GROUP_NORM:      dispatch_group_norm(gc, node); return true;
        case OP_NORM_BACK:       dispatch_norm_back(gc, node, 0); return true;
        case OP_RMS_NORM_BACK:   dispatch_norm_back(gc, node, 1); return true;
        case OP_GROUP_NORM_BACK: dispatch_group_norm_back(gc, node); return true;

        // softmax
        case OP_SOFT_MAX: dispatch_soft_max(gc, node); return true;
        case OP_CROSS_ENTROPY_LOSS: dispatch_cross_entropy_loss(gc, node); return true;
        case OP_CROSS_ENTROPY_LOSS_BACK: dispatch_cross_entropy_loss_back(gc, node); return true;

        default:
            TRC_ABORT("Vulkan 后端尚未实现算子 %s（op=%d，M1.4c/d 起补齐）", op_name(node->op),
                      (int) node->op);
    }
}

} // namespace traincpp
