// train.cpp - Vulkan 后端内部头（仅 src/ 内使用）
//
// 设计要点（参考 ggml v0.23.0 Vulkan 后端，见 docs/开发进度.md M1.4）：
//   - 每个 Buffer = 一个 VkBuffer + 一个 VkDeviceMemory。**默认优先 device-local**（M4.5e）：
//     不可映射 device-local（真显存）→ 用共享 staging 同步搬运（`base()==nullptr`）；否则可映射
//     device-local（核显/ReBAR，`base()` 可解引用）；否则 host-visible。`TRC_VK_HOST_MEMORY=1`
//     强制 host-visible（基线），`TRC_VK_DEVICE_LOCAL=1` 只优先可映射 device-local，
//     `TRC_VK_FORCE_STAGING=1` 强制 staging（测试）。分配/映射失败回退 host-visible。
//     （FIX-003 / Q26；M4.4b staging；M4.5e 默认切换）
//   - base() 在可映射时返回真实主机指针；不可映射（staging）时为 nullptr，主機读写经 Buffer::set/get_tensor
//   - 逐元素 kernel 通过 push constant 传 ne/nb（元素单位）与基址，支持视图与广播
//   - descriptor offset 向下对齐到 minStorageBufferOffsetAlignment，错位字节折成元素基址传给 shader
//     （与 ggml 的 misalign 处理思路一致，M1.4b 简化：要求 F32 且 4 字节对齐）
#pragma once

#include "traincpp/traincpp.h"

#include <vulkan/vulkan.h>

#include <cstdint>
#include <string>
#include <vector>

namespace traincpp {

// 单节点最多绑定：核心 MAX_SRC(10) + dst，参考 ggml MAX_PARAMETER_COUNT=12
constexpr uint32_t VK_MAX_BINDINGS = 12;

// 一元算子码（必须与 shaders/unary.comp 中的 TRC_UOP_* 一致）
enum class VulkanUnaryOp : uint32_t {
    NEG = 0,
    ABS,
    SQR,
    SQRT,
    EXP,
    LOG,
    SIN,
    COS,
    RELU,
    SIGMOID,
    TANH,
    SILU,
    GELU,
    GELU_ERF,
    ERF,
    SOFTPLUS,
    HARDSWISH,
    SGN,
    STEP,
    LEAKY_RELU,
    CLAMP,
    SCALE,
};

// 二元算子码（必须与 shaders/binary.comp 中的 TRC_BOP_* 一致）
enum class VulkanBinaryOp : uint32_t {
    ADD = 0,
    SUB,
    MUL,
    DIV,
};

// push constant 布局（std430；必须与 shader 中的 PC 块逐字段一致）
struct VulkanUnaryPC {
    uint32_t op;
    uint32_t ne[4];
    uint32_t nb_a[4];  // 元素单位步长
    uint32_t nb_d[4];
    uint32_t base_a;   // 相对 descriptor offset 的元素基址
    uint32_t base_d;
    uint32_t numel;
    float    p0;
    float    p1;
};
static_assert(sizeof(VulkanUnaryPC) == 72, "VulkanUnaryPC 布局必须与 unary.comp 一致");

struct VulkanBinaryPC {
    uint32_t op;
    uint32_t ne_a[4];  // 结果形状（= a 的形状）
    uint32_t ne_b[4];
    uint32_t nb_a[4];
    uint32_t nb_b[4];
    uint32_t nb_d[4];
    uint32_t base_a;
    uint32_t base_b;
    uint32_t base_d;
    uint32_t numel;
};
static_assert(sizeof(VulkanBinaryPC) == 100, "VulkanBinaryPC 布局必须与 binary.comp 一致");

// ---------------- push constant 布局（std430；必须与对应 .comp 中的 PC 块一致）----------------

struct VulkanMatMulPC {  // matmul.comp（mul_mat，batch tiled）：88 字节
    uint32_t k;
    uint32_t m;
    uint32_t n;
    uint32_t n_planes;  // b 的平面数 = b.ne2 * b.ne3（grid z 跨步上限）
    uint32_t b_ne2;     // 平面解码：i2 = p % b_ne2, i3 = p / b_ne2
    uint32_t a_ne2;     // a 的平面取模（batch 广播）
    uint32_t a_ne3;
    uint32_t nb_a0;     // 元素单位步长
    uint32_t nb_a1;
    uint32_t nb_a2;
    uint32_t nb_a3;
    uint32_t nb_b0;
    uint32_t nb_b1;
    uint32_t nb_b2;
    uint32_t nb_b3;
    uint32_t nb_d0;
    uint32_t nb_d1;
    uint32_t nb_d2;
    uint32_t nb_d3;
    uint32_t base_a;
    uint32_t base_b;
    uint32_t base_d;
};
static_assert(sizeof(VulkanMatMulPC) == 88, "VulkanMatMulPC 布局必须与 matmul.comp 一致");

struct VulkanIm2ColPC {  // im2col.comp / im2col_back.comp（共用布局）：108 字节
    uint32_t feat;         // IC*KH*KW
    uint32_t OW;
    uint32_t OH;
    uint32_t N;
    uint32_t IC;
    uint32_t IH;
    uint32_t IW;
    uint32_t KH;
    uint32_t KW;
    int32_t  s0;
    int32_t  s1;
    int32_t  p0;
    int32_t  p1;
    int32_t  d0;
    int32_t  d1;
    uint32_t is_2d;
    uint32_t nb_img[4];    // 元素单位步长（im2col 为输入图像；im2col_back 为输出梯度）
    uint32_t nb_d[4];      // im2col 为输出；im2col_back 为图像
    uint32_t base_img;
    uint32_t base_d;
    uint32_t numel;
};
static_assert(sizeof(VulkanIm2ColPC) == 108, "VulkanIm2ColPC 布局必须与 im2col.comp/im2col_back.comp 一致");

struct VulkanCol2ImPC {  // col2im.comp：48 字节
    uint32_t K;
    uint32_t OC;
    uint32_t T_in;
    int32_t  s0;
    int32_t  p0;
    uint32_t nb_s0;
    uint32_t nb_s1;
    uint32_t nb_d0;
    uint32_t nb_d1;
    uint32_t base_s;
    uint32_t base_d;
    uint32_t numel;
};
static_assert(sizeof(VulkanCol2ImPC) == 48, "VulkanCol2ImPC 布局必须与 col2im.comp 一致");

struct VulkanConvTranspose1DPC {  // conv_transpose_1d.comp：64 字节
    uint32_t K;
    uint32_t Cout;
    uint32_t Cin;
    uint32_t T_in;
    int32_t  s0;
    uint32_t nb_k0;
    uint32_t nb_k1;
    uint32_t nb_k2;
    uint32_t nb_x0;
    uint32_t nb_x1;
    uint32_t nb_d0;
    uint32_t nb_d1;
    uint32_t base_k;
    uint32_t base_x;
    uint32_t base_d;
    uint32_t numel;
};
static_assert(sizeof(VulkanConvTranspose1DPC) == 64,
              "VulkanConvTranspose1DPC 布局必须与 conv_transpose_1d.comp 一致");

struct VulkanPool2DPC {  // pool_2d.comp：96 字节
    int32_t  mode;   // 0=MAX、1=AVG（与 PoolMode/ggml_op_pool 一致）
    int32_t  k0;
    int32_t  k1;
    int32_t  s0;
    int32_t  s1;
    int32_t  p0;
    int32_t  p1;
    uint32_t OW;
    uint32_t OH;
    uint32_t C;
    uint32_t N;
    uint32_t W;
    uint32_t H;
    uint32_t nb_a[4];  // 输入 a 的元素单位步长
    uint32_t base_a;
    uint32_t nb_d[4];
    uint32_t base_d;
    uint32_t numel;
};
static_assert(sizeof(VulkanPool2DPC) == 96, "VulkanPool2DPC 布局必须与 pool_2d.comp 一致");

struct VulkanConvTranspose2DPC {  // conv_transpose_2d.comp：104 字节
    uint32_t KW;
    uint32_t KH;
    uint32_t Cout;
    uint32_t Cin;
    uint32_t W;
    uint32_t H;
    uint32_t N;
    uint32_t OW;
    uint32_t OH;
    int32_t  stride;
    uint32_t nb_w[4];  // 核 [KW,KH,Cout,Cin] 的元素单位步长
    uint32_t nb_x[4];  // 输入 [W,H,Cin,N]
    uint32_t nb_d[4];
    uint32_t base_w;
    uint32_t base_x;
    uint32_t base_d;
    uint32_t numel;
};
static_assert(sizeof(VulkanConvTranspose2DPC) == 104,
              "VulkanConvTranspose2DPC 布局必须与 conv_transpose_2d.comp 一致");

struct VulkanCastPC {  // cast_f32_to_f16.comp / cast_f16_to_f32.comp（共用布局）：60 字节
    uint32_t ne[4];
    uint32_t nb_s[4];  // 元素单位步长（按源类型宽度）
    uint32_t nb_d[4];  // 元素单位步长（按目标类型宽度）
    uint32_t base_s;
    uint32_t base_d;
    uint32_t numel;
};
static_assert(sizeof(VulkanCastPC) == 60, "VulkanCastPC 布局必须与 cast_*.comp 一致");

struct VulkanCopyPC {  // copy.comp（cont/cpy/dup/pad/concat 共用）：60 字节
    uint32_t ne[4];
    uint32_t nb_a[4];
    uint32_t nb_d[4];
    uint32_t base_a;
    uint32_t base_d;
    uint32_t numel;
};
static_assert(sizeof(VulkanCopyPC) == 60, "VulkanCopyPC 布局必须与 copy.comp 一致");

struct VulkanPadReflectPC {  // pad_reflect.comp（pad mode=reflect）：76 字节
    uint32_t ne_d[4];  // 输出形状
    uint32_t ne_a[4];  // 输入形状
    uint32_t lp[4];    // 每维左填充
    uint32_t nb_a[4];  // 输入元素单位步长
    uint32_t base_a;
    uint32_t base_d;
    uint32_t numel;
};
static_assert(sizeof(VulkanPadReflectPC) == 76, "VulkanPadReflectPC 布局必须与 pad_reflect.comp 一致");

struct VulkanPadBackPC {  // pad_back.comp（pad reflect 反向）：76 字节
    uint32_t ne_x[4];  // 前向输入形状（迭代域）
    uint32_t ne_g[4];  // 前向输出（grad）形状
    uint32_t lp[4];    // 每维左填充
    uint32_t nb_g[4];  // grad 元素单位步长
    uint32_t base_g;
    uint32_t base_x;
    uint32_t numel;
};
static_assert(sizeof(VulkanPadBackPC) == 76, "VulkanPadBackPC 布局必须与 pad_back.comp 一致");

struct VulkanReduceRowsPC {  // reduce_rows.comp（sum_rows/mean）：48 字节
    uint32_t mode;           // 0=sum, 1=mean
    uint32_t n_rows;
    uint32_t ne0;
    uint32_t ne1;
    uint32_t ne2;
    uint32_t nb_a[4];
    uint32_t base_a;
    uint32_t base_d;
    float    inv_ne0;
};
static_assert(sizeof(VulkanReduceRowsPC) == 48, "VulkanReduceRowsPC 布局必须与 reduce_rows.comp 一致");

struct VulkanSumAllPC {  // sum_all.comp（sum）：44 字节
    uint32_t ne[4];
    uint32_t nb_a[4];
    uint32_t base_a;
    uint32_t base_d;
    uint32_t numel;
};
static_assert(sizeof(VulkanSumAllPC) == 44, "VulkanSumAllPC 布局必须与 sum_all.comp 一致");

struct VulkanNormPC {  // norm.comp（norm/rms_norm）：64 字节
    uint32_t mode;     // 0=norm, 1=rms_norm
    uint32_t n_rows;
    uint32_t ne0;
    uint32_t ne1;
    uint32_t ne2;
    uint32_t nb_a[4];
    uint32_t nb_d[4];
    uint32_t base_a;
    uint32_t base_d;
    float    eps;
};
static_assert(sizeof(VulkanNormPC) == 64, "VulkanNormPC 布局必须与 norm.comp 一致");

struct VulkanGroupNormPC {  // group_norm.comp：68 字节
    uint32_t ne0;
    uint32_t ne1;
    uint32_t ne2;
    uint32_t ne3;
    uint32_t n_groups;
    uint32_t per_group;
    uint32_t nb_a[4];
    uint32_t nb_d[4];
    uint32_t base_a;
    uint32_t base_d;
    float    eps;
};
static_assert(sizeof(VulkanGroupNormPC) == 68, "VulkanGroupNormPC 布局必须与 group_norm.comp 一致");

struct VulkanNormBackPC {  // norm_back.comp（norm_back/rms_norm_back）：84 字节
    uint32_t mode;         // 0=norm_back, 1=rms_norm_back
    uint32_t n_rows;
    uint32_t ne0;
    uint32_t ne1;
    uint32_t ne2;
    uint32_t nb_dy[4];
    uint32_t nb_x[4];
    uint32_t nb_d[4];
    uint32_t base_dy;
    uint32_t base_x;
    uint32_t base_d;
    float    eps;
};
static_assert(sizeof(VulkanNormBackPC) == 84, "VulkanNormBackPC 布局必须与 norm_back.comp 一致");

struct VulkanGroupNormBackPC {  // group_norm_back.comp：88 字节
    uint32_t ne0;
    uint32_t ne1;
    uint32_t ne2;
    uint32_t ne3;
    uint32_t n_groups;
    uint32_t per_group;
    uint32_t nb_dy[4];
    uint32_t nb_x[4];
    uint32_t nb_d[4];
    uint32_t base_dy;
    uint32_t base_x;
    uint32_t base_d;
    float    eps;
};
static_assert(sizeof(VulkanGroupNormBackPC) == 88,
              "VulkanGroupNormBackPC 布局必须与 group_norm_back.comp 一致");

struct VulkanSoftMaxPC {  // softmax.comp（soft_max/soft_max_ext）：108 字节
    uint32_t ne0;
    uint32_t ne1;
    uint32_t ne2;
    uint32_t ne3;
    uint32_t nb_a[4];
    uint32_t nb_d[4];
    uint32_t nb_m[4];
    uint32_t base_a;
    uint32_t base_d;
    uint32_t base_m;
    uint32_t ne12;
    uint32_t ne13;
    uint32_t has_mask;
    uint32_t use_alibi;
    uint32_t n_head_log2;
    float    scale;
    float    m0;
    float    m1;
};
static_assert(sizeof(VulkanSoftMaxPC) == 108, "VulkanSoftMaxPC 布局必须与 softmax.comp 一致");

struct VulkanCrossEntropyPC {  // cross_entropy_loss.comp：64 字节
    uint32_t nc;      // ne0（类别数）
    uint32_t nr;      // ne1*ne2*ne3（行数）
    uint32_t ne1;
    uint32_t ne2;
    uint32_t nb_a[4];
    uint32_t nb_b[4];
    uint32_t base_a;
    uint32_t base_b;
    uint32_t base_d;
    float    inv_nr;
};
static_assert(sizeof(VulkanCrossEntropyPC) == 64,
              "VulkanCrossEntropyPC 布局必须与 cross_entropy_loss.comp 一致");

struct VulkanCrossEntropyBackPC {  // cross_entropy_loss_back.comp：84 字节
    uint32_t nc;
    uint32_t nr;
    uint32_t ne1;
    uint32_t ne2;
    uint32_t nb_a[4];
    uint32_t nb_b[4];
    uint32_t nb_d[4];
    uint32_t base_a;
    uint32_t base_b;
    uint32_t base_g;
    uint32_t base_d;
    float    inv_nr;
};
static_assert(sizeof(VulkanCrossEntropyBackPC) == 84,
              "VulkanCrossEntropyBackPC 布局必须与 cross_entropy_loss_back.comp 一致");

struct VulkanBroadcastPC {  // broadcast_copy.comp（repeat）：76 字节
    uint32_t ne_s[4];
    uint32_t ne_d[4];
    uint32_t nb_s[4];
    uint32_t nb_d[4];
    uint32_t base_s;
    uint32_t base_d;
    uint32_t numel;
};
static_assert(sizeof(VulkanBroadcastPC) == 76, "VulkanBroadcastPC 布局必须与 broadcast_copy.comp 一致");

struct VulkanRepeatBackPC {  // repeat_back.comp：76 字节
    uint32_t ne_d[4];
    uint32_t reps[4];  // a.ne[d] / dst.ne[d]
    uint32_t nb_a[4];
    uint32_t nb_d[4];
    uint32_t base_a;
    uint32_t base_d;
    uint32_t numel_d;
};
static_assert(sizeof(VulkanRepeatBackPC) == 76, "VulkanRepeatBackPC 布局必须与 repeat_back.comp 一致");

struct VulkanAccAddPC {  // acc_add.comp：60 字节
    uint32_t ne[4];
    uint32_t nb_b[4];
    uint32_t nb_d[4];
    uint32_t base_b;
    uint32_t base_d;
    uint32_t numel;
};
static_assert(sizeof(VulkanAccAddPC) == 60, "VulkanAccAddPC 布局必须与 acc_add.comp 一致");

// 各张量独立 descriptor（floor 对齐后 base_elem 可能不同），故每个 binding 单列 base
struct VulkanOptStepAdamwPC {  // opt_step_adamw.comp：76 字节
    uint32_t ne[4];
    uint32_t nb[4];  // 元素单位步长（param 连续）
    uint32_t base_p;
    uint32_t base_g;
    uint32_t base_m;
    uint32_t base_v;
    uint32_t numel;
    float    step_sz;   // lr / bc1
    float    bc2_sqrt;  // sqrt(1 - beta2^t)
    float    decay;     // 1 - lr*wd
    float    beta1;
    float    beta2;
    float    eps;
};
static_assert(sizeof(VulkanOptStepAdamwPC) == 76, "VulkanOptStepAdamwPC 布局必须与 opt_step_adamw.comp 一致");

struct VulkanOptStepSgdPC {  // opt_step_sgd.comp：72 字节
    uint32_t ne[4];
    uint32_t nb[4];
    uint32_t base_p;
    uint32_t base_g;
    uint32_t base_m;
    uint32_t numel;
    float    lr;
    float    momentum;   // 动量系数
    float    dampening;
    float    weight_decay;
    float    nesterov;   // 0/1
    float    is_first;   // 0/1（首个动量步直接取 d）
};
static_assert(sizeof(VulkanOptStepSgdPC) == 72, "VulkanOptStepSgdPC 布局必须与 opt_step_sgd.comp 一致");

// M4.5：梯度裁剪原语 / WeightNorm 同步
struct VulkanSumSqrAccPC {  // sum_sqr_acc.comp：12 字节
    uint32_t base_a;
    uint32_t base_acc;
    uint32_t numel;
};
static_assert(sizeof(VulkanSumSqrAccPC) == 12, "VulkanSumSqrAccPC 布局必须与 sum_sqr_acc.comp 一致");

struct VulkanClipScalePC {  // clip_scale_inplace.comp：20 字节
    uint32_t numel;
    uint32_t base_a;
    uint32_t base_norm;
    float    max_norm;
    float    eps;
};
static_assert(sizeof(VulkanClipScalePC) == 20, "VulkanClipScalePC 布局必须与 clip_scale_inplace.comp 一致");

struct VulkanWeightNormSyncPC {  // weightnorm_sync.comp：16 字节
    uint32_t numel;
    uint32_t n_rest;
    uint32_t base_v;
    uint32_t base_g;
};
static_assert(sizeof(VulkanWeightNormSyncPC) == 16,
              "VulkanWeightNormSyncPC 布局必须与 weightnorm_sync.comp 一致");

struct VulkanGetRowsPC {  // get_rows.comp：80 字节
    uint32_t ne0;
    uint32_t n_vocab;
    uint32_t nr;
    uint32_t ni1;
    uint32_t ni2;
    uint32_t nb_t[4];
    uint32_t nb_o[4];
    uint32_t nb_i[3];
    uint32_t base_t;
    uint32_t base_o;
    uint32_t base_i;
    uint32_t numel;
};
static_assert(sizeof(VulkanGetRowsPC) == 80, "VulkanGetRowsPC 布局必须与 get_rows.comp 一致");

struct VulkanSetRowsPC {  // set_rows.comp：88 字节
    uint32_t ne0;
    uint32_t nr;
    uint32_t ni2;
    uint32_t ni3;
    uint32_t idx_ne1;
    uint32_t idx_ne2;
    uint32_t n_vocab;
    uint32_t nb_v[4];
    uint32_t nb_t[4];
    uint32_t nb_i[3];
    uint32_t base_v;
    uint32_t base_t;
    uint32_t base_i;
    uint32_t numel;
};
static_assert(sizeof(VulkanSetRowsPC) == 88, "VulkanSetRowsPC 布局必须与 set_rows.comp 一致");

struct VulkanGetRowsBackPC {  // get_rows_back.comp：76 字节
    uint32_t ne0;
    uint32_t n_vocab;
    uint32_t nr;
    uint32_t ni1;
    uint32_t ni2;
    uint32_t nb_g[4];
    uint32_t nb_d[4];
    uint32_t nb_i[3];
    uint32_t base_g;
    uint32_t base_d;
    uint32_t base_i;
};
static_assert(sizeof(VulkanGetRowsBackPC) == 76, "VulkanGetRowsBackPC 布局必须与 get_rows_back.comp 一致");

// Vulkan 设备/队列/命令池等全局状态（进程内单例，惰性初始化）
struct VulkanState {
    bool                       available = false;
    VkInstance                 instance = VK_NULL_HANDLE;
    VkDebugUtilsMessengerEXT   debug_messenger = VK_NULL_HANDLE;
    VkPhysicalDevice           physical_device = VK_NULL_HANDLE;
    VkDevice                   device = VK_NULL_HANDLE;
    VkQueue                    queue = VK_NULL_HANDLE;
    uint32_t                   queue_family = 0;
    VkCommandPool              command_pool = VK_NULL_HANDLE;
    VkFence                    fence = VK_NULL_HANDLE;
    VkDescriptorSetLayout      descriptor_layout = VK_NULL_HANDLE;
    std::vector<VkDescriptorPool> descriptor_pools;
    size_t                     descriptor_pool_index = 0;
    // 设备端错误标志（索引越界等）：host-visible + coherent，graph_compute 结束时检查并中止
    // 布局：errors[0]=标志位（原子或）、errors[1]=首个越界索引（尽力而为）、errors[2]=算子码
    VkBuffer                   error_buffer = VK_NULL_HANDLE;
    VkDeviceMemory             error_memory = VK_NULL_HANDLE;
    uint32_t*                  error_mapped = nullptr;
    VkPhysicalDeviceProperties props{};
    VkPhysicalDeviceMemoryProperties mem_props{};
    uint32_t                   min_storage_offset_align = 256;
    size_t                     max_storage_buffer_range = 0;
    size_t                     max_buffer_size = 0;             // VkPhysicalDeviceMaintenance3Properties
    size_t                     max_memory_allocation_size = 0;  // 同上（单次 vkAllocateMemory 上限）
    size_t                     memory_total = 0;
    bool                       integrated = false;
    bool                       unified_memory = false;
    bool                       storage_16bit = false;  // 设备支持 16 位存储（F16 cast 需要）
    // M4.4b：共享 staging（不可映射 device-local 缓冲的主机↔设备传输，同步提交）
    VkBuffer                   stage_buffer = VK_NULL_HANDLE;
    VkDeviceMemory             stage_memory = VK_NULL_HANDLE;
    void*                      stage_mapped = nullptr;
    size_t                     stage_size = 0;
    VkCommandPool              transfer_pool = VK_NULL_HANDLE;
    VkFence                    transfer_fence = VK_NULL_HANDLE;
    // FIX-004 / Q28：失败诊断上下文（提交失败时输出）
    size_t                     live_buffer_bytes = 0;     // 进程存活计算缓冲字节（VulkanBuffer 构造/析构维护）
    uint32_t                   graph_dispatch_count = 0;  // 本次提交已录制的 dispatch 数（compute_begin 清零）
    std::string                name;
    std::string                driver;
};

// 惰性初始化；失败时 available=false（不 abort，由注册表跳过该后端）
VulkanState& vulkan_state();

// x 轴 dispatch 网格上限（min(设备 maxComputeWorkGroupCount[0], TRC_VK_MAX_GROUPS)）。
// TRC_VK_MAX_GROUPS 为测试用覆盖（模拟小上限设备，验证各 kernel 的 stride 循环）。
uint32_t vulkan_max_grid_x();

// z 轴 dispatch 网格上限（min(设备 maxComputeWorkGroupCount[2], TRC_VK_MAX_GROUPS)）。
// 用于 mul_mat batch 平面等按 z 跨步的 kernel（TRC_VK_MAX_GROUPS 同样覆盖 z）。
uint32_t vulkan_max_grid_z();

// descriptor storage range 上限（min(设备 maxStorageBufferRange, TRC_VK_MAX_STORAGE_RANGE)）。
// TRC_VK_MAX_STORAGE_RANGE（字节）为测试用覆盖（模拟小上限设备，验证 range 校验）。
size_t vulkan_max_storage_range();

// ---------------- Buffer ----------------

class VulkanBufferType;

class VulkanBuffer final : public Buffer {
public:
    static VulkanBuffer* create(VulkanBufferType* buft, size_t size);

    ~VulkanBuffer() override;

    BufferType* buffer_type() const override;
    size_t      size() const override { return size_; }
    void*       base() override { return mapped_; }   // 不可映射 device-local 缓冲返回 nullptr（M4.4b）
    bool        is_host() const override { return mapped_ != nullptr; }

    void set_tensor(Tensor* t, size_t offset, const void* data, size_t size) override;
    void get_tensor(const Tensor* t, size_t offset, void* data, size_t size) const override;
    void clear() override;

    VkBuffer vk_buffer() const { return buffer_; }

private:
    VulkanBuffer(VulkanBufferType* buft, VkBuffer buffer, VkDeviceMemory memory, void* mapped, size_t size);

    VulkanBufferType* buft_ = nullptr;
    VkBuffer          buffer_ = VK_NULL_HANDLE;
    VkDeviceMemory    memory_ = VK_NULL_HANDLE;
    void*             mapped_ = nullptr;
    size_t            size_ = 0;
};

class VulkanBufferType final : public BufferType {
public:
    explicit VulkanBufferType(Device* dev) : device_(dev) {}

    Device*     device() const override { return device_; }
    const char* name() const override { return "Vulkan"; }
    size_t      alignment() const override;
    Buffer*     alloc_buffer(size_t size) override;

private:
    Device* device_ = nullptr;
};

// ---------------- 管线 / 调度 ----------------

struct VulkanGraphContext {
    VkCommandBuffer cmd = VK_NULL_HANDLE;
};

struct VulkanPipelineHandle {
    VkPipeline       pipeline = VK_NULL_HANDLE;
    VkPipelineLayout layout = VK_NULL_HANDLE;
};

// descriptor 绑定信息 + 相对 descriptor offset 的元素基址
struct VulkanTensorBinding {
    VkDescriptorBufferInfo info{};
    uint32_t               base_elem = 0;
};

// 取（或首次创建）一个 compute 管线
VulkanPipelineHandle vulkan_pipeline(const char* name, uint32_t push_constant_size);

// 从当前池分配一个 descriptor set（池在每次 graph_compute 开始时重置）
VkDescriptorSet vulkan_alloc_descriptor_set();

// 计算张量在所属 VkBuffer 中的绑定（支持视图；要求 F32 + 4 字节对齐）
VulkanTensorBinding vulkan_bind_tensor(const Tensor* t);

// 设备端错误标志缓冲的绑定（索引类 shader 的 binding=3；见 VulkanState::error_buffer）
VulkanTensorBinding vulkan_bind_error_buffer();

// 记录一次 dispatch（含绑定、push constant 与保守的全局内存屏障）
// groups_x/y/z：3D 计算网格（tiled matmul 等用 2D；其余用默认 1）
void vulkan_dispatch(VulkanGraphContext& gc, const VulkanPipelineHandle& pipe, const void* pc,
                     uint32_t pc_size, const VulkanTensorBinding* bindings, uint32_t n_bindings,
                     uint32_t groups_x, uint32_t groups_y = 1, uint32_t groups_z = 1);

bool vulkan_supports_op(const Tensor* t);
bool vulkan_compute_node(VulkanGraphContext& gc, Tensor* node);

} // namespace traincpp
