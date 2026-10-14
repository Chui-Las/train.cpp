// train.cpp - Vulkan 后端 smoke 测试（M1.4a）
//
// 覆盖：设备注册与属性、缓冲 alloc/set/get/clear、视图共享 host-visible 内存（关键语义）、
//       空图 graph_compute。无 Vulkan 设备（或未启用 TRC_VULKAN）时自动跳过。
#include "test_util.h"
#include "traincpp/traincpp.h"

#include <cstdio>
#include <cstdlib>
#include <vector>

using namespace traincpp;

namespace {

Device* find_vulkan_device() {
    Device* dev = device_by_type(DeviceType::GPU);
    if (dev == nullptr) {
        dev = device_by_type(DeviceType::IGPU);
    }
    return dev;
}

} // namespace

TRC_TEST(vulkan_device_available) {
    Device* dev = find_vulkan_device();
    if (dev == nullptr) {
        std::printf("    跳过：未检测到 Vulkan 设备（或未启用 TRC_VULKAN）\n");
        return;
    }

    const DeviceProps& p = dev->props();
    std::printf("    Vulkan 设备: %s（unified=%d, 显存 %.0f MiB）\n", p.name.c_str(),
                (int) p.unified_memory, (double) p.memory_total / (1024.0 * 1024.0));
    TRC_EXPECT(!p.name.empty());
    TRC_EXPECT(p.type == DeviceType::GPU || p.type == DeviceType::IGPU);

    BufferType* buft = dev->default_buffer_type();
    TRC_EXPECT(buft != nullptr);
    const size_t align = buft->alignment();
    TRC_EXPECT(align > 0);
    TRC_EXPECT((align & (align - 1)) == 0);  // 必须是 2 的幂（核心 align_up 依赖）
}

TRC_TEST(vulkan_buffer_roundtrip) {
    Device* dev = find_vulkan_device();
    if (dev == nullptr) {
        return;
    }

    Context* ctx = context_new(1 << 20);

    Tensor* src  = new_tensor_2d(ctx, TYPE_F32, 4, 4);   // [ne0=4, ne1=4]
    Tensor* idx  = new_tensor_1d(ctx, TYPE_I32, 4);
    Tensor* view = view_2d(ctx, src, 4, 2, 16);          // 第 1~2 行（偏移 16 字节）

    Buffer* buf = buffer_alloc_ctx_tensors(ctx, dev->default_buffer_type());
    TRC_EXPECT(buf != nullptr);
    // 默认 host-visible（is_host=true）；M4.4b 强制 staging 时缓冲不可映射（is_host=false）
    const bool force_staging = std::getenv("TRC_VK_FORCE_STAGING") != nullptr;
    TRC_EXPECT(force_staging || buf->is_host());
    TRC_EXPECT(buf->size() > 0);

    // F32 往返
    std::vector<float> vals(16);
    for (int i = 0; i < 16; ++i) {
        vals[(size_t) i] = (float) i + 0.5f;
    }
    tensor_set(src, vals.data(), 0, vals.size() * sizeof(float));
    std::vector<float> got(16, -1.0f);
    tensor_get(src, got.data(), 0, got.size() * sizeof(float));
    for (size_t i = 0; i < vals.size(); ++i) {
        TRC_EXPECT_NEAR(got[i], vals[i], 0.0);
    }

    // 视图读取：视图沿 view_src 找到根的 buffer（含不可映射，经 staging/M4.4 契约）
    std::vector<float> vgot(8, -1.0f);
    tensor_get(view, vgot.data(), 0, vgot.size() * sizeof(float));
    for (int i = 0; i < 8; ++i) {
        TRC_EXPECT_NEAR(vgot[(size_t) i], vals[(size_t) (4 + i)], 0.0);
    }

    // 视图直写：写入视图第 0 行（对应源的 flat 元素 4..7）
    const float w[4] = {100.0f, 101.0f, 102.0f, 103.0f};
    tensor_set(view, w, 0, sizeof(w));
    std::vector<float> after(16, -1.0f);
    tensor_get(src, after.data(), 0, after.size() * sizeof(float));
    for (int i = 0; i < 4; ++i) {
        TRC_EXPECT_NEAR(after[(size_t) (4 + i)], w[i], 0.0);
    }

    // I32 往返
    const int32_t iv[4] = {7, -3, 42, 0};
    tensor_set(idx, iv, 0, sizeof(iv));
    int32_t iout[4] = {0, 0, 0, 0};
    tensor_get(idx, iout, 0, sizeof(iout));
    for (int i = 0; i < 4; ++i) {
        TRC_EXPECT_EQ(iout[i], iv[i]);
    }

    // clear 后应全 0
    buf->clear();
    std::vector<float> zero(16, 999.0f);
    tensor_get(src, zero.data(), 0, zero.size() * sizeof(float));
    for (float v : zero) {
        TRC_EXPECT_NEAR(v, 0.0, 0.0);
    }

    buffer_free(buf);
    context_free(ctx);
}

TRC_TEST(vulkan_supports_and_empty_graph) {
    Device* dev = find_vulkan_device();
    if (dev == nullptr) {
        return;
    }

    Context* ctx = context_new(64 * 1024);
    Tensor*  a   = new_tensor_1d(ctx, TYPE_F32, 4);
    Tensor*  b   = new_tensor_1d(ctx, TYPE_F32, 4);
    Tensor*  s   = add(ctx, a, b);
    Tensor*  mm  = mul_mat(ctx, a, b);      // M1.4d：已支持
    Tensor*  c16 = cast(ctx, a, TYPE_F16);  // M1.4e：F32↔F16（需 16 位存储特性）
    Tensor*  c8  = cast(ctx, a, TYPE_I8);   // 其他类型转换未实现

    // 叶子/视图与已实现算子为 true，未实现算子为 false
    TRC_EXPECT(dev->supports_op(a));      // OP_NONE
    TRC_EXPECT(dev->supports_op(s));      // OP_ADD（M1.4b）
    TRC_EXPECT(dev->supports_op(mm));     // OP_MUL_MAT（M1.4d）
    TRC_EXPECT(dev->supports_op(c16));    // OP_CAST F32->F16（M1.4e）
    TRC_EXPECT(!dev->supports_op(c8));    // OP_CAST 其他类型（未实现）

    Graph* g = graph_new(ctx);
    TRC_EXPECT(dev->graph_compute(g));

    graph_free(g);
    context_free(ctx);
}
