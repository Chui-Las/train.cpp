// train.cpp - checkpoint 保存期守卫死亡测试（M2.3f / schema v2 多优化器）
//
// 语义：多优化器分区的**名字重复**、**跨分区张量名重复**都必须在保存期明确中止
//       （否则载入分区歧义 / GGUF 张量同名互相覆盖）。
//   - 本程序按参数选择场景，期望在 checkpoint_save 处中止（退出码非 0）
//   - 未中止则打印错误并返回 0；CTest 经 tests/expect_abort.cmake 判定：中止=通过
#include "traincpp/traincpp.h"

#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>

using namespace traincpp;

namespace {

std::string guard_tmp_path(const char* name) {
    return (std::filesystem::temp_directory_path() /
            (std::string("trc_checkpoint_guard_") + name + ".gguf"))
        .string();
}

int run_dup_name() {
    Context* ctx = context_new(1 << 20);
    Tensor*  p1  = new_tensor_1d(ctx, TYPE_F32, 4);
    Tensor*  p2  = new_tensor_1d(ctx, TYPE_F32, 4);
    tensor_set_name(p1, "p1");
    tensor_set_name(p2, "p2");
    tensor_set_param(p1);
    tensor_set_param(p2);

    Tensor*    params1[1] = {p1};
    Tensor*    params2[1] = {p2};
    Optimizer* o1         = optim_adamw_new(ctx, params1, 1, AdamwOptions{});
    Optimizer* o2         = optim_adamw_new(ctx, params2, 1, AdamwOptions{});

    CheckpointOptimizer parts[2] = {{"G", o1, nullptr}, {"G", o2, nullptr}};
    CheckpointItems     items{};
    items.optimizers      = parts;
    items.optimizer_count = 2;
    checkpoint_save(guard_tmp_path("dup_name").c_str(), items, nullptr);  // 期望在此中止

    std::printf("[错误] 分区名重复未中止（守卫失效：载入会产生歧义）\n");
    optim_free(o2);
    optim_free(o1);
    context_free(ctx);
    return 0;
}

int run_dup_tensor() {
    Context* ctx = context_new(1 << 20);
    Tensor*  p   = new_tensor_1d(ctx, TYPE_F32, 4);
    tensor_set_name(p, "shared");
    tensor_set_param(p);
    Buffer* buf = buffer_alloc_ctx_tensors(ctx, device_cpu()->default_buffer_type());

    Tensor*    params[1] = {p};
    Optimizer* o1        = optim_adamw_new(ctx, params, 1, AdamwOptions{});
    Optimizer* o2        = optim_adamw_new(ctx, params, 1, AdamwOptions{});

    CheckpointOptimizer parts[2] = {{"G", o1, nullptr}, {"D", o2, nullptr}};
    CheckpointItems     items{};
    items.optimizers      = parts;
    items.optimizer_count = 2;
    checkpoint_save(guard_tmp_path("dup_tensor").c_str(), items, nullptr);  // 期望在此中止

    std::printf("[错误] 跨分区张量名重复未中止（守卫失效：状态张量会互相覆盖）\n");
    optim_free(o2);
    optim_free(o1);
    buffer_free(buf);
    context_free(ctx);
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    const char* mode = argc > 1 ? argv[1] : "";
    if (std::strcmp(mode, "dup_name") == 0) {
        return run_dup_name();
    }
    if (std::strcmp(mode, "dup_tensor") == 0) {
        return run_dup_tensor();
    }
    std::printf("用法: %s <dup_name|dup_tensor>\n", argc > 0 ? argv[0] : "test_checkpoint_guard");
    return 2;
}
