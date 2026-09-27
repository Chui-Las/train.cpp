// train.cpp - mul_mat 形状守卫死亡测试（M2.3a）
//
// 语义：mul_mat 按 ggml_can_mul_mat 约束——a.ne[0]==b.ne[0]、b.ne[2]%a.ne[2]==0、
//       b.ne[3]%a.ne[3]==0；不满足时必须**可读中止**（不得静默算错）。
//   - ARGS=inner ：内维不一致（a.ne0 != b.ne0）
//   - ARGS=batch2：b.ne2 不是 a.ne2 的整数倍
//   - ARGS=batch3：b.ne3 不是 a.ne3 的整数倍
//   - 未中止则打印错误并返回 0；CTest 经 tests/expect_abort.cmake 判定：中止=通过
#include "traincpp/traincpp.h"

#include <cstdio>
#include <cstring>

using namespace traincpp;

int main(int argc, char** argv) {
    const char* mode = argc > 1 ? argv[1] : "inner";

    Context* ctx = context_new(1 << 20);
    Tensor*  a   = nullptr;
    Tensor*  b   = nullptr;

    if (std::strcmp(mode, "inner") == 0) {
        a = new_tensor_4d(ctx, TYPE_F32, 3, 2, 1, 1);
        b = new_tensor_4d(ctx, TYPE_F32, 4, 5, 1, 1);  // ne0 不一致
    } else if (std::strcmp(mode, "batch2") == 0) {
        a = new_tensor_4d(ctx, TYPE_F32, 3, 2, 3, 1);
        b = new_tensor_4d(ctx, TYPE_F32, 3, 5, 2, 1);  // 2 % 3 != 0
    } else if (std::strcmp(mode, "batch3") == 0) {
        a = new_tensor_4d(ctx, TYPE_F32, 3, 2, 1, 3);
        b = new_tensor_4d(ctx, TYPE_F32, 3, 5, 1, 2);  // 2 % 3 != 0
    } else {
        std::printf("未知模式 %s\n", mode);
        context_free(ctx);
        return 77;
    }

    Tensor* out = mul_mat(ctx, a, b);
    (void) out;

    std::printf("[错误] mul_mat 形状非法（%s）但未中止（守卫失效，可能静默算错）\n", mode);
    context_free(ctx);
    return 0;
}
