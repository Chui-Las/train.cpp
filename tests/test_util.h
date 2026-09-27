// train.cpp - 轻量自研测试框架
// 设计：不引入第三方依赖；每个测试文件用 TRC_TEST 注册用例，由 test_main.cpp 统一运行。
#pragma once

#include "traincpp/traincpp.h"

#include <cstdio>
#include <cstdlib>
#include <functional>
#include <string>
#include <vector>

namespace trc_test {

// ---- 测试设备选择（M1.4d）----
// 环境变量 TRC_TEST_DEVICE=cpu（默认）| vulkan；未识别的值打印错误并返回 nullptr。
// Vulkan 不可用（无设备 / 未启用 TRC_VULKAN）时返回 nullptr，由调用方跳过。
inline traincpp::Device* test_device() {
    const char* env         = std::getenv("TRC_TEST_DEVICE");
    const std::string name  = (env == nullptr || *env == '\0') ? std::string("cpu") : std::string(env);
    if (name == "cpu") {
        return traincpp::device_cpu();
    }
    if (name == "vulkan") {
        traincpp::Device* dev = traincpp::device_by_type(traincpp::DeviceType::GPU);
        if (dev == nullptr) {
            dev = traincpp::device_by_type(traincpp::DeviceType::IGPU);
        }
        return dev;
    }
    std::fprintf(stderr, "[测试] 未知 TRC_TEST_DEVICE=%s（可选 cpu/vulkan）\n", name.c_str());
    return nullptr;
}

// 整图后端支持检查：返回第一个不支持的节点（nullptr 表示全部支持，可安全 compute）
inline const traincpp::Tensor* first_unsupported_node(const traincpp::Graph* graph,
                                                      const traincpp::Device* dev) {
    return traincpp::graph_first_unsupported(graph, dev);
}

struct TestCase {
    std::string           name;
    std::function<void()> fn;
};

std::vector<TestCase>& registry();
void add_failure(const std::string& msg);
int  run_all();

struct Registrar {
    Registrar(const char* name, std::function<void()> fn);
};

// ---- 断言实现 ----
inline void expect(bool cond, const char* expr, const char* file, int line) {
    if (!cond) {
        char buf[1024];
        std::snprintf(buf, sizeof(buf), "%s:%d: 断言失败: %s", file, line, expr);
        add_failure(buf);
    }
}

inline void expect_eq(long long a, long long b, const char* expr, const char* file, int line) {
    if (a != b) {
        char buf[1024];
        std::snprintf(buf, sizeof(buf), "%s:%d: 断言失败: %s (实际 %lld, 期望 %lld)", file, line, expr, a, b);
        add_failure(buf);
    }
}

inline void expect_near(double a, double b, double eps, const char* expr, const char* file, int line) {
    const double diff = a > b ? a - b : b - a;
    if (!(diff <= eps)) {
        char buf[1024];
        std::snprintf(buf, sizeof(buf), "%s:%d: 断言失败: %s (实际 %.9g, 期望 %.9g, 差 %.3g > %.3g)", file,
                      line, expr, a, b, diff, eps);
        add_failure(buf);
    }
}

} // namespace trc_test

#define TRC_TEST(name)                                                    \
    static void trc_test_##name();                                        \
    static ::trc_test::Registrar trc_reg_##name(#name, trc_test_##name);  \
    static void trc_test_##name()

#define TRC_EXPECT(cond) ::trc_test::expect((cond), #cond, __FILE__, __LINE__)

#define TRC_EXPECT_EQ(a, b) \
    ::trc_test::expect_eq((long long) (a), (long long) (b), #a " == " #b, __FILE__, __LINE__)

#define TRC_EXPECT_NEAR(a, b, eps) \
    ::trc_test::expect_near((double) (a), (double) (b), (eps), #a " ~= " #b, __FILE__, __LINE__)
