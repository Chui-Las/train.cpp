// train.cpp - 测试入口
#include "test_util.h"

#include <cstdio>
#include <string>
#include <vector>

namespace trc_test {

namespace {
std::vector<std::string> g_failures;
}

std::vector<TestCase>& registry() {
    static std::vector<TestCase> r;
    return r;
}

void add_failure(const std::string& msg) {
    g_failures.push_back(msg);
    std::fprintf(stderr, "    [FAIL] %s\n", msg.c_str());
}

Registrar::Registrar(const char* name, std::function<void()> fn) {
    registry().push_back({name, std::move(fn)});
}

int run_all() {
    int failed_cases = 0;
    std::printf("运行 %zu 个测试用例...\n", registry().size());
    for (const TestCase& tc : registry()) {
        g_failures.clear();
        std::printf("[ RUN  ] %s\n", tc.name.c_str());
        tc.fn();
        if (g_failures.empty()) {
            std::printf("[  OK  ] %s\n", tc.name.c_str());
        } else {
            std::printf("[ FAIL ] %s (%zu 个断言失败)\n", tc.name.c_str(), g_failures.size());
            ++failed_cases;
        }
    }
    if (failed_cases == 0) {
        std::printf("全部通过 (%zu 个用例)\n", registry().size());
        return 0;
    }
    std::printf("失败: %d / %zu 个用例\n", failed_cases, registry().size());
    return 1;
}

} // namespace trc_test

int main() {
    return trc_test::run_all();
}
