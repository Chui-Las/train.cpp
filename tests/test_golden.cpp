// train.cpp - PyTorch 黄金数据对拍（M1.2e）
//
// 数据由 scripts/gen_golden.py 生成到 tests/golden/：
//   cases.txt          用例清单
//   <name>_in<i>.npy   输入
//   <name>_out.npy     期望输出
//
// 若缺少黄金数据，本测试打印"跳过"并返回成功（便于无 Python 环境时通过 CTest）。
#include "npy.h"
#include "test_util.h"
#include "traincpp/traincpp.h"

#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#ifndef TRC_GOLDEN_DIR
#define TRC_GOLDEN_DIR ""
#endif

using namespace traincpp;
using namespace trc_test;

namespace {

struct InputSpec {
    Type    type = TYPE_F32;
    int64_t ne[4] = {1, 1, 1, 1};
};

struct Case {
    std::string          name;
    std::string          op;
    std::vector<double>  attrs;
    std::vector<InputSpec> inputs;
};

bool load_manifest(const std::string& path, std::vector<Case>& cases) {
    std::ifstream f(path);
    if (!f) {
        return false;
    }

    std::string line;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#') {
            continue;
        }
        std::istringstream ss(line);
        Case c;
        int n_attrs = 0;
        int n_inputs = 0;
        if (!(ss >> c.name >> c.op >> n_attrs)) {
            continue;
        }
        for (int i = 0; i < n_attrs; ++i) {
            double v = 0;
            ss >> v;
            c.attrs.push_back(v);
        }
        ss >> n_inputs;
        for (int i = 0; i < n_inputs; ++i) {
            std::string tok;
            ss >> tok;
            const size_t colon = tok.find(':');
            if (colon == std::string::npos) {
                return false;
            }
            InputSpec spec;
            spec.type = (tok.substr(colon + 1) == "i32") ? TYPE_I32 : TYPE_F32;

            const std::string shape_str = tok.substr(0, colon);
            int    dim = 0;
            size_t pos = 0;
            while (pos < shape_str.size() && dim < 4) {
                const size_t x = shape_str.find('x', pos);
                const std::string num = (x == std::string::npos) ? shape_str.substr(pos)
                                                                 : shape_str.substr(pos, x - pos);
                spec.ne[dim++] = std::stoll(num);
                if (x == std::string::npos) {
                    break;
                }
                pos = x + 1;
            }
            for (; dim < 4; ++dim) {
                spec.ne[dim] = 1;
            }
            c.inputs.push_back(spec);
        }
        cases.push_back(std::move(c));
    }
    return !cases.empty();
}

Tensor* build_op(Context* ctx, const Case& c, const std::vector<Tensor*>& in) {
    const std::string& op = c.op;
    auto Ai = [&](int i) { return (int) c.attrs[(size_t) i]; };
    auto Af = [&](int i) { return (float) c.attrs[(size_t) i]; };

    // 逐元素二元
    if (op == "add")    return add(ctx, in[0], in[1]);
    if (op == "add1")   return add1(ctx, in[0], in[1]);
    if (op == "sub")    return sub(ctx, in[0], in[1]);
    if (op == "mul")    return mul(ctx, in[0], in[1]);
    if (op == "div")    return div(ctx, in[0], in[1]);

    // 逐元素一元
    if (op == "neg")     return neg(ctx, in[0]);
    if (op == "abs")     return abs_op(ctx, in[0]);
    if (op == "sqr")     return sqr(ctx, in[0]);
    if (op == "sqrt")    return sqrt_op(ctx, in[0]);
    if (op == "exp")     return exp_op(ctx, in[0]);
    if (op == "log")     return log_op(ctx, in[0]);
    if (op == "sin")     return sin_op(ctx, in[0]);
    if (op == "cos")     return cos_op(ctx, in[0]);
    if (op == "relu")    return relu(ctx, in[0]);
    if (op == "sigmoid") return sigmoid(ctx, in[0]);
    if (op == "tanh")    return tanh_op(ctx, in[0]);
    if (op == "silu")    return silu(ctx, in[0]);
    if (op == "gelu")    return gelu(ctx, in[0]);
    if (op == "gelu_erf") return gelu_erf(ctx, in[0]);
    if (op == "softplus") return softplus(ctx, in[0]);
    if (op == "hardswish") return hardswish(ctx, in[0]);
    if (op == "leaky_relu") return leaky_relu(ctx, in[0], Af(0));
    if (op == "clamp")   return clamp(ctx, in[0], Af(0), Af(1));
    if (op == "scale")   return scale(ctx, in[0], Af(0));

    // 线代
    if (op == "mul_mat") return mul_mat(ctx, in[0], in[1]);

    // 归约
    if (op == "sum")      return sum(ctx, in[0]);
    if (op == "sum_rows") return sum_rows(ctx, in[0]);
    if (op == "mean")     return mean(ctx, in[0]);

    // 归一化
    if (op == "norm")       return norm(ctx, in[0], Af(0));
    if (op == "rms_norm")   return rms_norm(ctx, in[0], Af(0));
    if (op == "group_norm") return group_norm(ctx, in[0], Ai(0), Af(1));

    // 注意力
    if (op == "soft_max")     return soft_max(ctx, in[0]);
    if (op == "soft_max_ext") return soft_max_ext(ctx, in[0], in[1], Af(0), Af(1));
    if (op == "cross_entropy_loss") return cross_entropy_loss(ctx, in[0], in[1]);

    // 索引
    if (op == "get_rows") return get_rows(ctx, in[0], in[1]);
    if (op == "set_rows") return set_rows(ctx, in[0], in[1], in[2]);

    // 形状扩展
    if (op == "repeat") {
        const int64_t ne[4] = {Ai(0), Ai(1), Ai(2), Ai(3)};
        Tensor* target = new_tensor_nd(ctx, in[0]->type, MAX_DIMS, ne);
        return repeat(ctx, in[0], target);
    }
    if (op == "concat") return concat(ctx, in[0], in[1], Ai(0));
    if (op == "pad")    return pad(ctx, in[0], Ai(0), Ai(1), Ai(2), Ai(3));
    if (op == "pad_reflect") {
        return pad_ext(ctx, in[0], Ai(0), Ai(1), Ai(2), Ai(3), Ai(4), Ai(5), Ai(6), Ai(7),
                       PAD_REFLECT);
    }

    // 卷积
    if (op == "im2col") return im2col(ctx, in[0], in[1], Ai(0), Ai(1), Ai(2), Ai(3), Ai(4), Ai(5), Ai(6) == 1, TYPE_F32);
    if (op == "conv_1d") return conv_1d(ctx, in[0], in[1], Ai(0), Ai(1), Ai(2));
    if (op == "conv_2d") return conv_2d(ctx, in[0], in[1], Ai(0), Ai(1), Ai(2), Ai(3), Ai(4), Ai(5));
    if (op == "col2im_1d") return col2im_1d(ctx, in[0], Ai(0), Ai(1), Ai(2));
    if (op == "conv_transpose_1d") return conv_transpose_1d(ctx, in[0], in[1], Ai(0), Ai(1), Ai(2));
    if (op == "pool_2d") {
        return pool_2d(ctx, in[0], (PoolMode) Ai(0), Ai(1), Ai(2), Ai(3), Ai(4), Ai(5), Ai(6));
    }
    if (op == "conv_transpose_2d") return conv_transpose_2d(ctx, in[0], in[1], Ai(0));
    if (op == "scale_bias") return scale_bias(ctx, in[0], Af(0), Af(1));

    // 视图/形状
    if (op == "transpose_cont") return cont(ctx, transpose(ctx, in[0]));
    if (op == "reshape") {
        return reshape_4d(ctx, in[0], Ai(0), Ai(1), Ai(2), Ai(3));
    }

    // 类型转换
    if (op == "cast") return cast(ctx, cast(ctx, in[0], TYPE_F16), TYPE_F32);

    std::fprintf(stderr, "[黄金测试] 未知算子 %s（用例 %s）\n", op.c_str(), c.name.c_str());
    std::abort();
}

void tolerance_for(const std::string& op, double& atol, double& rtol) {
    if (op == "cast") {
        atol = 1e-3;
        rtol = 2e-3;
    } else if (op == "conv_1d" || op == "conv_2d" || op == "im2col" || op == "col2im_1d" ||
               op == "conv_transpose_1d" || op == "pool_2d" || op == "conv_transpose_2d") {
        atol = 1e-3;
        rtol = 2e-3;
    } else {
        atol = 1e-4;
        rtol = 1e-3;
    }
}

} // namespace

TRC_TEST(golden_cases) {
    const std::string dir = TRC_GOLDEN_DIR;
    const std::string manifest = dir + "/cases.txt";

    std::vector<Case> cases;
    if (!load_manifest(manifest, cases)) {
        std::printf("    跳过：未找到黄金数据 %s\n    请先运行: python scripts/gen_golden.py\n",
                    manifest.c_str());
        return;
    }

    Device* dev = trc_test::test_device();
    if (dev == nullptr) {
        std::printf("    跳过：TRC_TEST_DEVICE=%s 设备不可用\n",
                    std::getenv("TRC_TEST_DEVICE") != nullptr ? std::getenv("TRC_TEST_DEVICE") : "cpu");
        return;
    }

    int failed_cases  = 0;
    int skipped_cases = 0;
    for (const Case& c : cases) {
        Context* ctx = context_new(1 << 22);
        Graph*   graph = graph_new(ctx);

        std::vector<Tensor*> ins;
        for (const InputSpec& spec : c.inputs) {
            ins.push_back(new_tensor_nd(ctx, spec.type, MAX_DIMS, spec.ne));
        }

        Tensor* out = build_op(ctx, c, ins);
        Buffer* buf = buffer_alloc_ctx_tensors(ctx, dev->default_buffer_type());
        (void) buf;

        bool load_ok = true;
        std::vector<NpyArray> in_npy(ins.size());
        for (size_t i = 0; i < ins.size(); ++i) {
            const std::string path = dir + "/" + c.name + "_in" + std::to_string(i) + ".npy";
            if (!npy_load(path, in_npy[i]) || in_npy[i].raw.size() != tensor_nbytes(ins[i])) {
                trc_test::add_failure("黄金测试: 输入读取失败 " + path);
                load_ok = false;
                break;
            }
            tensor_set(ins[i], in_npy[i].raw.data(), 0, in_npy[i].raw.size());
        }

        NpyArray out_npy;
        if (load_ok && !npy_load(dir + "/" + c.name + "_out.npy", out_npy)) {
            trc_test::add_failure("黄金测试: 期望输出读取失败 " + c.name + "_out.npy");
            load_ok = false;
        }
        if (!load_ok) {
            ++failed_cases;
            buffer_free(buf);
            graph_free(graph);
            context_free(ctx);
            continue;
        }

        graph_build_forward_expand(ctx, graph, out);

        // 目标后端不支持整图时跳过该用例（如 Vulkan 尚未实现 cast / 归一化 _back 等）
        if (const Tensor* bad = trc_test::first_unsupported_node(graph, dev)) {
            std::printf("    [跳过] %-22s op=%-16s 后端不支持 %s\n", c.name.c_str(), c.op.c_str(),
                        op_name(bad->op));
            ++skipped_cases;
            buffer_free(buf);
            graph_free(graph);
            context_free(ctx);
            continue;
        }

        dev->graph_compute(graph);

        double atol = 0;
        double rtol = 0;
        tolerance_for(c.op, atol, rtol);

        // 形状校验：npy 形状（C 序）应等于 ne 反转
        bool shape_ok = (out_npy.shape.size() == 4);
        for (int d = 0; shape_ok && d < 4; ++d) {
            shape_ok = (out_npy.shape[(size_t) d] == out->ne[3 - d]);
        }

        const int64_t n = tensor_nelements(out);
        double max_abs = 0;
        double max_rel = 0;
        int64_t max_idx = 0;
        int64_t n_bad = 0;
        for (int64_t i = 0; i < n; ++i) {
            const size_t off = tensor_offset_linear(out, i);
            const float got = *(const float*) ((const uint8_t*) out->data + off);
            const float exp = ((const float*) out_npy.raw.data())[(size_t) i];
            const double diff = std::fabs((double) got - exp);
            const double tol = atol + rtol * std::fabs((double) exp);
            if (diff > max_abs) {
                max_abs = diff;
                max_idx = i;
            }
            const double rel = std::fabs((double) exp) > 1e-12 ? diff / std::fabs((double) exp) : diff;
            if (rel > max_rel) {
                max_rel = rel;
            }
            if (diff > tol) {
                ++n_bad;
            }
        }

        const bool ok = shape_ok && (n_bad == 0);
        if (!ok) {
            char msg[512];
            const float exp_v = ((const float*) out_npy.raw.data())[(size_t) max_idx];
            const size_t off = tensor_offset_linear(out, max_idx);
            const float got_v = *(const float*) ((const uint8_t*) out->data + off);
            std::snprintf(msg, sizeof(msg),
                          "黄金用例 %s (op=%s) 失败: 形状%s 超差 %lld/%lld 元素, 最大绝对差 %.3g "
                          "(下标 %lld: 实际 %.6g 期望 %.6g)",
                          c.name.c_str(), c.op.c_str(), shape_ok ? "OK" : "错误", (long long) n_bad,
                          (long long) n, max_abs, (long long) max_idx, (double) got_v, (double) exp_v);
            trc_test::add_failure(msg);
            ++failed_cases;
        } else {
            std::printf("    [ OK ] %-22s op=%-16s max_abs=%.3g\n", c.name.c_str(), c.op.c_str(), max_abs);
        }

        buffer_free(buf);
        graph_free(graph);
        context_free(ctx);
    }

    std::printf("    黄金用例: %zu 个, 失败 %d 个, 跳过 %d 个（设备 %s）\n", cases.size(), failed_cases,
                skipped_cases, dev->props().name.c_str());
    TRC_EXPECT(failed_cases == 0);
}
