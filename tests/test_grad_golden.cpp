// train.cpp - PyTorch 反向梯度对拍（M1.3c）
//
// 数据由 scripts/gen_golden.py 的 build_grad 生成到 tests/golden/：
//   grad_cases.txt        梯度用例清单（DSL，见脚本注释）
//   <name>_in<id>.npy     输入
//   <name>_loss.npy       前向 loss（1 元素）
//   <name>_grad<id>.npy   参数梯度（torch.autograd 参考值）
//
// 若缺少梯度黄金数据，本测试打印"跳过"并返回成功（便于无 Python 环境时通过 CTest）。
#include "npy.h"
#include "test_util.h"
#include "traincpp/traincpp.h"

#include <cmath>
#include <cstdio>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#ifndef TRC_GOLDEN_DIR
#define TRC_GOLDEN_DIR ""
#endif

using namespace traincpp;
using namespace trc_test;

namespace {

struct InSpec {
    int     id = 0;
    Type    type = TYPE_F32;
    int64_t ne[4] = {1, 1, 1, 1};
    bool    param = false;
};

struct NodeSpec {
    int                 id = 0;
    std::string         op;
    std::vector<double> attrs;
    std::vector<int>    srcs;
};

struct GCase {
    std::string           name;
    std::vector<InSpec>   ins;
    std::vector<NodeSpec> nodes;
    int                   loss_id = -1;
};

bool parse_shape(const std::string& s, int64_t ne[4]) {
    int    dim = 0;
    size_t pos = 0;
    while (pos < s.size() && dim < 4) {
        const size_t      x   = s.find('x', pos);
        const std::string num = (x == std::string::npos) ? s.substr(pos) : s.substr(pos, x - pos);
        try {
            ne[dim++] = std::stoll(num);
        } catch (...) {
            return false;
        }
        if (x == std::string::npos) {
            break;
        }
        pos = x + 1;
    }
    for (; dim < 4; ++dim) {
        ne[dim] = 1;
    }
    return dim > 0;
}

bool load_grad_manifest(const std::string& path, std::vector<GCase>& cases) {
    std::ifstream f(path);
    if (!f) {
        return false;
    }

    GCase cur;
    bool  open = false;
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#') {
            continue;
        }
        std::istringstream ss(line);
        std::string kw;
        ss >> kw;
        if (kw == "case") {
            if (open) {
                cases.push_back(std::move(cur));
            }
            cur = GCase();
            int n_inputs = 0;
            ss >> cur.name >> n_inputs;
            open = true;
        } else if (kw == "in") {
            InSpec      in;
            std::string type_str;
            std::string shape_str;
            if (!(ss >> in.id >> type_str >> shape_str)) {
                return false;
            }
            in.type = (type_str == "i32") ? TYPE_I32 : TYPE_F32;
            if (!parse_shape(shape_str, in.ne)) {
                return false;
            }
            cur.ins.push_back(in);
        } else if (kw == "param") {
            int id = 0;
            if (!(ss >> id)) {
                return false;
            }
            bool found = false;
            for (InSpec& in : cur.ins) {
                if (in.id == id) {
                    in.param = true;
                    found = true;
                }
            }
            if (!found) {
                return false;
            }
        } else if (kw == "node") {
            NodeSpec n;
            int      n_attrs = 0;
            int      n_srcs  = 0;
            if (!(ss >> n.id >> n.op >> n_attrs)) {
                return false;
            }
            for (int i = 0; i < n_attrs; ++i) {
                double v = 0;
                if (!(ss >> v)) {
                    return false;
                }
                n.attrs.push_back(v);
            }
            if (!(ss >> n_srcs)) {
                return false;
            }
            for (int i = 0; i < n_srcs; ++i) {
                int s = 0;
                if (!(ss >> s)) {
                    return false;
                }
                n.srcs.push_back(s);
            }
            cur.nodes.push_back(std::move(n));
        } else if (kw == "loss") {
            if (!(ss >> cur.loss_id)) {
                return false;
            }
        }
    }
    if (open) {
        cases.push_back(std::move(cur));
    }
    return !cases.empty();
}

Tensor* build_grad_node(Context* ctx, const NodeSpec& n, const std::map<int, Tensor*>& t) {
    auto S  = [&](int i) { return t.at(n.srcs[(size_t) i]); };
    auto Ai = [&](int i) { return (int) n.attrs[(size_t) i]; };
    auto Af = [&](int i) { return (float) n.attrs[(size_t) i]; };
    const std::string& op = n.op;

    // 逐元素
    if (op == "add")    return add(ctx, S(0), S(1));
    if (op == "sub")    return sub(ctx, S(0), S(1));
    if (op == "mul")    return mul(ctx, S(0), S(1));
    if (op == "div")    return div(ctx, S(0), S(1));
    if (op == "add1")   return add1(ctx, S(0), S(1));
    if (op == "neg")    return neg(ctx, S(0));
    if (op == "abs")    return abs_op(ctx, S(0));
    if (op == "sqr")    return sqr(ctx, S(0));
    if (op == "sqrt")   return sqrt_op(ctx, S(0));
    if (op == "exp")    return exp_op(ctx, S(0));
    if (op == "log")    return log_op(ctx, S(0));
    if (op == "sin")    return sin_op(ctx, S(0));
    if (op == "cos")    return cos_op(ctx, S(0));
    if (op == "relu")   return relu(ctx, S(0));
    if (op == "sigmoid") return sigmoid(ctx, S(0));
    if (op == "tanh")   return tanh_op(ctx, S(0));
    if (op == "silu")   return silu(ctx, S(0));
    if (op == "gelu")   return gelu(ctx, S(0));
    if (op == "gelu_erf") return gelu_erf(ctx, S(0));
    if (op == "softplus") return softplus(ctx, S(0));
    if (op == "hardswish") return hardswish(ctx, S(0));
    if (op == "erf")    return erf_op(ctx, S(0));
    if (op == "leaky_relu") return leaky_relu(ctx, S(0), Af(0));
    if (op == "clamp")  return clamp(ctx, S(0), Af(0), Af(1));
    if (op == "scale")  return scale(ctx, S(0), Af(0));

    // 归约
    if (op == "sum")      return sum(ctx, S(0));
    if (op == "sum_rows") return sum_rows(ctx, S(0));
    if (op == "mean")     return mean(ctx, S(0));

    // 归一化 / 注意力
    if (op == "norm")       return norm(ctx, S(0), Af(0));
    if (op == "rms_norm")   return rms_norm(ctx, S(0), Af(0));
    if (op == "group_norm") return group_norm(ctx, S(0), Ai(0), Af(1));
    if (op == "soft_max")     return soft_max(ctx, S(0));
    if (op == "soft_max_ext") return soft_max_ext(ctx, S(0), S(1), Af(0), Af(1));
    if (op == "cross_entropy_loss") return cross_entropy_loss(ctx, S(0), S(1));

    // 线代 / 索引 / 形状
    if (op == "mul_mat") return mul_mat(ctx, S(0), S(1));
    if (op == "get_rows") return get_rows(ctx, S(0), S(1));
    if (op == "concat") return concat(ctx, S(0), S(1), Ai(0));
    if (op == "pad")    return pad(ctx, S(0), Ai(0), Ai(1), Ai(2), Ai(3));
    if (op == "pad_reflect") {
        return pad_ext(ctx, S(0), Ai(0), Ai(1), Ai(2), Ai(3), Ai(4), Ai(5), Ai(6), Ai(7),
                       PAD_REFLECT);
    }
    if (op == "repeat") {
        const int64_t ne[4] = {Ai(0), Ai(1), Ai(2), Ai(3)};
        Tensor* target = new_tensor_nd(ctx, S(0)->type, MAX_DIMS, ne);
        return repeat(ctx, S(0), target);
    }
    if (op == "reshape") return reshape_4d(ctx, S(0), Ai(0), Ai(1), Ai(2), Ai(3));
    if (op == "transpose_cont") return cont(ctx, transpose(ctx, S(0)));
    if (op == "cont") return cont(ctx, S(0));

    // 卷积
    if (op == "conv_1d") return conv_1d(ctx, S(0), S(1), Ai(0), Ai(1), Ai(2));
    if (op == "conv_2d") return conv_2d(ctx, S(0), S(1), Ai(0), Ai(1), Ai(2), Ai(3), Ai(4), Ai(5));
    if (op == "conv_transpose_1d") return conv_transpose_1d(ctx, S(0), S(1), Ai(0), Ai(1), Ai(2));
    if (op == "pool_2d") {
        return pool_2d(ctx, S(0), (PoolMode) Ai(0), Ai(1), Ai(2), Ai(3), Ai(4), Ai(5), Ai(6));
    }
    if (op == "conv_transpose_2d") return conv_transpose_2d(ctx, S(0), S(1), Ai(0));

    std::fprintf(stderr, "[梯度黄金] 未知算子 %s\n", op.c_str());
    std::abort();
}

// 与期望 npy 逐元素比较（F32，atol=1e-4、rtol=1e-3）
bool compare_f32(const Tensor* t, const std::string& path, std::string& err) {
    NpyArray want;
    if (!npy_load(path, want)) {
        err = "读取失败 " + path;
        return false;
    }
    const int64_t n = tensor_nelements(t);
    if ((size_t) n * sizeof(float) != want.raw.size()) {
        err = "元素数不一致 " + path;
        return false;
    }

    double  max_abs = 0;
    int64_t max_idx = 0;
    int64_t n_bad   = 0;
    for (int64_t i = 0; i < n; ++i) {
        const size_t off = tensor_offset_linear(t, i);
        const float  got = *(const float*) ((const uint8_t*) t->data + off);
        const float  exp = ((const float*) want.raw.data())[(size_t) i];
        const double diff = std::fabs((double) got - exp);
        const double tol  = 1e-4 + 1e-3 * std::fabs((double) exp);
        if (diff > max_abs) {
            max_abs = diff;
            max_idx = i;
        }
        if (diff > tol) {
            ++n_bad;
        }
    }
    if (n_bad != 0) {
        char buf[512];
        const size_t off   = tensor_offset_linear(t, max_idx);
        const float  got_v = *(const float*) ((const uint8_t*) t->data + off);
        const float  exp_v = ((const float*) want.raw.data())[(size_t) max_idx];
        std::snprintf(buf, sizeof(buf),
                      "%s: 超差 %lld/%lld 元素, 最大绝对差 %.3g (下标 %lld: 实际 %.6g 期望 %.6g)",
                      path.c_str(), (long long) n_bad, (long long) n, max_abs, (long long) max_idx,
                      (double) got_v, (double) exp_v);
        err = buf;
        return false;
    }
    return true;
}

} // namespace

TRC_TEST(grad_golden_cases) {
    const std::string dir = TRC_GOLDEN_DIR;

    std::vector<GCase> cases;
    if (!load_grad_manifest(dir + "/grad_cases.txt", cases)) {
        std::printf("    跳过：未找到梯度黄金数据 %s/grad_cases.txt\n"
                    "    请先运行: python scripts/gen_golden.py\n",
                    dir.c_str());
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
    for (const GCase& c : cases) {
        Context* ctx   = context_new(1 << 22);
        Graph*   graph = graph_new(ctx);
        std::map<int, Tensor*> tensors;

        for (const InSpec& in : c.ins) {
            Tensor* t = new_tensor_nd(ctx, in.type, MAX_DIMS, in.ne);
            if (in.param) {
                tensor_set_param(t);
            }
            tensors[in.id] = t;
        }
        for (const NodeSpec& n : c.nodes) {
            tensors[n.id] = build_grad_node(ctx, n, tensors);
        }

        Tensor* loss = tensors.at(c.loss_id);
        tensor_set_loss(loss);
        graph_build_forward_expand(ctx, graph, loss);
        graph_build_backward_expand(graph);

        // 反向图含目标后端不支持的节点时跳过该用例（如 Vulkan 尚未实现归一化 _back）
        if (const Tensor* bad = trc_test::first_unsupported_node(graph, dev)) {
            std::printf("    [跳过] %-22s 后端不支持 %s\n", c.name.c_str(), op_name(bad->op));
            ++skipped_cases;
            graph_free(graph);
            context_free(ctx);
            continue;
        }

        Buffer* buf = buffer_alloc_ctx_tensors(ctx, dev->default_buffer_type());

        bool        ok = true;
        std::string err;
        for (const InSpec& in : c.ins) {
            const std::string path = dir + "/" + c.name + "_in" + std::to_string(in.id) + ".npy";
            NpyArray          a;
            if (!npy_load(path, a) || a.raw.size() != tensor_nbytes(tensors[in.id])) {
                trc_test::add_failure("梯度测试: 输入读取失败 " + path);
                ok = false;
                break;
            }
            tensor_set(tensors[in.id], a.raw.data(), 0, a.raw.size());
        }
        if (!ok) {
            ++failed_cases;
            buffer_free(buf);
            graph_free(graph);
            context_free(ctx);
            continue;
        }

        graph_reset(graph);
        dev->graph_compute(graph);

        if (!compare_f32(loss, dir + "/" + c.name + "_loss.npy", err)) {
            trc_test::add_failure("梯度用例 " + c.name + " loss 失败: " + err);
            ok = false;
        }
        for (const InSpec& in : c.ins) {
            if (!in.param) {
                continue;
            }
            Tensor* gx = graph_get_grad(graph, tensors[in.id]);
            TRC_EXPECT(gx != nullptr);
            if (gx == nullptr) {
                ok = false;
                continue;
            }
            if (!compare_f32(gx, dir + "/" + c.name + "_grad" + std::to_string(in.id) + ".npy", err)) {
                trc_test::add_failure("梯度用例 " + c.name + " 参数 " + std::to_string(in.id) + " 失败: " + err);
                ok = false;
            }
        }

        if (ok) {
            std::printf("    [ OK ] %s\n", c.name.c_str());
        } else {
            ++failed_cases;
        }

        buffer_free(buf);
        graph_free(graph);
        context_free(ctx);
    }

    std::printf("    梯度用例: %zu 个, 失败 %d 个, 跳过 %d 个（设备 %s）\n", cases.size(), failed_cases,
                skipped_cases, dev->props().name.c_str());
    TRC_EXPECT(failed_cases == 0);
}
