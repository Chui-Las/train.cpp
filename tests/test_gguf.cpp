// train.cpp - GGUF 读写测试（M1.6a）
//
// 覆盖：
//   - 全部元数据类型与数组的 round-trip（含 UTF-8 字符串）
//   - 张量 round-trip（F32/F16/I32）、F32->F16 转换、原始字节布局与 32 字节对齐
//   - 量化反量化（Q4_0/Q4_1/Q5_0/Q5_1/Q8_0，手算期望值）
//   - load_data=false（仅元数据）、载入已有张量、按文件形状创建张量
//   - 损坏文件与缺文件容错（返回 nullptr，不崩溃）
#include "test_util.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

using namespace traincpp;

namespace {

std::string gguf_tmp_path(const char* name) {
    const std::filesystem::path dir = std::filesystem::temp_directory_path();
    return (dir / (std::string("trc_gguf_test_") + name + ".gguf")).string();
}

void write_file_bytes(const std::string& path, const std::vector<uint8_t>& bytes) {
    std::FILE* fp = std::fopen(path.c_str(), "wb");
    if (fp == nullptr) {
        TRC_EXPECT(false && "无法创建临时文件");
        return;
    }
    if (!bytes.empty()) {
        std::fwrite(bytes.data(), 1, bytes.size(), fp);
    }
    std::fclose(fp);
}

std::vector<uint8_t> read_file_bytes(const std::string& path) {
    std::vector<uint8_t> bytes;
    std::FILE* fp = std::fopen(path.c_str(), "rb");
    if (fp == nullptr) {
        return bytes;
    }
    std::fseek(fp, 0, SEEK_END);
    const long size = std::ftell(fp);
    std::fseek(fp, 0, SEEK_SET);
    if (size > 0) {
        bytes.resize((size_t) size);
        std::fread(bytes.data(), 1, bytes.size(), fp);
    }
    std::fclose(fp);
    return bytes;
}

// GGUF 原始字节构造器（量化/损坏文件用例；仅小端）
struct ByteBuilder {
    std::vector<uint8_t> v;

    void raw(const void* p, size_t n) {
        const uint8_t* b = (const uint8_t*) p;
        v.insert(v.end(), b, b + n);
    }
    void u8(uint8_t x)   { v.push_back(x); }
    void u16(uint16_t x) { raw(&x, 2); }
    void u32(uint32_t x) { raw(&x, 4); }
    void u64(uint64_t x) { raw(&x, 8); }
    void i64(int64_t x)  { raw(&x, 8); }
    void str(const std::string& s) {
        u64((uint64_t) s.size());
        raw(s.data(), s.size());
    }
    void pad32() {
        while (v.size() % 32 != 0) v.push_back(0);
    }
    void header(uint64_t n_tensors, uint64_t n_kv) {
        raw("GGUF", 4);
        u32(3);
        u64(n_tensors);
        u64(n_kv);
    }
    void tensor_info(const std::string& name, Type type, const std::vector<int64_t>& ne, uint64_t offset) {
        str(name);
        u32((uint32_t) ne.size());
        for (int64_t d : ne) {
            i64(d);
        }
        u32((uint32_t) type);
        u64(offset);
    }
    void kv_u32(const std::string& key, uint32_t value) {
        str(key);
        u32((uint32_t) GgufType::U32);
        u32(value);
    }
};

std::vector<uint8_t> make_quant_file(Type type, const std::vector<uint8_t>& block_bytes, int64_t ne0) {
    ByteBuilder b;
    b.header(1, 0);
    b.tensor_info("w", type, {ne0}, 0);
    b.pad32();
    b.raw(block_bytes.data(), block_bytes.size());
    b.pad32();
    return b.v;
}

void check_dequant(const char* name, Type type, const std::vector<uint8_t>& blocks, int64_t ne0,
                   const std::vector<float>& expect) {
    const std::string path = gguf_tmp_path(name);
    write_file_bytes(path, make_quant_file(type, blocks, ne0));

    GgufFile* f = gguf_open(path.c_str());
    TRC_EXPECT(f != nullptr);
    if (f != nullptr) {
        TRC_EXPECT_EQ(gguf_tensor_count(f), 1);
        std::vector<float> got(expect.size(), 0.0f);
        const bool ok = gguf_tensor_to_f32(f, "w", got.data(), (int64_t) got.size());
        TRC_EXPECT(ok);
        if (ok) {
            for (size_t i = 0; i < expect.size(); ++i) {
                TRC_EXPECT_NEAR(got[i], expect[i], 1e-6);
            }
        }
        gguf_close(f);
    }
    std::remove(path.c_str());
}

} // namespace

// ---------------------------------------------------------------- 元数据

TRC_TEST(gguf_meta_roundtrip) {
    const std::string path = gguf_tmp_path("meta");

    GgufWriter* w = gguf_writer_new();
    gguf_writer_set_arch(w, "traincpp.test");
    gguf_writer_set_u8(w, "t.u8", 200);
    gguf_writer_set_i8(w, "t.i8", -100);
    gguf_writer_set_u16(w, "t.u16", 60000);
    gguf_writer_set_i16(w, "t.i16", -30000);
    gguf_writer_set_u32(w, "t.u32", 4000000000u);
    gguf_writer_set_i32(w, "t.i32", -2000000000);
    gguf_writer_set_f32(w, "t.f32", 3.25f);
    gguf_writer_set_bool(w, "t.bool", true);
    gguf_writer_set_u64(w, "t.u64", 12345678901234567890ull);
    gguf_writer_set_i64(w, "t.i64", -1234567890123456789ll);
    gguf_writer_set_f64(w, "t.f64", 2.5);
    gguf_writer_set_str(w, "t.str", "中文 UTF-8 / ascii");
    const int32_t a32[3] = {1, -2, 3};
    gguf_writer_set_arr_i32(w, "t.arr_i32", a32, 3);
    const float af[2] = {0.5f, -1.5f};
    gguf_writer_set_arr_f32(w, "t.arr_f32", af, 2);
    const int64_t a64[3] = {1, -5, 1099511627776ll};  // 1 << 40
    gguf_writer_set_arr_i64(w, "t.arr_i64", a64, 3);
    const char* astr[2] = {"alpha", "贝塔"};
    gguf_writer_set_arr_str(w, "t.arr_str", astr, 2);
    // 同名 key 覆盖：值取最后一次、计数不增长
    gguf_writer_set_u32(w, "t.u32", 7u);
    TRC_EXPECT(gguf_writer_write(w, path.c_str()));
    TRC_EXPECT_EQ(gguf_writer_tensor_count(w), 0);
    gguf_writer_free(w);

    GgufFile* f = gguf_open(path.c_str());
    TRC_EXPECT(f != nullptr);
    if (f == nullptr) {
        std::remove(path.c_str());
        return;
    }

    TRC_EXPECT_EQ(gguf_version(f), 3);
    TRC_EXPECT_EQ(gguf_alignment(f), 32);
    TRC_EXPECT_EQ(gguf_kv_count(f), 17);

    TRC_EXPECT_EQ(gguf_get_u8(f, "t.u8"), 200);
    TRC_EXPECT_EQ(gguf_get_i8(f, "t.i8"), -100);
    TRC_EXPECT_EQ(gguf_get_u16(f, "t.u16"), 60000);
    TRC_EXPECT_EQ(gguf_get_i16(f, "t.i16"), -30000);
    TRC_EXPECT_EQ(gguf_get_u32(f, "t.u32"), 7);
    TRC_EXPECT_EQ(gguf_get_i32(f, "t.i32"), -2000000000);
    TRC_EXPECT_NEAR(gguf_get_f32(f, "t.f32"), 3.25, 1e-9);
    TRC_EXPECT(gguf_get_bool(f, "t.bool"));
    TRC_EXPECT(gguf_get_u64(f, "t.u64") == 12345678901234567890ull);
    TRC_EXPECT_EQ(gguf_get_i64(f, "t.i64"), -1234567890123456789ll);
    TRC_EXPECT_NEAR(gguf_get_f64(f, "t.f64"), 2.5, 1e-12);
    TRC_EXPECT(std::string(gguf_get_str(f, "t.str")) == "中文 UTF-8 / ascii");
    TRC_EXPECT(std::string(gguf_get_str(f, "general.architecture")) == "traincpp.test");

    TRC_EXPECT(gguf_has_key(f, "t.u8"));
    TRC_EXPECT(!gguf_has_key(f, "t.nope"));

    int64_t n = 0;
    const int32_t* p32 = gguf_get_arr_i32(f, "t.arr_i32", &n);
    TRC_EXPECT(p32 != nullptr && n == 3);
    if (p32 != nullptr) {
        TRC_EXPECT_EQ(p32[0], 1);
        TRC_EXPECT_EQ(p32[1], -2);
        TRC_EXPECT_EQ(p32[2], 3);
    }
    const float* pf = gguf_get_arr_f32(f, "t.arr_f32", &n);
    TRC_EXPECT(pf != nullptr && n == 2);
    if (pf != nullptr) {
        TRC_EXPECT_NEAR(pf[0], 0.5, 1e-9);
        TRC_EXPECT_NEAR(pf[1], -1.5, 1e-9);
    }
    TRC_EXPECT(std::string(gguf_get_arr_str(f, "t.arr_str", 1)) == "贝塔");
    const int64_t* p64 = gguf_get_arr_i64(f, "t.arr_i64", &n);
    TRC_EXPECT(p64 != nullptr && n == 3);
    if (p64 != nullptr) {
        TRC_EXPECT(p64[0] == 1 && p64[1] == -5 && p64[2] == 1099511627776ll);
    }

    // 按索引访问
    const int64_t idx = gguf_find_key(f, "t.arr_str");
    TRC_EXPECT(idx >= 0);
    TRC_EXPECT_EQ(gguf_kv_type(f, idx), (int64_t) GgufType::ARRAY);
    TRC_EXPECT_EQ(gguf_arr_type(f, idx), (int64_t) GgufType::STRING);
    TRC_EXPECT_EQ(gguf_arr_count(f, idx), 2);
    TRC_EXPECT(std::string(gguf_key(f, idx)) == "t.arr_str");

    gguf_close(f);
    std::remove(path.c_str());
}

// ---------------------------------------------------------------- 张量

TRC_TEST(gguf_tensor_roundtrip) {
    Device* dev = device_cpu();
    TRC_EXPECT(dev != nullptr);
    if (dev == nullptr) {
        return;
    }

    Context* ctx = context_new(1 << 20);
    Tensor*  a   = new_tensor_2d(ctx, TYPE_F32, 5, 4);
    Tensor*  b   = new_tensor_1d(ctx, TYPE_I32, 6);
    tensor_set_name(a, "a");
    tensor_set_name(b, "b");
    Buffer* buf = buffer_alloc_ctx_tensors(ctx, dev->default_buffer_type());
    TRC_EXPECT(buf != nullptr);

    std::vector<float> av(20);
    for (int i = 0; i < 20; ++i) {
        av[(size_t) i] = (float) i * 0.5f - 2.0f;  // 均可被 F16 精确表示
    }
    std::vector<int32_t> bv(6);
    for (int i = 0; i < 6; ++i) {
        bv[(size_t) i] = i - 3;
    }
    tensor_set(a, av.data(), 0, av.size() * sizeof(float));
    tensor_set(b, bv.data(), 0, bv.size() * sizeof(int32_t));

    const std::string path = gguf_tmp_path("tensors");
    GgufWriter* w = gguf_writer_new();
    gguf_writer_set_arch(w, "traincpp.test");
    gguf_writer_add_tensor(w, "a", a, TYPE_F32);
    gguf_writer_add_tensor(w, "a16", a, TYPE_F16);
    gguf_writer_add_tensor(w, "b", b, TYPE_I32);
    TRC_EXPECT_EQ(gguf_writer_tensor_count(w), 3);
    TRC_EXPECT(gguf_writer_write(w, path.c_str()));
    gguf_writer_free(w);

    GgufFile* f = gguf_open(path.c_str());
    TRC_EXPECT(f != nullptr);
    if (f != nullptr) {
        TRC_EXPECT_EQ(gguf_tensor_count(f), 3);

        const int64_t ia = gguf_find_tensor(f, "a");
        const int64_t ih = gguf_find_tensor(f, "a16");
        const int64_t ib = gguf_find_tensor(f, "b");
        TRC_EXPECT(ia >= 0 && ih >= 0 && ib >= 0);
        TRC_EXPECT_EQ(gguf_find_tensor(f, "nope"), -1);
        TRC_EXPECT_EQ(gguf_tensor_nelements(f, ia), 20);

        const GgufTensorInfo* sa = gguf_tensor_info(f, ia);
        const GgufTensorInfo* sh = gguf_tensor_info(f, ih);
        const GgufTensorInfo* sb = gguf_tensor_info(f, ib);
        TRC_EXPECT(sa->type == TYPE_F32 && sa->ne[0] == 5 && sa->ne[1] == 4);
        TRC_EXPECT_EQ(sa->nbytes, 80);
        TRC_EXPECT(sh->type == TYPE_F16 && sh->ne[0] == 5);
        TRC_EXPECT_EQ(sh->nbytes, 40);
        TRC_EXPECT(sb->type == TYPE_I32);
        TRC_EXPECT_EQ(sb->nbytes, 24);

        // 数据段与张量偏移：32 字节对齐、连续无洞
        TRC_EXPECT_EQ(gguf_data_offset(f) % 32, 0);
        TRC_EXPECT_EQ(sa->offset, 0);
        TRC_EXPECT_EQ(sh->offset, 96);   // align_up(80, 32)
        TRC_EXPECT_EQ(sb->offset, 160);  // align_up(96 + 40, 32)

        // 原始字节布局：数据段起点 + 偏移处就是源张量字节（小端）
        const std::vector<uint8_t> raw = read_file_bytes(path);
        const size_t data_off = gguf_data_offset(f);
        TRC_EXPECT(raw.size() >= data_off + sb->offset + sb->nbytes);
        if (raw.size() >= data_off + sb->offset + sb->nbytes) {
            TRC_EXPECT(std::memcmp(raw.data() + data_off + sa->offset, av.data(),
                                   av.size() * sizeof(float)) == 0);
            TRC_EXPECT(std::memcmp(raw.data() + data_off + sb->offset, bv.data(),
                                   bv.size() * sizeof(int32_t)) == 0);
        }

        // 读出（F32 原样 / F16 转换 / I32 转 F32）
        std::vector<float> got(20, 0.0f);
        TRC_EXPECT(gguf_tensor_to_f32(f, "a", got.data(), 20));
        for (size_t i = 0; i < av.size(); ++i) {
            TRC_EXPECT(got[i] == av[i]);
        }
        std::fill(got.begin(), got.end(), 0.0f);
        TRC_EXPECT(gguf_tensor_to_f32(f, "a16", got.data(), 20));
        for (size_t i = 0; i < av.size(); ++i) {
            TRC_EXPECT(got[i] == av[i]);
        }
        std::vector<float> gotb(6, 0.0f);
        TRC_EXPECT(gguf_tensor_to_f32(f, "b", gotb.data(), 6));
        for (size_t i = 0; i < bv.size(); ++i) {
            TRC_EXPECT(gotb[i] == (float) bv[i]);
        }

        // 按文件形状创建张量并载入（no_alloc=false，数据随 ctx 分配）
        ContextParams params;
        params.mem_size = 1 << 20;
        params.no_alloc = false;
        Context* ctx2 = context_new(params);
        Tensor*  a2   = gguf_new_tensor(ctx2, f, "a", TYPE_F32);
        Tensor*  h2   = gguf_new_tensor(ctx2, f, "a16", TYPE_F32);
        Tensor*  b2   = gguf_new_tensor(ctx2, f, "b", TYPE_I32);
        TRC_EXPECT(a2 != nullptr && h2 != nullptr && b2 != nullptr);
        if (a2 != nullptr && h2 != nullptr && b2 != nullptr) {
            TRC_EXPECT(std::string(a2->name) == "a");
            TRC_EXPECT(gguf_load_tensor(f, "a", a2));
            TRC_EXPECT(gguf_load_tensor(f, "a16", h2));
            TRC_EXPECT(gguf_load_tensor(f, "b", b2));
            for (size_t i = 0; i < av.size(); ++i) {
                TRC_EXPECT(tensor_data_f32(a2)[i] == av[i]);
                TRC_EXPECT(tensor_data_f32(h2)[i] == av[i]);
            }
            const int32_t* pb = (const int32_t*) b2->data;
            for (size_t i = 0; i < bv.size(); ++i) {
                TRC_EXPECT_EQ(pb[i], bv[i]);
            }
        }
        context_free(ctx2);
        gguf_close(f);
    }

    buffer_free(buf);
    context_free(ctx);
    std::remove(path.c_str());
}

// ---------------------------------------------------------------- 量化反量化（手算）

TRC_TEST(gguf_dequant_q4_0) {
    // block[i]: d=0.5；qs[j] 低四位=j、高四位=15-j
    ByteBuilder blk;
    blk.u16(0x3800);
    for (int j = 0; j < 16; ++j) {
        blk.u8((uint8_t) ((j & 0x0F) | ((15 - j) << 4)));
    }
    // 第二块：d=0.25；低四位=15-j、高四位=j
    blk.u16(0x3400);
    for (int j = 0; j < 16; ++j) {
        blk.u8((uint8_t) (((15 - j) & 0x0F) | (j << 4)));
    }

    std::vector<float> expect(64);
    for (int j = 0; j < 16; ++j) {
        expect[(size_t) j]      = (float) (j - 8) * 0.5f;
        expect[(size_t) (16 + j)] = (float) (7 - j) * 0.5f;
        expect[(size_t) (32 + j)] = (float) (7 - j) * 0.25f;
        expect[(size_t) (48 + j)] = (float) (j - 8) * 0.25f;
    }
    check_dequant("q4_0", TYPE_Q4_0, blk.v, 64, expect);
}

TRC_TEST(gguf_dequant_q4_1) {
    ByteBuilder blk;
    blk.u16(0x3400);  // d = 0.25
    blk.u16(0xBC00);  // m = -1.0
    for (int j = 0; j < 16; ++j) {
        blk.u8((uint8_t) ((j & 0x0F) | ((15 - j) << 4)));
    }
    std::vector<float> expect(32);
    for (int j = 0; j < 16; ++j) {
        expect[(size_t) j]        = (float) j * 0.25f - 1.0f;
        expect[(size_t) (16 + j)] = (float) (15 - j) * 0.25f - 1.0f;
    }
    check_dequant("q4_1", TYPE_Q4_1, blk.v, 32, expect);
}

TRC_TEST(gguf_dequant_q5_0) {
    ByteBuilder blk;
    // 块 0：qh=0（高第 5 位为 0）
    blk.u16(0x3C00);  // d = 1.0
    blk.u32(0x00000000u);
    for (int j = 0; j < 16; ++j) {
        blk.u8((uint8_t) ((j & 0x0F) | ((15 - j) << 4)));
    }
    // 块 1：qh=全 1（所有高第 5 位置位）
    blk.u16(0x3C00);
    blk.u32(0xFFFFFFFFu);
    for (int j = 0; j < 16; ++j) {
        blk.u8((uint8_t) ((j & 0x0F) | ((15 - j) << 4)));
    }
    std::vector<float> expect(64);
    for (int j = 0; j < 16; ++j) {
        expect[(size_t) j]        = (float) (j - 16);
        expect[(size_t) (16 + j)] = (float) (-1 - j);
        expect[(size_t) (32 + j)] = (float) j;
        expect[(size_t) (48 + j)] = (float) (15 - j);
    }
    check_dequant("q5_0", TYPE_Q5_0, blk.v, 64, expect);
}

TRC_TEST(gguf_dequant_q5_1) {
    ByteBuilder blk;
    blk.u16(0x3800);  // d = 0.5
    blk.u16(0x3000);  // m = 0.125
    blk.u32(0x00000000u);
    for (int j = 0; j < 16; ++j) {
        blk.u8((uint8_t) ((j & 0x0F) | ((15 - j) << 4)));
    }
    std::vector<float> expect(32);
    for (int j = 0; j < 16; ++j) {
        expect[(size_t) j]        = (float) j * 0.5f + 0.125f;
        expect[(size_t) (16 + j)] = (float) (15 - j) * 0.5f + 0.125f;
    }
    check_dequant("q5_1", TYPE_Q5_1, blk.v, 32, expect);
}

TRC_TEST(gguf_dequant_q8_0) {
    ByteBuilder blk;
    blk.u16(0x3000);  // d = 0.125
    for (int j = 0; j < 32; ++j) {
        blk.u8((uint8_t) (int8_t) (j - 16));
    }
    std::vector<float> expect(32);
    for (int j = 0; j < 32; ++j) {
        expect[(size_t) j] = (float) (j - 16) * 0.125f;
    }
    check_dequant("q8_0", TYPE_Q8_0, blk.v, 32, expect);
}

// ---------------------------------------------------------------- 仅元数据 / 损坏文件

TRC_TEST(gguf_metadata_only) {
    ByteBuilder b;
    b.header(1, 0);
    b.tensor_info("w", TYPE_F32, {8}, 0);
    b.pad32();
    for (int i = 0; i < 8; ++i) {
        const float x = (float) i;
        b.raw(&x, 4);
    }
    b.pad32();

    const std::string path = gguf_tmp_path("meta_only");
    write_file_bytes(path, b.v);

    GgufFile* f = gguf_open(path.c_str(), false);
    TRC_EXPECT(f != nullptr);
    if (f != nullptr) {
        TRC_EXPECT_EQ(gguf_tensor_count(f), 1);
        std::vector<float> got(8, 0.0f);
        TRC_EXPECT(!gguf_tensor_to_f32(f, "w", got.data(), 8));  // 未加载数据
        gguf_close(f);
    }

    GgufFile* f2 = gguf_open(path.c_str(), true);
    TRC_EXPECT(f2 != nullptr);
    if (f2 != nullptr) {
        std::vector<float> got(8, 0.0f);
        TRC_EXPECT(gguf_tensor_to_f32(f2, "w", got.data(), 8));
        for (int i = 0; i < 8; ++i) {
            TRC_EXPECT(got[(size_t) i] == (float) i);
        }
        gguf_close(f2);
    }
    std::remove(path.c_str());
}

TRC_TEST(gguf_validate_files) {
    const std::string path = gguf_tmp_path("validate");

    // 合法文件（1 个 F32 张量）
    ByteBuilder b;
    b.header(1, 0);
    b.tensor_info("w", TYPE_F32, {8}, 0);
    b.pad32();
    for (int i = 0; i < 8; ++i) {
        const float x = (float) i;
        b.raw(&x, 4);
    }
    b.pad32();
    write_file_bytes(path, b.v);
    TRC_EXPECT(gguf_validate(path.c_str()));

    // 魔数损坏
    {
        std::vector<uint8_t> bad = b.v;
        bad[0] = 'X';
        write_file_bytes(path, bad);
        TRC_EXPECT(!gguf_validate(path.c_str()));
    }

    // 数据段被截断
    {
        write_file_bytes(path, b.v);
        GgufFile* vf = gguf_open(path.c_str());
        TRC_EXPECT(vf != nullptr);
        const size_t data_off = vf != nullptr ? gguf_data_offset(vf) : 0;
        gguf_close(vf);
        std::vector<uint8_t> bad = b.v;
        bad.resize(data_off + 4);  // 只保留 4 字节张量数据（声明为 32 字节）
        write_file_bytes(path, bad);
        TRC_EXPECT(!gguf_validate(path.c_str()));
    }

    // 缺文件
    std::remove(path.c_str());
    TRC_EXPECT(!gguf_validate(path.c_str()));
}

TRC_TEST(gguf_corrupt_rejected) {
    const std::string path = gguf_tmp_path("corrupt");

    const auto make_valid = []() {
        ByteBuilder b;
        b.header(1, 0);
        b.tensor_info("w", TYPE_F32, {8}, 0);
        b.pad32();
        for (int i = 0; i < 8; ++i) {
            const float x = (float) i;
            b.raw(&x, 4);
        }
        b.pad32();
        return b.v;
    };

    // 坏 magic
    {
        std::vector<uint8_t> v = make_valid();
        v[0] = 'X';
        write_file_bytes(path, v);
        TRC_EXPECT(gguf_open(path.c_str()) == nullptr);
    }
    // 版本 v1（不再支持）
    {
        std::vector<uint8_t> v = make_valid();
        v[4] = 1;
        write_file_bytes(path, v);
        TRC_EXPECT(gguf_open(path.c_str()) == nullptr);
    }
    // 版本 v4
    {
        std::vector<uint8_t> v = make_valid();
        v[4] = 4;
        write_file_bytes(path, v);
        TRC_EXPECT(gguf_open(path.c_str()) == nullptr);
    }
    // 张量数据段缺失（文件在 tensor info 后即结束）
    {
        ByteBuilder b;
        b.header(1, 0);
        b.tensor_info("w", TYPE_F32, {8}, 0);
        write_file_bytes(path, b.v);
        TRC_EXPECT(gguf_open(path.c_str()) == nullptr);
    }
    // 张量名重复
    {
        ByteBuilder b;
        b.header(2, 0);
        b.tensor_info("w", TYPE_F32, {8}, 0);
        b.tensor_info("w", TYPE_F32, {8}, 32);
        b.pad32();
        std::vector<uint8_t> zeros(64, 0);
        b.raw(zeros.data(), zeros.size());
        b.pad32();
        write_file_bytes(path, b.v);
        TRC_EXPECT(gguf_open(path.c_str()) == nullptr);
    }
    // 张量 offset 不连续
    {
        ByteBuilder b;
        b.header(2, 0);
        b.tensor_info("a", TYPE_F32, {8}, 0);
        b.tensor_info("b", TYPE_F32, {8}, 0);  // 应为 32
        b.pad32();
        std::vector<uint8_t> zeros(64, 0);
        b.raw(zeros.data(), zeros.size());
        b.pad32();
        write_file_bytes(path, b.v);
        TRC_EXPECT(gguf_open(path.c_str()) == nullptr);
    }
    // 类型尚未实现（本库类型表无此类型，如 IQ2_XXS）
    {
        ByteBuilder b;
        b.header(1, 0);
        b.tensor_info("w", TYPE_IQ2_XXS, {256}, 0);
        b.pad32();
        std::vector<uint8_t> zeros(66, 0);
        b.raw(zeros.data(), zeros.size());
        b.pad32();
        write_file_bytes(path, b.v);
        TRC_EXPECT(gguf_open(path.c_str()) == nullptr);
    }
    // ne[0] 不是块大小的整数倍
    {
        ByteBuilder b;
        b.header(1, 0);
        b.tensor_info("w", TYPE_Q4_0, {30}, 0);
        b.pad32();
        std::vector<uint8_t> zeros(18, 0);
        b.raw(zeros.data(), zeros.size());
        b.pad32();
        write_file_bytes(path, b.v);
        TRC_EXPECT(gguf_open(path.c_str()) == nullptr);
    }
    // 元数据 key 重复
    {
        ByteBuilder b;
        b.header(0, 2);
        b.kv_u32("k", 1);
        b.kv_u32("k", 2);
        b.pad32();
        write_file_bytes(path, b.v);
        TRC_EXPECT(gguf_open(path.c_str()) == nullptr);
    }
    // 文件不存在
    {
        const std::string missing = gguf_tmp_path("missing");
        std::remove(missing.c_str());
        TRC_EXPECT(gguf_open(missing.c_str()) == nullptr);
    }

    std::remove(path.c_str());
}
