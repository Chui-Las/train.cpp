#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GGUF 独立校验脚本（仅标准库，无第三方依赖）。

用途：独立校验 train.cpp（或任意 ggml 生态工具）写出的 GGUF 文件结构：
  - 文件头 24 字节：magic/version/tensor_count/metadata_kv_count
  - KV 区：key（u64 长度 + UTF-8）+ 类型（int32）+ 值；ARRAY 带元素类型与个数
  - tensor info 区：name + n_dims + ne[0..]（不反转）+ type + offset（相对数据段）
  - 数据段：按 general.alignment（默认 32）对齐、offset 连续无洞、数据在文件范围内

用法：
  python scripts/verify_gguf.py file.gguf [more.gguf ...] [--hash]

退出码：全部通过 0；任一文件失败 1。

与 C++ 自检（gguf_validate）的输出可相互印证；如需与 ggml 本体交叉验证，见 tools/ggml_crosscheck。
"""
import argparse
import struct
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

GGUF_MAGIC = b"GGUF"
GGUF_VERSION_MIN = 2
GGUF_VERSION_MAX = 3
GGUF_MAX_DIMS = 4
GGML_MAX_NAME = 64
GGUF_DEFAULT_ALIGNMENT = 32

# type id -> (block_size, type_bytes, name)；与 ggml enum ggml_type / trc_types.h 对齐
TYPE_INFO = {
    0: (1, 4, "f32"),
    1: (1, 2, "f16"),
    2: (32, 18, "q4_0"),
    3: (32, 20, "q4_1"),
    6: (32, 22, "q5_0"),
    7: (32, 24, "q5_1"),
    8: (32, 34, "q8_0"),
    9: (32, 36, "q8_1"),
    10: (256, 84, "q2_K"),
    11: (256, 110, "q3_K"),
    12: (256, 144, "q4_K"),
    13: (256, 176, "q5_K"),
    14: (256, 210, "q6_K"),
    15: (256, 292, "q8_K"),
    16: (256, 66, "iq2_xxs"),
    17: (256, 74, "iq2_xs"),
    18: (256, 98, "iq3_xxs"),
    19: (256, 50, "iq1_s"),
    20: (32, 18, "iq4_nl"),
    21: (256, 110, "iq3_s"),
    22: (256, 82, "iq2_s"),
    23: (256, 136, "iq4_xs"),
    24: (1, 1, "i8"),
    25: (1, 2, "i16"),
    26: (1, 4, "i32"),
    27: (1, 8, "i64"),
    28: (1, 8, "f64"),
    29: (256, 56, "iq1_m"),
    30: (1, 2, "bf16"),
    34: (256, 54, "tq1_0"),
    35: (256, 66, "tq2_0"),
    39: (32, 17, "mxfp4"),
    40: (64, 36, "nvfp4"),
    41: (128, 18, "q1_0"),
    42: (64, 18, "q2_0"),
}

GGUF_TYPE_NAMES = {
    0: "u8", 1: "i8", 2: "u16", 3: "i16", 4: "u32", 5: "i32", 6: "f32",
    7: "bool", 8: "str", 9: "arr", 10: "u64", 11: "i64", 12: "f64",
}

# 各标量类型的字节数（BOOL 落盘 1 字节）
SCALAR_SIZES = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}


class ParseError(Exception):
    pass


class Reader:
    def __init__(self, data):
        self.data = data
        self.pos = 0

    def remaining(self):
        return len(self.data) - self.pos

    def read(self, n):
        if self.pos + n > len(self.data):
            raise ParseError("文件在偏移 %d 处截断（还需 %d 字节）" % (self.pos, n))
        b = self.data[self.pos:self.pos + n]
        self.pos += n
        return b

    def u8(self):
        return self.read(1)[0]

    def u16(self):
        return struct.unpack("<H", self.read(2))[0]

    def u32(self):
        return struct.unpack("<I", self.read(4))[0]

    def i32(self):
        return struct.unpack("<i", self.read(4))[0]

    def u64(self):
        return struct.unpack("<Q", self.read(8))[0]

    def i64(self):
        return struct.unpack("<q", self.read(8))[0]

    def f32(self):
        return struct.unpack("<f", self.read(4))[0]

    def f64(self):
        return struct.unpack("<d", self.read(8))[0]

    def string(self):
        n = self.u64()
        if n > self.remaining():
            raise ParseError("字符串长度 %d 超出文件剩余大小" % n)
        return self.read(n).decode("utf-8", errors="strict")


def align_up(v, a):
    return (v + a - 1) & ~(a - 1)


def read_scalar(r, tid):
    if tid == 0:
        return r.u8()
    if tid == 1:
        return struct.unpack("<b", r.read(1))[0]
    if tid == 2:
        return r.u16()
    if tid == 3:
        return struct.unpack("<h", r.read(2))[0]
    if tid == 4:
        return r.u32()
    if tid == 5:
        return r.i32()
    if tid == 6:
        return r.f32()
    if tid == 7:
        return r.u8() != 0
    if tid == 10:
        return r.u64()
    if tid == 11:
        return r.i64()
    if tid == 12:
        return r.f64()
    raise ParseError("未知标量类型 id %d" % tid)


def read_kv(r):
    key = r.string()
    tid = r.i32()
    if tid == 8:  # STRING
        return key, tid, r.string()
    if tid == 9:  # ARRAY
        et = r.i32()
        n = r.u64()
        if et == 9:
            raise ParseError("不支持嵌套数组（key=%s）" % key)
        if et == 8:
            vals = [r.string() for _ in range(n)]
        else:
            size = SCALAR_SIZES.get(et)
            if size is None:
                raise ParseError("数组元素类型未知 id %d（key=%s）" % (et, key))
            if n * size > r.remaining():
                raise ParseError("数组元素个数 %d 超出文件剩余大小（key=%s）" % (n, key))
            vals = [read_scalar(r, et) for _ in range(n)]
        return key, tid, (et, vals)
    if tid not in SCALAR_SIZES:
        raise ParseError("未知元数据类型 id %d（key=%s）" % (tid, key))
    return key, tid, read_scalar(r, tid)


def tensor_nbytes(ne, blck, es):
    if ne[0] % blck != 0:
        raise ParseError("ne[0]=%d 不是块大小 %d 的整数倍" % (ne[0], blck))
    nb = [0, 0, 0, 0]
    nb[0] = es
    for i in range(1, GGUF_MAX_DIMS):
        nb[i] = nb[i - 1] * (ne[i - 1] // blck)
    nbytes = (ne[0] // blck) * es
    for i in range(1, GGUF_MAX_DIMS):
        if ne[i] > 1:
            nbytes += (ne[i] - 1) * nb[i]
    return nbytes


def fnv1a64(data):
    h = 0xCBF29CE484222325
    for b in data:
        h ^= b
        h = (h * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return h


def verify_file(path, show_hash):
    data = open(path, "rb").read()
    r = Reader(data)

    if r.read(4) != GGUF_MAGIC:
        raise ParseError("magic 不是 GGUF")
    version = r.u32()
    if version < GGUF_VERSION_MIN or version > GGUF_VERSION_MAX:
        raise ParseError("不支持的版本 %d（支持 v2/v3）" % version)
    n_tensors = r.u64()
    n_kv = r.u64()

    kv_map = {}
    for _ in range(n_kv):
        key, tid, val = read_kv(r)
        if key == "":
            raise ParseError("元数据 key 为空")
        if key in kv_map:
            raise ParseError("元数据 key 重复: %s" % key)
        kv_map[key] = (tid, val)

    alignment = GGUF_DEFAULT_ALIGNMENT
    if "general.alignment" in kv_map:
        tid, val = kv_map["general.alignment"]
        if tid != 4 or val == 0 or (val & (val - 1)) != 0:
            raise ParseError("general.alignment 非法: %r" % (val,))
        alignment = val

    tensors = []
    names = set()
    expected_off = 0
    for _ in range(n_tensors):
        name = r.string()
        if name == "":
            raise ParseError("张量名为空")
        if len(name.encode("utf-8")) >= GGML_MAX_NAME:
            raise ParseError("张量名过长（>= %d 字节）: %s" % (GGML_MAX_NAME, name))
        if name in names:
            raise ParseError("张量名重复: %s" % name)
        names.add(name)

        n_dims = r.u32()
        if n_dims < 1 or n_dims > GGUF_MAX_DIMS:
            raise ParseError("张量 %s 的 n_dims 非法: %d" % (name, n_dims))
        ne = [1, 1, 1, 1]
        for i in range(n_dims):
            ne[i] = r.i64()
            if ne[i] <= 0:
                raise ParseError("张量 %s 的 ne[%d] 非法: %d" % (name, i, ne[i]))

        tid = r.i32()
        if tid not in TYPE_INFO:
            raise ParseError("张量 %s 的类型 id %d 未知" % (name, tid))
        blck, es, tname = TYPE_INFO[tid]

        offset = r.u64()
        if offset != expected_off:
            raise ParseError("张量 %s 的 offset=%d 不连续（期望 %d）" % (name, offset, expected_off))

        nbytes = tensor_nbytes(ne, blck, es)
        expected_off += align_up(nbytes, alignment)
        tensors.append({
            "name": name, "type": tid, "type_name": tname, "ne": ne, "n_dims": n_dims,
            "offset": offset, "nbytes": nbytes,
        })

    data_offset = align_up(r.pos, alignment)
    if data_offset > len(data):
        raise ParseError("数据段偏移 %d 超出文件大小 %d" % (data_offset, len(data)))
    data_size = len(data) - data_offset
    if tensors:
        last_end = tensors[-1]["offset"] + tensors[-1]["nbytes"]
        if data_offset + last_end > len(data):
            raise ParseError("张量数据段越界（需要 %d，文件 %d）" % (data_offset + last_end, len(data)))

    print("[OK] %s" % path)
    print("  版本=%d 对齐=%d KV=%d 张量=%d 文件=%d 数据段(off=%d size=%d)" %
          (version, alignment, n_kv, n_tensors, len(data), data_offset, data_size))
    for t in tensors:
        line = "  %-24s %-6s ne=%s offset=%d nbytes=%d" % (
            t["name"], t["type_name"], t["ne"][:t["n_dims"]], t["offset"], t["nbytes"])
        if show_hash:
            blob = data[data_offset + t["offset"]: data_offset + t["offset"] + t["nbytes"]]
            line += " fnv1a64=0x%016x" % fnv1a64(blob)
        print(line)
    return True


def main():
    ap = argparse.ArgumentParser(description="GGUF 结构校验（无第三方依赖）")
    ap.add_argument("files", nargs="+", help="待校验的 .gguf 文件")
    ap.add_argument("--hash", action="store_true", help="打印每个张量数据的 FNV-1a 64 哈希")
    args = ap.parse_args()

    failed = 0
    for path in args.files:
        try:
            verify_file(path, args.hash)
        except (ParseError, OSError, struct.error, UnicodeDecodeError) as exc:
            print("[FAIL] %s: %s" % (path, exc))
            failed += 1
        print()
    if failed:
        print("校验失败：%d/%d 个文件" % (failed, len(args.files)))
        return 1
    print("全部通过：%d 个文件" % len(args.files))
    return 0


if __name__ == "__main__":
    sys.exit(main())
