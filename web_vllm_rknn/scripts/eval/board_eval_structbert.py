#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""板端（RK3588）StructBERT RKNN 评测：前向输出 + 推理速度。ctypes 直调 librknnrt，
无需 rknn_toolkit_lite2。

用法（板上）：
  python3 board_eval_structbert.py <model.rknn> <tower|pair> <test_dir> <out_npy> \
      [lib路径] [runs]
test_dir 内 list_tower.txt/list_pair.txt 行序 = 前缀 t/p_XXX 的三个 npy 三元组。
输出：out_npy 存全部用例前向结果 [N,768] 或 [N,2]；stdout 打印逐次/统计延迟。
"""
import ctypes
import os
import sys
import time

import numpy as np

u32 = ctypes.c_uint32


class RknnInput(ctypes.Structure):
    _fields_ = [("index", u32), ("buf", ctypes.c_void_p), ("size", u32),
                ("pass_through", ctypes.c_uint8), ("type", ctypes.c_int),
                ("fmt", ctypes.c_int)]


class RknnOutput(ctypes.Structure):
    _fields_ = [("want_float", ctypes.c_uint8), ("is_prealloc", ctypes.c_uint8),
                ("index", u32), ("buf", ctypes.c_void_p), ("size", u32)]


class InOutNum(ctypes.Structure):
    _fields_ = [("n_input", u32), ("n_output", u32),
                ("reserved", ctypes.c_uint8 * 60)]


class TensorAttr(ctypes.Structure):
    _fields_ = [("index", u32), ("n_dims", u32),
                ("dims", u32 * 16), ("name", ctypes.c_char * 256),
                ("n_elems", u32), ("size", u32), ("fmt", ctypes.c_int),
                ("type", ctypes.c_int), ("qnt_type", ctypes.c_int),
                ("fl", ctypes.c_int8), ("zp", ctypes.c_int32),
                ("scale", ctypes.c_float), ("w_stride", u32),
                ("size_with_stride", u32), ("pass_through", ctypes.c_uint8),
                ("h_stride", u32)]


TYPE_NP = {0: np.float32, 1: np.float16, 2: np.int8, 3: np.uint8, 6: np.int32,
           8: np.int64}


def main():
    model_path, mode, test_dir, out_npy = sys.argv[1:5]
    lib = sys.argv[5] if len(sys.argv) > 5 else \
        "/home/web_vllm_rknn/lib/librknnrt.so"
    runs = int(sys.argv[6]) if len(sys.argv) > 6 else 30

    lib_ = ctypes.CDLL(lib)
    ctx = ctypes.c_void_p()
    blob = open(model_path, "rb").read()
    buf = ctypes.create_string_buffer(blob, len(blob))
    rc = lib_.rknn_init(ctypes.byref(ctx), buf, u32(len(blob)), u32(0), None)
    assert rc == 0, f"rknn_init {rc}"

    io = InOutNum()
    assert lib_.rknn_query(ctx, 0, ctypes.byref(io),
                           ctypes.sizeof(io)) == 0
    print(f"inputs={io.n_input} outputs={io.n_output}")

    prefix, listfile = (("t", "list_tower.txt") if mode == "tower"
                        else ("p", "list_pair.txt"))
    cases = [l.split() for l in
             open(os.path.join(test_dir, listfile)).read().split("\n") if l.strip()]

    # 输出属性（want_float 路径在部分版本不可靠，读原始字节按 native 类型解码）
    out_attrs = []
    for i in range(io.n_output):
        a = TensorAttr()
        a.index = i
        assert lib_.rknn_query(ctx, 2, ctypes.byref(a), ctypes.sizeof(a)) == 0
        print(f"  out[{i}] {a.name.decode()} dims={list(a.dims[:a.n_dims])} "
              f"type={a.type} qnt={a.qnt_type} zp={a.zp} scale={a.scale}")
        out_attrs.append(a)

    NP_OF = {0: np.float32, 1: np.float16, 2: np.int8, 3: np.uint8,
             5: np.uint16, 6: np.int32}

    # 查询输入属性：按模型真实 dtype/fmt 喂数据
    attrs = []
    for i in range(io.n_input):
        a = TensorAttr()
        a.index = i
        rc = lib_.rknn_query(ctx, 1, ctypes.byref(a), ctypes.sizeof(a))  # INPUT_ATTR
        assert rc == 0, f"query input attr {rc}"
        print(f"  in[{i}] {a.name.decode()} dims={list(a.dims[:a.n_dims])} "
              f"type={a.type} fmt={a.fmt} qnt={a.qnt_type} scale={a.scale}")
        attrs.append(a)
    n_in = io.n_input

    data = []
    for line in cases:
        arrs = [np.ascontiguousarray(np.load(os.path.join(test_dir, n)),
                                     dtype=TYPE_NP[attrs[i].type]).reshape(-1)
                for i, n in enumerate(line)]
        data.append(arrs)

    def run_case(arrs):
        ins = (RknnInput * n_in)()
        keep = []
        for i, a in enumerate(arrs):
            ptr = a.ctypes.data_as(ctypes.c_void_p)
            keep.append((a, ptr))
            ins[i].index = i
            ins[i].buf = ptr
            ins[i].size = u32(a.nbytes)
            pt = int(os.environ.get("SBERT_PT", "0"))   # 喂入模式可切换调试
            ins[i].pass_through = pt
            ins[i].type = attrs[i].type
            ins[i].fmt = int(os.environ.get("SBERT_FMT", "0"))  # NCHW=0
        rc = lib_.rknn_inputs_set(ctx, u32(n_in), ins)
        assert rc == 0, f"inputs_set {rc}"
        rc = lib_.rknn_run(ctx, None)
        assert rc == 0, f"rknn_run {rc}"
        outs = (RknnOutput * io.n_output)()
        for i in range(io.n_output):
            outs[i].index = i
            outs[i].want_float = 0     # 读 native，手动解码
            outs[i].is_prealloc = 0
        rc = lib_.rknn_outputs_get(ctx, u32(io.n_output), outs, None)
        assert rc == 0, f"outputs_get {rc}"
        a = out_attrs[0]
        raw = np.ctypeslib.as_array(
            ctypes.cast(outs[0].buf, ctypes.POINTER(ctypes.c_uint8)),
            shape=(outs[0].size,)).copy()
        lib_.rknn_outputs_release(ctx, u32(io.n_output), outs)
        v = raw.view(NP_OF[a.type])
        n = a.n_elems
        v = v[:n] if v.size >= n else v
        if a.type in (2, 3):           # int8/uint8 仿射反量化
            v = (v.astype(np.float32) - a.zp) * a.scale
        return v.astype(np.float32)

    # 前向全部用例
    res = np.stack([run_case(a) for a in data])
    np.save(out_npy, res)
    print("saved", out_npy, res.shape)
    print("row0[:8]:", np.round(res[0][:8], 3).tolist())
    if mode == "pair":
        e = np.exp(res - res.max(-1, keepdims=True))
        print("P(相似):", np.round((e[:, 1] / e.sum(-1)), 4).tolist())

    # 速度：用例 0 重复 runs 次
    for _ in range(3):
        run_case(data[0])
    ts = []
    for _ in range(runs):
        t0 = time.perf_counter()
        run_case(data[0])
        ts.append((time.perf_counter() - t0) * 1000)
    ts = np.array(ts)
    print(f"[{os.path.basename(model_path)}] runs={runs} "
          f"mean={ts.mean():.1f}ms p50={np.percentile(ts,50):.1f} "
          f"p90={np.percentile(ts,90):.1f} min={ts.min():.1f}ms")

    lib_.rknn_destroy(ctx)


if __name__ == "__main__":
    main()
