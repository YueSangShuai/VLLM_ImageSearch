#!/usr/bin/env python3
"""板端 ctypes 直调 librknnrt 跑文本塔 golden（与 C++/x86 模拟器三方对拍）。
板上执行: python3 board_ctypes_check.py <rknn_path> <raw_dir>"""
import ctypes, json, sys
import numpy as np

rknn_path, raw = sys.argv[1], sys.argv[2]
lib = ctypes.CDLL("librknnrt.so")


class IO_NUM(ctypes.Structure):
    _fields_ = [("n_input", ctypes.c_uint32), ("n_output", ctypes.c_uint32)]


class TENSOR_ATTR(ctypes.Structure):
    _fields_ = [("index", ctypes.c_uint32), ("type", ctypes.c_uint32),
                ("fmt", ctypes.c_uint32), ("size", ctypes.c_uint32),
                ("dims", ctypes.c_uint32 * 5), ("n_elems", ctypes.c_uint32),
                ("name", ctypes.c_char * 32)]


class TENSOR_BUF(ctypes.Structure):
    _fields_ = [("buf", ctypes.c_void_p), ("size", ctypes.c_uint32),
                ("pass_through", ctypes.c_uint8)]


class IO_TENSOR(ctypes.Structure):
    _fields_ = [("attr", TENSOR_ATTR), ("tab", TENSOR_BUF * 3),
                ("is_prealloc", ctypes.c_uint8), ("idx", ctypes.c_uint32)]


class INPUT(ctypes.Structure):
    _fields_ = [("index", ctypes.c_uint32), ("type", ctypes.c_uint32),
                ("fmt", ctypes.c_uint32), ("size", ctypes.c_uint32),
                ("pass_through", ctypes.c_uint8), ("buf", ctypes.c_void_p)]


class OUTPUT(ctypes.Structure):
    _fields_ = [("index", ctypes.c_uint32), ("buf", ctypes.c_void_p),
                ("size", ctypes.c_uint32), ("want_float", ctypes.c_uint8),
                ("is_prealloc", ctypes.c_uint8)]


meta = json.load(open(f"{raw}/meta.json"))
M, SEQ, DIM = meta["M"], meta["seq"], meta["dim"]
ids = np.fromfile(f"{raw}/ids.raw", dtype=np.int32).reshape(M, SEQ)
attn = np.fromfile(f"{raw}/attn.raw", dtype=np.float32).reshape(M, SEQ)
t_emb = np.fromfile(f"{raw}/t_emb.raw", dtype=np.float32).reshape(M, DIM)

ctx = ctypes.c_void_p()
assert lib.rknn_init(ctypes.byref(ctx), rknn_path.encode(), 0, 0, None) >= 0
io = IO_NUM()
lib.rknn_query(ctx, 1, ctypes.byref(io), ctypes.sizeof(io))  # RKNN_QUERY_IN_OUT_NUM=1
print("n_input", io.n_input, "n_output", io.n_output)
for i in range(io.n_input):
    a = TENSOR_ATTR(index=i)
    lib.rknn_query(ctx, 3, ctypes.byref(a), ctypes.sizeof(a))  # INPUT_ATTR=3
    print(f"in{i}: {a.name.decode()} type={a.type} dims={a.dims[:3]} size={a.size}")

i0 = INPUT(0, 6, 0, SEQ * 4, 1, None)
i1 = INPUT(1, 1, 0, SEQ * 4, 1, None)
inp = (INPUT * 2)(i0, i1)
out = OUTPUT(0, None, 0, 1, 0)

for qi in [0, 1]:
    ids_c = np.ascontiguousarray(ids[qi], dtype=np.int32)
    attn_c = np.ascontiguousarray(attn[qi], dtype=np.float32)
    inp[0].buf = ids_c.ctypes.data_as(ctypes.c_void_p)
    inp[1].buf = attn_c.ctypes.data_as(ctypes.c_void_p)
    assert lib.rknn_inputs_set(ctx, 2, inp) >= 0
    assert lib.rknn_run(ctx, None) >= 0
    assert lib.rknn_outputs_get(ctx, 1, ctypes.byref(out), None) >= 0
    got = (ctypes.c_float * DIM).from_address(out.buf)
    g = np.array(got)
    cos = float(g @ t_emb[qi] / (np.linalg.norm(g) * np.linalg.norm(t_emb[qi]) + 1e-12))
    print(f"q{qi}: ctypes vs torch cos = {cos:.5f}  emb[0:3]={g[:3]} torch={t_emb[qi][:3]}")
lib.rknn_outputs_release(ctx, 1, ctypes.byref(out))
