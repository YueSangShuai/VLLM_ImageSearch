#!/usr/bin/env python3
# StructBERT ONNX (固定 1×128, int32 输入) -> rk3588 .rknn
# 用法（rknn env, PYTHONNOUSERSITE=1）：
#   python convert_structbert_rknn.py <onnx> <out.rknn> [--int8 --calib dataset.txt]
import sys, argparse, os

ap = argparse.ArgumentParser()
ap.add_argument("onnx")
ap.add_argument("out")
ap.add_argument("--int8", action="store_true")
ap.add_argument("--calib", default=None)
args = ap.parse_args()

from rknn.api import RKNN

rknn = RKNN(verbose=False)
rknn.config(target_platform="rk3588", optimization_level=1)

print("== load_onnx", args.onnx)
ret = rknn.load_onnx(
    model=args.onnx,
    inputs=["input_ids", "attention_mask", "token_type_ids"],
    input_size_list=[[1, 128], [1, 128], [1, 128]])
if ret != 0:
    sys.exit(f"load_onnx failed {ret}")

if args.int8:
    if not args.calib:
        sys.exit("--int8 需要 --calib")
    print("== build w8a8, calib:", args.calib)
    ret = rknn.build(do_quantization=True, dataset=args.calib)
else:
    print("== build fp16")
    ret = rknn.build(do_quantization=False)
if ret != 0:
    sys.exit("build failed")

ret = rknn.export_rknn(args.out)
if ret != 0:
    sys.exit("export failed")
print("OK ->", args.out, os.path.getsize(args.out) // 2**20, "MB")
