#!/usr/bin/env python3
# yolov8n-dyn.onnx (dynamic) -> rk3588 .rknn，固定 1x3x640x640。
# 用法（rknn env, PYTHONNOUSERSITE=1）：python convert_yolo_rknn.py <onnx> <out.rknn> [--int8]
import sys, argparse, os

ap = argparse.ArgumentParser()
ap.add_argument("onnx")
ap.add_argument("out")
ap.add_argument("--int8", action="store_true", help="w8a8 量化（默认 fp16 免校准，先保精度）")
ap.add_argument("--calib", default=None, help="int8 校准列表文件（每行一个 npy/txt 目录）")
args = ap.parse_args()

from rknn.api import RKNN

rknn = RKNN(verbose=False)

# mean/std 两分支都要设：板侧直接喂 uint8 NHWC 原图，/255 折进模型归一化
# （float NCHW 输入在 NHWC 模型上会触发 "unsupported src layout for normalize"）
if args.int8:
    rknn.config(target_platform="rk3588", mean_values=[[0, 0, 0]],
                std_values=[[255, 255, 255]], optimization_level=1)
else:
    rknn.config(target_platform="rk3588", mean_values=[[0, 0, 0]],
                std_values=[[255, 255, 255]], optimization_level=1)

print("== load_onnx", args.onnx)
ret = rknn.load_onnx(model=args.onnx, inputs=["images"], input_size_list=[[1, 3, 640, 640]])
if ret != 0:
    sys.exit(f"load_onnx failed {ret}")

print("== build (quant=", args.int8, ")", sep="")
if args.int8:
    if not args.calib:
        sys.exit("--int8 需要 --calib 数据集列表")
    ret = rknn.build(do_quantization=True, dataset=args.calib)
else:
    ret = rknn.build(do_quantization=False)
if ret != 0:
    sys.exit("build failed")

ret = rknn.export_rknn(args.out)
if ret != 0:
    sys.exit("export failed")
print("OK ->", args.out, os.path.getsize(args.out) // (1024 * 1024), "MB")
