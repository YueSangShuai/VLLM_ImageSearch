#!/usr/bin/env python
"""x86 侧：把 golden 输入/期望输出导出为板端 C++ checker 可用的 raw 格式。

产出（scripts/golden/raw/）：
  input{i}.raw   uint8 [3*224*224] NHWC 0-255（rknn 输入约定）
  expect{i}.raw  fp32 [768]（ONNX/torch 的 L2 归一化图像向量）
  meta.txt       每行：idx cos阈值（默认 0.995）
用法：/home/yuesang/miniconda3/envs/rknn/bin/python scripts/golden/export_golden.py
"""
import os
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
GOLDEN = os.path.join(ROOT, "golden")
RAW = os.path.join(os.path.dirname(os.path.abspath(__file__)), "raw")
os.makedirs(RAW, exist_ok=True)

pixel = np.load(os.path.join(GOLDEN, "pixel_pm1.npy"))          # [N,3,H,W] [-1,1]
emb = np.load(os.path.join(GOLDEN, "img_emb_torch.npy"))        # [N,768]
n = pixel.shape[0]
for i in range(n):
    x = ((pixel[i] + 1.0) * 127.5).clip(0, 255).round().astype(np.uint8)
    x = x.transpose(1, 2, 0)                                     # NHWC
    x.tofile(os.path.join(RAW, f"input{i}.raw"))
    emb[i].astype(np.float32).tofile(os.path.join(RAW, f"expect{i}.raw"))
with open(os.path.join(RAW, "meta.txt"), "w") as f:
    for i in range(n):
        f.write(f"{i} 0.995\n")
print("wrote", n, "cases to", RAW)
