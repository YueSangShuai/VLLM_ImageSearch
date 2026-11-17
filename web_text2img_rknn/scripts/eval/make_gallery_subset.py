#!/usr/bin/env python3
"""从 pa100k 取前 K 张可读图作为板端初始图库（按名排序，确定性）。
用法: python3 scripts/eval/make_gallery_subset.py [K] [dest]"""
import os, shutil, sys
import cv2

SRC = "/data/yuesang/person_data/pa100k/images"
K = int(sys.argv[1]) if len(sys.argv) > 1 else 500
DEST = sys.argv[2] if len(sys.argv) > 2 else os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "output", "gallery")
os.makedirs(DEST, exist_ok=True)
n = 0
for name in sorted(os.listdir(SRC)):
    if not name.endswith(".jpg"):
        continue
    p = os.path.join(SRC, name)
    if cv2.imread(p) is None:
        continue
    shutil.copy2(p, os.path.join(DEST, name))
    n += 1
    if n >= K:
        break
print(f"copied {n} -> {DEST}")
