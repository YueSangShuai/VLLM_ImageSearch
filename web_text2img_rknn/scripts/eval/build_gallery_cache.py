#!/usr/bin/env python3
"""离线构建文搜图向量缓存：x86 torch 精确图像向量 → 板上 data/gallery/ 直接加载。

板上 10 万张 NPU 编码要 ~2h，且带 stb 解码噪声（±1 LSB）。本脚本用与训练/基线
完全一致的管线（cv2 解码 + cv2.resize + torch 塔）离线算向量，产出与板端
Gallery::load_cache 字节兼容的 manifest.json + vectors.bin；推上板后建库即刻完成，
图库侧漂移归零（剩余漂移仅查询文本向量的 NPU 侧 ~0.0002）。

前提：--gallery 必须与板上 GALLERY_DIR 内容逐字节一致且 mtime 相同
（用 tar 保 mtime 部署，即 /data/yuesang/person_data/pa100k/images 之于板上 gallery/），
否则板端 sync 会判定全部过期并重编。板端新增/enroll 图仍走板内增量编码，互不影响。

用法（MARS env）：
  /home/yuesang/miniconda3/envs/MARS/bin/python scripts/eval/build_gallery_cache.py \
      [--gallery /data/yuesang/person_data/pa100k/images] [--out /tmp/gcache] [--batch 64]
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "export"))
from onnx_export import build_model, load_u8, u8_to_nchw, ImageTower  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gallery", default="/data/yuesang/person_data/pa100k/images")
    ap.add_argument("--out", default="/tmp/gcache")
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args()

    files = []
    for dirpath, _, names in os.walk(args.gallery):
        for n in sorted(names):
            if n.lower().rsplit(".", 1)[-1] in ("jpg", "jpeg", "png"):
                files.append(os.path.relpath(os.path.join(dirpath, n), args.gallery))
    files.sort()
    print(f"图库 {len(files)} 张（相对 {args.gallery}）")

    model, tok, cfg = build_model()
    it = ImageTower(model.visual_encoder, model.vision_proj)

    os.makedirs(args.out, exist_ok=True)
    vec_path = os.path.join(args.out, "vectors.bin")
    entries, dim = [], None
    with open(vec_path, "wb") as vf, torch.no_grad():
        for i in range(0, len(files), args.batch):
            chunk = files[i:i + args.batch]
            xs, keep = [], []
            for rel in chunk:
                try:
                    xs.append(u8_to_nchw(load_u8(os.path.join(args.gallery, rel))))
                    keep.append(rel)
                except Exception as e:  # 坏图跳过并警告（与板端增量行为一致）
                    print(f"[warn] 跳过 {rel}: {e}")
            if not xs:
                continue
            emb = it(torch.from_numpy(np.concatenate(xs)).float()).numpy()
            for j, rel in enumerate(keep):
                v = np.ascontiguousarray(emb[j], dtype=np.float32)
                if dim is None:
                    dim = int(v.size)
                entries.append({"file": rel,
                                "mtime": int(os.stat(os.path.join(args.gallery, rel)).st_mtime),
                                "off": len(entries), "alive": True})
                vf.write(v.tobytes())
            if (i // args.batch) % 32 == 0:
                print(f"  {len(entries)}/{len(files)}", flush=True)

    json.dump({"dim": dim, "entries": entries},
              open(os.path.join(args.out, "manifest.json"), "w"))
    ok = os.path.getsize(vec_path) == len(entries) * dim * 4
    print(f"完成: {len(entries)} 条 dim={dim} vectors.bin={os.path.getsize(vec_path)}B "
          f"自检={'OK' if ok else '大小不符!'} -> {args.out}")
    sys.exit(0 if ok and len(entries) == len(files) else 1)


if __name__ == "__main__":
    main()
