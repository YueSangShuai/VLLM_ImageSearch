# -*- coding: utf-8 -*-
"""阶段一：冻结 Qwen3-VL 视觉塔，对 json 条目逐张提 mean-pool 2048 维特征并分块缓存。

分块原子写盘 + 断点续提 + .done 标记（沿用 qwen_classifer/extract_features.py 模式）。
产物 <cache-dir>/<name>_c<idx>_n<count>.pt：feats fp16 [N,2048] / ids / paths / meta。
顺序与 json 条目顺序一致，训练侧按序拼接。

例：
  python extract_image_feats.py --json data/json/train.json --name train_i --limit 500
  python extract_image_feats.py --json data/json/train.json --name train_i
  python extract_image_feats.py --json data/json/test.json  --name test_i
"""
import argparse
import glob
import json
import os
import re
import time

import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from common import abspath, load_config
from vision import (FixedSizeImageDataset, extract_features, load_processor,
                    load_vision, make_collate)

ROOT = os.path.dirname(os.path.abspath(__file__))
CHUNK_NAME = re.compile(r"^c(\d+)_n(\d+)\.pt$")


def find_chunks(cache_dir, name):
    out = []
    for p in glob.glob(os.path.join(cache_dir, f"{name}_c*.pt")):
        m = CHUNK_NAME.match(os.path.basename(p)[len(name) + 1:])
        if m:
            out.append((int(m.group(1)), int(m.group(2)), p))
    return sorted(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--json", nargs="+", required=True)
    ap.add_argument("--name", required=True, help="缓存前缀，如 train_i / test_i")
    ap.add_argument("--cache-dir", default="cache")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--chunk-size", type=int, default=5000)
    args = ap.parse_args()

    cfg = load_config(args.config)
    vcfg = cfg["vision"]
    cache_dir = abspath(args.cache_dir)
    os.makedirs(cache_dir, exist_ok=True)
    done_path = os.path.join(cache_dir, args.name + ".done")
    if os.path.exists(done_path):
        raise SystemExit(f"{args.name} 已完成（{done_path} 存在），重跑先删 .done")

    from common import load_entries
    entries = load_entries([abspath(p) for p in args.json])
    if args.limit:
        entries = entries[:args.limit]
    ids = torch.tensor([int(e["id"]) for e in entries], dtype=torch.long)
    total = len(entries)
    print(f"images: {total}  size: {vcfg['size']}")

    chunks = find_chunks(cache_dir, args.name)
    skip = sum(n for _, n, _ in chunks)
    chunk_idx = (chunks[-1][0] + 1) if chunks else 0
    if skip:
        if skip >= total:
            raise SystemExit(f"已有分块 {skip} >= 总量 {total}，视为已完成？")
        m = torch.load(chunks[0][2], mmap=True)
        for key, want in (("size", int(vcfg["size"])), ("model", vcfg["model_path"])):
            if m.get(key) != want:
                raise SystemExit(f"已有分块 {key}={m.get(key)} 与本次 {want} 不一致，先清理缓存")
        print(f"续提：跳过前 {skip} 张")

    ds = FixedSizeImageDataset(entries, size=vcfg["size"])
    processor = load_processor(vcfg["model_path"])
    visual = load_vision(vcfg["model_path"])
    loader = DataLoader(Subset(ds, range(skip, total)), batch_size=args.batch_size,
                        shuffle=False, num_workers=args.workers,
                        collate_fn=make_collate(processor))

    feats, fids, fpaths = [], [], []
    filled = 0   # 当前未落盘样本数
    made = 0     # 本次新提取总数

    def flush():
        nonlocal feats, fids, fpaths, filled, chunk_idx
        if not feats:
            return
        x = torch.stack(feats)
        tmp = os.path.join(cache_dir, f"{args.name}_c{chunk_idx}_n{len(feats)}.pt.tmp")
        torch.save({"feats": x, "ids": torch.stack(fids), "paths": list(fpaths),
                    "size": int(vcfg["size"]), "model": vcfg["model_path"]}, tmp)
        os.replace(tmp, tmp[:-4])
        print(f"chunk {chunk_idx}: {len(feats)} 张 (dim={x.shape[1]})")
        chunk_idx += 1
        feats, fids, fpaths, filled = [], [], [], 0

    t0 = time.time()
    for pixel_values, grid_thw, idxs in tqdm(loader, initial=skip, total=total):
        f = extract_features(visual, pixel_values, grid_thw).half()
        for j in range(f.shape[0]):
            feats.append(f[j])
            fids.append(ids[int(idxs[j])])
            fpaths.append(entries[int(idxs[j])]["image_path"])
            filled += 1
            made += 1
            if filled >= args.chunk_size:
                flush()
    flush()

    done = {"name": args.name, "total": skip + made,
            "seconds": round(time.time() - t0, 1), "size": int(vcfg["size"])}
    with open(done_path, "w") as fp:
        json.dump(done, fp, ensure_ascii=False, indent=1)
    print(f"done -> {done_path}")


if __name__ == "__main__":
    main()
