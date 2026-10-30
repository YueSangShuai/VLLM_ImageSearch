# -*- coding: utf-8 -*-
"""阶段一：用冻结的 Qwen3-VL 视觉塔对数据集提特征并缓存到磁盘。

特征按 --chunk-size 分块写盘（默认 30 万张/块 ≈ 1.2GB），每块原子写入，
中断后重跑同一命令会跳过已完成样本续提；全部分片完成后写 .done 标记。
产物：<cache>/<name>_s<shard>_c<chunk>_n<count>.pt，配对的 train_head 直接吃通配符。

例：
  python extract_features.py --json /data/yuesang/person_data/pa100k/json/train.json --name train
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

from data import Pa100kDataset, make_collate
from vision import DEFAULT_MODEL, extract_features, load_processor, load_vision

CHUNK_NAME = re.compile(r"^c(\d+)_n(\d+)\.pt$")


def find_chunks(cache_dir, base):
    """-> [(chunk_idx, n_samples, path)] 按 idx 排序。"""
    out = []
    for p in glob.glob(os.path.join(cache_dir, f"{base}_c*.pt")):
        m = CHUNK_NAME.match(os.path.basename(p)[len(base) + 1:])
        if m:
            out.append((int(m.group(1)), int(m.group(2)), p))
    return sorted(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", nargs="+", required=True,
                    help="PA-100K 风格 json，可给多个文件自动拼接")
    ap.add_argument("--name", default="train", help="缓存文件名前缀")
    ap.add_argument("--cache-dir", default="cache")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--size", type=int, default=448, help="输入边长，需为 32 的倍数")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shard-id", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1,
                    help="多 GPU 并行：每台一张卡跑 shard-id=0..N-1")
    ap.add_argument("--chunk-size", type=int, default=300_000,
                    help="每块样本数（0=单文件）。30 万张 2048 维 fp16 约 1.2GB")
    ap.add_argument("--use-deepstack", action="store_true",
                    help="拼接 3 个 deepstack 特征的 mean pooling（F=8192）")
    args = ap.parse_args()

    os.makedirs(args.cache_dir, exist_ok=True)
    base = args.name + (f"_s{args.shard_id}" if args.num_shards > 1 else "")
    done_path = os.path.join(args.cache_dir, base + ".done")
    if os.path.exists(done_path) or os.path.exists(
            os.path.join(args.cache_dir, base + ".pt")):
        raise SystemExit(f"{base} 已完成（{done_path} 存在），要重跑先删除")

    ds = Pa100kDataset(args.json, size=args.size, limit=args.limit,
                       shard_id=args.shard_id, num_shards=args.num_shards)
    total = len(ds)
    print(f"images: {total}  size: {args.size}")

    # ---- 断点续跑：已有分块覆盖的前缀样本直接跳过 ----
    chunks = find_chunks(args.cache_dir, base)
    skip = sum(n for _, n, _ in chunks)
    chunk_idx = (chunks[-1][0] + 1) if chunks else 0
    if skip:
        if skip >= total:
            raise SystemExit(f"已有分块 {skip} 张 >= 总量 {total}，视为已完成？检查 cache")
        # 续提前校验已有分块的提取配置，防止 size/deepstack 不同的缓存被混接
        m = torch.load(chunks[0][2], mmap=True)
        for key, want in (("size", args.size), ("use_deepstack", args.use_deepstack),
                          ("model", args.model)):
            got = m.get(key)
            if got != want:
                raise SystemExit(f"续提中止：已有分块 {os.path.basename(chunks[0][2])} 的 "
                                 f"{key}={got}，与本次 {key}={want} 不一致，请换 --name 或清理 cache")
        del m
        print(f"resume: 已有 {len(chunks)} 块共 {skip} 张，从第 {skip} 张继续")
        ds = Subset(ds, range(skip, total))

    visual = load_vision(args.model)
    processor = load_processor(args.model)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, collate_fn=make_collate(processor),
                        pin_memory=True)

    meta = {
        "use_deepstack": args.use_deepstack,
        "size": args.size,
        "model": args.model,
        "json": [os.path.abspath(p) for p in args.json],
    }
    buf, n_buf, n_saved = [], 0, skip

    def flush(tag=""):
        nonlocal buf, n_buf, chunk_idx, n_saved
        if not buf:
            return
        fe = torch.cat([b[0] for b in buf])
        la = torch.cat([b[1] for b in buf])
        pa = [x for b in buf for x in b[2]]
        if args.chunk_size > 0:
            fname = f"{base}_c{chunk_idx:04d}_n{len(pa)}.pt"
        else:  # 单文件模式：全存一个文件，靠 .done 标记防重复
            fname = f"{base}.pt"
        path = os.path.join(args.cache_dir, fname)
        tmp = path + ".tmp"
        torch.save({"feats": fe, "labels": la, "paths": pa, **meta}, tmp)
        os.replace(tmp, path)
        n_saved += len(pa)
        chunk_idx += 1
        buf, n_buf = [], 0
        tqdm.write(f"  saved {fname}{tag}")

    pbar = tqdm(total=total, initial=skip, unit="img", smoothing=0.05, desc=base)
    t0 = time.time()
    for pv, grid, lab, p in loader:
        pv = pv.cuda(non_blocking=True)
        f = extract_features(visual, pv, grid, use_deepstack=args.use_deepstack)
        buf.append((f.half().cpu(), lab, list(p)))
        n_buf += len(p)
        sps = (pbar.n - skip) / max(time.time() - t0, 1e-6)
        pbar.update(len(p))
        pbar.set_postfix_str(f"{sps:.0f} img/s  "
                             f"eta {(total - pbar.n) / max(sps, 1e-6) / 60:.0f}min")
        if args.chunk_size > 0 and n_buf >= args.chunk_size:
            flush(f"  ({n_saved}/{total})")
    pbar.close()
    flush()

    if n_saved >= total:
        with open(done_path, "w") as f:
            json.dump({"n": n_saved, "total": total, **meta}, f, ensure_ascii=False)
        print(f"DONE {base}: {n_saved} 张 -> {args.cache_dir}/{base}_c*.pt，"
              f"耗时 {(time.time() - t0) / 60:.1f} min")
    else:
        print(f"!! 提前结束：已存 {n_saved}/{total}，重跑本命令可续提")


if __name__ == "__main__":
    main()
