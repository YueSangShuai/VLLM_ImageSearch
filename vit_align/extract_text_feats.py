# -*- coding: utf-8 -*-
"""阶段二：冻结 BGE-small-zh 对 caption 提 512 维文本向量（单文件缓存）。

caption 顺序 = build_index(json) 展开顺序，训练/评估两侧用同一函数保证对齐。
--prefix 时给每条文本加生产官方 query 前缀（中文查询上线口径），另存一个 name。

例：
  python extract_text_feats.py --json data/json/train.json --name train_t
  python extract_text_feats.py --json data/json/test.json  --name test_t
  python extract_text_feats.py --json data/json/test.json  --name test_t_q --prefix
"""
import argparse
import json
import os

import torch

from common import abspath, load_config
from data.dataset import build_index
from text import BGETextTower


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--json", nargs="+", required=True)
    ap.add_argument("--name", required=True, help="缓存前缀，如 train_t / test_t_q")
    ap.add_argument("--cache-dir", default="cache")
    ap.add_argument("--prefix", action="store_true",
                    help="加生产 query 前缀（上线中文查询口径），doc/图像对齐侧不要加")
    ap.add_argument("--batch-size", type=int, default=256)
    args = ap.parse_args()

    cfg = load_config(args.config)
    tcfg = cfg["text"]
    jsons = [abspath(p) for p in args.json]
    _, captions, cap_ids, pairs = build_index(jsons)
    texts = captions
    if args.prefix:
        texts = [tcfg["query_prefix"] + t for t in captions]
    print(f"captions: {len(texts)}  pairs: {len(pairs)}  prefix={args.prefix}")

    tower = BGETextTower(model_path=tcfg["model_path"], max_length=tcfg["max_length"])
    emb = tower.encode(texts, batch_size=args.batch_size)

    cache_dir = abspath(args.cache_dir)
    os.makedirs(cache_dir, exist_ok=True)
    out = os.path.join(cache_dir, f"{args.name}_c0_n{len(texts)}.pt")
    tmp = out + ".tmp"
    torch.save({"feats": emb.half(), "ids": cap_ids, "texts": texts,
                "model": tcfg["model_path"], "prefix": bool(args.prefix),
                "max_length": int(tcfg["max_length"])}, tmp)
    os.replace(tmp, out)
    norms = torch.linalg.vector_norm(emb, dim=-1)
    print(f"wrote {out}  shape={tuple(emb.shape)}  "
          f"L2 norm min/mean/max = {norms.min():.4f}/{norms.mean():.4f}/{norms.max():.4f}")


if __name__ == "__main__":
    main()
