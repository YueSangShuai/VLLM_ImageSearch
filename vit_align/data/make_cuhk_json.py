# -*- coding: utf-8 -*-
"""扫 CUHK-PEDES 目录生成 MARS 风格 json（train.json / test.json）。

源目录约定（../MARS 同款）：imgs/train_query/ 与 imgs/test_query/ 下每张
pXXXX_sYYYY.jpg 配同名 .txt（每行一条英文描述），id = pXXXX。

例：python data/make_cuhk_json.py          # 全量
    python data/make_cuhk_json.py --limit-per 200   # 冒烟
"""
import argparse
import json
import os
import re
from collections import defaultdict

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import load_config  # noqa: E402

SPLIT_DIRS = {"train": "train_query", "test": "test_query"}
NAME = re.compile(r"^p(\d+)_s\d+\.(jpg|jpeg|png)$")


def scan(root, subdir, limit_per=0):
    d = os.path.join(root, "imgs", subdir)
    entries, skipped = [], 0
    id_count = defaultdict(int)
    for fn in sorted(os.listdir(d)):
        m = NAME.match(fn)
        if not m:
            continue
        stem = os.path.splitext(fn)[0]
        txt = os.path.join(d, stem + ".txt")
        if not os.path.exists(txt):
            skipped += 1
            continue
        with open(txt, "r", encoding="utf-8") as f:
            captions = [ln.strip() for ln in f if ln.strip()]
        if not captions:
            skipped += 1
            continue
        pid = int(m.group(1))
        if limit_per and id_count[pid] >= limit_per:
            continue
        id_count[pid] += 1
        entries.append({
            "id": pid,
            "image_path": os.path.join(d, fn),
            "text_path": txt,
            "text": captions[0],
            "captions": captions,
        })
    return entries, skipped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--limit-per", type=int, default=0, help="每人最多取几张图（冒烟用）")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    root = cfg["data"]["root"]
    out_dir = args.out_dir or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), cfg["data"]["json_dir"])
    os.makedirs(out_dir, exist_ok=True)

    for split, subdir in SPLIT_DIRS.items():
        entries, skipped = scan(root, subdir, args.limit_per)
        n_captions = sum(len(e["captions"]) for e in entries)
        n_ids = len({e["id"] for e in entries})
        out = os.path.join(out_dir, f"{split}.json")
        with open(out, "w", encoding="utf-8") as f:
            json.dump(entries, f, ensure_ascii=False)
        print(f"{split}: {len(entries)} imgs / {n_ids} ids / {n_captions} captions "
              f"(no-txt skipped {skipped}) -> {out}")


if __name__ == "__main__":
    main()
