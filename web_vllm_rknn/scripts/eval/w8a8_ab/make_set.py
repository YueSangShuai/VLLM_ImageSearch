# -*- coding: utf-8 -*-
"""抽样 pa100k test 构建 W8A8 A/B 评测集：images/ 平铺图 + gt.json。

文件名 {idx:03d}_{原名} 前缀保证板端 bench_models 的目录排序 = gt.json 顺序。
"""
import argparse, json, os, random, shutil, sys

ap = argparse.ArgumentParser()
ap.add_argument("--json", default="/data/yuesang/person_data/pa100k/json/test_new_processed.json")
ap.add_argument("--n", type=int, default=50)
ap.add_argument("--seed", type=int, default=20261103)
ap.add_argument("--out", default=os.path.dirname(os.path.abspath(__file__)))
args = ap.parse_args()

sys.path.insert(0, "/data/yuesang/LLM/FG-CLIP/qwen_classifer")
from common import ATTRS

entries = json.load(open(args.json))
rng = random.Random(args.seed)
picked = rng.sample(range(len(entries)), args.n)

img_dir = os.path.join(args.out, "images")
os.makedirs(img_dir, exist_ok=True)
gt = []
for k, i in enumerate(sorted(picked)):
    e = entries[i]
    src = e["image_path"]
    name = f"{k:03d}_{os.path.basename(src)}"
    shutil.copyfile(src, os.path.join(img_dir, name))
    gt.append({"file": name, "src": src,
               "labels": {a: int(e.get(a, -1)) for a in ATTRS}})

with open(os.path.join(args.out, "gt.json"), "w") as f:
    json.dump({"seed": args.seed, "n": args.n, "attrs": ATTRS, "items": gt}, f,
              ensure_ascii=False, indent=1)

npos = sum(sum(1 for a in ATTRS if g["labels"][a] == 1) for g in gt)
print(f"sampled {len(gt)} images -> {img_dir}  (positives {npos}, avg {npos/len(gt):.1f}/img)")
