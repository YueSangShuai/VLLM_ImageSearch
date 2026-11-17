#!/usr/bin/env python3
"""golden/*.npy → scripts/golden/raw/*.raw（板端 golden_check 免 npy 解析）。
任意带 numpy 的 env 可跑。"""
import json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
G = os.path.join(ROOT, "golden")
OUT = os.path.join(HERE, "raw")
os.makedirs(OUT, exist_ok=True)

px = np.load(f"{G}/gallery_px_u8.npy").astype(np.uint8)          # [N,224,224,3] NHWC
ids = np.load(f"{G}/query_input_ids.npy").astype(np.int32)        # [M,128]
attn = np.load(f"{G}/query_attn.npy").astype(np.float32)          # [M,128]
t_emb = np.load(f"{G}/query_emb_torch.npy").astype(np.float32)    # [M,256]
g_emb = np.load(f"{G}/gallery_emb_torch.npy").astype(np.float32)  # [N,256]

px.tofile(f"{OUT}/px_u8.raw")
ids.tofile(f"{OUT}/ids.raw")
attn.tofile(f"{OUT}/attn.raw")
t_emb.tofile(f"{OUT}/t_emb.raw")
g_emb.tofile(f"{OUT}/g_emb.raw")
for name in ("queries.json", "gallery_files.json", "expected_topk.json"):
    open(f"{OUT}/{name}", "w").write(open(f"{G}/{name}").read())

meta = {"N": int(px.shape[0]), "M": int(ids.shape[0]), "seq": int(ids.shape[1]),
        "dim": int(g_emb.shape[1])}
json.dump(meta, open(f"{OUT}/meta.json", "w"))
print("raw ->", OUT, meta)
