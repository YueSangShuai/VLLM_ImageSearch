# -*- coding: utf-8 -*-
"""评估：test 集 i2t / t2i 的 R@1/5/10 + mAP；有前缀版 query 缓存时另报上线口径。

例：
  python eval.py --ckpt runs/full/best.pt
  python eval.py --ckpt runs/full/best.pt --prefix-cache cache/test_t_q_*

口径：图像向量 = ProjMLP(视觉塔 mean-pool)，文本向量 = BGE CLS（冻结）。
中文查询上线时应加官方前缀（query 空间）检索图像——--prefix-cache 即该口径的预演。
"""
import argparse
import os

import torch

from common import abspath, load_config
from data.dataset import build_index
from metrics import recall_ap_chunked
from model import load_model
from train import stack_cache


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--ckpt", default="runs/full/best.pt")
    ap.add_argument("--prefix-cache", default=None,
                    help="test caption 加前缀的文本缓存 glob（extract_text_feats.py --prefix 产物）")
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, ck = load_model(abspath(args.ckpt), cfg, device)
    if ck.get("i2t"):
        print(f"ckpt 记录: epoch {ck.get('epoch')} i2t R@1 {ck['i2t']['R@1']:.4f} "
              f"t2i R@1 {ck['t2i']['R@1']:.4f}")

    json_dir = abspath(cfg["data"]["json_dir"])
    entries, caps, cap_ids, _ = build_index([os.path.join(json_dir, "test.json")])
    img_x, img_ids = stack_cache(cfg["eval"]["img_cache"], cfg["vision"]["feat_dim"])
    txt_x, _ = stack_cache(cfg["eval"]["txt_cache"], cfg["text"]["feat_dim"])
    assert img_x.shape[0] == len(entries) and txt_x.shape[0] == len(caps)

    img_ids, cap_ids = img_ids.to(device), cap_ids.to(device)
    with torch.no_grad():
        img = model.encode_image(img_x.to(device))
        txt = model.encode_text(txt_x.to(device))
        i2t = recall_ap_chunked(img @ txt.t(), img_ids, cap_ids)
        t2i = recall_ap_chunked(txt @ img.t(), cap_ids, img_ids)
    print(f"[无前缀] i2t: " + "  ".join(f"{k} {v:.4f}" for k, v in i2t.items()))
    print(f"[无前缀] t2i: " + "  ".join(f"{k} {v:.4f}" for k, v in t2i.items()))

    if args.prefix_cache:
        qx, _ = stack_cache(args.prefix_cache, cfg["text"]["feat_dim"])
        assert qx.shape[0] == len(caps), "前缀缓存与 test.json caption 数不一致"
        with torch.no_grad():
            q = model.encode_text(qx.to(device))
            t2i_p = recall_ap_chunked(q @ img.t(), cap_ids.to(device), img_ids.to(device))
        print(f"[加前缀 query] t2i: " + "  ".join(f"{k} {v:.4f}" for k, v in t2i_p.items()))


if __name__ == "__main__":
    main()
