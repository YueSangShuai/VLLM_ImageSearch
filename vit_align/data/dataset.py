# -*- coding: utf-8 -*-
"""json 条目 -> (图像, 描述) 样本对索引。

约定：图像缓存按 json 条目顺序对齐（extract_image_feats.py 保序写盘）；
文本缓存按 build_index 展开出的 caption 顺序对齐（extract_text_feats.py 用同一函数）。
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import load_entries  # noqa: E402


def build_index(json_paths):
    """-> images (json 条目列表，顺序即图像缓存顺序),
        captions (list[str]), cap_ids (tensor long),
        pairs: list[(img_pos, cap_pos, id)]
    """
    entries = load_entries(json_paths)
    captions, cap_ids, pairs = [], [], []
    for img_pos, e in enumerate(entries):
        caps = e.get("captions") or [e["text"]]
        for txt in caps:
            pairs.append((img_pos, len(captions), int(e["id"])))
            captions.append(txt)
            cap_ids.append(int(e["id"]))
    return entries, captions, torch.tensor(cap_ids, dtype=torch.long), pairs


class PairDataset:
    """缓存矩阵 + pair 索引的轻量训练集（全部驻留内存，batch 时才上 GPU）。"""

    def __init__(self, img_matrix, txt_matrix, pairs):
        self.img = img_matrix          # [Ni, 2048] float32
        self.txt = txt_matrix          # [Nc, 512]  float32
        p = torch.tensor(pairs, dtype=torch.long, device=self.img.device)
        self.img_pos = p[:, 0]
        self.cap_pos = p[:, 1]
        self.ids = p[:, 2]

    def __len__(self):
        return len(self.ids)

    def batch(self, idx):
        ip = self.img_pos[idx]
        cp = self.cap_pos[idx]
        return self.img[ip], self.txt[cp], self.ids[idx]
