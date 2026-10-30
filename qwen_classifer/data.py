# -*- coding: utf-8 -*-
"""图像数据集：与板端部署一致的预处理（expand2square(127.5) -> resize 固定尺寸），
保证训练/部署特征对齐。像素值交给 Qwen3VLProcessor 做 patch 化与归一化。"""
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from common import load_entries, entries_to_labels

BG = 127.5  # 与 web_vllm_rknn 部署端 img_encoder.cpp 的 background_color 一致


def expand2square(img, bg=BG):
    """等比放置到正方形画布，空白填灰色，与部署端 cv2 实现等价。"""
    w, h = img.size
    side = max(w, h)
    canvas = Image.new("RGB", (side, side), (int(bg), int(bg), int(bg)))
    canvas.paste(img, ((side - w) // 2, (side - h) // 2))
    return canvas


class Pa100kDataset(Dataset):
    """返回 (uint8 CHW tensor [3,S,S], label int8 [54], path)。

    S 固定（默认 448，是 2*patch_size=32 的整数倍），这样每张图 grid 恒为
    [1, S/16, S/16]，pixel_values 可以按 batch 直接拼接。
    """

    def __init__(self, json_paths, size=448, limit=0, shard_id=0, num_shards=1):
        self.entries = load_entries(json_paths, limit)
        if num_shards > 1:  # 跨 GPU 并行提特征：按条目序切片
            self.entries = self.entries[shard_id::num_shards]
        self.paths = [e["image_path"] for e in self.entries]
        self.labels = torch.from_numpy(entries_to_labels(self.entries))
        self.size = int(size)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        img = expand2square(img)
        if img.size != (self.size, self.size):
            img = img.resize((self.size, self.size), Image.BILINEAR)
        arr = np.array(img, dtype=np.uint8)  # HWC，copy 保证可写
        pv = torch.from_numpy(arr).permute(2, 0, 1).contiguous()  # CHW uint8
        return pv, self.labels[idx], self.paths[idx]


def make_collate(image_processor):
    """把 uint8 图像过 Qwen3VL image_processor（patch 化 + (x-0.5)/0.5 归一化），
    输出可直接喂给 visual 的 pixel_values / grid_thw。"""

    def collate(batch):
        pvs, labels, paths = zip(*batch)
        imgs = [np.asarray(pv.permute(1, 2, 0).numpy()) for pv in pvs]
        out = image_processor(images=imgs, return_tensors="pt")
        # processor 会把 batch 内 pixel_values 拼成 [total_patches, D]，grid 每图一行
        return out["pixel_values"], out["image_grid_thw"], torch.stack(labels), list(paths)

    return collate
