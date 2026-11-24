# -*- coding: utf-8 -*-
"""Qwen3-VL 视觉塔加载与特征提取（移植自 ../qwen_classifer/vision.py，修正权重路径）。

预处理与板端部署一致：expand2square(127.5) -> resize 固定边长 -> image_processor
patch 化 + (x-0.5)/0.5 归一化，保证训练/部署特征对齐。
"""
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

DEFAULT_MODEL = ("/home/yuesang/.cache/huggingface/hub/"
                 "models--Qwen--Qwen3-VL-2B-Instruct/snapshots/"
                 "89644892e4d85e24eaac8bacfd4f463576704203")
MERGE_SIZE = 2
BG = 127.5  # 与 web_vllm_rknn 部署端 img_encoder.cpp 的 background_color 一致


def load_vision(model_path=DEFAULT_MODEL, dtype=torch.bfloat16, device="cuda"):
    """加载 Qwen3-VL 并取出冻结的视觉塔 model.model.visual（释放 LLM 权重）。"""
    from transformers import Qwen3VLForConditionalGeneration

    try:  # transformers 5.x 用 dtype，4.x 用 torch_dtype
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path, dtype=dtype, low_cpu_mem_usage=True)
    except TypeError:
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=dtype, low_cpu_mem_usage=True)
    visual = model.model.visual
    model.model.visual = None
    del model
    torch.cuda.empty_cache()
    return visual.eval().requires_grad_(False).to(device)


def load_processor(model_path=DEFAULT_MODEL):
    """只取 image_processor（纯图像任务避免 Processor 的 text 参数要求）。"""
    from transformers import AutoImageProcessor
    return AutoImageProcessor.from_pretrained(model_path)


@torch.no_grad()
def extract_features(visual, pixel_values, grid_thw, use_deepstack=False):
    """pixel_values [total_patches, D], grid_thw [B,3] -> feats [B, F] float32。

    对每张图 merge 后的 token 做 mean pooling（与板端 AttrHead 的 pooling 口径一致）。
    """
    dev = next(visual.parameters()).device
    out = visual(hidden_states=pixel_values.to(device=dev, dtype=visual.dtype),
                 grid_thw=grid_thw.to(device=dev, dtype=torch.int64))
    main = out.pooler_output                      # [total_merged, 2048]
    ds = list(out.deepstack_features) if use_deepstack else []

    counts = (grid_thw.prod(-1) // MERGE_SIZE ** 2).tolist()
    mains = torch.stack([p.mean(0) for p in main.split(counts)])       # [B, 2048]
    if not ds:
        return mains.float()
    parts = [mains]
    for d in ds:
        parts.append(torch.stack([p.mean(0) for p in d.split(counts)]))
    return torch.cat(parts, dim=-1).float()


def expand2square(img, bg=BG):
    """等比放置到正方形画布，空白填灰，与部署端 cv2 实现等价。"""
    w, h = img.size
    side = max(w, h)
    canvas = Image.new("RGB", (side, side), (int(bg), int(bg), int(bg)))
    canvas.paste(img, ((side - w) // 2, (side - h) // 2))
    return canvas


class FixedSizeImageDataset(Dataset):
    """按条目列表（含 image_path）输出固定边长的 uint8 CHW 图像。"""

    def __init__(self, entries, size=448):
        self.paths = [e["image_path"] for e in entries]
        self.size = int(size)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        img = expand2square(img)
        if img.size != (self.size, self.size):
            img = img.resize((self.size, self.size), Image.BILINEAR)
        arr = np.array(img, dtype=np.uint8)
        return torch.from_numpy(arr).permute(2, 0, 1).contiguous(), idx


def make_collate(image_processor):
    """uint8 图像过 Qwen3VL image_processor -> visual 可吃的 pixel_values/grid_thw。"""

    def collate(batch):
        pvs, idxs = zip(*batch)
        imgs = [np.asarray(pv.permute(1, 2, 0).numpy()) for pv in pvs]
        out = image_processor(images=imgs, return_tensors="pt")
        return out["pixel_values"], out["image_grid_thw"], torch.tensor(idxs)

    return collate
