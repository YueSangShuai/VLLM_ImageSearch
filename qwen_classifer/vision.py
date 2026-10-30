# -*- coding: utf-8 -*-
"""Qwen3-VL 视觉塔加载与特征提取（方案 A：整个 visual 冻结，只做前向）。"""
import torch

from common import NUM_ATTRS  # noqa: F401  (供调用方引用)

DEFAULT_MODEL = "/data/yuesang/LLM/Qwen3-VL-2B-Instruct"
MERGE_SIZE = 2  # patch merger 2x2，merge 后 token 数 = t*h*w / 4
DEEPSTACK_LAYERS = 3  # Qwen3-VL 有 3 个 deepstack 分支


def load_vision(model_path=DEFAULT_MODEL, dtype=torch.bfloat16, device="cuda"):
    """加载 Qwen3-VL 并取出冻结的视觉塔 model.model.visual。"""
    from transformers import Qwen3VLForConditionalGeneration

    try:  # transformers 5.x 用 dtype，4.x 用 torch_dtype
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path, dtype=dtype, low_cpu_mem_usage=True)
    except TypeError:
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=dtype, low_cpu_mem_usage=True)
    visual = model.model.visual
    model.model.visual = None  # 释放 LLM 权重显存，只留视觉塔
    del model
    torch.cuda.empty_cache()
    visual = visual.eval().requires_grad_(False).to(device)
    return visual


def load_processor(model_path=DEFAULT_MODEL):
    """只取 image_processor（transformers 5.x 的 Processor 强制要求 text 参数，
    纯图像任务直接用底层 image_processor 更稳）。"""
    from transformers import AutoImageProcessor
    return AutoImageProcessor.from_pretrained(model_path)


@torch.no_grad()
def extract_features(visual, pixel_values, grid_thw, use_deepstack=False):
    """pixel_values [total_patches, D], grid_thw [B,3] -> feats [B, F] (float32)。

    F = hidden(2048)，开启 deepstack 后 F = hidden * (1+3)。
    对每张图的 merge 后 token 做 mean pooling。
    """
    out = visual(hidden_states=pixel_values.to(visual.dtype),
                 grid_thw=grid_thw.to(torch.int64))
    main = out.pooler_output                      # [total_merged, 2048]
    ds = list(out.deepstack_features) if use_deepstack else []

    merge = MERGE_SIZE
    counts = (grid_thw.prod(-1) // merge ** 2).tolist()
    mains = torch.stack([p.mean(0) for p in main.split(counts)])       # [B, 2048]
    if not ds:
        return mains.float()
    parts = [mains]
    for d in ds:
        parts.append(torch.stack([p.mean(0) for p in d.split(counts)]))
    return torch.cat(parts, dim=-1).float()
