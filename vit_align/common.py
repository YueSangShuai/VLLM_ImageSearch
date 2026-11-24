# -*- coding: utf-8 -*-
"""配置加载、随机种子、json 条目读取与缓存分块工具。"""
import glob
import json
import os
import random

import numpy as np
import torch
import yaml

ROOT = os.path.dirname(os.path.abspath(__file__))


def load_config(path=None):
    path = path or os.path.join(ROOT, "configs", "default.yaml")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def abspath(p):
    """配置里的相对路径一律相对 vit_align/ 目录解析。"""
    return p if os.path.isabs(p) else os.path.join(ROOT, p)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_entries(json_paths):
    """多个 json（list of dict）拼接为一个条目列表。"""
    entries = []
    for p in json_paths:
        with open(p, "r", encoding="utf-8") as f:
            part = json.load(f)
        assert isinstance(part, list), f"{p} 应为 json list"
        entries.extend(part)
    return entries


def expand_cache_paths(patterns):
    """glob 通配（如 cache/train_i_*）；无通配时按前缀补 _*.pt。"""
    out = []
    for pat in patterns:
        pat = abspath(pat)
        if os.path.isfile(pat):  # shell 已展开成具体文件的路径
            out.append(pat)
            continue
        hits = sorted(glob.glob(pat)) if "*" in pat else sorted(glob.glob(pat + "_*.pt"))
        if not hits:
            raise FileNotFoundError(f"找不到缓存分片: {pat}")
        out.extend(hits)
    return out


def read_chunk(path):
    """读一个提取缓存分片（extract_*.py 产物），返回 dict（feats 转 float32）。"""
    d = torch.load(path, map_location="cpu")
    d["feats"] = d["feats"].float()
    return d
