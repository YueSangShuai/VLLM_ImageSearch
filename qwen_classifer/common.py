# -*- coding: utf-8 -*-
"""共享常量与工具：54 个属性定义、数据加载、标签 mask、AP/mAP 指标。"""
import json
import numpy as np

# 54 个 PA-100K 属性，顺序与 train_sample_128.json 保持一致
ATTRS = [
    "Female", "AgeOver60", "Age18-60", "AgeLess18",
    "Front", "Side", "Back",
    "Hat", "Glasses",
    "HandBag", "ShoulderBag", "Backpack", "HoldObjectsInFront",
    "ShortSleeve", "LongSleeve", "UpperStride", "UpperLogo", "UpperPlaid", "UpperSplice",
    "LowerStripe", "LowerPattern", "LongCoat", "Trousers", "Shorts", "Skirt&Dress", "boots",
    "ublack", "ugray", "ublue", "ugreen", "uwhite", "upurple", "ured", "ubrown",
    "uyellow", "upink", "uorange", "ubeige", "ustriped_color", "umulticolor",
    "lwhite", "lpink", "lred", "lgreen", "lyellow", "lpurple", "lbrown", "lblack",
    "lorange", "lblue", "lgray", "lbeige", "lstriped_color", "lmulticolor",
]
NUM_ATTRS = len(ATTRS)  # 54

NON_LABEL_KEYS = {"id", "image_path", "text_path"}


def load_entries(json_paths, limit=0):
    """读取 PA-100K 风格 json（list of dict），支持多个文件拼接。

    额外字段（hair_length/attributes_path 等）自动忽略；
    缺失的属性按 -1（unknown）处理，训练与评测会自动 mask。
    """
    if isinstance(json_paths, str):
        json_paths = [json_paths]
    entries = []
    for p in json_paths:
        with open(p) as f:
            part = json.load(f)
        bad = [e for e in part[:20] if not any(k in e for k in ATTRS)]
        if len(bad) == min(20, len(part)):
            raise ValueError(f"{p} 前 20 条均无属性字段，格式不兼容")
        entries.extend(part)
    if limit > 0:
        entries = entries[:limit]
    return entries


def entries_to_labels(entries):
    """-> labels int8 [N, 54]，保留原始 -1/0/1（-1 = unknown，训练与评测都要 mask 掉）。"""
    N = len(entries)
    labels = np.zeros((N, NUM_ATTRS), dtype=np.int8)
    for i, e in enumerate(entries):
        for j, a in enumerate(ATTRS):
            labels[i, j] = int(e.get(a, -1))
    return labels


def average_precision(scores, labels, mask):
    """单属性 AP（all-point 插值），mask=False 的样本不参与。

    scores: float [N]；labels: 0/1 [N]；mask: 有效样本 [N] bool。
    """
    scores = scores[mask]
    labels = labels[mask]
    pos = labels.sum()
    if pos == 0 or pos == len(labels):
        return None  # 无正样本或全正，PA-100K mAP 惯例中跳过该属性
    order = np.argsort(-scores)
    labels = labels[order]
    tp = np.cumsum(labels)
    fp = np.cumsum(1 - labels)
    recall = tp / pos
    precision = tp / (tp + fp)
    # all-point 插值：precision 从后往前取后缀最大
    precision = np.maximum.accumulate(precision[::-1])[::-1]
    rec_prev = np.concatenate([[0.0], recall[:-1]])
    return float(np.sum((recall - rec_prev) * precision))


def compute_map(scores, labels):
    """scores/labels: [N, 54]。逐属性 AP（-1 位置 mask），返回 (mAP, per-attr dict)。"""
    mask = labels >= 0
    y = (labels == 1).astype(np.int32)
    aps = {}
    for j, a in enumerate(ATTRS):
        ap = average_precision(scores[:, j].astype(np.float64), y[:, j], mask[:, j])
        if ap is not None:
            aps[a] = ap
    map_ = float(np.mean(list(aps.values()))) if aps else 0.0
    return map_, aps


def per_attribute_accuracy(scores, labels, thresh=0.5):
    """逐属性二分类准确率（阈值 0.5，-1 样本 mask 后计算）。

    scores/labels: [N, 54]。返回 (平均准确率, per-attr dict)。
    某属性全为 -1 时无有效样本，该属性跳过。
    """
    mask = labels >= 0
    y = (labels == 1)
    pred = scores >= thresh
    correct = (pred == y) & mask[:, :]  # -1 处 pred==y 可能巧合成立，必须交 mask
    accs = {}
    for j, a in enumerate(ATTRS):
        n = mask[:, j].sum()
        if n:
            accs[a] = float(correct[:, j][mask[:, j]].sum() / n)
    mean = float(np.mean(list(accs.values()))) if accs else 0.0
    return mean, accs


def valid_pos_weight(labels):
    """按有效样本（!=-1）计算每属性 BCE pos_weight = neg/pos，clamp 到 [1, 20]。"""
    valid = labels >= 0
    pos = (labels == 1).sum(0).astype(np.float64)
    neg = (labels == 0).sum(0).astype(np.float64)
    pos = np.maximum(pos, 1.0)
    w = neg / pos
    return np.clip(w, 1.0, 20.0).astype(np.float32)
