# -*- coding: utf-8 -*-
"""检索指标：Recall@K 与 mAP（按 id 判相关，同人多图为正例）。"""
import torch


def recall_ap_chunked(sim, q_ids, g_ids, ks=(1, 5, 10), chunk=1024):
    """sim [Q,G] 相似度（任意设备），流式分块，返回 dict。"""
    device = sim.device
    rec = {k: 0.0 for k in ks}
    ap_sum = 0.0
    n = sim.shape[0]
    for i in range(0, n, chunk):
        s = sim[i:i + chunk]
        qi = q_ids[i:i + chunk]
        rel = (g_ids.unsqueeze(0) == qi.unsqueeze(1)).float()   # [C,G]
        order = s.argsort(dim=-1, descending=True)
        rel_sorted = torch.gather(rel, 1, order)
        for k in ks:
            rec[k] += rel_sorted[:, :k].sum(-1).gt(0).float().sum().item()
        prec = rel_sorted.cumsum(-1) / torch.arange(
            1, sim.shape[1] + 1, device=device, dtype=rel.dtype).unsqueeze(0)
        ap_sum += (prec * rel_sorted).sum(-1).div(rel.sum(-1).clamp(min=1)).sum().item()
    return {**{f"R@{k}": rec[k] / n for k in ks}, "mAP": ap_sum / n}
