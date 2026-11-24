# -*- coding: utf-8 -*-
"""对称 InfoNCE。多正例分支同 MARS multiple_texts_loss 的 label_matrix 思路：
同 id 的描述/图像互为正例（CUHK-PEDES 一图多描述、同人多描述）。
"""
import torch
import torch.nn.functional as F


def _multi_positive_ce(sim, ids):
    """sim [B,B] logits, ids [B]（图文同一批 pair，行列同一 id 序）。"""
    labels = (ids.unsqueeze(1) == ids.unsqueeze(0)).float()
    log_prob = sim - torch.logsumexp(sim, dim=-1, keepdim=True)
    return -(log_prob * labels).sum(-1) / labels.sum(-1).clamp(min=1.0)


def info_nce(img_emb, txt_emb, logit_scale, ids=None, multi_positive=False):
    """img/txt_emb 已 L2 归一化；logit_scale 为对数温度参数（内部 clamp exp<=100，CLIP 惯例）。
    返回 (loss, i2t_acc@1)。"""
    sim = logit_scale.exp().clamp(max=100.0) * img_emb @ txt_emb.t()
    if multi_positive and ids is not None:
        loss = 0.5 * (_multi_positive_ce(sim, ids).mean()
                      + _multi_positive_ce(sim.t().contiguous(), ids).mean())
    else:
        b = sim.shape[0]
        eye = torch.arange(b, device=sim.device)
        loss = 0.5 * (F.cross_entropy(sim, eye) + F.cross_entropy(sim.t(), eye))
    with torch.no_grad():
        acc = (sim.argmax(-1) == torch.arange(sim.shape[0], device=sim.device)).float().mean()
    return loss, acc
