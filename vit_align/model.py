# -*- coding: utf-8 -*-
"""唯一可训练件：图像侧 ProjMLP（结构模板取自 MARS vision_proj）。

图像路径: mean-pool 2048 -> Linear -> LayerNorm -> GELU -> Linear -> 512 -> L2
文本路径: BGE 512 (冻结) -> [可选 text_proj 恒等初始化线性投影] -> L2
温度:     logit_scale 可学习，初值 log(10)（同 MARS temp: 10）。
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ProjMLP(nn.Module):
    def __init__(self, in_dim=2048, hidden=1024, out_dim=512,
                 init_logit_scale=10.0, text_proj=False, text_dim=None):
        super().__init__()
        text_dim = text_dim or out_dim
        self.in_dim, self.hidden, self.out_dim = in_dim, hidden, out_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, out_dim),
        )
        self.logit_scale = nn.Parameter(torch.tensor(math.log(float(init_logit_scale))))
        # 文侧轻投影扩展位：恒等初始化，默认关（yaml head.text_proj）
        self.text_net = None
        if text_proj:
            self.text_net = nn.Linear(text_dim, text_dim)
            with torch.no_grad():
                self.text_net.weight.copy_(torch.eye(text_dim))
                self.text_net.bias.zero_()

    def encode_image(self, feats):
        return F.normalize(self.net(feats), dim=-1)

    def encode_text(self, emb):
        if self.text_net is not None:
            emb = self.text_net(emb)
        return F.normalize(emb, dim=-1)

    def trainable_state(self):
        return {"net": self.net.state_dict(),
                "logit_scale": float(self.logit_scale.detach().exp()),
                "text_net": self.text_net.state_dict() if self.text_net else None}

    def load_trainable(self, sd):
        self.net.load_state_dict(sd["net"])
        self.logit_scale.data = torch.tensor(math.log(sd["logit_scale"]))
        if sd.get("text_net") and self.text_net is not None:
            self.text_net.load_state_dict(sd["text_net"])


def load_model(ckpt_path, cfg, device="cpu"):
    ck = torch.load(ckpt_path, map_location="cpu")
    h = cfg["head"]
    model = ProjMLP(int(cfg["vision"]["feat_dim"]), int(h["hidden"]), int(h["out_dim"]),
                    float(h["init_logit_scale"]), bool(h.get("text_proj", False)),
                    int(cfg["text"]["feat_dim"]))
    model.load_trainable(ck["trainable"])
    return model.eval().to(device), ck
