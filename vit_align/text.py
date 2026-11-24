# -*- coding: utf-8 -*-
"""BGE-small-zh-v1.5 文本塔（冻结）：CLS pooling + L2，与生产 C++ text_embedder.cpp 同口径。

注意（记忆 [[bge-st-do-lower-case]]）：sentence_bert_config.json 的 do_lower_case
会覆盖 tokenizer_config 的 lowercase 设置，生产 C++ 按 sentence_bert_config 逐字小写。
这里显式读 sentence_bert_config 并手动小写，保证与板端一致。

自检：python text.py --selfcheck --onnx <生产 bge-small-zh.onnx>
"""
import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from common import load_config

DEFAULT_MODEL = "/data/yuesang/LLM/FG-CLIP/models/bge-small-zh-v1.5"
DEFAULT_ONNX = ("/data/yuesang/LLM/FG-CLIP/web_vllm_rknn/output_cpp/models/"
                "onnx/bge-small-zh.onnx")


class BGETextTower:
    def __init__(self, model_path=DEFAULT_MODEL, max_length=128, device="cuda",
                 dtype=torch.float32):
        from transformers import AutoTokenizer, BertModel

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = BertModel.from_pretrained(model_path).to(dtype).to(device)
        self.model.eval().requires_grad_(False)
        self.max_length = int(max_length)
        self.device = device
        self.dtype = dtype
        sc_path = os.path.join(model_path, "sentence_bert_config.json")
        self.do_lower_case = True
        if os.path.exists(sc_path):
            with open(sc_path, "r", encoding="utf-8") as f:
                self.do_lower_case = bool(json.load(f).get("do_lower_case", True))
        cfg = self.model.config
        assert cfg.hidden_size in (384, 512, 768), f"意外 hidden={cfg.hidden_size}"

    @torch.no_grad()
    def encode(self, texts, batch_size=128):
        """texts -> L2 归一化 [N, hidden] float32（doc 侧，不加前缀）。"""
        if self.do_lower_case:
            texts = [t.lower() for t in texts]
        embs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            enc = self.tokenizer(batch, padding=True, truncation=True,
                                 max_length=self.max_length, return_tensors="pt")
            enc = {k: v.to(self.device) for k, v in enc.items()}
            hidden = self.model(**enc).last_hidden_state          # [B, L, H]
            cls = hidden[:, 0]                                     # CLS pooling
            embs.append(F.normalize(cls.float(), dim=-1).cpu())
        return torch.cat(embs) if embs else torch.zeros(0, 384)


def selfcheck(onnx_path=DEFAULT_ONNX, model_path=DEFAULT_MODEL):
    """HF fp32 vs 生产 ONNX 同文本比对，要求 cos>=0.999。"""
    import onnxruntime as ort

    samples = [
        "一名穿红色外套戴眼镜的男子",
        "The man is wearing a blue and white striped tank top.",
        "为这个句子生成表示以用于检索相关文章：穿黑色羽绒服背着双肩包的女孩",
        "手持雨伞、灰色长裤、中年男性",
    ]
    t = BGETextTower(model_path=model_path, device="cpu")
    hf = t.encode(samples)

    sc_path = os.path.join(model_path, "sentence_bert_config.json")
    lower = True
    if os.path.exists(sc_path):
        lower = bool(json.load(open(sc_path)).get("do_lower_case", True))
    toks = t.tokenizer([s.lower() if lower else s for s in samples],
                       padding=True, truncation=True, max_length=t.max_length,
                       return_tensors="np")
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    feeds = {k: v.astype(np.int64) for k, v in toks.items()
             if k in [i.name for i in sess.get_inputs()]}
    on = sess.run(None, feeds)[0]                                  # last_hidden_state
    on = on[:, 0, :]
    on = on / np.maximum(np.linalg.norm(on, axis=1, keepdims=True), 1e-12)

    ok = True
    for s, a, b in zip(samples, hf.numpy(), on):
        cos = float(a @ b)
        flag = "OK " if cos >= 0.999 else "FAIL"
        ok &= cos >= 0.999
        print(f"[{flag}] cos={cos:.6f}  {s[:40]}")
    print("SELFCHECK", "PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selfcheck", action="store_true")
    ap.add_argument("--onnx", default=DEFAULT_ONNX)
    args = ap.parse_args()
    if args.selfcheck:
        cfg = load_config()
        ok = selfcheck(args.onnx, cfg["text"]["model_path"])
        raise SystemExit(0 if ok else 1)
