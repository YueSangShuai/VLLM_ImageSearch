#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""StructBERT (damo/nlp_structbert_sentence-similarity_chinese-base) → ONNX，
为 RKNN 转换准备：固定 seq_len=128、batch=1、int32 输入。

导出两个：
  sbert_tower.onnx  孪生塔：input_ids/attention_mask/token_type_ids → pooled[1,768]
  sbert_pair.onnx   交叉编码：pair 拼接输入 → logits[1,2]（P(相似)=softmax[1]）
运行：/home/yuesang/miniconda3/envs/nomic/bin/python export_structbert_onnx.py
"""
import os

import torch

MODEL_DIR = ("/data/yuesang/LLM/FG-CLIP/models/.ms_cache/damo/"
             "nlp_structbert_sentence-similarity_chinese-base")
OUT = "/data/yuesang/LLM/FG-CLIP/web_vllm_rknn/models/rk3588"
SEQ = 128


def main():
    os.makedirs(OUT, exist_ok=True)
    from modelscope.models import Model
    model = Model.from_pretrained(MODEL_DIR)
    model.eval()

    # 关键：attention_mask 以 float32 扩展掩码 [1,1,1,SEQ]（0/-10000）作输入。
    # 图内由 int mask 现算 extended mask 会引入 (1-m)*finfo.min(≈-3.4e38) 大常数，
    # RKNN fp16 下变 -inf → softmax NaN、输出恒定，实测不可绕（opt_level 0/1/2 全中）。
    ids = torch.ones(1, SEQ, dtype=torch.int32)
    am = torch.zeros(1, 1, 1, SEQ, dtype=torch.float32)
    tt = torch.zeros(1, SEQ, dtype=torch.int32)

    class Tower(torch.nn.Module):
        def __init__(self, enc):
            super().__init__()
            self.enc = enc

        def forward(self, input_ids, attention_mask, token_type_ids):
            out = self.enc(input_ids=input_ids.long(),
                           attention_mask=attention_mask,
                           token_type_ids=token_type_ids.long())
            vec = getattr(out, "pooled_output", None)
            if vec is None:
                hs = out[1] if isinstance(out, tuple) and len(out) > 1 else out[0]
                vec = hs[:, 0]
            return vec

    class Pair(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, input_ids, attention_mask, token_type_ids):
            out = self.m(input_ids=input_ids.long(),
                         attention_mask=attention_mask,
                         token_type_ids=token_type_ids.long())
            return out["logits"] if isinstance(out, dict) else out.logits

    for name, mod in (("sbert_tower", Tower(model.encoder)),
                      ("sbert_pair", Pair(model))):
        path = os.path.join(OUT, name + ".onnx")
        with torch.no_grad():
            torch.onnx.export(
                mod, (ids, am, tt), path,
                input_names=["input_ids", "attention_mask", "token_type_ids"],
                output_names=["output"],
                opset_version=11, do_constant_folding=True)
        print("exported", path, os.path.getsize(path) // 2**20, "MB")

        import onnxruntime as ort
        so = ort.SessionOptions()
        s = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
        y = s.run(None, {"input_ids": ids.numpy(), "attention_mask": am.numpy(),
                         "token_type_ids": tt.numpy()})[0]
        print("  onnxruntime 自检 ok, out shape:", y.shape)


if __name__ == "__main__":
    main()
