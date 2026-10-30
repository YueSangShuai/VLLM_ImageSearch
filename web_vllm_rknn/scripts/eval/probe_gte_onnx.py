#!/usr/bin/env python3
"""探 GTE-ModernColBERT ONNX 的输入/输出签名与 CPU 延迟。"""
import os, sys, time
import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

D = "/data/yuesang/LLM/FG-CLIP/models/GTE-ModernColBERT-v1"
onnx = os.environ.get("GTE_ONNX", os.path.join(D, "model.onnx"))
so = ort.SessionOptions(); so.intra_op_num_threads = 4
s = ort.InferenceSession(onnx, so, providers=["CPUExecutionProvider"])
print("inputs:", [(i.name, i.shape, i.type) for i in s.get_inputs()])
print("outputs:", [(o.name, o.shape, o.type) for o in s.get_outputs()])

t = Tokenizer.from_file(os.path.join(D, "tokenizer.json"))
ids = np.array([t.encode("[Q] 穿黑色上衣的男生").ids])
am = np.ones_like(ids)
out = s.run(None, {"input_ids": ids, "attention_mask": am})
print("out shapes:", [o.shape for o in out])
for _ in range(5):
    t0 = time.time(); s.run(None, {"input_ids": ids, "attention_mask": am})
    print("  %.1f ms" % ((time.time() - t0) * 1000))
