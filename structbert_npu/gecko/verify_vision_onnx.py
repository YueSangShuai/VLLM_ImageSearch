#!/usr/bin/env python
# ONNX(ORT) vs torch fp32 extract_feature 数值对齐：真实 A/B 行人图，max|Δ| + cos。
# 用法（Qwen env）：python verify_vision_onnx.py
import os

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from transformers import AutoModel

MODEL = "/data/yuesang/LLM/FG-CLIP/structbert_npu/models/gecko-guard-1b"
ONNX = "/data/yuesang/LLM/FG-CLIP/structbert_npu/gecko/onnx/internvl3-1b_vision.onnx"
IMAGES = "/data/yuesang/LLM/FG-CLIP/web_vllm_rknn/scripts/eval/w8a8_ab/images"

import onnxruntime as ort

tfm = T.Compose([
    T.Lambda(lambda im: im.convert("RGB")),
    T.Resize((448, 448), interpolation=T.InterpolationMode.BICUBIC),
    T.ToTensor(),
    T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

torch.manual_seed(0)
sess = ort.InferenceSession(ONNX, providers=["CPUExecutionProvider"])
in_name = sess.get_inputs()[0].name

# 随机噪声也要对齐（rknn 校准/精度探针口径）
cases = [("rand", np.random.randn(1, 3, 448, 448).astype(np.float32))]
for name in sorted(os.listdir(IMAGES))[:3]:
    img = Image.open(os.path.join(IMAGES, name))
    cases.append((name, tfm(img).unsqueeze(0).numpy()))

model = AutoModel.from_pretrained(MODEL, torch_dtype=torch.float32,
                                  trust_remote_code=True,
                                  low_cpu_mem_usage=False).eval()

for name, x in cases:
    with torch.no_grad():
        ref = model.extract_feature(torch.from_numpy(x)).numpy()
    got = sess.run(None, {in_name: x})[0]
    d = float(np.max(np.abs(ref - got)))
    cos = float((ref.ravel() @ got.ravel()) /
                (np.linalg.norm(ref) * np.linalg.norm(got) + 1e-12))
    print(f"{name}: shape {got.shape} max|Δ|={d:.2e} cos={cos:.6f}")
