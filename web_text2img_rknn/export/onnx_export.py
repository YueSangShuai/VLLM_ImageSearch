#!/usr/bin/env python
"""MARS(260320.pth) 双塔拆分导出 ONNX + golden（x86 主机，MARS conda env）。

产物（onnx/）：
  mars_image.onnx  pixel_values[1,3,224,224] float ([-1,1], 由 u8 RGB (x-127.5)/127.5 而来)
                   -> img_emb[1,256] (图内已 L2 归一化)
  mars_text.onnx   input_ids[1,128] int32 + attention_mask[1,128] float32 (1/0)
                   -> txt_emb[1,256] (图内已 L2 归一化)
  文本塔只导出前 fusion_layer=6 层（xbert mode='text' 路径），扩展掩码
  (1-mask)*-10000 在图内计算（-10000 在 fp16 内安全，规避 finfo.min 常数坑）。

约定与 RKNN 侧对齐：
  图像塔板端喂 uint8 RGB NHWC 0-255，mean/std=127.5 归一化 ≡ 本 ONNX 的 [-1,1] 输入。

golden/（板端对拍 + topk 基线）：
  gallery_files.json      图库相对文件名（pa100k/images 前 N_GOLD 张）
  gallery_px_u8.npy       [N,224,224,3] uint8 RGB（cv2 resize 224，C++ 侧用同法复现）
  gallery_emb_torch.npy   [N,256] L2 归一化
  queries.json            清洗后英文查询（原图 caption，走 dataset.utils.pre_caption）
  query_input_ids.npy     [M,128] int32
  query_attn.npy          [M,128] float32
  query_emb_torch.npy     [M,256]
  expected_topk.json      每条 query 的 topk（文件名+分数），板端排序基线

用法：cd /data/yuesang/LLM/MARS && \
  /home/yuesang/miniconda3/envs/MARS/bin/python \
  /data/yuesang/LLM/FG-CLIP/web_text2img_rknn/export/onnx_export.py
"""
import json
import os
import sys

MARS = "/data/yuesang/LLM/MARS"
os.chdir(MARS)
sys.path.insert(0, MARS)

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from models.model_person_search import ALBEF
from models.tokenization_bert import BertTokenizer
from dataset.utils import pre_caption

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "onnx")
GOLDEN = os.path.join(ROOT, "golden")
os.makedirs(OUT, exist_ok=True)
os.makedirs(GOLDEN, exist_ok=True)

CKPT = "/data/yuesang/LLM/260320.pth"
BERT = os.path.join(MARS, "bert-base-uncased")
GALLERY_DIR = "/data/yuesang/person_data/pa100k/images"
TXT_DIR = "/data/yuesang/person_data/pa100k/txts"
MAXLEN = 128
N_GOLD = 16  # golden 图库张数（另从 train_sample_128.json 混入几张不同来源图）


def build_model():
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    cfg = dict(ck["config"])
    if not os.path.isabs(cfg.get("bert_config", "")):
        cfg["bert_config"] = os.path.join(MARS, cfg["bert_config"])
    tok = BertTokenizer.from_pretrained(BERT)
    model = ALBEF(config=cfg, text_encoder=BERT, tokenizer=tok)
    missing, unexpected = model.load_state_dict(ck["model"], strict=False)
    missing = [k for k in missing if "teacher" not in k]
    print(f"missing={len(missing)} unexpected={len(unexpected)}")
    assert not missing and not unexpected, (missing[:5], unexpected[:5])
    model.eval()
    torch.set_grad_enabled(False)
    return model, tok, cfg


class ImageTower(torch.nn.Module):
    """EfficientNet-V2-S backbone -> classifier -> CLS token -> vision_proj。"""

    def __init__(self, ve, vp):
        super().__init__()
        self.ve = ve
        self.vp = vp

    def forward(self, pixel_values):
        tokens = self.ve(pixel_values)  # [B, 1+49, 768]
        return F.normalize(self.vp(tokens[:, 0, :]), dim=-1)


class TextTower(torch.nn.Module):
    """前 fusion_layer 层 BERT（mode='text' 路径）-> CLS -> text_proj。
    attention_mask 为 1/0 float [B,L]，扩展掩码 (1-m)*-10000 在图内算。"""

    def __init__(self, bert, proj, n_layer):
        super().__init__()
        self.bert = bert
        self.proj = proj
        self.n_layer = n_layer

    def forward(self, input_ids, attention_mask):
        hs = self.bert.embeddings(input_ids.long())
        ext = (1.0 - attention_mask[:, None, None, :].float()) * -10000.0
        for layer in self.bert.encoder.layer[: self.n_layer]:
            hs = layer(hs, ext)[0]
        return F.normalize(self.proj(hs[:, 0, :]), dim=-1)


def load_u8(path):
    """训练/评估同款预处理的前半段：cv2 resize 224 + BGR→RGB uint8。"""
    bgr = cv2.imread(path)
    assert bgr is not None, path
    rgb = cv2.resize(bgr, (224, 224))[:, :, ::-1].copy()
    return rgb


def u8_to_nchw(rgb):
    f = (rgb.astype(np.float32) / 255.0 - 0.5) / 0.5
    return np.transpose(f, (2, 0, 1))[None]  # [1,3,224,224] in [-1,1]


def main():
    model, tok, cfg = build_model()

    img_tower = ImageTower(model.visual_encoder, model.vision_proj)
    fusion_layer = int(
        getattr(model.text_encoder.config, "fusion_layer", 6)
    )
    print("fusion_layer(text 层数) =", fusion_layer)
    txt_tower = TextTower(model.text_encoder.bert, model.text_proj, fusion_layer)

    # ---------- 与原始 forward 对拍（导出 wrapper 正确性闸门） ----------
    demo_px = u8_to_nchw(load_u8(f"{GALLERY_DIR}/000001.jpg"))
    with torch.no_grad():
        x = torch.from_numpy(demo_px).float()
        v = model.visual_encoder(x)
        ref_img = F.normalize(model.vision_proj(v[:, 0, :]), dim=-1)
        got_img = img_tower(x)
        d_img = (ref_img - got_img).abs().max().item()

        q = "a woman in a white short-sleeve shirt and black trousers"
        ti = tok(
            pre_caption(q),
            padding="max_length",
            truncation=True,
            max_length=MAXLEN,
            return_tensors="pt",
        )
        ids = ti.input_ids[:, :MAXLEN]
        attn = (ids != 0).long()
        o = model.text_encoder.bert(ids, attention_mask=attn, mode="text")
        ref_txt = F.normalize(model.text_proj(o.last_hidden_state[:, 0, :]), dim=-1)
        got_txt = txt_tower(ids.int(), attn.float())
        d_txt = (ref_txt - got_txt).abs().max().item()
    print(f"wrapper check: img maxdiff={d_img:.2e} txt maxdiff={d_txt:.2e}")
    assert d_img < 1e-4 and d_txt < 1e-4

    # ---------- golden 素材 ----------
    def pick(names):
        """按序取前 N_GOLD 张可读图，坏图跳过。"""
        out = []
        for n in names:
            try:
                load_u8(os.path.join(GALLERY_DIR, n))
            except Exception:
                continue
            out.append(n)
            if len(out) >= N_GOLD:
                break
        return out

    all_names = [n for n in sorted(os.listdir(GALLERY_DIR)) if n.endswith(".jpg")]
    gallery = pick(all_names)
    # 混入 train_sample 里的几张（不同 pid，caption 现成）
    sample = json.load(open(os.path.join(MARS, "train_sample_128.json")))
    extra = [os.path.basename(e["image_path"]) for e in sample[:8]]
    for name in extra:
        if name not in gallery:
            try:
                load_u8(os.path.join(GALLERY_DIR, name))
                gallery.append(name)
            except Exception:
                pass

    px = np.stack([load_u8(os.path.join(GALLERY_DIR, n)) for n in gallery])
    with torch.no_grad():
        g_emb = torch.cat(
            [img_tower(torch.from_numpy(u8_to_nchw(p)).float()) for p in px]
        ).numpy().astype(np.float32)

    # 查询 = 每张 golden 图自己的 caption（清洗后）+ 两条自选描述
    queries = []
    for n in gallery[:N_GOLD]:
        cap = open(os.path.join(TXT_DIR, os.path.splitext(n)[0] + ".txt")).read()
        queries.append(pre_caption(cap))
    queries.append("man with red jacket and backpack")
    queries.append("person wearing black coat carrying handbag")

    ids_list, attn_list, emb_list = [], [], []
    with torch.no_grad():
        for q in queries:
            ti = tok(
                q,
                padding="max_length",
                truncation=True,
                max_length=MAXLEN,
                return_tensors="pt",
            )
            ids = ti.input_ids[:, :MAXLEN]
            attn = (ids != 0).long()
            e = txt_tower(ids.int(), attn.float()).numpy()[0]
            ids_list.append(ids.numpy()[0].astype(np.int32))
            attn_list.append(attn.numpy()[0].astype(np.float32))
            emb_list.append(e)
    q_emb = np.stack(emb_list).astype(np.float32)

    sims = q_emb @ g_emb.T
    k = 10
    expected = []
    for qi, q in enumerate(queries):
        idx = np.argsort(-sims[qi])[:k]
        expected.append(
            {
                "query": q[:80],
                "topk": [
                    {"file": gallery[i], "score": round(float(sims[qi][i]), 4)}
                    for i in idx
                ],
            }
        )

    # ---------- 导出 ONNX ----------
    x = torch.from_numpy(demo_px).float()
    torch.onnx.export(
        img_tower, (x,), os.path.join(OUT, "mars_image.onnx"),
        input_names=["pixel_values"], output_names=["img_emb"],
        opset_version=17, dynamo=False,
    )
    ids0 = torch.from_numpy(ids_list[0]).int()[None]
    attn0 = torch.from_numpy(attn_list[0])[None]
    torch.onnx.export(
        txt_tower, (ids0, attn0), os.path.join(OUT, "mars_text.onnx"),
        input_names=["input_ids", "attention_mask"], output_names=["txt_emb"],
        opset_version=17, dynamo=False,
    )

    # ---------- golden 落盘 ----------
    np.save(os.path.join(GOLDEN, "gallery_px_u8.npy"), px)
    np.save(os.path.join(GOLDEN, "gallery_emb_torch.npy"), g_emb)
    np.save(os.path.join(GOLDEN, "query_input_ids.npy"), np.stack(ids_list))
    np.save(os.path.join(GOLDEN, "query_attn.npy"), np.stack(attn_list))
    np.save(os.path.join(GOLDEN, "query_emb_torch.npy"), q_emb)
    json.dump(gallery, open(os.path.join(GOLDEN, "gallery_files.json"), "w"))
    json.dump(queries, open(os.path.join(GOLDEN, "queries.json"), "w"), indent=1)
    json.dump(expected, open(os.path.join(GOLDEN, "expected_topk.json"), "w"), indent=1)

    # ---------- ORT 对拍 ----------
    try:
        import onnxruntime as ort

        s_img = ort.InferenceSession(os.path.join(OUT, "mars_image.onnx"), providers=["CPUExecutionProvider"])
        s_txt = ort.InferenceSession(os.path.join(OUT, "mars_text.onnx"), providers=["CPUExecutionProvider"])
        got = np.concatenate([s_img.run(None, {"pixel_values": u8_to_nchw(p)})[0] for p in px])
        print("image onnx-vs-torch maxdiff:", np.abs(got - g_emb).max())
        got_t = np.concatenate([
            s_txt.run(None, {"input_ids": ids_list[i][None], "attention_mask": attn_list[i][None]})[0]
            for i in range(len(queries))
        ])
        print("text onnx-vs-torch maxdiff:", np.abs(got_t - q_emb).max())
    except ImportError:
        print("onnxruntime 不可用，跳过 ORT 对拍（rknn_convert.py 会做模拟器对拍）")

    print("OK. gallery=%d queries=%d" % (len(gallery), len(queries)))


if __name__ == "__main__":
    main()
