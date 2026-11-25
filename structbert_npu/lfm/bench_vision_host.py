#!/usr/bin/env python3
# 宿主机 A800 bf16 视觉塔单独测速：Qwen3-VL ViT / LFM2.5-VL SigLIP2-NaFlex。
# （gecko-InternViT 部分见 bench_vision_host_gecko 内联命令，vllm env 跑，15.1ms 已测）
# 同一张 001 行人图，各自 processor 出真实输入，只跑 vision(+projector)，warmup 后取中位。
import time

import numpy as np
import torch
from PIL import Image

IMG = "/data/yuesang/LLM/FG-CLIP/web_vllm_rknn/scripts/eval/w8a8_ab/images/001_090662.jpg"
N = 10


def bench(name, fn):
    for _ in range(3):
        fn()
    ts = []
    for _ in range(N):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1000)
    print(f"{name}: median {np.median(ts):.1f} ms (min {min(ts):.1f})")


@torch.inference_mode()
def main():
    from transformers import AutoModelForImageTextToText, AutoProcessor

    # ---- Qwen3-VL-2B ViT ----
    import glob, os
    Q = os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen3-VL-2B-Instruct")
    qpath = sorted(glob.glob(os.path.join(Q, "snapshots/*")))[0]
    proc = AutoProcessor.from_pretrained(qpath)
    qm = AutoModelForImageTextToText.from_pretrained(qpath, dtype=torch.bfloat16).cuda().eval()
    img = Image.open(IMG).convert("RGB")
    inp = proc(images=[img], text=" ", return_tensors="pt").to("cuda")
    def qwen_vit():
        _ = qm.model.visual(inp["pixel_values"].to(torch.bfloat16),
                            grid_thw=inp["image_grid_thw"])
    qwen_vit()
    bench("Qwen3-VL-2B ViT", qwen_vit)
    print("  grid_thw =", inp["image_grid_thw"].tolist(), "pixel", tuple(inp["pixel_values"].shape))
    del qm
    torch.cuda.empty_cache()

    # ---- LFM2.5-VL SigLIP2 NaFlex ----
    L = "/data/yuesang/LLM/FG-CLIP/structbert_npu/models/LFM2.5-VL-1.6B"
    lproc = AutoProcessor.from_pretrained(L)
    lm = AutoModelForImageTextToText.from_pretrained(L, dtype=torch.bfloat16).cuda().eval()
    limg = Image.open(IMG).convert("RGB")
    conv = [{"role": "user", "content": [{"type": "image", "image": limg},
                                         {"type": "text", "text": "x"}]}]
    text = lproc.apply_chat_template(conv, tokenize=False, add_generation_prompt=False)
    enc = lproc(images=[limg], text=[text], return_tensors="pt", return_dict=True).to("cuda")
    px = enc["pixel_values"]
    pam = enc.get("pixel_attention_mask", None)
    print("  LFM pixel_values", tuple(px.shape), "mask", None if pam is None else tuple(pam.shape))
    called = {"f": False}
    def lfm_vit():
        feats = lm.model.vision_tower(pixel_values=px.to(torch.bfloat16),
                                      pixel_attention_mask=pam)
        out = feats.last_hidden_state if hasattr(feats, "last_hidden_state") else feats[0]
        _ = lm.model.multi_modal_projector(out)
        called["f"] = True
    try:
        lfm_vit()
        bench("LFM2.5-VL SigLIP2-NaFlex + projector", lfm_vit)
    except Exception as e:
        print("direct vision_tower failed:", repr(e)[:300])
        sp = enc["spatial_shapes"]
        def lfm_vit2():
            _ = lm.model.get_image_features(pixel_values=px.to(torch.bfloat16),
                                            pixel_attention_mask=pam,
                                            spatial_shapes=sp)
        lfm_vit2()
        bench("LFM2.5-VL get_image_features", lfm_vit2)


main()
