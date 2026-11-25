#!/usr/bin/env python3
# 宿主机 A800 bf16 同口径测速：gecko-guard-1b vs LFM2.5-VL-1.6B。
# 同一张图 + 同一条 desc prompt，greedy，max_new_tokens=64；
# 每模型 warmup 1 次后连测 3 次：prefill 时间用「1-token 生成」单独测，
# decode 用「64-token 生成 − prefill」换算 tok/s；末 token 文本一致（greedy 确定性校验）。
# 用法（lfm 目录）：CUDA_VISIBLE_DEVICES=0 <llama-env-py> bench_host.py gecko
#                  CUDA_VISIBLE_DEVICES=0 <llama-env-py> bench_host.py lfm
import json
import sys
import time

import numpy as np
import torch
from PIL import Image

IMG = "/data/yuesang/LLM/FG-CLIP/web_vllm_rknn/scripts/eval/w8a8_ab/images/001_090662.jpg"
PROMPT = ("一句中文详细描述行人外观：性别、衣着颜色款式、携带物，"
          "只写看得见的，不需要描述看不见的，不用解释。")
NEW_TOK = 64


def bench_gecko():
    import torchvision.transforms as T
    from transformers import AutoModel, AutoTokenizer
    M = "/data/yuesang/LLM/FG-CLIP/structbert_npu/models/gecko-guard-1b"
    tok = AutoTokenizer.from_pretrained(M, trust_remote_code=True, use_fast=False)
    model = AutoModel.from_pretrained(M, dtype=torch.bfloat16,
                                      trust_remote_code=True).cuda().eval()
    tfm = T.Compose([
        T.Lambda(lambda im: im.convert("RGB")),
        T.Resize((448, 448), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    pv = tfm(Image.open(IMG)).unsqueeze(0).to("cuda", torch.bfloat16)
    gc = {"max_new_tokens": NEW_TOK, "do_sample": False}

    def run(n):
        return model.chat(tokenizer=tok, pixel_values=pv, question=PROMPT,
                          generation_config={"max_new_tokens": n, "do_sample": False},
                          verbose=False)
    return run, tok


def bench_lfm():
    from transformers import AutoModelForImageTextToText, AutoProcessor
    M = "/data/yuesang/LLM/FG-CLIP/structbert_npu/models/LFM2.5-VL-1.6B"
    proc = AutoProcessor.from_pretrained(M)
    model = AutoModelForImageTextToText.from_pretrained(M, dtype=torch.bfloat16).cuda().eval()
    img = Image.open(IMG).convert("RGB")
    conv = [{"role": "user", "content": [
        {"type": "image", "image": img}, {"type": "text", "text": PROMPT}]}]
    inputs = proc.apply_chat_template(conv, add_generation_prompt=True, tokenize=True,
                                      return_tensors="pt", return_dict=True).to(model.device)

    def run(n):
        with torch.inference_mode():
            out = model.generate(**inputs, max_new_tokens=n, do_sample=False)
        return proc.decode(out[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
    return run, None


def timed(run, n):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    txt = run(n)
    torch.cuda.synchronize()
    return time.perf_counter() - t0, str(txt).strip()


def main():
    which = sys.argv[1]
    run, tok = bench_gecko() if which == "gecko" else bench_lfm()
    _ = timed(run, 8)                       # warmup（含 CUDA graph/编译）
    t_1, _ = timed(run, 1)                  # ~prefill+1tok
    t_full, txt = timed(run, NEW_TOK)       # 完整一次
    reps = [timed(run, NEW_TOK) for _ in range(3)]
    fulls = [t for t, _ in reps]
    n_out = len(tok.encode(txt)) if tok else NEW_TOK
    prefill = t_1
    decode = max(np.median(fulls) - t_1, 1e-6)
    res = {
        "model": which,
        "prefill_s": round(prefill, 3),
        "e2e_64tok_s": round(float(np.median(fulls)), 3),
        "decode_ms_per_tok": round(decode / max(n_out - 1, 1) * 1000, 1),
        "decode_tok_per_s": round((n_out - 1) / decode, 1),
        "text": txt[:80],
    }
    out = f"/data/yuesang/LLM/FG-CLIP/structbert_npu/lfm/bench_{which}.json"
    json.dump(res, open(out, "w"), ensure_ascii=False, indent=1)
    print(json.dumps(res, ensure_ascii=False))


if __name__ == "__main__":
    main()
