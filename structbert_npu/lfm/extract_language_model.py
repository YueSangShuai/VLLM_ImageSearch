#!/usr/bin/env python3
"""从 LiquidAI/LFM2.5-VL-1.6B 提取语言塔，落成标准 Lfm2ForCausalLM 目录，
供 rkllm-toolkit load_huggingface 直转（loader.so 支持列表含 Lfm2ForCausalLM）。

依据：safetensors 键前缀 model.language_model.* / model.vision_tower.* /
model.multi_modal_projector.*；text_config 即完整 lfm2 config；
无 lm_head（tie_word_embeddings 与 embedding 绑定）。与 gecko 拆塔同法。
"""
import argparse
import json
import os
import shutil

from safetensors.torch import load_file, save_file


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="../models/LFM2.5-VL-1.6B")
    ap.add_argument("--dst", default="../models/LFM2.5-VL-1.6B-lm-lfm2")
    args = ap.parse_args()

    src, dst = os.path.abspath(args.src), os.path.abspath(args.dst)
    os.makedirs(dst, exist_ok=True)

    cfg = json.load(open(os.path.join(src, "config.json")))
    t = dict(cfg["text_config"])
    t["architectures"] = ["Lfm2ForCausalLM"]
    t["model_type"] = "lfm2"
    t["tie_word_embeddings"] = True
    for k in ("_name_or_path", "transformers_version", "_attn_implementation_autoset"):
        t.pop(k, None)
    json.dump(t, open(os.path.join(dst, "config.json"), "w"), indent=2,
              ensure_ascii=False)

    sd = load_file(os.path.join(src, "model.safetensors"))
    P = "model.language_model."
    lm = {k: v for k, v in ((f"model.{kk[len(P):]}", v)
                            for kk, v in sd.items() if kk.startswith(P))}
    dropped = len(sd) - len(lm)
    print(f"lm tensors={len(lm)} (lm_head 缺失,与embedding绑定), 丢弃 vision/projector={dropped}")
    save_file(lm, os.path.join(dst, "model.safetensors"), metadata={"format": "pt"})

    for f in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
              "generation_config.json"):
        s = os.path.join(src, f)
        if os.path.exists(s):
            shutil.copyfile(s, os.path.join(dst, f))
    print("tokenizer/chat_template 已复制 →", dst)


if __name__ == "__main__":
    main()
