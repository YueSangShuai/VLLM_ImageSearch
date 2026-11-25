#!/usr/bin/env python3
"""从 OpenSparX-gecko-guard-1B-v1（InternVLChatModel）里提取语言塔，
落成标准 Qwen2.5-0.5B（Qwen2ForCausalLM）目录，供 rkllm-toolkit load_huggingface 直转。

依据：config.json 的 llm_config 是 Qwen2ForCausalLM（hidden 896 / 24 层 / FFN 4864），
safetensors 键前缀 language_model.* / vision_model.* / mlp1.*。
rkllm-toolkit 1.3.1 wheel 内无 InternVL 加载逻辑（strings 实锤），故走「拆塔直转」路线，
与 web_vllm_rknn 既有链路同构（vision 走 rknn，LLM 走 rkllm）。
"""
import argparse
import json
import os
import shutil

from safetensors.torch import load_file, save_file

QWEN25_05B_TEMPLATE_KEYS = {  # Qwen2.5-0.5B-Instruct config 的字段模板（数值以 llm_config 为准）
    "architectures": ["Qwen2ForCausalLM"],
    "model_type": "qwen2",
    "auto_map": None,  # 明确删除 auto_map，杜绝 custom code
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="../models/gecko-guard-1b")
    ap.add_argument("--dst", default="../models/gecko-lm-qwen25-05b")
    args = ap.parse_args()

    src, dst = os.path.abspath(args.src), os.path.abspath(args.dst)
    os.makedirs(dst, exist_ok=True)

    cfg = json.load(open(os.path.join(src, "config.json")))
    llm_cfg = dict(cfg["llm_config"])
    llm_cfg.update(QWEN25_05B_TEMPLATE_KEYS)
    llm_cfg.pop("_name_or_path", None)
    llm_cfg.pop("transformers_version", None)
    llm_cfg.pop("_attn_implementation_autoset", None)
    # 删 None/空占位与非 qwen2 字段
    for k in ("auto_map", "image_fold", "downsample_ratio"):
        llm_cfg.pop(k, None)
    json.dump(llm_cfg, open(os.path.join(dst, "config.json"), "w"), indent=2, ensure_ascii=False)

    # 权重：剥 language_model. 前缀
    st = os.path.join(src, "model.safetensors")
    sd = load_file(st)
    lm, other = {}, {}
    for k, v in sd.items():
        if k.startswith("language_model."):
            lm[k[len("language_model."):]] = v
        else:
            other[k] = v
    tied = "lm_head.weight" not in lm
    print(f"llm tensors={len(lm)} (lm_head {'缺失,与embedding绑定' if tied else '存在'})")
    print(f"非LM张数={len(other)}（vision/mlp1，本脚本丢弃，另行导出）")
    save_file(lm, os.path.join(dst, "model.safetensors"),
              metadata={"format": "pt"})

    # tokenizer 全套照抄 gecko 的（含 <|TAG|> 等 added tokens 与 chat template）
    for f in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt",
              "added_tokens.json", "special_tokens_map.json", "chat_template.jinja"):
        s = os.path.join(src, f)
        if os.path.exists(s):
            shutil.copyfile(s, os.path.join(dst, f))
    print("tokenizer/chat_template 已复制 →", dst)


if __name__ == "__main__":
    main()
