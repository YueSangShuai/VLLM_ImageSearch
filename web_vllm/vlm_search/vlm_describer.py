# -*- coding: utf-8 -*-
"""VLMDescriber：Qwen-VL 图像描述封装（重建版）。

支持两种后端：
  1. HuggingFace transformers 模型目录（推荐，如合并 LoRA 后的 Qwen3-VL）
  2. GGUF（llama_cpp + mmproj，由 VLM_MMPROJ_PATH 指定视觉投影文件）

processor.py 会剥掉提示词里的 <image> 占位符，图片由本类插入消息。
"""

import os
import re
import threading

import torch


def _extract_image_placeholder(prompt: str):
    """把 <image> 占位符从文本中拆出来，返回 (text, 图片是否出现在文本中)。"""
    text, n = re.subn(r"[ \t]*<\s*image\s*>[ \t]*", "\n", prompt or "", flags=re.IGNORECASE)
    return text.strip(), n > 0


class VLMDescriber:
    """输入 PIL 图像 + 文本提示，输出模型生成的描述文本。"""

    def __init__(self, model_name: str, device: str = "cpu", template: str = None,
                 torch_dtype=torch.bfloat16):
        self.model_name = model_name
        self.device = device
        self.template = template
        self._lock = threading.Lock()  # 监控多线程共用同一个 describer

        if model_name.lower().endswith(".gguf"):
            self.backend = "gguf"
            self._init_gguf(model_name, device)
        else:
            self.backend = "hf"
            self._init_hf(model_name, device, torch_dtype)

    # ---------- HuggingFace 后端 ----------

    def _init_hf(self, model_name: str, device: str, torch_dtype):
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.model = AutoModelForImageTextToText.from_pretrained(
            model_name, dtype=torch_dtype, trust_remote_code=True
        )
        self.model.to(device)
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

    @torch.inference_mode()
    def _describe_hf(self, image, prompt, max_new_tokens, temperature, repetition_penalty):
        text, _ = _extract_image_placeholder(prompt)
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": text},
            ],
        }]
        text_input = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            images=[image], text=[text_input], return_tensors="pt", padding=True
        ).to(self.device)

        gen_kwargs = {
            "max_new_tokens": int(max_new_tokens),
            "repetition_penalty": float(repetition_penalty),
            "do_sample": temperature is not None and float(temperature) > 0,
        }
        if gen_kwargs["do_sample"]:
            gen_kwargs["temperature"] = float(temperature)

        with self._lock:
            output = self.model.generate(**inputs, **gen_kwargs)

        generated = output[0][inputs["input_ids"].shape[1]:]
        return self.processor.decode(generated, skip_special_tokens=True).strip()

    # ---------- GGUF 后端 ----------

    def _init_gguf(self, model_name: str, device: str):
        from llama_cpp import Llama

        if not os.path.isfile(model_name):
            raise FileNotFoundError(f"GGUF 模型不存在: {model_name}")
        mmproj = os.getenv("VLM_MMPROJ_PATH", "")
        if not mmproj:
            raise ValueError("GGUF 后端需要设置 VLM_MMPROJ_PATH（视觉投影文件）")
        n_gpu_layers = int(os.getenv("VLM_GGUF_N_GPU_LAYERS", "-1"))
        if device.startswith("cpu"):
            n_gpu_layers = 0
        self.llm = Llama(
            model_path=model_name,
            chat_format=os.getenv("VLM_GGUF_CHAT_HANDLER", "qwen25vl"),
            chat_handler=None,
            n_ctx=int(os.getenv("VLM_GGUF_N_CTX", "4096")),
            n_gpu_layers=n_gpu_layers,
            verbose=False,
            **({"mmproj": mmproj} if os.path.isfile(mmproj) else {}),
        )

    def _describe_gguf(self, image, prompt, max_new_tokens, temperature, repetition_penalty):
        import base6
        import io

        text, _ = _extract_image_placeholder(prompt)
        buf = io.BytesIO()
        image.convert("RGB").save(buf, format="JPEG", quality=90)
        b64 = base64.b64encode(buf.getvalue()).decode()

        resp = self.llm.create_chat_completion(
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": text},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                ],
            }],
            max_tokens=int(max_new_tokens),
            temperature=float(temperature) if temperature else 0.0,
            repeat_penalty=float(repetition_penalty),
        )
        return (resp["choices"][0]["message"]["content"] or "").strip()

    # ---------- 对外接口 ----------

    def describe(self, image, prompt: str = None, max_new_tokens: int = 120,
                 temperature: float = 0.0, repetition_penalty: float = 1.0) -> str:
        """生成图像描述。prompt 缺省时使用初始化传入的 template。"""
        prompt = prompt or self.template or ""
        if self.backend == "gguf":
            return self._describe_gguf(image, prompt, max_new_tokens,
                                       temperature, repetition_penalty)
        return self._describe_hf(image, prompt, max_new_tokens,
                                 temperature, repetition_penalty)
