# rkllm / rknn 转换环境自装说明

代理：`export https_proxy=http://10.8.18.5:7890 http_proxy=http://10.8.18.5:7890`
（pip 走代理也可用 `--proxy http://10.8.18.5:7890`）

转换需要**两个独立 conda 环境**（LLM 与视觉塔的 torch/transformers 版本互斥）。

---

## 环境 A：LLM 转换 → `rkllm`（Python 3.10）

依赖清单：`requirements_rkllm.txt`（即 rknn-llm/rkllm-toolkit/packages/requirements.txt 的副本）

⚠️ 关键：先装 **CPU 版 torch**，否则默认源会连带拉 ~2.5GB CUDA 包（本机用不上）。

```bash
conda create -y -n rkllm python=3.10
RKLLM=/home/yuesang/miniconda3/envs/rkllm/bin

# 1) 先装 CPU torch（走 pytorch 官方 cpu 源）
$RKLLM/pip install --index-url https://download.pytorch.org/whl/cpu torch==2.6.0 torchvision==0.21.0

# 2) 再装其余依赖 + rkllm-toolkit 本体
cd /data/yuesang/LLM/FG-CLIP/web_vllm_rknn/rknn-llm
$RKLLM/pip install -r rkllm-toolkit/packages/requirements.txt \
    rkllm-toolkit/packages/rkllm_toolkit-1.3.1-cp310-cp310-linux_x86_64.whl
```

- 只有 `torch==2.6.0` 一行从 cpu 源装；`requirements_rkllm.txt` 里同名的 torch/torchvision 会被 pip 判为已满足、跳过
- `auto_gptq==0.7.1` 若无对应 wheel 会现场编译，需要 gcc，耗时较久
- 装完自检：`$RKLLM/python -c "import torch, transformers, rkllm; print(torch.__version__, transformers.__version__)"`
  期望 `2.6.0+cpu 5.8.0`

## 环境 B：视觉塔转换 → `rknn`（Python 3.10，**已建**）

rknn-toolkit2==2.5.0 + torch 2.4.0+cpu 已装好，**只差 transformers==4.57.0**：

```bash
RKNN=/home/yuesang/miniconda3/envs/rknn/bin
$RKNN/pip install transformers==4.57.0
# 自检：$RKNN/python -c "from rknn.api import RKNN; import transformers; print(transformers.__version__)"  → 4.57.0
```

---

## 版本对照（别混）

| 环境 | 用途 | torch | transformers | 关键包 |
|---|---|---|---|---|
| rkllm | LLM → .rkllm | 2.6.0 | **5.8.0** | rkllm-toolkit 1.3.1 |
| rknn  | 视觉塔 → .rknn | 2.4.0 | **4.57.0** | rknn-toolkit2 2.5.0 |

装完两个环境后告诉我，我来跑视觉塔 ONNX 导出（Phase 0 闸门）。
