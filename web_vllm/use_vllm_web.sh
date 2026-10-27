#!/bin/bash
set -euo pipefail

# 杀掉旧的app.py进程（限定本项目路径，避免误杀其他人同名 app.py 进程）
pkill -f "python.*web_vllm/app.py" 2>/dev/null || true
sleep 1

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate FGCLIP2
cd /data/yuesang/LLM/FG-CLIP/web_vllm

export HF_HOME="${HF_HOME:-/data/yuesang/LLM/FG-CLIP/.hf_cache}"
# LoRA 微调合并模型（原始基座：/data/yuesang/LLM/Qwen3-VL-2B-Instruct）
export VLM_MODEL_NAME="${VLM_MODEL_NAME:-/data/yuesang/LLM/saves/Qwen3-VL-2B-Instruct/lora/train_2026-09-09/checkpoint-17254/lora}"
# 中文规则 + 中文描述 → 使用中文 BGE
export EMBEDDING_MODEL_NAME="${EMBEDDING_MODEL_NAME:-/data/yuesang/LLM/FG-CLIP/models/bge-small-zh-v1.5}"
export VLM_SEARCH_DIR="${VLM_SEARCH_DIR:-/data/yuesang/LLM/FG-CLIP/web_vllm/vlm_search}"
# 匹配模式: structured=仅结构化属性(符号匹配) | description=仅自由文本描述(向量匹配)
# combined=同时生成结构化属性与图像描述；匹配用图像描述，属性用于符号匹配规则和界面展示
# rule_direct=将规则列表代入提示词，由VLM直接判断图像是否含有该规则：
#   命中只输出规则名（多条用「、」分隔），未命中只输出「不匹配」，不走BGE向量匹配
export VLM_MATCH_MODE="${VLM_MATCH_MODE:-rule_direct}"

# 规则直判提示词（rule_direct 模式使用）：{rules} 占位符处由代码填入规则列表，不可删除
export VLM_RULE_JUDGE_PROMPT="${VLM_RULE_JUDGE_PROMPT:-请判断图像中的行人是否符合以下每一条规则。
{rules}
输出要求：只输出符合的规则名称，多条之间用顿号「、」分隔；如果没有任何规则符合，只输出：不匹配
除规则名称或「不匹配」外，不要输出任何其他文字、编号、标点或解释；只依据图像中清晰可见的内容判断，不要依据规则名称猜测。 <image>}"
export VLM_RULE_JUDGE_MAX_NEW_TOKENS="${VLM_RULE_JUDGE_MAX_NEW_TOKENS:-128}"

# 文本搜索提示词（中文指令 + 中文输出；description 模式使用，结构化模式下备用）
export VLM_DESCRIPTION_PROMPT="${VLM_DESCRIPTION_PROMPT:-请详细描述图中行人的外观，包括性别、大致年龄段、朝向、衣着款式与颜色、携带物品等清晰可见的属性。只描述画面中清晰可见的内容，不得猜测看不见或无法确认的特征；行人背对镜头或看不到脸部时，禁止断言眼镜等面部特征。对于无法看清或无法确定的属性，直接不要提及，严禁出现「是否佩戴眼镜难以判断」「看不清是否戴眼镜」这类包含疑问字样的表述。用连贯的中文陈述句输出，只写确定的事实，不要输出英文，不要输出多余的解释。 <image>}"

# 结构化提示词（structured模式使用）：只输出值为1的属性名，逗号分隔
export VLM_STRUCTURED_PROMPT="${VLM_STRUCTURED_PROMPT:-请分析这张行人图像，从以下预定义属性中找出所有存在的属性（即值为1的属性），只输出这些属性名，用逗号分隔，不要输出任何其他文字或值为0的属性。
预定义属性列表：Female,AgeOver60,Age18-60,AgeLess18,Front,Side,Back,Hat,Glasses,HandBag,ShoulderBag,Backpack,HoldObjectsInFront,ShortSleeve,LongSleeve,UpperStride,UpperLogo,UpperPlaid,UpperSplice,LowerStripe,LowerPattern,LongCoat,Trousers,Shorts,Skirt&Dress,boots,ublack,ugray,ublue,ugreen,uwhite,upurple,ured,ubrown,uyellow,upink,uorange,ubeige,ustriped_color,umulticolor,lwhite,lpink,lred,lgreen,lyellow,lpurple,lbrown,lblack,lorange,lblue,lgray,lbeige,lstriped_color,lmulticolor

<image>}"

export VLM_STRUCTURED_MAX_NEW_TOKENS="${VLM_STRUCTURED_MAX_NEW_TOKENS:-128}"
export VLM_RULE_PROMPT_MAX_TEXTS="${VLM_RULE_PROMPT_MAX_TEXTS:-3}"
export VLM_GENERATION_TEMPERATURE="${VLM_GENERATION_TEMPERATURE:-0.0}"
export VLM_MAX_NEW_TOKENS="${VLM_MAX_NEW_TOKENS:-320}"
export VLM_RULE_DESCRIPTION_ENABLED="${VLM_RULE_DESCRIPTION_ENABLED:-0}"
export VLM_RULE_DESCRIPTION_VERIFY="${VLM_RULE_DESCRIPTION_VERIFY:-1}"
export VLM_RULE_DESCRIPTION_MAX_NEW_TOKENS="${VLM_RULE_DESCRIPTION_MAX_NEW_TOKENS:-60}"
export VLM_REPETITION_PENALTY="${VLM_REPETITION_PENALTY:-1.2}"
export VLM_DEVICE="${VLM_DEVICE:-cuda:0}"
export EMBEDDING_DEVICE="${EMBEDDING_DEVICE:-$VLM_DEVICE}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

if [[ "$VLM_MODEL_NAME" == *.gguf ]]; then
    export VLM_MODEL_FORMAT="${VLM_MODEL_FORMAT:-gguf}"
    export VLM_MMPROJ_PATH="${VLM_MMPROJ_PATH:-/data/yuesang/LLM/test_qwen/mmproj-Qwen3VL-2B-Instruct-F16.gguf}"
    export VLM_GGUF_N_CTX="${VLM_GGUF_N_CTX:-4096}"
    export VLM_GGUF_N_GPU_LAYERS="${VLM_GGUF_N_GPU_LAYERS:--1}"
    export VLM_GGUF_CHAT_HANDLER="${VLM_GGUF_CHAT_HANDLER:-qwen25vl}"

    python - <<'PY'
import importlib.util
import os
import sys

model = os.environ["VLM_MODEL_NAME"]
if not os.path.isfile(model):
    raise SystemExit(f"VLM_MODEL_NAME 不存在: {model}")

if importlib.util.find_spec("llama_cpp") is None:
    raise SystemExit("缺少 llama-cpp-python；请在 FGCLIP2 环境中安装后再启动。")

mmproj = os.environ.get("VLM_MMPROJ_PATH", "")
if not mmproj:
    print(
        "[WARN] 当前选择 GGUF 模型，但未设置 VLM_MMPROJ_PATH。"
        "这个后端只能在提供对应 mmproj 视觉投影文件后处理图片；"
        "如要立即使用同目录 transformers 权重，可执行："
        "VLM_MODEL_NAME=/data/yuesang/LLM/test_qwen bash use_vllm_web.sh",
        file=sys.stderr,
    )
elif not os.path.isfile(mmproj):
    raise SystemExit(f"VLM_MMPROJ_PATH 不存在: {mmproj}")
PY
fi

python /data/yuesang/LLM/FG-CLIP/web_vllm/app.py
