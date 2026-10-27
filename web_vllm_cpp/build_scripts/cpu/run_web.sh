#!/bin/bash
# C++ 版【纯 CPU】启动脚本：无 GPU 依赖。由 build_scripts/cpu/build_app.sh 拷成
# output/cpu/run_web.sh，日常启动用 output/cpu/run_web.sh。
# 本脚本预期位于 <仓库根>/output/cpu/ 下（ROOT 由上两级推得；部署到别处请自带环境变量覆盖）。
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"

# 单实例：只匹配 output/cpu 路径，不误杀 Python 版 / 其他同名进程
pkill -f "web_vllm_cpp/output/cpu/web_vllm_cpp" 2>/dev/null || true
sleep 1

export LD_LIBRARY_PATH="$HERE/lib:${LD_LIBRARY_PATH:-}"

# 本项目自带的 data/ 目录（仓库根下）
export WEB_VLLM_DATA_DIR="${WEB_VLLM_DATA_DIR:-$ROOT/data}"
export WEB_VLLM_STATIC_DIR="${WEB_VLLM_STATIC_DIR:-$HERE/static}"

# VLM（GGUF 微调模型）——CPU 推理：VLM_DEVICE=cpu 时代码内强制 n_gpu_layers=0
export VLM_MODEL_NAME="${VLM_MODEL_NAME:-$ROOT/models/qwen3vl-ft-20260909/qwen3vl-ft-f16.gguf}"
export VLM_MMPROJ_PATH="${VLM_MMPROJ_PATH:-$ROOT/models/qwen3vl-ft-20260909/mmproj-qwen3vl-ft-f16.gguf}"
export VLM_GGUF_N_CTX="${VLM_GGUF_N_CTX:-4096}"
export VLM_DEVICE="${VLM_DEVICE:-cpu}"
export VLM_GGUF_N_GPU_LAYERS=0

# 默认 combined：结构化属性 + 行人自由描述同时开启
export VLM_MATCH_MODE="${VLM_MATCH_MODE:-combined}"

# 规则直判提示词（{rules} 占位符处由代码填入规则列表，不可删除）
export VLM_RULE_JUDGE_PROMPT="${VLM_RULE_JUDGE_PROMPT:-请判断图像中的行人是否符合以下每一条规则。
{rules}
输出要求：只输出符合的规则名称，多条之间用顿号「、」分隔；如果没有任何规则符合，只输出：不匹配
除规则名称或「不匹配」外，不要输出任何其他文字、编号、标点或解释；只依据图像中清晰可见的内容判断，不要依据规则名称猜测。 <image>}"
export VLM_RULE_JUDGE_MAX_NEW_TOKENS="${VLM_RULE_JUDGE_MAX_NEW_TOKENS:-128}"

export VLM_DESCRIPTION_PROMPT="${VLM_DESCRIPTION_PROMPT:-请详细描述图中行人的外观，包括性别、大致年龄段、朝向、衣着款式与颜色、携带物品等清晰可见的属性。只描述画面中清晰可见的内容，不得猜测看不见或无法确认的特征；行人背对镜头或看不到脸部时，禁止断言眼镜等面部特征。对于无法看清或无法确定的属性，直接不要提及，严禁出现「是否佩戴眼镜难以判断」「看不清是否戴眼镜」这类包含疑问字样的表述。用连贯的中文陈述句输出，只写确定的事实，不要输出英文，不要输出多余的解释。 <image>}"

# 结构化提示词不设置时由 C++ 内置列表生成；如需覆盖可设 VLM_STRUCTURED_PROMPT

export VLM_STRUCTURED_MAX_NEW_TOKENS="${VLM_STRUCTURED_MAX_NEW_TOKENS:-128}"
export VLM_RULE_PROMPT_MAX_TEXTS="${VLM_RULE_PROMPT_MAX_TEXTS:-3}"
export VLM_GENERATION_TEMPERATURE="${VLM_GENERATION_TEMPERATURE:-0.0}"
export VLM_MAX_NEW_TOKENS="${VLM_MAX_NEW_TOKENS:-320}"
export VLM_RULE_DESCRIPTION_ENABLED="${VLM_RULE_DESCRIPTION_ENABLED:-0}"
export VLM_RULE_DESCRIPTION_VERIFY="${VLM_RULE_DESCRIPTION_VERIFY:-1}"
export VLM_RULE_DESCRIPTION_MAX_NEW_TOKENS="${VLM_RULE_DESCRIPTION_MAX_NEW_TOKENS:-60}"
export VLM_REPETITION_PENALTY="${VLM_REPETITION_PENALTY:-1.2}"

export EMBEDDING_ONNX_PATH="${EMBEDDING_ONNX_PATH:-$ROOT/models/onnx/bge-small-zh.onnx}"
export EMBEDDING_VOCAB_PATH="${EMBEDDING_VOCAB_PATH:-$ROOT/models/bge-small-zh-v1.5/vocab.txt}"
export DETECTOR_ONNX_PATH="${DETECTOR_ONNX_PATH:-$ROOT/models/onnx/yolov8n-dyn.onnx}"

export WEB_VLLM_PORT="${WEB_VLLM_PORT:-5003}"

# CPU 吞吐提示：4B F16 VLM 每帧需数次推理（属性+描述），全链路单帧可达十秒级；
# 高吞吐场景建议 VLM_MATCH_MODE=rule_direct 或换量化 GGUF（如 Q4_K_M）
exec "$HERE/web_vllm_cpp"
