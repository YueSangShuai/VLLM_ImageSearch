#!/bin/bash
# 模型推理耗时基准：output/aarch64/run_bench.sh（产物为 aarch64 交叉编译版） [bench_models 参数...]
# 环境变量与 run_web.sh 保持一致（改了 run_web.sh 的模型/提示词配置请同步这里，
# 或直接用 `VLM_...=... ./run_bench.sh` 覆盖）。bench 不会停在线服务。
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"

export LD_LIBRARY_PATH="$HERE/lib:${LD_LIBRARY_PATH:-}"

# VLM（GGUF 微调模型）——CPU 推理
export VLM_MODEL_NAME="${VLM_MODEL_NAME:-$ROOT/models/qwen3vl-ft-20260909/qwen3vl-ft-f16.gguf}"
export VLM_MMPROJ_PATH="${VLM_MMPROJ_PATH:-$ROOT/models/qwen3vl-ft-20260909/mmproj-qwen3vl-ft-f16.gguf}"
export VLM_GGUF_N_CTX="${VLM_GGUF_N_CTX:-4096}"
export VLM_DEVICE="${VLM_DEVICE:-cpu}"
export VLM_GGUF_N_GPU_LAYERS=0

export VLM_MATCH_MODE="${VLM_MATCH_MODE:-combined}"

# 规则直判提示词（{rules} 占位符处由代码填入规则列表，不可删除）
export VLM_RULE_JUDGE_PROMPT="${VLM_RULE_JUDGE_PROMPT:-请判断图像中的行人是否符合以下每一条规则。
{rules}
输出要求：只输出符合的规则名称，多条之间用顿号「、」分隔；如果没有任何规则符合，只输出：不匹配
除规则名称或「不匹配」外，不要输出任何其他文字、编号、标点或解释；只依据图像中清晰可见的内容判断，不要依据规则名称猜测。 <image>}"
export VLM_RULE_JUDGE_MAX_NEW_TOKENS="${VLM_RULE_JUDGE_MAX_NEW_TOKENS:-128}"

export VLM_DESCRIPTION_PROMPT="${VLM_DESCRIPTION_PROMPT:-请详细描述图中行人的外观，包括性别、大致年龄段、朝向、衣着款式与颜色、携带物品等清晰可见的属性。只描述画面中清晰可见的内容，不得猜测看不见或无法确认的特征；行人背对镜头或看不到脸部时，禁止断言眼镜等面部特征。对于无法看清或无法确定的属性，直接不要提及，严禁出现「是否佩戴眼镜难以判断」「看不清是否戴眼镜」这类包含疑问字样的表述。用连贯的中文陈述句输出，只写确定的事实，不要输出英文，不要输出多余的解释。 <image>}"

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

exec "$HERE/bench_models" "$@"
