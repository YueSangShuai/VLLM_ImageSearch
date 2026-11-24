#!/usr/bin/env bash
# 评估 + 导出头 + bin 零差校验
# 用法: CFG=configs/tp600.yaml bash scripts/eval.sh runs/tproj_long/best.pt
set -e
PY=${PY:-/home/yuesang/miniconda3/envs/llama/bin/python}   # transformers 5.8，支持 Qwen3-VL；可用 PY= 覆盖
cd "$(dirname "$0")/.."
CKPT=${1:-runs/full/best.pt}
CFG=${CFG:-configs/default.yaml}
"$PY" eval.py --config "$CFG" --ckpt "$CKPT" --prefix-cache cache/test_t_q_*
"$PY" export_head.py --config "$CFG" --ckpt "$CKPT" --out models/vproj_head.bin
"$PY" export_head.py --config "$CFG" --verify --bin models/vproj_head.bin --ckpt "$CKPT"
echo "注意：导出含 text_proj 时产物为 models/vproj_head.bin，板端需按 bin 内 has_text_proj 对文本侧补投影"
