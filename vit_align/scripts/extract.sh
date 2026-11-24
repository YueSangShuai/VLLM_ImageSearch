#!/usr/bin/env bash
# 离线提特征：视觉塔(2048) 与 BGE(512) 各 train/test 两份 + test 加前缀查询版（全量）
# 提取器冒烟请单独跑: python extract_image_feats.py --json ... --name smoke_i --limit 500
set -e
PY=${PY:-/home/yuesang/miniconda3/envs/llama/bin/python}   # transformers 5.8，支持 Qwen3-VL；可用 PY= 覆盖
cd "$(dirname "$0")/.."

"$PY" extract_image_feats.py --json data/json/train.json --name train_i
"$PY" extract_image_feats.py --json data/json/test.json  --name test_i
"$PY" extract_text_feats.py  --json data/json/train.json --name train_t
"$PY" extract_text_feats.py  --json data/json/test.json  --name test_t
"$PY" extract_text_feats.py  --json data/json/test.json  --name test_t_q --prefix
echo "全部特征提取完成"
