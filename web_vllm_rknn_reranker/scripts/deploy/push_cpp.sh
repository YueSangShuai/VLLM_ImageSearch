#!/bin/bash
# 推送 output_cpp 关键产物到板并校验
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"   # scripts/deploy -> 项目根
cd "$ROOT"
PY="$SCRIPT_DIR"
O=output_cpp
for pair in \
  "$O/models/yolov8n_640_rk3588.rknn:/home/pi/output_cpp/models/yolov8n_640_rk3588.rknn" \
  "$O/web_vllm_rknn:/home/pi/output_cpp/web_vllm_rknn" \
  "$O/bench_models:/home/pi/output_cpp/bench_models"
do
  src=${pair%%:*}; dst=${pair##*:}
  # 注意：scp_pi.py 必须带 push 子命令，否则静默变成 pull（把板上旧文件反下载）！
  timeout 120 python3 "$PY/scp_pi.py" push "$src" --dest "$(dirname "$dst")" >/dev/null 2>&1 \
    && echo "pushed $(basename "$src")" || echo "FAIL $(basename "$src")"
done
echo "== board md5"
timeout 30 python3 "$PY/ssh_pi.py" 'cd ~/output_cpp && md5sum web_vllm_rknn bench_models models/yolov8n_640_rk3588.rknn'
echo "== host md5"
md5sum "$O/web_vllm_rknn" "$O/bench_models" "$O/models/yolov8n_640_rk3588.rknn"
