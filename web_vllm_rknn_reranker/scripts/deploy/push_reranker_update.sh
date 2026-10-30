#!/bin/bash
# 小更新推送：二进制 + rerank_eval + 启动脚本（模型/数据不动），带 md5 核对。
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
cd "$ROOT"
PY="$SCRIPT_DIR"
O=output_cpp
BOARD_DIR=/home/pi/output_cpp_reranker

for f in web_vllm_rknn rerank_eval run_web_rk_reranker.sh; do
  # scp_pi.py 必须带 push 子命令，否则静默变 pull！
  timeout 120 python3 "$PY/scp_pi.py" push "$O/$f" --dest "$BOARD_DIR" >/dev/null 2>&1 \
    && echo "pushed $f" || echo "FAIL $f"
done
echo "== board md5"
timeout 30 python3 "$PY/ssh_pi.py" "cd $BOARD_DIR && md5sum web_vllm_rknn rerank_eval"
echo "== host md5"
md5sum "$O/web_vllm_rknn" "$O/rerank_eval"
