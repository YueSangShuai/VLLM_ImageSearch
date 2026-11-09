#!/bin/bash
# 整包推送 output/ 到板 pi@10.8.18.118（沿用 web_vllm_rknn 的推送通道，板子相同）。
# 交互输一次密码（pi）。传完板上：cd ~/output && bash run_web_rk.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

SSH_OPTS="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"
DEST="${1:-/home/pi}"

echo "== streaming $(du -shL output | cut -f1) to pi@10.8.18.118:$DEST (输入密码 pi)"
tar cf - -h output | ssh $SSH_OPTS pi@10.8.18.118 "tar xf - -C $DEST"

echo "== verify on board"
DIR="$DEST/$(basename output)"
ssh $SSH_OPTS pi@10.8.18.118 "cd $DIR && ls -la models/ && pgrep -x web_siglip_reid >/dev/null && echo 'NOTE: 旧服务仍在跑，记得重启' || true"
echo "DONE. 板上启动: ssh pi@10.8.18.118 -> cd ~/$DIR && bash run_web_rk.sh"
