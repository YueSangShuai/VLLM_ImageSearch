#!/bin/bash
# 整包推送 output_cpp/ 到板子 ~ 下（自包含：二进制+lib+模型+数据，一个文件夹全搞定）。
# 关键：tar -h 把包内 models/ 的软链（-> ../../npu_models/...）解引用成真实文件，
#       板上拿到的是完整模型；本地只留软链，不占双倍空间。权重不可压缩，不用 gzip。
# 交互输一次板子密码（pi）。传完板上直接：cd ~/output_cpp && bash run_web_rk.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

SSH_OPTS="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"
DEST="${1:-/home/pi}"

echo "== streaming $(du -shL output_cpp | cut -f1) to pi@10.8.18.118:$DEST (输入密码 pi)"
tar cf - -h output_cpp | ssh $SSH_OPTS pi@10.8.18.118 "tar xf - -C $DEST"

echo "== verify on board"
DIR="$DEST/$(basename output_cpp)"
ssh $SSH_OPTS pi@10.8.18.118 "cd $DIR && ls -la models/ && pgrep -x web_vllm_rknn >/dev/null && echo 'NOTE: service still running old code; restart it' || true"
echo "DONE. 板上启动: ssh pi@10.8.18.118 -> cd ~/$DIR && bash run_web_rk.sh"
