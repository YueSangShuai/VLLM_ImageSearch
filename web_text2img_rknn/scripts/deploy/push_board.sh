#!/bin/bash
# 推 output/ 全量到板上 ~/web_text2img_rknn/（保留板上 gallery/ 与 data/，不覆盖）。
set -euo pipefail
DEPLOY_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$DEPLOY_DIR/../.." && pwd)"
D=pi@10.8.18.118
SSHOPT="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"

cd "$ROOT/output"
tar czf /tmp/t2i_deploy.tgz --exclude './data/gallery/manifest.json' \
    --exclude './data/gallery/vectors.bin' --exclude './gallery' \
    web_text2img golden_check run_web_rk.sh run_golden_rk.sh lib static models scripts 2>/dev/null
python3 "$DEPLOY_DIR/scp_pi.py" push /tmp/t2i_deploy.tgz "$DEPLOY_DIR/start_svc_board.sh" --dest /tmp
python3 "$DEPLOY_DIR/ssh_pi.py" 120 \
  "mkdir -p ~/web_text2img_rknn && tar xzf /tmp/t2i_deploy.tgz -C ~/web_text2img_rknn && mv /tmp/start_svc_board.sh ~/web_text2img_rknn/ && chmod +x ~/web_text2img_rknn/*.sh"
echo "PUSH OK -> ~/web_text2img_rknn  重启: python3 scripts/deploy/ssh_pi.py 30 'bash ~/web_text2img_rknn/start_svc_board.sh'"
