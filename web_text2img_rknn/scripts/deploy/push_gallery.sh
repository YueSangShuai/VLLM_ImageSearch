#!/bin/bash
# 推初始图库子集到板上 ~/web_text2img_rknn/gallery/（本地 output/gallery 内容）。
# 注意：不覆盖 data/ 下向量缓存；服务重启后 sync 只会增量编码新图。
set -euo pipefail
DEPLOY_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$DEPLOY_DIR/../.." && pwd)"

cd "$ROOT/output"
tar czf /tmp/t2i_gallery.tgz gallery
python3 "$DEPLOY_DIR/scp_pi.py" push /tmp/t2i_gallery.tgz --dest /tmp
python3 "$DEPLOY_DIR/ssh_pi.py" 300 \
  "mkdir -p ~/web_text2img_rknn && tar xzf /tmp/t2i_gallery.tgz -C ~/web_text2img_rknn && ls ~/web_text2img_rknn/gallery | wc -l"
echo "GALLERY PUSH OK"
