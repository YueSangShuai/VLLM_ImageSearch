#!/bin/bash
# 从板上抽样 ROI 告警图（跨规则去重），拉回 gecko/board_images/，
# 同时拉 alerts_history（用于对照线上模型的描述）。
set -uo pipefail
Q=/home/yuesang/miniconda3/envs/Qwen/bin/python
D=/data/yuesang/LLM/FG-CLIP/web_vllm_rknn/scripts/deploy
OUT=/data/yuesang/LLM/FG-CLIP/structbert_npu/gecko/board_images
mkdir -p "$OUT"

# 每条规则取最新一张 roi 图（按 rule 目录去重）
PATHS=$($Q $D/ssh_pi.py "find /home/web_vllm_rknn/data/alerts_images -path '*roi*' -name '*.jpg' -printf '%T@ %p\n' 2>/dev/null | sort -rn | awk '{p=\$2; rule=p; sub(/\\/roi\\/.*/,\"\",rule); if(!(rule in seen)){seen[rule]=1; print p}}' | head -12" \
  | grep '^/home')
echo "$PATHS" | while read -r p; do
  [ -z "$p" ] && continue
  tag=$(echo "$p" | awk -F/ '{print $(NF-3)"__"$(NF-2)"__"$NF}' | tr ' ' '_')
  $Q $D/scp_pi.py pull "$p" --dest "$OUT" >/dev/null 2>&1
  [ -f "$OUT/$(basename "$p")" ] && mv "$OUT/$(basename "$p")" "$OUT/$tag" && echo "ok $tag"
done

$Q $D/scp_pi.py pull /home/web_vllm_rknn/data/alerts_history --dest "$OUT" >/dev/null 2>&1 && echo "ok alerts_history"
ls -la "$OUT" | head -16
