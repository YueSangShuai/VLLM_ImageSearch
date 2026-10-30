#!/bin/bash
# 续传（SMALL/LIB/STATIC 已推）：data + calib + 板端脚本 + 大模型（vision 855MB / rkllm 2.3GB）。
set -uo pipefail
cd "$(dirname "$0")/../.."
S=scripts/deploy/scp_pi.py; B=/home/pi/output_cpp_reranker
timeout 60 python3 scripts/deploy/ssh_pi.py "mkdir -p $B/data $B/calib" || exit 1
timeout 120 python3 $S push output_cpp/data/*.json --dest $B/data && echo DATA_OK || echo FAIL_data
timeout 600 python3 $S push -r calib --dest $B && echo CALIB_OK || echo FAIL_calib
timeout 120 python3 $S push scripts/deploy/run_eval_board.sh scripts/deploy/restart_5003_board.sh \
    scripts/e2e/e2e_reranker_add_rule.sh scripts/e2e/e2e_reranker_rm_rule.sh --dest $B && echo SCRIPTS_OK || echo FAIL_scripts
timeout 600 python3 $S push output_cpp/models/yolov8n_640_rk3588.rknn --dest $B/models && echo YOLO_OK || echo FAIL_yolo
timeout 3000 python3 $S push output_cpp/models/qwen3vl-reranker_vision_rk3588.rknn --dest $B/models && echo VISION_OK || echo FAIL_vision
timeout 3000 python3 $S push output_cpp/models/qwen3vl-reranker-2b_w8a8_rk3588.rkllm --dest $B/models && echo RKLLM_OK || echo FAIL_rkllm
timeout 60 python3 scripts/deploy/ssh_pi.py "cd $B && du -sh models/* calib | tail -5; md5sum web_vllm_rknn rerank_eval"
echo "== host md5"; md5sum output_cpp/web_vllm_rknn output_cpp/rerank_eval
