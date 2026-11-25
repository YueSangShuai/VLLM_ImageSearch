#!/bin/bash
# 独占 NPU 视觉塔裸速：停服务→测两塔→无论如何都重启服务（trap 保障）。
set -u
cd /home/pi/rrbench
export LD_LIBRARY_PATH=/home/web_vllm_rknn/lib
restore() {
  cd /home/web_vllm_rknn && setsid nohup ./run_web_rk.sh > web.log 2>&1 < /dev/null &
  sleep 3
  pgrep -x web_vllm_rknn >/dev/null && echo SERVICE_RESTARTED || echo "!! SERVICE DOWN — 手动跑 run_web_rk.sh"
}
trap restore EXIT
pkill -x web_vllm_rknn; sleep 2
echo "== Qwen3-VL-2B ViT (zkteco ft, 448)"
taskset -c 0-3 ./vision_bench /home/web_vllm_rknn/models/zkteco-person-describe-ft-vl-2b-rk3588.rknn 10 16 | grep -E "^#|iter (0|1|2)"
echo "== gecko InternViT-300M (448)"
taskset -c 0-3 ./vision_bench internvl3-1b_vision_rk3588.rknn 10 16 | grep -E "^#|iter (0|1|2)"
