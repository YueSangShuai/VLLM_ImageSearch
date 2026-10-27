#!/bin/bash
# 板上执行：后台启动 q35 平台实例（端口 5004），setsid 脱离 ssh 通道；日志 app.log。
# ⚠️ 只杀 q35 自己的进程名，绝不用 pkill -x web_vllm_rknn（那是线上 5003 实例）。
cd /home/web_vllm_rknn_q35
# 注意：comm 名 15 字符截断（web_vllm_rknn_q35 → web_vllm_rknn_q），pgrep/pkill -x 永远匹配不到，
# 一律用 -f 匹配完整命令行；模式含 q35 不会误伤线上 web_vllm_rknn。
# [.] 只匹配 exec 后的 "./web_vllm_rknn_q35"，不会杀掉本启动脚本自身（路径形式 /home/... 不含 "./"）
pkill -f "[.]/web_vllm_rknn_q35" 2>/dev/null
sleep 1
setsid nohup bash run_web_rk.sh > app.log 2>&1 < /dev/null &
sleep 2
pgrep -f "[.]/web_vllm_rknn_q35" >/dev/null && echo Q35_STARTING || echo Q35_FAILED
