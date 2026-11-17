#!/bin/bash
# 纯净环境重启线上服务（避免测试脚本导出的 VLM_* 环境变量泄漏进服务）
pkill -x web_vllm_rknn || true
sleep 3
# 线上目录 2026-11 起为 /home/web_vllm_rknn（旧 ~/output_cpp、/home/Desktop/output_cpp 均已废弃）
cd /home/web_vllm_rknn
env -i HOME=/home/pi PATH=/usr/bin:/bin bash -c 'cd /home/web_vllm_rknn && setsid nohup ./run_web_rk.sh > web.log 2>&1 < /dev/null &'
echo RESTARTED
