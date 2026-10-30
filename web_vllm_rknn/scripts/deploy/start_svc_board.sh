#!/bin/bash
# 纯净环境重启线上服务（避免测试脚本导出的 VLM_* 环境变量泄漏进服务）
pkill -f '[w]eb_vllm_rknn_q35' || true
pkill -x web_vllm_rknn || true
sleep 3
cd /home/Desktop/output_cpp
env -i HOME=/home/pi PATH=/usr/bin:/bin bash -c 'cd /home/Desktop/output_cpp && setsid nohup ./run_web_rk.sh > web.log 2>&1 < /dev/null &'
echo RESTARTED
