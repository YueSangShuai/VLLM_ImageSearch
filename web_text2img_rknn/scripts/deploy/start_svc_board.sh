#!/bin/bash
# 纯净环境重启线上服务（避免测试脚本导出的环境变量泄漏进服务）
pkill -x web_text2img || true
sleep 2
cd /home/web_text2img_rknn
env -i HOME=/home/pi PATH=/usr/bin:/bin bash -c 'cd /home/web_text2img_rknn && setsid nohup ./run_web_rk.sh > web.log 2>&1 < /dev/null &'
echo RESTARTED
