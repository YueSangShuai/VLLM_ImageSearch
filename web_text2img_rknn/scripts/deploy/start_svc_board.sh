#!/bin/bash
# 纯净环境重启线上服务（避免测试脚本导出的环境变量泄漏进服务）
DIR="${T2I_BOARD_DIR:-$HOME/web_text2img_rknn}"
pkill -x web_text2img || true
sleep 2
env -i HOME=/home/pi PATH=/usr/bin:/bin bash -c "cd $DIR && setsid nohup ./run_web_rk.sh > web.log 2>&1 < /dev/null &"
echo RESTARTED
