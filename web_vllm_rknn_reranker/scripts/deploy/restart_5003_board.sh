#!/bin/bash
# 评测结束后恢复 5003 线上服务（路径以板上实际部署位置为准）
for D in /home/Desktop/output_cpp /home/pi/output_cpp; do
    if [ -x "$D/run_web_rk.sh" ]; then
        cd "$D"
        setsid nohup bash run_web_rk.sh > web.log 2>&1 < /dev/null &
        echo "RESTARTED 5003 from $D"
        exit 0
    fi
done
echo "WARN: 未找到 run_web_rk.sh，请手动恢复 5003"
