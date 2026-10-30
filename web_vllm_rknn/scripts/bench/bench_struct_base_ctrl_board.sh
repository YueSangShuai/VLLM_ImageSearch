#!/bin/bash
# 对照组：基座 Qwen3-VL-2B-Instruct W8A8 同一张图同一结构化 prompt（用户已批准停/重启服务）
set -uo pipefail
cd /home/Desktop/output_cpp
export LD_LIBRARY_PATH="$PWD/lib" WEB_VLLM_MODELS_DIR="$PWD/models"
pkill -f "[w]eb_vllm_rknn_q35" || true   # 括号防自匹配：-f 带引号原样会杀掉 ssh 会话自己
pkill -x web_vllm_rknn || true
sleep 3
export VLM_MODEL_NAME="$PWD/models/qwen3-vl_vision_rk3588.rknn"
export VLM_MMPROJ_PATH="$PWD/models/qwen3-vl-2b-instruct_w8a8_rk3588.rkllm"
export VLM_DESCRIPTION_PROMPT='请分析这张行人图像，从以下预定义属性中找出所有存在的属性（即值为1的属性），只输出这些属性名，用逗号分隔，不要输出任何其他文字或值为0的属性。
预定义属性列表：Female,AgeOver60,Age18-60,AgeLess18,Front,Side,Back,Hat,Glasses,HandBag,ShoulderBag,Backpack,HoldObjectsInFront,ShortSleeve,LongSleeve,UpperStride,UpperLogo,UpperPlaid,UpperSplice,LowerStripe,LowerPattern,LongCoat,Trousers,Shorts,Skirt&Dress,boots,ublack,ugray,ublue,ugreen,uwhite,upurple,ured,ubrown,uyellow,upink,uorange,ubeige,ustriped_color,umulticolor,lwhite,lpink,lred,lgreen,lyellow,lpurple,lbrown,lblack,lorange,lblue,lgray,lbeige,lstriped_color,lmulticolor

<image>'
for img in \
  'data/alerts_images/2026-10-19/一楼电梯厅/戴眼镜的男生/roi/alert_1792370895049_895048.jpg' \
  'data/alerts_images/2026-10-19/一楼电梯厅/带帽子的人/roi/alert_1792370129563_129548.jpg' \
  'data/alerts_images/2026-10-19/一楼电梯厅/穿裙子的女生/roi/alert_1792370697027_697026.jpg'; do
    ./bench_models --skip yolo,bge --image "$img" --iters 1 --warmup 0 --vlm-max-tokens 128 2>&1 | grep -aE '末次输出|失败|fail'
done
setsid nohup bash ./run_web_rk.sh > web.log 2>&1 < /dev/null &
echo RESTART_TRIGGERED
