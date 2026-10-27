#!/bin/bash
# 杀掉旧的app.py进程
pkill -f "python.*app.py" 2>/dev/null
sleep 1

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate MARS
cd /data/yuesang/LLM/FG-CLIP/web
CUDA_VISIBLE_DEVICES=5 python /data/yuesang/LLM/FG-CLIP/web/app.py