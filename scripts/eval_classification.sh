#!/bin/bash

# FG-CLIP Classification Evaluation Script
# Usage: bash scripts/eval_classification.sh

set -e

# Model configuration
model_path="./checkpoints/fgclip-classification"
model_base="qihoo360/fg-clip-base"

# Data configuration
eval_jsonfiles="/data/yuesang/person_data/pa100k/json/test_new_processed.json"
image_root="/data/yuesang/person_data/"

# Classifier configuration
classifier_yaml="./configs/classifier_config.yaml"

# Evaluation settings
batch_size=64
image_size=224

python -m fgclip.eval.eval_classification \
    --model_path $model_path \
    --model_base $model_base \
    --data_path $eval_jsonfiles \
    --image_folder $image_root \
    --classifier_yaml $classifier_yaml \
    --batch_size $batch_size \
    --image_size $image_size