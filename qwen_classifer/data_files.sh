# MARS PS_cuhk_pedes.yaml 同款数据集清单，供 run_extract.sh 引用
# train 合计 6,625,567 张；val 合计 127,532 张

TRAIN_JSONS=(
  /data/yuesang/person_data/SYNTH-PEDES/json/train_new_processed.json
  /data/yuesang/person_data/MARS/json/val_new_processed.json
  /data/yuesang/person_data/MARS/json/train_new_processed.json
  /data/yuesang/person_data/PAB/json/val_new_processed.json
  /data/yuesang/person_data/PAB/json/train_new_processed.json
  /data/yuesang/person_data/pa100k/json/train_new_processed.json
  /data/yuesang/person_data/19w/json/train_new_relabel.json
  /data/yuesang/person_data/T4/V1/json/train_new_relabel.json
  /data/yuesang/person_data/T4/V2/json/train_new_relabel.json
)

VAL_JSONS=(
  /data/yuesang/person_data/pa100k/json/test_new_processed.json
  /data/yuesang/person_data/pa100k/json/val_new_processed.json
  /data/yuesang/person_data/19w/json/val_new_relabel.json
  /data/yuesang/person_data/T4/V1/json/val_new_relabel.json
)

# 基线用：train 只取 pa100k（8 万张，三卡 15~20 分钟出全链路真实指标）
TRAIN_JSONS_PA100K=(
  /data/yuesang/person_data/pa100k/json/train_new_processed.json
)
