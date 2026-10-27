# FG-CLIP — 细粒度图文对齐模型与行人布控检索系统

本仓库基于开源 [FG-CLIP（360CVGroup，ICML 2025，arXiv:2505.05071）](https://github.com/360CVGroup/FG-CLIP) 改造，包含三部分工作：

1. **FG-CLIP 细粒度图文模型**的训练与评测（上游能力：长文本、region 级 caption、正/负例硬样本）；
2. **行人属性分类扩展**：在 FG-CLIP 上加 54 个二分类属性头（性别/年龄/帽子/眼镜/上下装颜色款式等），在 PEDRA/PA-100K 风格数据上训练；
3. **端侧部署管线**：把训练产物拆分为 image/text 双模型并导出 ONNX，供下游多个推理服务（`web*/` 各目录）消费。

整体定位：**FG-CLIP 训练/导出侧 + 多端行人布控推理服务** 的 monorepo。

---

## 目录结构

```
FG-CLIP/
├── fgclip/                  # 实际使用的 Python 包（模型、训练、评测）
│   ├── model/clip_strc/     #   FGCLIPModel、ClassificationHead（本地新增）、modeling_clip
│   ├── train/               #   train_fgclip.py（训练主入口）、clean_clip_trainer.py
│   └── eval/                #   COCO 检索 / IN1K / IN-V2 / FG-OVD / 行人属性分类评测
├── configs/                 # classifier_config.yaml（54 属性）、train_classification.yaml
├── scripts/                 # 训练/评测 shell 与 YAML 配置
├── tools/                   # export_fgclip_to_onnx.py（独立 ONNX 导出线）、fgclip_loader.py
├── data/                    # 上游 FineHARD 教程(data.md)、get_data.sh、fgovd/、IN1K_val/
├── checkpoints/             # 已训练产物：fgclip-classification（model.safetensors + deepspeed ckpt）
├── models/                  # 部署配套权重：bge-small-zh/en、siglip2、qwen2.5-vl-3b、
│                            #   qwen3vl-ft GGUF+mmproj、YOLOv8n ONNX、minimind-3v
├── split_models/            # split_fgclip_model.py 产物（image_model/ text_model/）
├── onnx_exports/            # image_model.onnx / text_model.onnx
└── DEPENDENCIES.md          # monorepo 全部第三方依赖与来源说明
```

> 所有权重均不入库（.gitignore 排除 `models/`、`checkpoints/`、`split_models/`、`onnx_exports/` 等），需自行下载或从训练产物导出。

### 下游推理服务（各目录均有独立 README）

| 目录 | 说明 | 端口 |
|---|---|---|
| [`web/`](web/README.md) | Flask + FG-CLIP 匹配的 RTSP 行人布控原型 | 5001 |
| [`web_vllm/`](web_vllm/README.md) | Flask + Qwen-VL 描述 + BGE 匹配的布控平台 | 5003 |
| [`web_vllm_cpp/`](web_vllm_cpp/README.md) | web_vllm 的 C++17 CPU 移植（llama.cpp + ONNX Runtime） | 5003 |
| [`web_vllm_rknn/`](web_vllm_rknn/README.md) | 向 RK3588 NPU（NanoPi R6S）的移植（RKNN + RKLLM） | 5003(板) |
| [`web_vllm_rknn_q35/`](web_vllm_rknn_q35/README.md) | rknn 版的 Qwen3.5-0.8B 并行实验实例 | 5004(板) |

---

## 安装

```bash
# 训练/评测环境（CUDA 12.4）
pip install -r requirements_fgclip2_minimind3v.txt
```

要点：

- 要求 `torch==2.6.0+cu124`、`transformers>=4.53,<5.0`。注意 `pyproject.toml` 中 pin 的 `transformers==4.41.0` 是上游遗留，**以 requirements 为准**。
- 基座模型 `qihoo360/fg-clip-base`（224px）或 `qihoo360/fg-clip-large`；本仓库脚本硬编码的本地副本位于 `/data/yuesang/LLM/MARS/fg-clip`，换机器需自行改路径。
- 上游 FineHARD 训练数据的下载与改名流程见 [`data/data.md`](data/data.md)。

---

## 训练

### 行人属性分类（本地主训练线）

```bash
# 方式一：YAML 一键启动（配置见 scripts/train_classification.yaml）
bash scripts/train_classification.sh

# 方式二：直接看底层 deepspeed 命令
python scripts/run_train_classification.py --config scripts/train_classification.yaml
```

当前配置：冻结 vision/text 主干，只训 54 个分类头 + 投影层，bf16 + ZeRO-2，输出到 `./checkpoints/fgclip-classification`。训练数据来自 `/data/yuesang/person_data/{SYNTH-PEDES,MARS,PAB,pa100k,19w,T4/V1,T4/V2}/json/*_new_*.json`（路径硬编码，需按机器修改）。

属性定义在 `configs/classifier_config.yaml`：每个属性一对 yes/no 英文句子，同时用于文本侧对齐与分类头监督。

### FG-CLIP 二阶段微调（上游）

```bash
bash scripts/train.sh   # GRIT-12M + FineHARD，from_openai 时做位置嵌入插值扩展
```

---

## 评测

| 脚本 | 内容 |
|---|---|
| `scripts/eval_classification.sh` | 行人属性分类（PA-100K test） |
| `scripts/eval/eval_in1k.sh` / `eval_inv2.sh` | ImageNet-1K / V2 zero-shot |
| `scripts/eval/eval_coco_retrieval.sh` | COCO 图文检索 |
| `scripts/eval/eval_fgovd.sh` | FG-OVD 细粒度开放词汇检测 |
| `scripts/eval/eval_long.sh` | 长文本检索（ShareGPT4V-1K + DCI，max_length 248） |

```bash
# 例：属性分类评测
python -m fgclip.eval.eval_classification \
    --model_path ./checkpoints/fgclip-classification \
    --model_base qihoo360/fg-clip-base \
    --classifier_yaml ./configs/classifier_config.yaml --batch_size 64
```

---

## 模型拆分与 ONNX 导出（供部署侧使用）

```bash
# 1) 把带分类头的 checkpoint 拆成 image_model / text_model 两个独立模型
python split_fgclip_model.py --checkpoint ./checkpoints/fgclip-classification --output ./split_models

# 2) 拆分模型 → ONNX（image 侧输出 image_features[B,512] + classification_output[B,108]）
python export_to_onnx.py

# 另一条独立导出线（不依赖 fgclip 包，直接读 HF 权重，自带数值校验）
python tools/export_fgclip_to_onnx.py --model_path /data/yuesang/LLM/MARS/fg-clip \
    --output_dir ./onnx_exports --opset 14 --verify
```

产物被下游 web 服务消费：`onnx_exports/`（web/ 的属性分类）、`models/onnx/*`（BGE/YOLO，web_vllm*）、`models/qwen3vl-ft-*/…gguf`（web_vllm_cpp）。

---

## 已知坑

- **`script.py` 是混淆的 Linux 本地提权漏洞利用代码（AF_ALG splice），与本项目无关，切勿执行，建议删除并排查来源。**
- `fg_clip` 是指向不存在目录的坏符号链接（上游包名残留），新代码一律 `import fgclip`；根目录 `export_onnx.py` 因依赖 `fg_clip` 已失效。
- `scripts/eval/eval_coco_boxcls.sh` 中模块名 `myclip.` 是旧包名残留，应为 `fgclip.`。
- `fgclip/eval/shareg4v_retrieval.py` 顶部 `import torch_npu`，GPU 环境需注释掉才能跑长文本评测。
- 大量配置硬编码 `/data/yuesang/...` 绝对路径，迁移机器时全局搜索替换。

## 许可

[Apache-2.0](LICENSE)（随上游 FG-CLIP）。
