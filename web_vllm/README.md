# web_vllm — Qwen-VL + BGE 行人布控平台（端口 5003）

RTSP 取帧 → YOLOv8 行人检测 → Qwen-VL 生成描述/判断 → BGE 文本向量与布控规则匹配 → 告警落盘 + Web 展示。VLM 为进程内加载（transformers 或 llama-cpp-python GGUF 二选一），不依赖 vLLM 服务。

## 所需模型与下载

| 模型 | 用途 | 下载地址 |
|---|---|---|
| Qwen3-VL-2B-Instruct | VLM 基座 | https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct |
| LoRA 合并模型（默认） | VLM 微调权重 | 本地微调（LLaMA-Factory 导出），无公开下载；脚本默认指向本地微调模型目录，用 `VLM_MODEL_NAME` 覆盖即可 |
| Qwen3-VL GGUF + mmproj | VLM 备选 GGUF 后端 | https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct-GGUF （需同时下 `.gguf` 与 `mmproj*.gguf`） |
| bge-small-zh-v1.5 | 文本向量 | https://huggingface.co/BAAI/bge-small-zh-v1.5 |
| YOLOv8n | 行人检测 | https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt |

```bash
export HF_ENDPOINT=https://hf-mirror.com   # 国内镜像
huggingface-cli download Qwen/Qwen3-VL-2B-Instruct --local-dir <本地目录>
huggingface-cli download BAAI/bge-small-zh-v1.5 --local-dir ../models/bge-small-zh-v1.5
wget -O yolov8n.pt https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt
```

> LoRA 合并模型目录必须带完整 base metadata（config/processor/chat_template），否则 HF 后端加载失败；换用基座只需把 `VLM_MODEL_NAME` 指向上面下载的基座目录。

## 安装

```bash
conda activate FGCLIP2     # 或等价环境（GPU + 系统 ffmpeg）
pip install -r requirements.txt
```

## 运行

```bash
bash use_vllm_web.sh       # 杀旧进程 → conda FGCLIP2 → 设环境变量 → python app.py
# 访问 http://localhost:5003
```

覆盖默认配置（`use_vllm_web.sh` 内变量均可用环境变量替换）：

```bash
# 用未微调基座（不设置则用脚本默认指向的本地微调模型）
VLM_MODEL_NAME=<Qwen3-VL-2B-Instruct 本地目录> bash use_vllm_web.sh

# 用 GGUF 后端（免 transformers，必须带 mmproj 才能处理图片）
VLM_MODEL_FORMAT=gguf \
VLM_MODEL_NAME=/path/to/Qwen3VL-2B-Instruct-Q8_0.gguf \
VLM_MMPROJ_PATH=/path/to/mmproj-F16.gguf \
bash use_vllm_web.sh

# 切匹配模式：rule_direct（默认，VLM 直判规则）| description（+BGE）| structured（54 属性符号）| combined
VLM_MATCH_MODE=description bash use_vllm_web.sh
```

关键默认值：`EMBEDDING_MODEL_NAME=../models/bge-small-zh-v1.5`、`VLM_DEVICE=cuda:0`、`CUDA_VISIBLE_DEVICES=0`。

## 使用流程

1. `/devices` 添加 RTSP 设备（每 5s 自动心跳探测在线）；
2. `/rules` 创建规则：中文 positive/negative 文本、threshold、score_diff、interval（默认 60s）；保存写 `data/tasks.json` 并热刷新；
3. 启动即自动加载模型并开始监控；
4. `/alerts` 查看告警。

触发条件（向量路径）：`best_pos ≥ threshold 且 > max_neg + score_diff`。

## 数据文件（data/，纯 JSON）

```
devices.json                             设备与在线状态
tasks.json                               任务 = 设备 + 规则（规则持久化于此）
detection_log.json                       检测日志（≤1000 条）
alerts_history/alerts_YYYY-MM-DD.json    按天告警
alerts_images/<日期>/<设备>/<规则>/{roi,original}/alert_<ms>_<n>.jpg
```

## API

`/api/devices|rules|tasks|alerts|logs`（CRUD）、`/api/video/frame/<device_id>`、`/api/monitoring/status|start|stop`、`/data/alerts_images/<path>`。

## 注意

- 与 web_vllm_cpp（同端口 5003）不要同时运行。
- `models/` 目录是占位，实际权重 fallback 到 `../models/`。
