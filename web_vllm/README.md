# web_vllm — Qwen-VL + BGE 行人布控平台

基于 RTSP 流的行人检测 + 文本布控告警平台（Flask，**端口 5003**）。

流水线：RTSP 取帧 → YOLOv8 行人检测 → 行人裁剪图送 **Qwen-VL** 生成描述/判断 → **BGE** 文本向量与布控规则匹配 → 告警落盘 + Web 展示。

> 目录名带 "vllm" 是历史遗留——当前代码**不使用 vLLM 服务**，VLM 为进程内加载（transformers 或 llama-cpp-python GGUF 二选一）。本目录是 [web/](../web/README.md)（FG-CLIP 版）的多模态大模型升级版，也是 [web_vllm_cpp/](../web_vllm_cpp/README.md)（C++ 移植）与 [web_vllm_rknn/](../web_vllm_rknn/README.md)（NPU 移植）的功能基准。

## 代码结构

| 文件 | 作用 |
|---|---|
| `app.py` | Flask 入口：设备/规则/任务/告警/日志 REST API、RTSP 心跳线程（5s）、ffmpeg 单帧截图、启动即预加载模型并自动开始监控 |
| `processor.py` | 核心引擎：`PersonDetector`（YOLO）/ `VLMTextMatcher`（四种匹配模式）/ `StreamProcessor`（每设备一线程，ffmpeg 逐帧拉流、指数退避重连）/ `MonitoringManager`（全局单例，规则热更新，告警原子写） |
| `vlm_search/vlm_describer.py` | VLM 双后端：HF（`AutoModelForImageTextToText`）或 GGUF（`llama_cpp` + mmproj 视觉投影） |
| `vlm_search/text_search.py` | `TextEmbedder`：sentence-transformers 封装，归一化输出，max_seq_length 512 |

## 匹配模式（`VLM_MATCH_MODE`）

| 模式 | 机制 |
|---|---|
| `rule_direct`（脚本默认） | 规则列表代入提示词，VLM 直接输出命中规则名，不走 BGE |
| `description` | VLM 自由描述 + BGE 向量 cosine 匹配规则 positive 文本 |
| `structured` | VLM 输出预定义行人属性名（54 属性），符号匹配 |
| `combined` | structured + description 都做 |

命中条件（向量路径）：`best_pos ≥ threshold 且 > max_neg + score_diff`，受 `interval` 去抖（默认 60s）；`VLM_RULE_DESCRIPTION_VERIFY=1` 时命中后 VLM 再生成一句规则相关描述二次校验，判"不满足"则撤销告警。

## 安装与运行

```bash
# 前置：conda 环境 FGCLIP2、GPU、系统 ffmpeg
cd /data/yuesang/LLM/FG-CLIP/web_vllm
pip install -r requirements.txt

bash use_vllm_web.sh          # 杀旧进程 → conda FGCLIP2 → 设环境变量 → python app.py
# 访问 http://localhost:5003
```

`use_vllm_web.sh` 内所有模型路径/提示词均为可覆盖的环境变量，默认值：

- `VLM_MODEL_NAME=/data/yuesang/LLM/saves/Qwen3-VL-2B-Instruct/lora/train_2026-09-09/checkpoint-17254/lora`（**LoRA 合并模型目录**，须带完整 base metadata：config/processor/chat_template 等）
- `EMBEDDING_MODEL_NAME=../models/bge-small-zh-v1.5`（中文 384 维）
- `VLM_MATCH_MODE=rule_direct`、`VLM_DEVICE=cuda:0`、`CUDA_VISIBLE_DEVICES=0`

GGUF 后端（免 transformers）：

```bash
VLM_MODEL_FORMAT=gguf \
VLM_MODEL_NAME=/data/yuesang/LLM/test_qwen/qwen3-vl-2b-Q8_0.gguf \
VLM_MMPROJ_PATH=/data/yuesang/LLM/test_qwen/mmproj-Qwen3VL-2B-Instruct-F16.gguf \
bash use_vllm_web.sh
```

无 mmproj 时 GGUF 后端只能跑纯文本，无法处理图片。

## API 与页面

页面：`/`（概览）、`/devices`、`/rules`、`/alerts`。
REST：`/api/devices|rules|tasks|logs|alerts`（CRUD）、`/api/video/frame/<device_id>`、`/api/monitoring/status|start|stop`、`/data/alerts_images/<path>`。

## 数据（data/，纯 JSON 无数据库）

```
devices.json                            设备与在线状态
tasks.json                              任务 = 设备 + 规则集合（规则持久化于此）
detection_log.json                      检测日志（≤1000 条）
alerts_history/alerts_YYYY-MM-DD.json   按天告警（加锁 + tmp 原子替换）
alerts_images/<日期>/<设备>/<规则>/{roi,original}/alert_<ms>_<n>.jpg
```

> `rules.json` / `alerts.json` 为兼容旧文件，当前流程不读写。

## 注意

- `models/` 目录是占位空目录，实际权重 fallback 到 `../models/`（qwen2.5-vl-3b-instruct、bge-small-zh-v1.5）。
- 与线上 C++ 版（web_vllm_cpp，同为 5003 端口）**不要同时运行**。
- 告警图含真实行人，注意脱敏与访问控制。
