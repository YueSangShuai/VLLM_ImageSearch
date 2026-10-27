# 多模态大模型文搜实测平台

基于Qwen-VL图像描述和BGE-small文本向量相似度的RTSP流行人检测与文本布控系统。

## 功能特性

1. **设备管理** - 添加、编辑、删除RTSP流设备（IP、端口、用户名、密码）
2. **规则配置** - 设置文本布控规则和匹配分数阈值
3. **告警展示** - 展示满足条件的图片、时间、触发规则

## 安装

```bash
cd /data/yuesang/LLM/FG-CLIP/web_vllm
pip install -r requirements.txt
```

## 运行

```bash
bash /data/yuesang/LLM/FG-CLIP/web_vllm/use_vllm_web.sh
```

访问 http://localhost:5003

## 页面说明

- `/` - 首页，监控状态概览
- `/devices` - 设备管理
- `/rules` - 规则配置
- `/alerts` - 告警记录

## 配置

默认模型路径：
- `VLM_MODEL_NAME=/data/yuesang/LLM/test_qwen/qwen3-vl-2b-Q8_0.gguf`
- `EMBEDDING_MODEL_NAME=/data/yuesang/LLM/FG-CLIP/models/bge-small-zh-v1.5`

兼容原来的 transformers 目录模型加载：

```bash
VLM_MODEL_NAME=/data/yuesang/LLM/test_qwen \
bash /data/yuesang/LLM/FG-CLIP/web_vllm/use_vllm_web.sh
```

GGUF 多模态加载需要 `llama-cpp-python` 和对应的 `mmproj` 视觉投影文件：

```bash
VLM_MODEL_NAME=/data/yuesang/LLM/test_qwen/qwen3-vl-2b-Q8_0.gguf \
VLM_MMPROJ_PATH=/data/yuesang/LLM/test_qwen/mmproj-Qwen3VL-2B-Instruct-F16.gguf \
bash /data/yuesang/LLM/FG-CLIP/web_vllm/use_vllm_web.sh
```

如果没有 `mmproj` 文件，GGUF 后端只能加载语言模型，不能处理图片；可以先使用同目录的 transformers 权重 `/data/yuesang/LLM/test_qwen`。

数据存储在 `web_vllm/data/` 目录：
- `devices.json` - 设备信息
- `tasks.json` - 任务和布控规则
- `detection_log.json` - 检测日志
- `rules.json` - 兼容旧规则文件
- `alerts.json` - 兼容旧告警文件
- `alerts_images/` - 告警图片
