# web — FG-CLIP 行人布控系统（端口 5001）

RTSP 取帧 → YOLOv8 行人检测 → FG-CLIP 图文匹配（正/负样本 + 阈值）→ 属性分类头校验 → 告警落盘 + Web 展示。

## 所需模型与下载

| 模型 | 放置路径 | 下载地址 |
|---|---|---|
| FG-CLIP base（图文匹配） | 本地目录（`processor.py` 硬编码默认路径，可改） | https://huggingface.co/qihoo360/fg-clip-base （large 版：https://huggingface.co/qihoo360/fg-clip-large ） |
| 行人属性分类头权重 | `../checkpoints/fgclip-classification/checkpoint-1060/global_step1060/mp_rank_00_model_states.pt` | 无公开下载，由本仓库训练产出：`bash ../scripts/train_classification.sh`；缺失时自动降级为无属性校验 |
| YOLOv8n | `./yolov8n.pt`（本目录已带） | https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt |

```bash
# 国内可用镜像：export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download qihoo360/fg-clip-base --local-dir <本地目录>
```

## 安装

```bash
conda activate MARS        # 或等价的 torch/transformers 环境
pip install -r requirements.txt
# requirements 缺两项，需手动补：
pip install ultralytics pyyaml
# 系统依赖：ffmpeg
```

## 运行

```bash
bash use_web.sh     # 等价于：CUDA_VISIBLE_DEVICES=5 python app.py（须在本目录运行）
# 访问 http://localhost:5001
```

## 使用流程

1. `/devices` 添加 RTSP 设备（IP/端口/账密/主子码流），后台每 5s 自动探测在线状态；
2. `/rules` 创建布控规则：英文 positive/negative 文本、threshold、score_diff、interval（触发冷却，默认 60s）、结构化属性条件；保存后自动写入 `data/tasks.json` 并热刷新监控；
3. 服务启动即自动加载模型并开始监控（也可 `/api/monitoring/start|stop` 控制）；
4. `/alerts` 查看告警（ROI 224×224 裁剪 + 带框原图）。

触发条件：最佳正样本分数 ≥ threshold 且 > 最高负样本分数 + score_diff；ROI 宽<64 或高<32 跳过。

## 数据文件（data/，纯 JSON）

```
devices.json                             设备与在线状态
tasks.json                               任务 = 设备 + 规则（规则持久化于此）
detection_log.json                       检测日志（≤1000 条）
alerts_history/alerts_YYYY-MM-DD.json    按天告警
alerts_images/<日期>/<设备>/<规则>/{roi,original}/alert_<ms>_<n>.jpg
```

## API

`/api/devices|rules|tasks|alerts|logs`（CRUD）、`/api/video/frame/<device_id>`（单帧截图）、`/api/monitoring/status|start|stop`、`/data/alerts_images/<path>`。
