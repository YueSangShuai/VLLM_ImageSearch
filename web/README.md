# web — FG-CLIP 行人布控原型系统

基于 **FG-CLIP 图文匹配** 的 RTSP 流行人检测与文本布控告警系统（Flask Web 应用，**端口 5001**）。

流水线：RTSP 取帧 → YOLOv8 行人检测 → FG-CLIP 图像/文本余弦匹配（正/负样本 + 阈值 + score_diff）→ FG-CLIP 属性分类头做结构化条件校验 → 告警落盘 + Web 展示。

> 这是最早的 FG-CLIP 版本；后续演进为 [web_vllm/](../web_vllm/README.md)（Qwen-VL + BGE）及其 C++/NPU 移植。

## 功能

1. **设备管理** — RTSP 设备 CRUD（IP/端口/账密/主子码流），后台线程每 5s 心跳探测在线状态
2. **规则配置** — 自然语言正/负样本文本、阈值、score_diff、触发间隔、结构化属性条件；改规则热刷新监控
3. **告警展示** — 按设备/规则/类型筛选、分页、ROI 与全景图查看
4. **监控控制** — 启动/停止全部设备监控（启动服务时自动加载模型并开始监控）

## 安装与运行

```bash
conda activate MARS            # 或自行准备等价的 torch/transformers 环境
pip install -r requirements.txt
# 另需：ultralytics（YOLO）、pyyaml、系统 ffmpeg；FG-CLIP 包经父目录 sys.path 导入

bash /data/yuesang/LLM/FG-CLIP/web/use_web.sh    # 杀旧进程 → conda MARS → GPU 5 → 前台运行
# 访问 http://localhost:5001
```

`use_web.sh` 等价于：`CUDA_VISIBLE_DEVICES=5 python app.py`（须在本目录运行，YOLO 权重 `yolov8n.pt` 是相对路径）。

## 前置模型（路径硬编码在 processor.py）

| 用途 | 路径 |
|---|---|
| FG-CLIP 匹配模型 | `/data/yuesang/LLM/MARS/fg-clip` |
| 属性分类头权重 | `../checkpoints/fgclip-classification/checkpoint-1060/global_step1060/mp_rank_00_model_states.pt`（DeepSpeed 格式） |
| 属性分类配置 | `../configs/classifier_config.yaml`（分类头加载失败会自动降级为无属性校验） |
| YOLO | `./yolov8n.pt`（COCO class 0，conf=0.5） |

## 匹配逻辑

- 规则多条 positive 文本预计算 FG-CLIP 文本嵌入（max_length 77，`walk_short_pos=True`，L2 归一化）；
- **触发条件**：最佳正样本分数 ≥ threshold 且 > 最高负样本分数 + score_diff；
- `struct_conditions` 用属性分类头输出（Female/长发/上衣颜色等 54 属性）做符号校验；
- `interval`（默认 60s）触发冷却；ROI 宽<64 或高<32 标记 low_quality 跳过。

## API 与页面

页面：`/`（概览）、`/devices`、`/rules`、`/alerts`。
REST：`/api/devices|rules|tasks|logs|alerts`（CRUD）、`/api/video/frame/<device_id>`（ffmpeg 单帧截图）、`/api/monitoring/status|start|stop`、`/data/alerts_images/<path>`。

## 数据（data/，无数据库，纯 JSON）

```
devices.json                        设备列表与在线状态
tasks.json                          任务 = 设备 + 规则集合（规则实际持久化在这里）
detection_log.json                  最近检测日志（≤1000 条）
alerts_history/alerts_YYYY-MM-DD.json   按天告警（原子写）
alerts_images/<日期>/<设备>/<规则>/{roi,original}/alert_<ms>_<n>.jpg
```

> 注意：旧文档提到的 `rules.json` / `alerts.json` 已不再使用。

## 已知问题

- `requirements.txt` 缺 `ultralytics`、`pyyaml`，需手动补装；
- 每帧起一个 ffmpeg 子进程取图，性能一般——追求吞吐请用 web_vllm_cpp / web_vllm_rknn。
