# web_vllm_rknn_q35 — Qwen3.5-0.8B 板端实例（端口 5004）

[web_vllm_rknn/](../web_vllm_rknn/README.md) 的克隆：模型换为 Qwen3.5-0.8B 多模态 w8a8 基座，与线上 5003 实例同板并行。**共用同一代码库**（`cpp/src` 零改动），差异仅在模型、端口、部署目录、进程名。

## 所需模型与下载

| 模型 | 获取方式 |
|---|---|
| Qwen3.5-0.8B（HF 权重，转换输入） | https://huggingface.co/Qwen/Qwen3.5-0.8B |
| qwen3.5_vision_rk3588.rknn（219M）+ qwen3.5-0.8b_w8a8_rk3588.rkllm（1.3G） | 用 `../web_vllm_rknn` 的转换流程（见其 README「模型转换」，`export_rkllm.py --num_npu_core 3 --quantized_dtype w8a8`）从上述 HF 权重转出；母本在 `../web_vllm_rknn/models/rk3588/`，本目录 `output_cpp/models/` 已含部署副本 |
| yolov8n_640_rk3588.rknn、bge-small-zh.onnx、vocab.txt | 同 web_vllm_rknn（YOLO：https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt 转 RKNN；BGE：https://huggingface.co/BAAI/bge-small-zh-v1.5 导 ONNX） |

```bash
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download Qwen/Qwen3.5-0.8B --local-dir <本地目录>
```

## 与 web_vllm_rknn 的差异（速查）

| 项 | 值 |
|---|---|
| 端口 | **5004**（`WEB_VLLM_PORT`，代码默认 5003） |
| 板端目录 / 进程名 | `/home/web_vllm_rknn_q35` / `web_vllm_rknn_q35` |
| `rknn-llm/` | 软链 → `../web_vllm_rknn/rknn-llm`，勿在此目录改动 |
| `build_aarch64.sh` | 产物改名 `_q35`；data 仅在缺 devices.json 时播种 |
| `scripts/` | 仅 `deploy/`（push_full_py.py、push_cpp_full.sh、start_board_q35.sh） |

## 构建与部署

```bash
# x86 交叉编译（依赖上游 web_vllm_cpp 工具链/ORT 与软链的 rknn-llm）
bash cpp/build_scripts/build_aarch64.sh              # SKIP_CLEAN=1 增量

# 推板（板 pi@<板IP> → /home/web_vllm_rknn_q35，整包 ~1.6G）
python scripts/deploy/push_full_py.py                # 或 bash scripts/deploy/push_cpp_full.sh

# 板上启停（日志 app.log）
bash /home/web_vllm_rknn_q35/start_board_q35.sh
```

换模型/换端口只改 `output_cpp/run_web_rk.sh`：`VLM_MODEL_NAME`=vision `.rknn`、`VLM_MMPROJ_PATH`=`.rkllm`、`WEB_VLLM_PORT=5004`、`RKLLM_NPU_CORE_NUM=3`。

## 验证与操作要点

- 确认服务看监听，别用 pgrep：板上 `ss -tlnp | grep 5004`，或 `curl http://<板IP>:5004/`。
- **comm 15 字符截断**：`pgrep/pkill -x web_vllm_rknn_q35` 永远匹配不到；用 `pgrep -f "[.]/web_vllm_rknn_q35"`（不误杀启动脚本，也不误伤线上 5003）。
- data 为纯 JSON、**以板端为准**：构建只在缺 devices.json 时播种，运行数据勿盲推覆盖。
- 与 5003 共 NPU：跑 bench 前先停线上服务独占 NPU；`run_bench_rk.sh` 的 env 默认指向 ft 模型，跑前改。
- 板盘剩余 ~2.6G，勿再放大文件。

## bench_q35_vs_ft_board.sh

```bash
bash bench_q35_vs_ft_board.sh [image.jpg] [超时秒=600]
```

在线上目录 `/home/web_vllm_rknn` 内对比 ft-2B 与 q35：flock 防并发 → 停线上服务独占 NPU → 取最新告警图 + 线上同款中文 prompt 各测一轮 → 测完自动拉起。要点：demo 须 `printf '<image>提示词\nexit\n' | ./demo ...`（否则死循环）；解析输出先 `tr -d '\r'` 再 `grep -ao 'robot:.*'`；拉起时 `9>&-` 必带（防继承 flock fd9）。
