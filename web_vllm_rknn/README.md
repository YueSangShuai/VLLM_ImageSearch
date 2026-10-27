# web_vllm_rknn — 行人布控平台 RK3588 NPU 版（板端端口 5003）

[web_vllm_cpp/](../web_vllm_cpp/README.md) 向 RK3588（NanoPi R6S）的移植：VLM 走 RKNN（视觉塔）+ RKLLM（语言塔），YOLO 走 RKNN，BGE 走 ORT-CPU；业务层与 C++ 版一致。

## 所需模型与下载

| 模型 | 用途 | 获取方式 |
|---|---|---|
| Qwen3-VL-2B-Instruct | VLM 基座（微调/转换的输入） | https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct |
| qwen3vl-ft-merged | LoRA 合并 HF 权重 | 无公开下载：LLaMA-Factory `llamafactory-cli export` 产出 → `models/qwen3vl-ft-merged/` |
| qwen3vl vision `.rknn` + `.rkllm` | 板端 VLM | 本地转换产物（流程见下「模型转换」）；`models/rk3588/`，`npu_models/` 为软链合集 |
| yolov8n_640_rk3588.rknn | 板端行人检测 | 由 https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt 导出 ONNX 后转换：`python cpp/tools/convert_yolo_rknn.py yolov8n.onnx yolov8n_640_rk3588.rknn` |
| bge-small-zh-v1.5（→ bge-small-zh.onnx） | 文本向量（ORT-CPU） | https://huggingface.co/BAAI/bge-small-zh-v1.5 |
| rknn-llm v1.3.1 | 转换工具链 + 板端 runtime | `git clone https://github.com/airockchip/rknn-llm.git` → `rknn-llm/` |

```bash
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download Qwen/Qwen3-VL-2B-Instruct --local-dir <本地目录>
huggingface-cli download BAAI/bge-small-zh-v1.5 --local-dir ../models/bge-small-zh-v1.5
```

## 一、x86 主机模型转换（两个互斥 conda 环境）

| 环境 | 关键包 | 用途 |
|---|---|---|
| `rknn`（py3.10） | rknn-toolkit2==2.5.0、torch 2.4.0+**cpu**、transformers==4.57.0 | 视觉塔 ONNX→.rknn、YOLO |
| `rkllm`（py3.10） | rkllm-toolkit 1.3.1 wheel、torch 2.6.0+cpu、transformers==5.8.0 | LLM→.rkllm |

安装细节见 `INSTALL.md`、`requirements_rknn.txt` / `requirements_rkllm.txt`（互斥勿同装；全程 `PYTHONNOUSERSITE=1`）。

```bash
# 1) LoRA 合并（rkllm-toolkit 无 --lora，必须喂合并后模型）
llamafactory-cli export ...   # → models/qwen3vl-ft-merged/
# 2) 视觉塔（rknn 环境，在 rknn-llm/examples/multimodal_model_demo/ 下）
python export/export_vision.py --model_name=qwen3-vl --height=448 --width=448   # → ONNX
python export/export_vision_rknn.py                                             # → .rknn
# 3) LLM（rkllm 环境）⚠️ CWD 必须在 ext4（如 /tmp），xfs 上会误报空间不足
python export/export_rkllm.py --target-platform rk3588 --num_npu_core 3 --quantized_dtype w8a8 --device cpu
# 4) 量化校准：make_input_embeds_for_quantize.py 生成 llm_inputs.json，另需 ln -s 成 inputs.json
# 5) YOLO：python cpp/tools/convert_yolo_rknn.py yolov8n.onnx out.rknn（默认 fp16，不要 trim）
```

## 二、板端固件前提（kernel ≥6.1.141 / rknpu ≥0.9.x）

原 kernel 5.10 的 builtin rknpu 0.8.2 跑不了 RKLLM w8a8。换 boot-chain 4 分区（不动 rootfs），物料 `firmware/`，流程 `RECOVERY.md`：

```bash
ssh pi@<板IP> 'sudo bash /home/pi/board_backup.sh'   # 先备份分区 + sha256
ssh pi@<板IP> 'sudo bash /home/pi/board_flash.sh'    # 写新 dtbo/resource/kernel/boot，失败自动回滚
ssh pi@<板IP> 'sudo reboot'
# 验证：uname -r = 6.1.141；dmesg | grep rknpu 显示 ≥0.9.x
# 救砖：USB-C maskrom + rkdeveloptool 全量刷 firmware/ubuntu-jammy-minimal-arm64/
```

## 三、交叉编译与部署

```bash
bash cpp/build_scripts/build_aarch64.sh      # x86 上唯一构建入口，组装 output_cpp/
# ⚠️ 组装会用 web_vllm_cpp/data 覆盖 output_cpp/data，整包 push 前先备份板上数据

bash scripts/deploy/push_cpp_full.sh         # tar -h 解引用软链推整包（板 pi@<板IP>）
bash scripts/deploy/push_cpp.sh              # 小更新：仅 3 二进制+YOLO，带 md5 核对

cd ~/output_cpp && bash run_web_rk.sh        # 板上启动（必须 bash，dash 不认 pipefail）
```

`scp_pi.py` 必须带 `push` 子命令（否则静默变 pull）；带 `pkill -f` 的命令写成脚本推上去跑，勿走 ssh_pi.py 内联。

## 运行配置（run_web_rk.sh 环境变量）

- `VLM_MODEL_NAME` = vision `.rknn`；`VLM_MMPROJ_PATH` = `.rkllm`（变量名沿用，实指 LLM 权重）
- `DETECTOR_ONNX_PATH`（rknn/onnx 兼容）、`EMBEDDING_ONNX_PATH` / `EMBEDDING_VOCAB_PATH`
- `VLM_MATCH_MODE`（默认 combined）、`WEB_VLLM_PORT=5003`、`RKLLM_NPU_CORE_NUM=3`
- **提示词必须中文**（英文 prompt 会导致 BGE 分数 0.3x 永不报警）；线上规则 threshold 0.45

## 使用流程与数据

添加设备 → 配规则（`/rules`）→ 自动监控；告警落盘 `data/alerts_images/<日期>/<设备>/<规则>/{roi,original}/`、`alerts_history/alerts_YYYY-MM-DD.json`、`detection_log.json`。触发条件 `positive ≥ threshold 且 > max_neg + score_diff` + interval 冷却。页面与 API 同 web_vllm_cpp。

## 其他

- `scripts/bench/` 板端基准（flock 单实例，先停服务独占 NPU；demo 是交互 REPL，须 `printf '<image>提示词\nexit\n' | ./demo ...`）；`scripts/e2e/` 告警链路冒烟；`scripts/bench/stop_all_board.sh` 清残留。
- 板上 `output_cpp/static/` 真源在 web_vllm_cpp/templates，构建时拷入，别直接改。
- 告警图含真实行人 PII，勿外传。
