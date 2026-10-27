# web_vllm_rknn — 行人布控平台的 RK3588 NPU 移植

[web_vllm_cpp/](../web_vllm_cpp/README.md)（x86/CPU C++ 版）向 **RK3588 NPU（NanoPi R6S）** 的移植：推理栈换成 **RKNN（视觉塔/YOLO）+ RKLLM（语言塔）**，matcher/monitor/server 业务层零改动复用。**板端服务端口 5003**。

性能基线（NPU ×3 核）：VLM RKLLM w8a8 **7.9s/120token**（CPU f16 的 3.9 倍）、YOLO RKNN 114.7ms（8.7fps）、视觉塔 448×448 ~1.9s、BGE 仍走 ORT-CPU ~7ms/条。

## 目录结构

| 目录 | 作用 |
|---|---|
| `cpp/` | aarch64 C++ 源码。`src/`：`main/server/monitor/stream_processor/matcher/detector`（YOLO 双后端 `#ifdef WEBVLLM_RKNN`）/`text_embedder`（BGE，仍 ORT-CPU）/`vlm_describer`（**整体重写为 RKLLM 后端**）/`rknn_image_enc`（视觉塔封装，deepstack 交错拼接）等；`build_scripts/build_aarch64.sh` 为唯一构建入口；`tools/` YOLO 转换脚本 |
| `rknn-llm/` | 官方 [airockchip/rknn-llm](https://github.com/airockchip/rknn-llm) **v1.3.1** checkout：rkllm-toolkit（x86 转换工具）、librkllmrt、驱动包、`examples/multimodal_model_demo/`（实际转换工作区） |
| `firmware/` | 换内核救砖物料（kernel 6.1.141 全套镜像 + boot-chain 分区备份 + 救砖流程见 RECOVERY.md） |
| `models/` | `qwen3vl-ft-merged{,-compat}/`（HF 权重与 rknn-toolkit 兼容目录）、`rk3588/`（各模型 .rknn/.rkllm 最终产物 ~7.8G） |
| `npu_models/` | 7 个 NPU 产物的软链合集（~7.5G，方便一次传输；**传输必须解引用**：rsync -L / tar -h） |
| `output_cpp/` | **自包含板上部署包**（~3.4G）：3 个 aarch64 二进制 + `lib/`（librknnrt/librkllmrt/libonnxruntime）+ models/ + data/ + static/ + `run_web_rk.sh` |
| `scripts/` | bench/（8 个板端基准）、deploy/（推送）、e2e/（告警链路冒烟）、firmware/（板端分区脚本） |

文档：**PLAN.md**（主进度日志）、**INSTALL.md**（模型转换环境）、**BUILD_CHANGES.md**（构建与踩坑）、**RECOVERY.md**（换内核/救砖）、**BGE_LARGE_PLAN.md**。

## 一、x86 主机模型转换（两个互斥 conda 环境）

| 环境 | 关键包 | 用途 |
|---|---|---|
| `rknn`（py3.10） | rknn-toolkit2==2.5.0、torch 2.4.0+**cpu**、transformers==4.57.0 | 视觉塔 ONNX→.rknn、YOLO |
| `rkllm`（py3.10） | rkllm-toolkit 1.3.1 wheel、torch 2.6.0+cpu、transformers==5.8.0 | LLM→.rkllm |

安装细节与依赖冲突处理见 INSTALL.md 与 `requirements_rknn.txt` / `requirements_rkllm.txt`（互斥，勿装同一环境；全程 `PYTHONNOUSERSITE=1`）。

流程（详见 PLAN.md 2026-10-15）：

```bash
# 1) LoRA 合并（rkllm-toolkit 无 --lora，必须喂合并后模型）
llamafactory-cli export ...   # → models/qwen3vl-ft-merged/

# 2) 视觉塔（rknn 环境，在 rknn-llm/.../examples/multimodal_model_demo/ 下）
python export/export_vision.py --model_name=qwen3-vl --height=448 --width=448   # → ONNX
python export/export_vision_rknn.py                                            # → .rknn

# 3) LLM（rkllm 环境）⚠️ CWD 必须在 ext4（如 /tmp）——toolkit 在 xfs 上 O_TMPFILE 预检查误报空间不足
python export/export_rkllm.py --target-platform rk3588 --num_npu_core 3 --quantized_dtype w8a8 --device cpu

# 4) 量化校准：make_input_embeds_for_quantize.py 生成 llm_inputs.json，另需 ln -s 成 inputs.json

# 5) YOLO（默认 fp16，mean/std 折进模型、板侧直喂 uint8 NHWC；不要 trim）
python cpp/tools/convert_yolo_rknn.py yolov8n.onnx yolov8n_640_rk3588.rknn [--int8]
```

## 二、板端固件前提

板子原 kernel 5.10 的 rknpu 驱动 0.8.2 是 builtin，**跑不了 RKLLM w8a8**。解法：原地换 eMMC boot-chain 4 分区（不动 rootfs），物料在 `firmware/`，脚本在 `scripts/firmware/`：

```bash
ssh pi@<板IP> 'sudo bash /home/pi/board_backup.sh'   # 先备份 4 分区 + sha256
ssh pi@<板IP> 'sudo bash /home/pi/board_flash.sh'    # 校验后写新 dtbo/resource/kernel/boot，失败自动回滚
ssh pi@<板IP> 'sudo reboot'
# 验证：uname -r = 6.1.141；dmesg | grep rknpu 显示 ≥0.9.x
```

救砖：USB-C maskrom + rkdeveloptool 全量刷 `firmware/ubuntu-jammy-minimal-arm64/`。全流程见 **RECOVERY.md**。

## 三、交叉编译与部署

```bash
# x86 主机（复用 web_vllm_cpp 的交叉工具链与 ORT aarch64 库 + 软链的 rknn-llm）
bash cpp/build_scripts/build_aarch64.sh     # 一条命令组装出 output_cpp/
# ⚠️ 组装时会用 web_vllm_cpp/data 覆盖 output_cpp/data，重编后整包 push 前先备份板上数据

# 推板（板 pi@10.8.18.118，密码见脚本；~3.4G）
bash scripts/deploy/push_cpp_full.sh        # tar -h 解引用软链流式推送
bash scripts/deploy/push_cpp.sh             # 小更新快速通道：仅 3 二进制+YOLO，带 md5 核对

# 板上启动（必须 bash，板上 sh=dash 不认 pipefail）
cd ~/output_cpp && bash run_web_rk.sh       # 生产实例部署在 /home/web_vllm_rknn/，端口 5003
```

`scripts/deploy/scp_pi.py` / `ssh_pi.py` 为 pexpect 密码自动化包装；**scp_pi.py 必须带 `push` 子命令，否则静默变 pull**；带 `pkill -f` 的复合命令勿走 ssh_pi.py 内联（会匹配 ssh 自身导致断连），写成脚本文件推上去跑。

## 四、告警链路

`main` → `Monitor` 加载 devices.json，全局单例 Detector(RKNN)/VlmDescriber(RKLLM)/TextEmbedder(BGE-ORT)；每设备一个 `StreamProcessor`：ffmpeg 单帧拉 RTSP（指数退避重连）→ YOLO 检测 → ROI → `Matcher`（默认 **combined**：先属性符号匹配，再 RKLLM 中文描述 + BGE 分句矩阵非对称检索）→ `positive ≥ threshold 且 > max_neg + score_diff` 且过 interval 冷却 → 告警落盘。

落盘：`data/alerts_images/<日期>/<设备>/<规则>/{roi,original}/alert_<ms>_<n>.jpg`、`alerts_history/alerts_YYYY-MM-DD.json`、`detection_log.json`（头插截断 1000）。Web UI（`/ /devices /rules /alerts` + `/api/*`，与 web_vllm_cpp 同契约）读取展示。

调参经验：提示词必须**中文**（英文 prompt → 中文规则 BGE 只有 0.3x，永不报警）；线上 18 条规则 threshold 0.45（中文分数分布 0.27~0.59）。

## 五、常用脚本

`scripts/bench/`：YOLO/BGE/VLM 的 NPU vs CPU 基准、Qwen3.5 对比等（共性：flock 单实例、先 `pkill -x web_vllm_rknn` 独占 NPU、测完 `setsid nohup ... 9>&-` 拉起服务）。⚠️ demo 是交互 REPL，须 `printf '<image>提示词\nexit\n' | ./demo ...`。
`scripts/e2e/`：`e2e_add_rule.sh` / `e2e_rm_rule.sh` / `e2e_finalize.sh` 告警链路冒烟一条龙。
`scripts/bench/stop_all_board.sh`：板端清理残留。

## 关键参数速查

- 板：NanoPi R6S（RK3588 8G）`pi@10.8.18.118`，Ubuntu 22.04，kernel 6.1.141，rknpu 0.9.8，`RKLLM_NPU_CORE_NUM=3`
- `run_web_rk.sh` 关键 env：`VLM_MODEL_NAME`（vision .rknn）、**`VLM_MMPROJ_PATH`（实际指向 .rkllm）**、`DETECTOR_ONNX_PATH`、`EMBEDDING_ONNX_PATH/VOCAB_PATH`、`VLM_MATCH_MODE`、`WEB_VLLM_PORT=5003`
- **YOLO RKNN 接口坑**：`rknn_outputs_get` 必须按 n_output(6) 全量请求且 `want_float=1`（单取 index 0 会报 index out of range）
- 其他坑：板时钟比 host 慢约 5 天（告警目录以板端时间为准）；板盘剩余紧张（~2.7G）；板重启不自启服务；告警图含真实 PII，禁止 scp 回本地；`output_cpp/static/` 真源在 web_vllm_cpp/templates，构建时拷入、直接改会被覆盖。
