# web_vllm_rknn_q35 — Qwen3.5-0.8B 并行实验实例（RK3588）

[web_vllm_rknn/](../web_vllm_rknn/README.md)（线上 Qwen3VL-2B-ft 实例，端口 5003）的**克隆实验版**：换用 **Qwen3.5-0.8B 多模态 w8a8 未微调基座**，与线上实例**同板并行**运行，**端口 5004**。共用同一代码库——业务代码零改动，差异只在模型产物、部署目录与进程名。

立项依据（2026-10-26 实测）：基座反超 ft-2B（13.5s vs 21.5s/轮）。v2 速度矩阵结论：q35 优势在**冷启动 −37%、视觉 ×2.3、模型小 1G、文本质量好**，常驻吞吐与 ft-2B 基本打平。详见 PLAN.md 与 `../web_vllm_rknn/BUILD_CHANGES.md`（注意 10-19「线性注意力不走 NPU」是 REPL 死循环假象，10-26 已翻案）。

## 与 web_vllm_rknn 的差异

| 项 | web_vllm_rknn（线上） | 本目录（q35 实验） |
|---|---|---|
| VLM | qwen3vl-ft_vision.rknn + ft-merged_w8a8.rkllm | **qwen3.5_vision.rknn + qwen3.5-0.8b_w8a8.rkllm** |
| 端口 | 5003 | **5004**（`WEB_VLLM_PORT`，代码默认 5003） |
| 板端目录/进程名 | `/home/web_vllm_rknn` | `/home/web_vllm_rknn_q35` |
| `rknn-llm/` | 真身 | **软链 → ../web_vllm_rknn/rknn-llm** |
| `cpp/src` | — | 与上游完全一致（`rknn_image_enc.cc` 本就支持 n_output==1，q35 视觉单输出 [196,1024]） |
| `cpp/build_scripts/build_aarch64.sh` | 无条件覆盖 data | 产物改名 `web_vllm_rknn_q35`；data **仅在缺 devices.json 时播种** |
| `scripts/` | bench/e2e/firmware/deploy 全量 | 仅 `deploy/`（新增 push_full_py.py、start_board_q35.sh） |
| data | 各自独立累积 | 种子为 2026-10-26 从板上拉回的 18 条规则（阈值 0.44~0.6） |

## 目录结构

```
cpp/            aarch64 源码（与上游零差异）+ build_scripts/build_aarch64.sh（唯一构建入口）
rknn-llm/       软链 → ../web_vllm_rknn/rknn-llm（v1.3.1，勿在此目录改动）
output_cpp/     自包含部署包（推到板上的就是它，~1.6G）：
                web_vllm_rknn_q35 二进制、run_web_rk.sh / run_bench_rk.sh、
                models/（qwen3.5_vision_rk3588.rknn 219M + qwen3.5-0.8b_w8a8_rk3588.rkllm 1.3G
                         + yolov8n_640_rk3588.rknn + onnx/{bge,yolo} + vocab.txt）、
                lib/（librknnrt/librkllmrt/libonnxruntime）、data/ 种子、static/ 前端
scripts/deploy/ push_full_py.py（pexpect 整包推送）、push_cpp_full.sh、start_board_q35.sh（板上启停）
bench_q35_vs_ft_board.sh  q35 vs ft-2B 质量对比（板上跑）
```

模型来源：HF 权重 `/data/yuesang/LLM/Qwen3.5-0.8B`；RKNN 转换产物母本在 `../web_vllm_rknn/models/rk3588/`（转换流程见上游 README）。**换模型只需改 `output_cpp/run_web_rk.sh` 的模型路径 + 端口**（`VLM_MODEL_NAME`=vision .rknn、`VLM_MMPROJ_PATH`=.rkllm、`WEB_VLLM_PORT=5004`、`RKLLM_NPU_CORE_NUM=3`）。

## 构建与部署

```bash
# x86 交叉编译（依赖上游 web_vllm_cpp 的工具链/ORT 与软链的 rknn-llm）
bash cpp/build_scripts/build_aarch64.sh              # SKIP_CLEAN=1 增量

# 推板（板 pi@10.8.18.118 → /home/web_vllm_rknn_q35）
python scripts/deploy/push_full_py.py                # 或 bash scripts/deploy/push_cpp_full.sh

# 板上启停（日志 app.log）
bash /home/web_vllm_rknn_q35/start_board_q35.sh
```

**验证端口用监听而非 pgrep**：板上 `ss -tlnp | grep 5004`，或 host 侧 `curl http://10.8.18.118:5004/`（与 5003 同一套 UI）。

## 关键坑

- **comm 15 字符截断**：`web_vllm_rknn_q35` 的 comm 被截成 `web_vllm_rknn_q`，`pgrep/pkill -x web_vllm_rknn_q35` **永远匹配不到**。正确写法（start_board_q35.sh 已固化）：`pgrep -f "[.]/web_vllm_rknn_q35"`——`[.]` 不误杀启动脚本本身，也不误伤线上 5003。`push_cpp_full.sh` 尾部的 `pgrep -x` 自检因此恒为假，属已知失效点。
- **data 为 json-only、以板端为准**：业务配置无数据库，全是 `data/` 下 JSON（devices/tasks/rules + alerts_history 按天）。构建仅在缺 devices.json 时播种，**运行数据勿盲推覆盖**（上游有 host/板端 threshold 不一致的教训）。
- 与 5003 共 NPU：跑 bench 前需独占 NPU（先停线上服务）；`run_bench_rk.sh` 的 env 默认仍指向 ft 模型，跑前注意。
- 板盘剩 ~2.6G，两套部署包并存放不下多余大文件；双实例共 NPU 的磁盘问题与后续 LoRA 微调 q35 待办见 PLAN.md。

## bench_q35_vs_ft_board.sh

用法 `bash bench_q35_vs_ft_board.sh [image.jpg] [超时秒=600]`，在**线上目录** `/home/web_vllm_rknn` 内做 demo 级对比：flock 防并发 → `pkill -x web_vllm_rknn` 独占 NPU → 取最新告警图 + 线上同款中文 prompt，分别喂 ft-2B 与 q35-0.8B 各一轮 → 测完自动拉起服务。要点：demo 是交互 REPL，必须 `printf '<image>提示词\nexit\n' | ./demo ...` 否则 EOF 空 prompt 死循环；解析输出先 `tr -d '\r'` 再 `grep -ao 'robot:.*'`；拉起时 `9>&-` 必带（否则继承 flock 的 fd9 导致锁永久占用）。
