# web_vllm_cpp — 行人布控平台 C++17 CPU 版（端口 5003）

[web_vllm/](../web_vllm/README.md) 的 C++ 移植：前端页面与 REST 契约不变，全链路进程内推理（llama.cpp + ONNX Runtime），纯 CPU 运行，无 vLLM 等外部服务。

## 所需模型与下载（models/）

| 模型 | 用途 | 获取方式 |
|---|---|---|
| qwen3vl-ft-f16.gguf + mmproj | VLM | 本地由微调模型转 GGUF（基座 https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct ）；**未微调可直接用官方 GGUF 替代**：https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct-GGUF （下 model + mmproj 两个文件，改 `run_web.sh` 路径即可） |
| bge-small-zh-v1.5 | 文本向量 | https://huggingface.co/BAAI/bge-small-zh-v1.5 （`vocab.txt` 取自此仓库；`onnx/bge-small-zh.onnx` 由 `torch.onnx.export` 从该仓库导出） |
| yolov8n-dyn.onnx | 行人检测 | 由 https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt 以动态 H/W 导出：`YOLO('yolov8n.pt').export(format='onnx', dynamic=True, imgsz=640)` |

## 构建（x86 CPU，全源码编译）

```bash
# 前置：cmake、g++、Python>=3.10（ORT 构建用，ORT_PYTHON 可指定）、系统 ffmpeg
# 代理：PROXY=http://<代理地址:端口> 指定、PROXY= 关闭；增量编译加 SKIP_CLEAN=1

bash build_scripts/cpu/build_thirdparty.sh   # ① ONNX Runtime 1.18.1 源码编译（首次 15~30 分钟）
                                             # ② llama_buildsrc CPU 共享库
bash build_scripts/cpu/build_app.sh          # 编译应用 + 组装 output/cpu/ + ldd 自检
```

注意：本项目只能宿主机构建（无 Docker 路线）；构建体系权威说明见 BUILD_CHANGES.md，PLAN/PROGRESS 中 `build_host.sh` 等均为旧脚本名。

## 运行

```bash
./output/cpu/run_web.sh                              # 默认端口 5003，模型指仓库内 models/
VLM_MATCH_MODE=rule_direct ./output/cpu/run_web.sh   # 默认 combined；追求吞吐用 rule_direct 或 Q4 GGUF
pkill -f "web_vllm_cpp/output/cpu/web_vllm_cpp"      # 停止
./output/cpu/run_bench.sh                            # 模型耗时基准（--help 看参数）
```

页面 `/ /devices /rules /alerts`，添加设备 → 配规则 → 自动监控告警，操作与 web_vllm 一致。

## 交叉编译（aarch64）与离线构建

```bash
# x86 主机交叉编译（免 root，官方 jammy 交叉 deb 解包到 third_party/aarch64-cross/）
bash build_scripts/aarch64/setup_cross_toolchain.sh    # 仅首次
bash build_scripts/aarch64/build_thirdparty.sh
bash build_scripts/aarch64/build_app.sh
# rsync output/aarch64/（保持 仓库根/output/aarch64 布局）到目标机，跑其 run_web.sh

# 无外网目标机原生编译：有网机拉物料（~300MB），目标机 apply 后免联网
bash build_scripts/offline/fetch_offline.sh
# 目标机：bash build_scripts/offline/apply_offline.sh && cpu/ 两脚本
```

## 数值对齐 / 契约回归

```bash
# golden/ 为 Python 版基准：BGE 以 bge_golden.json 的 vectors 为准（tokenized 记录不可信）
bash scripts/rest_ab_test.sh    # 起 Python 版(5004) vs C++ 版(5005) 逐端点 diff，需 Python 版环境
```

## 目录速查

`src/` 源码；`llama_buildsrc/` vendored llama.cpp；`third_party/` 依赖与工具链；`build_scripts/{cpu,aarch64,common,offline}/`；`data/` JSON 持久化与告警落盘（与 Python 版解耦）；`output/{cpu,aarch64}/` 产物。

## 注意

- 与 web_vllm（同端口 5003）不要同时运行。
- `scripts/package_deploy.sh` 是已删除的 GPU 线遗留，勿用。
