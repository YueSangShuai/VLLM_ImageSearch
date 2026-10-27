# web_vllm_cpp — 行人布控平台的 C++17 CPU 移植

[web_vllm/](../web_vllm/README.md)（Flask + transformers）的**完整 C++ 移植**：前端页面原样复用，REST 契约与 Python 版逐字段一致，全链路进程内推理、**无 vLLM / 无任何外部推理服务**，纯 CPU 运行。**端口 5003**。

## 架构映射（Python → C++）

| 功能 | Python 版 | C++ 版 |
|---|---|---|
| Qwen3-VL 描述 | transformers / llama-cpp-python | llama.cpp `libllama + libmtmd` 直接链接（GGUF） |
| YOLOv8n 检测 | ultralytics | 动态 H/W ONNX + ONNX Runtime CPU，手写 letterbox/NMS |
| BGE 文本向量 | sentence-transformers | ONNX + C++ 手写 WordPiece 分词 + CLS pooling + L2 归一化 |
| 取帧/编解码 | OpenCV | ffmpeg 子进程取 RTSP 帧 + stb 编解码（不引入 OpenCV） |
| Web | Flask + flask-cors | cpp-httplib |
| 持久化 | JSON 文件 | nlohmann/json + mutex + tmp 写后 rename 原子替换 |

`src/` 12 模块：`config / util / frame / detector / text_embedder / vlm_describer / matcher（四种匹配模式）/ stream_processor / monitor / json_store / server / main`，另有 `bench_models` 模型耗时基准。启动序列对齐 Python 版 app.py：加载配置 → 预加载模型 → 自动开监控 → HTTP 常驻。

数值对齐由 `golden/` 黄金样例验证：BGE cos=1.000000（9/9）、YOLO IoU=1.0（Python 生产环境产出的基准；验收以 `bge_golden.json` 的 vectors 为准，tokenized 记录对含 ASCII 文本不可信）。

## 目录结构

```
src/                  应用源码（CMake 目标：web_vllm_cpp / bench_models）
llama_buildsrc/       vendored llama.cpp 完整源码树（构建自包含，不依赖外部 llama.cpp）
third_party/          httplib/json/stb 头文件；onnxruntime(-src/-aarch64)；aarch64-cross 工具链；eigen-src；offline/ 离线物料
build_scripts/        构建脚本（cpu/ aarch64/ common/ offline/）—— 权威说明见 BUILD_CHANGES.md
scripts/              package_deploy.sh（GPU 遗留，已过时）、rest_ab_test.sh（Python vs C++ 契约 A/B diff）
golden/               数值对齐基准
models/               qwen3vl-ft GGUF(f16 3.4G + mmproj 819M)、onnx/{bge,yolov8n-dyn}、vocab.txt
data/                 JSON 持久化 + 告警落盘（已与 Python 版解耦，各写各的）
output/{cpu,aarch64}/ 构建产物（二进制 + lib/ + static/ + 启动脚本）
```

## 构建（x86 CPU，全源码编译、零预编译二进制）

> 当前体系以 **BUILD_CHANGES.md** 为准：Docker/GPU 构建线已全部删除（宿主机连不上 docker.io；`/usr/bin/nvcc` 是 CUDA 11.5 与 gcc≥11.3 不兼容，为历史删除原因）。PLAN/PROGRESS 里的 `build_cpu.sh`、`build_host.sh`、`deploy-cpu/` 均为旧名。

```bash
# 前置：cmake、g++、Python>=3.10（ORT 构建用，可用 ORT_PYTHON 指定）、系统 ffmpeg
# 默认走代理 http://10.8.18.5:7890，PROXY=xxx 覆盖、PROXY= 关闭

bash build_scripts/cpu/build_thirdparty.sh   # ① 源码编译 ONNX Runtime 1.18.1（首次 15~30 分钟）
                                             # ② 编译 llama_buildsrc CPU 共享库
bash build_scripts/cpu/build_app.sh          # cmake 编应用 + 组装 output/cpu/ + ldd 自检
# 增量编译：SKIP_CLEAN=1
```

## 运行

```bash
./output/cpu/run_web.sh                        # 默认端口 5003，模型默认指仓库内 models/
VLM_MATCH_MODE=rule_direct ./output/cpu/run_web.sh   # 高吞吐建议 rule_direct 或 Q4 GGUF（默认 combined）
pkill -f "web_vllm_cpp/output/cpu/web_vllm_cpp"      # 停止（路径锚定，不误杀 Python 版）
./output/cpu/run_bench.sh                      # 模型耗时基准（--help 看参数）
```

CPU 性能预期：4B F16 VLM 全链路单帧十秒级。路由与 Python 版逐字段对齐：`/ /devices /rules /alerts` + `/api/*` + `/data/alerts_images/*`。

## 交叉编译（NanoPi R6S / aarch64）与离线构建

```bash
# x86 主机交叉编译（免 root 解包官方 jammy 交叉 deb 到 third_party/aarch64-cross/）
bash build_scripts/aarch64/setup_cross_toolchain.sh    # 仅首次
bash build_scripts/aarch64/build_thirdparty.sh && bash build_scripts/aarch64/build_app.sh
# rsync output/aarch64/（保持仓库根/output/aarch64 布局）到板端跑 run_web.sh

# 无外网目标机原生编译：有网机拉离线物料（约 300MB），目标机 apply 后全程免联网
bash build_scripts/offline/fetch_offline.sh
# 目标机：bash build_scripts/offline/apply_offline.sh && cpu/ 两脚本
```

## 契约回归

```bash
bash scripts/rest_ab_test.sh   # 同一数据快照起 Python 版(5004, GGUF) vs C++ 版(5005)，逐端点归一化后 diff
```

## 其他文档

- **BUILD_CHANGES.md** — 构建体系权威（2026-10-13~14 重构记录、ORT 源码编译四个坑、交叉编译 libgomp 坑）
- **PLAN.md / PROGRESS.md** — 移植对账表、GGUF 制作、行为对齐备忘（匹配判定式、心跳 5s、退避策略、BGE 误报修复与规则文本方法论）
