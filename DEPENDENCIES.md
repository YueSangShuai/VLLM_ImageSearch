# 第三方依赖配置指南

以下目录体积大且可从公开源重建，已被 `.gitignore` 排除。克隆本仓库后请按本文重建。

## 1. web_vllm_rknn/rknn-llm/ — Rockchip RKLLM SDK

```bash
cd web_vllm_rknn
git clone https://github.com/airockchip/rknn-llm.git
```

包含 `rkllm-runtime`（设备端运行时）、`rkllm-toolkit`（PC 端转换工具）、`rknpu-driver`。
Python 侧安装见本仓库自带的 `web_vllm_rknn/requirements_rkllm.txt`、`requirements_rknn.txt`。

## 2. web_vllm_cpp/third_party/ — C++ 第三方依赖

```bash
mkdir -p web_vllm_cpp/third_party && cd web_vllm_cpp/third_party
```

| 组件 | 获取方式 |
|---|---|
| **onnxruntime 1.18.1**（预编译 C++ 库） | 从 https://github.com/microsoft/onnxruntime/releases/tag/v1.18.1 下载 `onnxruntime-linux-x64-1.18.1.tgz`（x86 开发机）和 `onnxruntime-linux-aarch64-1.18.1.tgz`（ARM 目标机），解压后分别重命名为 `onnxruntime/`、`onnxruntime-aarch64/` |
| **llama.cpp 头文件/库** | 见下文第 3 节，编好后通过 CMake 变量 `LLAMA_LIB_DIR` / `LLAMA_INC_DIRS` 指过来 |
| **Eigen** | https://gitlab.com/libeigen/eigen/-/releases （本仓库用的是 commit e7248b2 的 zip，解压为 `eigen-src/`） |
| **nlohmann/json** | 下载单头文件 https://github.com/nlohmann/json/releases/latest → `single_include/nlohmann/json.hpp` |
| **cpp-httplib** | 下载单头文件 https://github.com/yhirose/cpp-httplib/releases/latest → `httplib.h` |
| **stb** | https://github.com/nothings/stb → `stb_image.h`、`stb_image_write.h`、`stb_image_resize2.h` |

`onnxruntime-src/` + `offline/`：仅当需要**从源码编译 onnxruntime**（如为 aarch64 交叉编译）时使用：

```bash
git clone --branch v1.18.1 https://github.com/microsoft/onnxruntime.git onnxruntime-src
cd onnxruntime-src && ./build.sh --config Release --build_shared_lib --parallel
```

`offline/`（含 MANIFEST.txt、34 个 deps zip、aarch64 wheels）由项目内的 fetch 脚本生成，
用于离线环境构建；联网环境下从源码编译时可由构建脚本自行拉取，无需还原。

## 3. web_vllm_cpp/llama_buildsrc/ — llama.cpp 源码及构建

```bash
cd web_vllm_cpp
git clone https://github.com/ggml-org/llama.cpp llama_buildsrc
cd llama_buildsrc
cmake -B build -DLLAMA_CURL=OFF -DGGML_CUDA=ON \      # 目标机部署可改交叉编译 toolchain
      -DLLAMA_BUILD_MTMD=ON
cmake --build build -j -- llama-cli mtmd-cli llama-common ggml ggml-base
```

> 需要带 `mtmd`（多模态）支持的版本；本项目链接
> `mtmd llama-common llama ggml ggml-base`（见 `web_vllm_cpp/CMakeLists.txt`）。
> 注意：本机 `/usr/bin/nvcc` 是 CUDA 11.5，仅支持 host 构建（无 docker.io）。

## 4. Python 依赖

| 场景 | 命令 |
|---|---|
| 训练/推理 (FG-CLIP) | `pip install -r requirements_fgclip2_minimind3v.txt` |
| web 服务 | `cd web && pip install -r requirements.txt` |
| RKNN 平台 | `pip install -r web_vllm_rknn/requirements_rknn.txt -r web_vllm_rknn/requirements_rkllm.txt` |

## 5. 模型文件

所有模型权重（`models/`、`checkpoints/`、`onnx_exports/`、`split_models/`、`*.rknn`、`*.gguf` 等）均不入库，
需按各子项目脚本自行导出/转换：
- ONNX 导出：`export_onnx.py` / `export_to_onnx.py` / `tools/export_fgclip_to_onnx.py`
- 模型切分：`split_fgclip_model.py`
- RKLLM 转换：见 `web_vllm_rknn/scripts/`
