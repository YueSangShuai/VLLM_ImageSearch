#!/bin/bash
# 【CPU 版】三方库单独编译（全部源码编译，无预编译二进制）：
#   1. ONNX Runtime v1.18.1 —— common/build_onnxruntime.sh（CPU EP）
#   2. llama.cpp            —— 仓库自带 llama_buildsrc/，编出 CPU 共享库（GGML_CUDA=OFF）
# httplib/nlohmann-json/stb 为 header-only，无需编译。
#
# 产物：third_party/onnxruntime/{include,lib} + build/llama-cpu/bin/*.so
# 编完后再跑同目录 build_app.sh 编译程序本体。
# 默认全量重编；SKIP_CLEAN=1 走增量。
set -euo pipefail

HERE="$(cd "$(dirname "$0")/../.." && pwd)"   # 仓库根
LLAMA_SRC="$HERE/llama_buildsrc"
BUILD="$HERE/build/llama-cpu"

[ -d "$LLAMA_SRC" ] || { echo "缺 $LLAMA_SRC（仓库自带的 llama.cpp 源码）"; exit 1; }
command -v cmake >/dev/null && command -v g++ >/dev/null || { echo "需要 cmake + g++"; exit 1; }

echo "==> 1/2 源码编译 ONNX Runtime（CPU EP；首次含 clone+装依赖，全量约 15~30 分钟）"
bash "$HERE/build_scripts/common/build_onnxruntime.sh"

# 强制重编：清空旧的 cmake 缓存/目标文件；SKIP_CLEAN=1 保留增量
if [ "${SKIP_CLEAN:-0}" != "1" ]; then
    echo "    清理旧构建：$BUILD"
    rm -rf "$BUILD"
fi

echo "==> 2/2 编译 llama.cpp CPU 共享库（GGML_CUDA=OFF，全量重编约 3~8 分钟）"
cmake -S "$LLAMA_SRC" -B "$BUILD" \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_CUDA=OFF \
    -DGGML_NATIVE=OFF \
    -DBUILD_SHARED_LIBS=ON \
    -DLLAMA_CURL=OFF -DGGML_CCACHE=OFF \
    -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF
cmake --build "$BUILD" -j"$(nproc)" \
    --target llama ggml ggml-base llama-common mtmd

echo "==> 三方库完成：ORT=$HERE/third_party/onnxruntime，llama=$BUILD/bin"
echo "    下一步：bash build_scripts/cpu/build_app.sh"
