#!/bin/bash
# 【aarch64 交叉编译（NanoPi R6S）】三方库单独编译，全部源码：
#   1. ONNX Runtime v1.18.1 —— common/build_onnxruntime.sh（ORT_ARCH_TAG=aarch64）
#      → third_party/onnxruntime-aarch64/
#   2. llama.cpp —— 仓库自带 llama_buildsrc/ → build/llama-aarch64/
# 工具链：third_party/aarch64-cross（没有则先跑 setup_cross_toolchain.sh）
# 默认全量重编；SKIP_CLEAN=1 走增量。
set -euo pipefail

HERE="$(cd "$(dirname "$0")/../.." && pwd)"
TC="$HERE/third_party/aarch64-cross"
LLAMA_SRC="$HERE/llama_buildsrc"
BUILD="$HERE/build/llama-aarch64"
TCFILE="$HERE/build_scripts/aarch64/aarch64-linux-gnu.toolchain.cmake"

[ -x "$TC/wrap/aarch64-linux-gnu-g++" ] || { echo "缺交叉工具链 —— 先跑 build_scripts/aarch64/setup_cross_toolchain.sh"; exit 1; }
[ -d "$LLAMA_SRC" ] || { echo "缺 $LLAMA_SRC"; exit 1; }

echo "==> 1/2 源码交叉编译 ONNX Runtime（aarch64）"
export ORT_ARCH_TAG=aarch64
export ORT_CROSS_TOOLCHAIN="$TCFILE"
[ -x /usr/bin/qemu-aarch64-static ] && export ORT_CROSS_EMULATOR="$HERE/build_scripts/aarch64/qemu-emulator.sh"
bash "$HERE/build_scripts/common/build_onnxruntime.sh"

if [ "${SKIP_CLEAN:-0}" != "1" ]; then
    echo "    清理旧构建：$BUILD"
    rm -rf "$BUILD"
fi

echo "==> 2/2 交叉编译 llama.cpp（aarch64，GGML_NATIVE=OFF）"
cmake -S "$LLAMA_SRC" -B "$BUILD" \
    -DCMAKE_TOOLCHAIN_FILE="$TCFILE" \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_NATIVE=OFF -DGGML_CPU_ARM_ARCH=armv8-a \
    -DGGML_OPENMP=OFF \
    -DBUILD_SHARED_LIBS=ON \
    -DLLAMA_CURL=OFF -DGGML_CCACHE=OFF \
    -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF
cmake --build "$BUILD" -j"$(nproc)" \
    --target llama ggml ggml-base llama-common mtmd

echo "==> 三方库完成：ORT=$HERE/third_party/onnxruntime-aarch64，llama=$BUILD/bin"
echo "    下一步：bash build_scripts/aarch64/build_app.sh"
