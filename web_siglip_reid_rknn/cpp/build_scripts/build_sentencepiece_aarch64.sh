#!/bin/bash
# 交叉编译 sentencepiece（Gemma 分词，静态库，aarch64）。
# 源码：third_party/sentencepiece-src（v0.2.0 GitHub 克隆；pypi sdist 不带 CMake 文件，勿用）
#   git clone --depth 1 --branch v0.2.0 https://github.com/google/sentencepiece sentencepiece-src
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"          # web_siglip_reid_rknn/
VLLM="$(cd "$ROOT/.." && pwd)/web_vllm_cpp"          # 交叉工具链来源
SP="$ROOT/third_party/sentencepiece-src"
TCFILE="$VLLM/build_scripts/aarch64/aarch64-linux-gnu.toolchain.cmake"

[ -e "$SP/src/CMakeLists.txt" ] || { echo "缺 $SP/src/CMakeLists.txt（先 git clone v0.2.0）"; exit 1; }

cd "$SP"
# 必须从仓库根 configure：config.h 由根 CMakeLists 生成（-S src 会缺 config.h）
cmake -S . -B build-aarch64 \
    -DCMAKE_TOOLCHAIN_FILE="$TCFILE" \
    -DCMAKE_BUILD_TYPE=Release \
    -DSPM_ENABLE_SHARED=OFF -DSPM_BUILD_EXAMPLES=OFF -DSPM_BUILD_python=OFF \
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON
cmake --build build-aarch64 -j"$(nproc)" --target sentencepiece-static
find build-aarch64 -name "*.a" | sed 's/^/    /'
