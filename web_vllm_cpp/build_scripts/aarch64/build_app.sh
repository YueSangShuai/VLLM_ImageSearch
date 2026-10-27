#!/bin/bash
# 【aarch64 交叉编译】程序编译：web_vllm_cpp 本体 + 组装 output/aarch64/。
# 前提：先跑过同目录 build_thirdparty.sh。
# 产物拷到 NanoPi R6S 上运行：output/aarch64/run_web.sh（相对布局需保持 仓库根/output/aarch64，
# data/ models/ 在仓库根，与 x86 版同构；也可整体 rsync 整个仓库）。
set -euo pipefail

HERE="$(cd "$(dirname "$0")/../.." && pwd)"
TC="$HERE/third_party/aarch64-cross"
LLAMA_SRC="$HERE/llama_buildsrc"
BUILD="$HERE/build/llama-aarch64"
ORT="$HERE/third_party/onnxruntime-aarch64"
DEPLOY="$HERE/output/aarch64"
TCFILE="$HERE/build_scripts/aarch64/aarch64-linux-gnu.toolchain.cmake"

[ -e "$BUILD/bin/libllama.so" ] || { echo "缺 $BUILD/bin/libllama.so —— 先跑 build_scripts/aarch64/build_thirdparty.sh"; exit 1; }
[ -e "$ORT/lib/libonnxruntime.so" ] || { echo "缺 $ORT —— 先跑 build_scripts/aarch64/build_thirdparty.sh"; exit 1; }

if [ "${SKIP_CLEAN:-0}" != "1" ]; then
    echo "    清理旧构建：$HERE/build/app-aarch64"
    rm -rf "$HERE/build/app-aarch64"
fi

echo "==> 1/2 交叉编译 web_vllm_cpp（aarch64）"
cmake -S "$HERE" -B "$HERE/build/app-aarch64" \
    -DCMAKE_TOOLCHAIN_FILE="$TCFILE" \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLAMA_LIB_DIR="$BUILD/bin" \
    -DLLAMA_INC_DIRS="$LLAMA_SRC/include;$LLAMA_SRC/ggml/include;$LLAMA_SRC/common;$LLAMA_SRC/tools/mtmd;$LLAMA_SRC/vendor" \
    -DORT_DIR="$ORT"
cmake --build "$HERE/build/app-aarch64" -j"$(nproc)"

echo "==> 2/2 组装 output/aarch64/"
mkdir -p "$DEPLOY/lib" "$DEPLOY/static/static"
cp -f "$HERE/build/app-aarch64/web_vllm_cpp" "$DEPLOY/web_vllm_cpp"
cp -f "$HERE/build/app-aarch64/bench_models" "$DEPLOY/bench_models"
cp -a -f "$BUILD"/bin/*.so* "$DEPLOY/lib/"
cp -a -f "$ORT"/lib/*.so* "$DEPLOY/lib/"
cp -a "$HERE/static/." "$DEPLOY/static/static/"
cp -a "$HERE/templates/"*.html "$DEPLOY/static/"
cp -f "$HERE/build_scripts/aarch64/run_web.sh" "$DEPLOY/run_web.sh"
cp -f "$HERE/build_scripts/aarch64/run_bench.sh" "$DEPLOY/run_bench.sh"
chmod +x "$DEPLOY/web_vllm_cpp" "$DEPLOY/run_web.sh" "$DEPLOY/bench_models" "$DEPLOY/run_bench.sh"

# 自检：用 qemu + sysroot 动态加载器检查依赖（本机 ldd 跑不了 aarch64）
if [ -x /usr/bin/qemu-aarch64-static ]; then
    miss=$(/usr/bin/qemu-aarch64-static -L "$TC/root" "$TC/root/lib/ld-linux-aarch64.so.1" \
        --list --library-path "$DEPLOY/lib" "$DEPLOY/web_vllm_cpp" "$DEPLOY/bench_models" 2>&1 | grep "not found" || true)
    [ -z "$miss" ] || { echo "缺依赖：$miss"; exit 1; }
    echo "    依赖自检通过（qemu ld.so --list）"
fi
file "$DEPLOY/web_vllm_cpp" | grep -q "ARM aarch64" || { echo "产物架构不对！"; exit 1; }

echo "==> 完成：$DEPLOY"
echo "    部署：把整个仓库（或至少 output/ + data/ + models/）拷到 NanoPi 同一相对布局，"
echo "          运行 output/aarch64/run_web.sh"
echo "    耗时基准：output/aarch64/run_bench.sh（平台上跑全部模型的加载/推理耗时）"
