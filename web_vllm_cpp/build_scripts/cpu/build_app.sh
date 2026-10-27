#!/bin/bash
# 【CPU 版】程序编译：web_vllm_cpp 本体 + 组装 output/cpu/。
# 前提：先跑过同目录 build_thirdparty.sh（ORT + llama.cpp CPU 共享库已就位）。
# 产物 → output/cpu/（所有编译产物统一放 output/ 下）。
# 运行：output/cpu/run_web.sh（由本脚本从 build_scripts/cpu/run_web.sh 拷入）
set -euo pipefail

HERE="$(cd "$(dirname "$0")/../.." && pwd)"   # 仓库根
LLAMA_SRC="$HERE/llama_buildsrc"
BUILD="$HERE/build/llama-cpu"
DEPLOY="$HERE/output/cpu"

[ -e "$BUILD/bin/libllama.so" ] || { echo "缺 $BUILD/bin/libllama.so —— 先跑 build_scripts/cpu/build_thirdparty.sh"; exit 1; }
for f in httplib.h json.hpp stb_image.h stb_image_write.h stb_image_resize2.h \
         onnxruntime/include/onnxruntime_cxx_api.h onnxruntime/lib/libonnxruntime.so; do
    [ -e "$HERE/third_party/$f" ] || { echo "缺少 third_party/$f —— 先跑 build_scripts/cpu/build_thirdparty.sh"; exit 1; }
done

if [ "${SKIP_CLEAN:-0}" != "1" ]; then
    echo "    清理旧构建：$HERE/build/app-cpu"
    rm -rf "$HERE/build/app-cpu"
fi

echo "==> 1/2 编译 web_vllm_cpp（CPU 链接）"
cmake -S "$HERE" -B "$HERE/build/app-cpu" \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLAMA_LIB_DIR="$BUILD/bin" \
    -DLLAMA_INC_DIRS="$LLAMA_SRC/include;$LLAMA_SRC/ggml/include;$LLAMA_SRC/common;$LLAMA_SRC/tools/mtmd;$LLAMA_SRC/vendor" \
    -DORT_DIR="$HERE/third_party/onnxruntime"
cmake --build "$HERE/build/app-cpu" -j"$(nproc)"

echo "==> 2/2 组装 output/cpu/"
mkdir -p "$DEPLOY/lib" "$DEPLOY/static/static"
cp -f "$HERE/build/app-cpu/web_vllm_cpp" "$DEPLOY/web_vllm_cpp"
cp -f "$HERE/build/app-cpu/bench_models" "$DEPLOY/bench_models"
cp -a -f "$BUILD"/bin/*.so* "$DEPLOY/lib/"
cp -a -f "$HERE"/third_party/onnxruntime/lib/*.so* "$DEPLOY/lib/"
cp -a "$HERE/static/." "$DEPLOY/static/static/"
cp -a "$HERE/templates/"*.html "$DEPLOY/static/"
cp -f "$HERE/build_scripts/cpu/run_web.sh" "$DEPLOY/run_web.sh"
cp -f "$HERE/build_scripts/cpu/run_bench.sh" "$DEPLOY/run_bench.sh"
chmod +x "$DEPLOY/web_vllm_cpp" "$DEPLOY/run_web.sh" "$DEPLOY/bench_models" "$DEPLOY/run_bench.sh"

# 自检：除 libcuda（CPU 版不应出现）外不得有 not found
miss=$(LD_LIBRARY_PATH="$DEPLOY/lib" ldd "$DEPLOY/web_vllm_cpp" "$DEPLOY/bench_models" 2>&1 | grep "not found" || true)
[ -z "$miss" ] || { echo "缺依赖：$miss"; exit 1; }

echo "==> 完成：$DEPLOY"
echo "    启动：$DEPLOY/run_web.sh（纯 CPU，默认端口 5003，模型路径同环境变量可覆盖）"
echo "    耗时基准：$DEPLOY/run_bench.sh（全部模型在本机的加载/推理耗时，--help 看参数）"
