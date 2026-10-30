#!/bin/bash
# 【aarch64 交叉编译】web_vllm_rknn（RK3588 NPU 版）：复用 web_vllm_cpp 的交叉工具链与 ORT aarch64。
# 产物 output_cpp/：可整体 rsync 到板上 ~/web_vllm_rknn/。
set -euo pipefail

HERE="$(cd "$(dirname "$0")/.." && pwd)"            # cpp/
CPP_ROOT="$(cd "$HERE/.." && pwd)"                  # web_vllm_rknn/
VLLM="$(cd "$CPP_ROOT/.." && pwd)/web_vllm_cpp"     # x86 版仓库（工具链/ORT/前端资产来源）
RKLLM_REPO="$CPP_ROOT/rknn-llm"
RKNN_INC="$RKLLM_REPO/examples/multimodal_model_demo/deploy/3rdparty/librknnrt/Linux/librknn_api/include"
RKNN_LIB="$RKLLM_REPO/examples/multimodal_model_demo/deploy/3rdparty/librknnrt/Linux/librknn_api/aarch64"
RKLLM_INC="$RKLLM_REPO/rkllm-runtime/Linux/librkllm_api/include"
RKLLM_LIB="$RKLLM_REPO/rkllm-runtime/Linux/librkllm_api/aarch64"
ORT="$VLLM/third_party/onnxruntime-aarch64"
TCFILE="$VLLM/build_scripts/aarch64/aarch64-linux-gnu.toolchain.cmake"
BUILD="$HERE/build/aarch64"
DEPLOY="$CPP_ROOT/output_cpp"

for p in "$TCFILE" "$ORT/lib/libonnxruntime.so" "$RKNN_LIB/librknnrt.so" "$RKLLM_LIB/librkllmrt.so"; do
    [ -e "$p" ] || { echo "缺依赖：$p"; exit 1; }
done

if [ "${SKIP_CLEAN:-0}" != "1" ]; then
    echo "    清理旧构建：$BUILD"
    rm -rf "$BUILD"
fi

echo "==> 1/2 交叉编译 web_vllm_rknn（aarch64）"
cmake -S "$HERE" -B "$BUILD" \
    -DCMAKE_TOOLCHAIN_FILE="$TCFILE" \
    -DCMAKE_BUILD_TYPE=Release \
    -DRKNN_INC="$RKNN_INC" \
    -DRKNN_LIB_DIR="$RKNN_LIB" \
    -DRKLLM_INC="$RKLLM_INC" \
    -DRKLLM_LIB_DIR="$RKLLM_LIB" \
    -DORT_DIR="$ORT" \
    -DTHIRD_PARTY_DIR="$VLLM/third_party"
cmake --build "$BUILD" -j"$(nproc)"

echo "==> 2/2 组装 output_cpp/"
mkdir -p "$DEPLOY/lib" "$DEPLOY/static/static" "$DEPLOY/models/onnx"
cp -f "$BUILD/web_vllm_rknn" "$DEPLOY/web_vllm_rknn"
cp -f "$BUILD/bench_models" "$DEPLOY/bench_models"
cp -f "$BUILD/yolo_probe" "$DEPLOY/yolo_probe"
cp -f "$BUILD/rerank_eval" "$DEPLOY/rerank_eval"
[ -e "$HERE/../output_cpp/run_bench_rk.sh" ] && chmod +x "$HERE/../output_cpp/run_bench_rk.sh" || true
cp -a -f "$RKNN_LIB/librknnrt.so"* "$DEPLOY/lib/"
cp -a -f "$RKLLM_LIB/librkllmrt.so"* "$DEPLOY/lib/"
cp -a -f "$ORT"/lib/*.so* "$DEPLOY/lib/"
cp -a "$VLLM/static/." "$DEPLOY/static/static/"
cp -a "$VLLM/templates/"*.html "$DEPLOY/static/"
# 业务数据：仅当 output_cpp/data 为空时用 web_vllm_cpp/data 播种；非空绝不覆盖
# （旧项目曾被这一步冲掉板端配置——本项目硬性保护）
if [ -z "$(ls -A "$DEPLOY/data" 2>/dev/null)" ]; then
    mkdir -p "$DEPLOY/data"
    cp -a "$VLLM/data/." "$DEPLOY/data/"
fi
# reranker 双件套 + hidden 兜底权重：软链进包（tar -h push 时解引用；小更新 push 勿带）
for m in qwen3vl-reranker_vision_rk3588.rknn qwen3vl-reranker-2b_w8a8_rk3588.rkllm; do
    [ -e "$CPP_ROOT/models/rk3588/$m" ] && ln -sfn "$CPP_ROOT/models/rk3588/$m" "$DEPLOY/models/$m" || true
done
[ -e "$CPP_ROOT/calib/rerank_yes_no_diff.bin" ] && ln -sfn "$CPP_ROOT/calib/rerank_yes_no_diff.bin" "$DEPLOY/rerank_yes_no_diff.bin" || true
# YOLO / BGE：上板先用 ORT-CPU 跑（Phase 1 再转 RKNN）
[ -e "$VLLM/models/onnx/yolov8n-dyn.onnx" ] && cp -a "$VLLM/models/onnx/yolov8n-dyn.onnx" "$DEPLOY/models/onnx/" || true
[ -e "$VLLM/models/onnx/bge-small-zh.onnx" ] && cp -a "$VLLM/models/onnx/bge-small-zh.onnx" "$DEPLOY/models/onnx/" || true
# YOLO：RKNN 版（cpp/models/，由 tools/convert_yolo_rknn.py 生成）
for m in "$HERE"/models/*.rknn; do [ -e "$m" ] && cp -a "$m" "$DEPLOY/models/"; done

file "$DEPLOY/web_vllm_rknn" | grep -q "ARM aarch64" || { echo "产物架构不对！"; exit 1; }

# 依赖自检（qemu 可用时）
if [ -x /usr/bin/qemu-aarch64-static ]; then
    TC="$VLLM/third_party/aarch64-cross/root"
    miss=$(/usr/bin/qemu-aarch64-static -L "$TC" "$TC/lib/ld-linux-aarch64.so.1" \
        --list --library-path "$DEPLOY/lib" "$DEPLOY/web_vllm_rknn" 2>&1 | grep "not found" || true)
    [ -z "$miss" ] || { echo "缺依赖：$miss"; exit 1; }
    echo "    依赖自检通过（qemu ld.so --list）"
fi

echo "==> 完成：$DEPLOY"
