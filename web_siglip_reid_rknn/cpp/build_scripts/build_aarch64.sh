#!/bin/bash
# 【aarch64 交叉编译】web_siglip_reid_rknn（RK3588 NPU 版）：
# 复用 web_vllm_rknn 的 rknn-llm 头/库 + web_vllm_cpp 的交叉工具链/ORT/third_party。
# 前置：build_scripts/build_sentencepiece_aarch64.sh
# 产物 output/：可整体 scp 到板上 ~/web_siglip_reid_rknn/。
set -euo pipefail

HERE="$(cd "$(dirname "$0")/.." && pwd)"             # cpp/
ROOT="$(cd "$HERE/.." && pwd)"                        # web_siglip_reid_rknn/
VLLM="$(cd "$ROOT/.." && pwd)/web_vllm_cpp"           # 工具链/ORT/前端资产来源
RKNN_REPO="$(cd "$ROOT/.." && pwd)/web_vllm_rknn"     # 借用其 rknn-llm 里的 rknn_api 头与 librknnrt
RKNN_INC="$RKNN_REPO/rknn-llm/examples/multimodal_model_demo/deploy/3rdparty/librknnrt/Linux/librknn_api/include"
RKNN_LIB="$RKNN_REPO/rknn-llm/examples/multimodal_model_demo/deploy/3rdparty/librknnrt/Linux/librknn_api/aarch64"
ORT="$VLLM/third_party/onnxruntime-aarch64"
SP="$ROOT/third_party/sentencepiece-src"
TCFILE="$VLLM/build_scripts/aarch64/aarch64-linux-gnu.toolchain.cmake"
BUILD="$HERE/build/aarch64"
DEPLOY="$ROOT/output"

for p in "$TCFILE" "$ORT/lib/libonnxruntime.so" "$RKNN_LIB/librknnrt.so"; do
    [ -e "$p" ] || { echo "缺依赖：$p"; exit 1; }
done
[ -n "$(find "$SP/build-aarch64" -name "*.a" 2>/dev/null | head -1)" ] \
    || { echo "缺 sentencepiece 静态库：先跑 build_scripts/build_sentencepiece_aarch64.sh"; exit 1; }

if [ "${SKIP_CLEAN:-0}" != "1" ]; then
    echo "    清理旧构建：$BUILD"
    rm -rf "$BUILD"
fi

echo "==> 1/2 交叉编译 web_siglip_reid（aarch64）"
cmake -S "$HERE" -B "$BUILD" \
    -DCMAKE_TOOLCHAIN_FILE="$TCFILE" \
    -DCMAKE_BUILD_TYPE=Release \
    -DRKNN_INC="$RKNN_INC" \
    -DRKNN_LIB_DIR="$RKNN_LIB" \
    -DORT_DIR="$ORT" \
    -DTHIRD_PARTY_DIR="$VLLM/third_party" \
    -DSP_DIR="$SP"
cmake --build "$BUILD" -j"$(nproc)"

echo "==> 2/2 组装 output/"
mkdir -p "$DEPLOY/lib" "$DEPLOY/static/static" "$DEPLOY/models/onnx" "$DEPLOY/data"
cp -f "$BUILD/web_siglip_reid" "$DEPLOY/web_siglip_reid"
cp -f "$BUILD/bench_models" "$DEPLOY/bench_models"
cp -f "$BUILD/rknn_check" "$DEPLOY/rknn_check"
# golden 对拍数据 + 运行脚本（板端 scripts/golden/raw + run_*.sh）
mkdir -p "$DEPLOY/scripts/golden"
cp -a "$ROOT"/scripts/golden/raw "$DEPLOY/scripts/golden/" 2>/dev/null || echo "    （提示：先跑 scripts/golden/export_golden.py 生成 raw）"
cp -a "$ROOT"/output/run_web_rk.sh "$ROOT"/output/run_bench_rk.sh "$DEPLOY/" 2>/dev/null || true
chmod +x "$DEPLOY"/run_*.sh 2>/dev/null || true
cp -a -f "$RKNN_LIB/librknnrt.so"* "$DEPLOY/lib/"
cp -a -f "$ORT"/lib/*.so* "$DEPLOY/lib/"
# 前端：共享模板来自 web_vllm_cpp（与原框架同源），rknn 特有页（debug.html）来自本项目
cp -a "$VLLM/static/." "$DEPLOY/static/static/"
cp -a "$VLLM/templates/"*.html "$DEPLOY/static/"
cp -a "$HERE"/static/*.html "$DEPLOY/static/" 2>/dev/null || true
# 模型：视觉塔 rknn / 文本塔 onnx / Gemma tokenizer / YOLO rknn（自 web_vllm_rknn 产物）
cp -a "$ROOT"/rknn/siglip_vision_fp16.rknn "$DEPLOY/models/" 2>/dev/null || true
# 文本塔用 int8 动态量化版（269MB vs fp32 1077MB；板子 available ~2.8GB 放不下 fp32。
# golden 规则向量 cos≥0.992，分值尺度 0.15 下漂移可忽略。x86 侧重出：见 onnx/ 同级命令注释）
cp -a "$ROOT"/onnx/siglip_text_int8.onnx "$DEPLOY/models/onnx/"
cp -a "$ROOT"/models/siglip2-base-patch16-224/tokenizer.model "$DEPLOY/models/"
for m in "$RKNN_REPO"/cpp/models/yolov8n*.rknn "$RKNN_REPO"/output_cpp/models/yolov8n*.rknn; do
    [ -e "$m" ] && { cp -a "$m" "$DEPLOY/models/"; break; }
done
# 数据种子：本项目 data/（tasks.json 英文规则样例）+ 设备清单沿用 web_vllm_cpp
cp -a "$ROOT"/data/. "$DEPLOY/data/" 2>/dev/null || true
[ -e "$DEPLOY"/data/devices.json ] || cp -a "$VLLM/data/devices.json" "$DEPLOY/data/" 2>/dev/null || true

file "$DEPLOY/web_siglip_reid" | grep -q "ARM aarch64" || { echo "产物架构不对！"; exit 1; }

# 依赖自检（qemu 可用时）
if [ -x /usr/bin/qemu-aarch64-static ]; then
    TC="$VLLM/third_party/aarch64-cross/root"
    miss=$(/usr/bin/qemu-aarch64-static -L "$TC" "$TC/lib/ld-linux-aarch64.so.1" \
        --list --library-path "$DEPLOY/lib" "$DEPLOY/web_siglip_reid" 2>&1 | grep "not found" || true)
    [ -z "$miss" ] || { echo "缺依赖：$miss"; exit 1; }
    echo "    依赖自检通过（qemu ld.so --list）"
fi

echo "==> 完成：$DEPLOY"
