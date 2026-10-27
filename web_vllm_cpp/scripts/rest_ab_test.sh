#!/bin/bash
# REST 契约动态 A/B：同一份数据快照上分别起 Python 版(5004) 与 C++ 版(5005)，
# 逐端点抓响应做归一化 diff（剔除时间戳/uuid/设备状态等易变字段）。
set -uo pipefail

HERE="$(cd "$(dirname "$0")/.." && pwd)"
PY_SRC="/data/yuesang/LLM/FG-CLIP/web_vllm"
WORK="/tmp/rest_ab"
SNAP="$WORK/data_snapshot"
PY_DIR="$WORK/py_app"
OUT="$WORK/out"

echo "==> 快照数据目录（只读基准）"
rm -rf "$WORK"; mkdir -p "$OUT"
cp -a "$PY_SRC/data" "$SNAP"

echo "==> 复制 Python 版代码到隔离目录（避免动生产 data/）"
mkdir -p "$PY_DIR"
cp -a "$PY_SRC/app.py" "$PY_SRC/processor.py" "$PY_SRC/templates" "$PY_SRC/static" "$PY_SRC/vlm_search" "$PY_DIR/"
cp -a "$SNAP" "$PY_DIR/data"
sed -i "s/port=5003/port=5004/" "$PY_DIR/app.py"   # app.py 端口写死，隔离副本改到 5004

norm() {  # 归一化 JSON：易变字段替换为 <X>
    python3 -c '
import json,sys,re
raw=sys.stdin.read()
try: j=json.loads(raw)
except Exception: print("PARSE_FAIL:"+raw[:200]); sys.exit(0)
VOL={"created_at","timestamp","status","id","base_description","description","base_desc"}
def walk(o):
    if isinstance(o,dict):
        return {k:("<X>" if k in VOL else walk(v)) for k,v in o.items()}
    if isinstance(o,list): return [walk(x) for x in o]
    return o
print(json.dumps(walk(j),ensure_ascii=False,sort_keys=True,indent=1))'
}

probe() {  # probe <port> <name> <method> <path> [body]
    local port=$1 tag=$2 m=$3 p=$4 body=${5:-}
    local url="http://127.0.0.1:$port$p"
    local f="$OUT/${tag}_${m}_${p//[^a-zA-Z0-9]/_}.json"
    if [ "$m" = "GET" ]; then
        curl -s --max-time 30 "$url" -o "$f.raw" -w "%{http_code}" > "$f.code"
    else
        curl -s --max-time 30 -X "$m" -H 'Content-Type: application/json' \
             -d "$body" "$url" -o "$f.raw" -w "%{http_code}" > "$f.code"
    fi
    if [ "$(head -c 2 "$f.raw" | od -An -tx1 | tr -d ' \n')" = "ffd8" ]; then
        echo "<JPEG>" > "$f.norm"   # 图片端点只比 magic
    else
        norm < "$f.raw" > "$f.norm"
    fi
}

probe_all() {  # probe_all <port> <tag>
    local port=$1 tag=$2
    # 先停监控并清空实时写入的 alerts/logs（摄像头有人时会持续写告警），保证两边起点一致
    probe $port $tag POST /api/monitoring/stop '{}'
    sleep 2
    probe $port $tag POST /api/alerts/clear '{}'
    probe $port $tag POST /api/logs/clear '{}'
    probe $port $tag GET  /api/monitoring/status
    probe $port $tag GET  /api/devices
    probe $port $tag GET  /api/rules
    probe $port $tag GET  /api/tasks
    probe $port $tag GET  /api/alerts
    probe $port $tag GET  /api/logs
    probe $port $tag POST /api/tasks '{"name":"ab_test_task","device_id":"nonexist","rules":[]}'
    probe $port $tag GET  /api/tasks/does-not-exist
    probe $port $tag POST /api/rules '{"name":"AB测试规则","text":"测试","positive":["测试"],"negative":[],"threshold":0.5}'
    probe $port $tag GET  /api/devices
    probe $port $tag GET  /api/tasks
}

echo "==> 起 Python 版 (5004, GGUF 后端)"
cd "$PY_DIR"
VLM_MODEL_FORMAT=gguf \
VLM_MODEL_NAME=/data/yuesang/LLM/FG-CLIP/models/qwen3vl-ft-20260909/qwen3vl-ft-f16.gguf \
VLM_MMPROJ_PATH=/data/yuesang/LLM/FG-CLIP/models/qwen3vl-ft-20260909/mmproj-qwen3vl-ft-f16.gguf \
VLM_MATCH_MODE=rule_direct \
VLM_GGUF_N_GPU_LAYERS=-1 \
CUDA_VISIBLE_DEVICES=0 \
nohup /home/yuesang/miniconda3/envs/FGCLIP2/bin/python app.py > "$OUT/py.log" 2>&1 &
PY_PID=$!
echo "    pid=$PY_PID，等待模型加载…"
for i in $(seq 1 120); do
    sleep 2
    curl -s --max-time 3 "http://127.0.0.1:5004/api/monitoring/status" >/dev/null 2>&1 && break
done
if ! curl -s --max-time 3 "http://127.0.0.1:5004/api/monitoring/status" >/dev/null; then
    echo "Python 版没起来，看 $OUT/py.log"; exit 1
fi
probe_all 5004 py

echo "==> 恢复数据快照（Python 侧 POST 已污染副本），并停 Python"
kill $PY_PID 2>/dev/null; wait $PY_PID 2>/dev/null || true
sleep 1
rm -rf "$PY_DIR/data"; cp -a "$SNAP" "$PY_DIR/data"

echo "==> 起 C++ 版 (5005)"
DEPLOY="$HERE/output/cpu"
[ -x "$DEPLOY/web_vllm_cpp" ] || { echo "缺 $DEPLOY/web_vllm_cpp，先跑 build_scripts/cpu/build_app.sh"; exit 1; }
cp -a "$SNAP" "$WORK/cpp_data"
WEB_VLLM_ROOT="$HERE" \
WEB_VLLM_DATA_DIR="$WORK/cpp_data" \
WEB_VLLM_STATIC_DIR="$HERE/output/cpu/static" \
LD_LIBRARY_PATH="$DEPLOY/lib:${LD_LIBRARY_PATH:-}" \
WEB_VLLM_PORT=5005 \
VLM_MODEL_NAME=/data/yuesang/LLM/FG-CLIP/models/qwen3vl-ft-20260909/qwen3vl-ft-f16.gguf \
VLM_MMPROJ_PATH=/data/yuesang/LLM/FG-CLIP/models/qwen3vl-ft-20260909/mmproj-qwen3vl-ft-f16.gguf \
VLM_MATCH_MODE=rule_direct VLM_DEVICE=cpu VLM_GGUF_N_GPU_LAYERS=0 \
nohup "$DEPLOY/web_vllm_cpp" > "$OUT/cpp.log" 2>&1 &
CPP_PID=$!
echo "    pid=$CPP_PID，等待模型加载…"
for i in $(seq 1 120); do
    sleep 2
    curl -s --max-time 3 "http://127.0.0.1:5005/api/monitoring/status" >/dev/null 2>&1 && break
done
if ! curl -s --max-time 3 "http://127.0.0.1:5005/api/monitoring/status" >/dev/null; then
    echo "C++ 版没起来，看 $OUT/cpp.log"; exit 1
fi
probe_all 5005 cpp

echo "==> 停 C++"
kill $CPP_PID 2>/dev/null; wait $CPP_PID 2>/dev/null || true

echo "==> 逐端点 diff（py vs cpp）"
FAIL=0
for f in "$OUT"/py_*.json.norm; do
    base=$(basename "$f"); c="${base/py_/cpp_}"
    cf="$OUT/$c"
    if [ ! -e "$cf" ]; then echo "缺 C++ 响应: $c"; FAIL=1; continue; fi
    if ! diff -q "$f" "$cf" > /dev/null; then
        echo "--- DIFF $base ---"; diff "$f" "$cf" | head -20; FAIL=1
    fi
done
cp "$OUT"/py_*.json.code "$OUT"/ 2>/dev/null
for f in "$OUT"/py_*.json.code; do
    base=$(basename "$f"); c="${base/py_/cpp_}"
    [ "$(cat "$f")" = "$(cat "$OUT/$c" 2>/dev/null)" ] || { echo "HTTP code 不一致 $base: $(cat $f) vs $(cat $OUT/$c 2>/dev/null)"; FAIL=1; }
done
[ $FAIL -eq 0 ] && echo "✅ REST 契约 A/B 全部一致" || echo "❌ 存在差异，见上"
exit $FAIL
