"""
多模态大模型文搜实测平台
基于RTSP流的行人检测和文本布控
"""
import os
import json
import uuid
import cv2
import numpy as np
import subprocess
import threading
import time

# 禁用OpenCV多线程避免RTSP问题
os.environ['OPENCV_FFMPEG_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_from_directory, Response
from flask_cors import CORS


def test_rtsp_connection(device):
    """测试RTSP连接"""
    import urllib.parse
    ip = device.get('ip', '')
    port = device.get('port', '554')
    user = urllib.parse.quote(device.get('username', ''), safe='')
    pwd = urllib.parse.quote(device.get('password', ''), safe='')
    path = device.get('main_stream_path', '/ch01')
    url = f"rtsp://{user}:{pwd}@{ip}:{port}{path}"
    cap = None
    try:
        cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if cap.isOpened():
            cap.read()
            return 'online'
    except Exception as e:
        print(f"RTSP连接错误: {e}")
    finally:
        if cap is not None:
            cap.release()
    return 'offline'


# 定时检测设备状态
heartbeat_running = True

def device_heartbeat():
    """定时检测所有设备状态"""
    while heartbeat_running:
        time.sleep(5)  # 每5秒检测一次
        if not heartbeat_running:
            break
        try:
            if os.path.exists(DEVICES_FILE):
                with open(DEVICES_FILE, 'r', encoding='utf-8') as f:
                    devices = json.load(f)

                # 逐台 RTSP 探测最长数秒×N 台，这期间 /filters、PUT /api/devices
                # 都可能改文件。曾按「读整份→改→整体写回」做，把并发写入的 roi_* 等
                # 字段整片覆盖回旧值（实测把刚存的区域配置「复活」，区域形同失效）。
                # 故这里只暂存 status，写前重读文件、按 id 只合并 status，别的字段一概不碰。
                new_status = {}   # device id -> status
                for device in devices:
                    old_status = device.get('status', 'offline')
                    probe = test_rtsp_connection(device)
                    if old_status != probe:
                        new_status[device.get('id', '')] = probe
                        print(f"📡 设备 '{device['name']}' 状态变更: {old_status} -> {probe}")

                if not new_status:
                    continue

                with open(DEVICES_FILE, 'r', encoding='utf-8') as f:
                    fresh = json.load(f)
                updated = False
                for device in fresh:
                    if not isinstance(device, dict):
                        continue
                    dev_id = device.get('id', '')
                    if dev_id not in new_status or device.get('status') == new_status[dev_id]:
                        continue        # 设备已被删除或状态又被别处改了，不越权
                    device['status'] = new_status[dev_id]
                    updated = True
                if updated:
                    tmp = DEVICES_FILE + '.tmp'
                    with open(tmp, 'w', encoding='utf-8') as f:
                        json.dump(fresh, f, ensure_ascii=False, indent=2)
                    os.replace(tmp, DEVICES_FILE)
        except Exception as e:
            print(f"心跳检测异常: {e}")


# 启动心跳线程
heartbeat_thread = threading.Thread(target=device_heartbeat, daemon=True)
heartbeat_thread.daemon = True  # 确保daemon属性设置
heartbeat_thread.start()


# 注册atexit处理确保线程正确退出
import atexit
@atexit.register
def shutdown_heartbeat():
    global heartbeat_running
    heartbeat_running = False
    time.sleep(0.5)  # 等待线程自然退出

app = Flask(__name__,
            template_folder='templates',
            static_folder='static')
CORS(app)

# 数据存储路径
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
os.makedirs(DATA_DIR, exist_ok=True)

DEVICES_FILE = os.path.join(DATA_DIR, 'devices.json')
RULES_FILE = os.path.join(DATA_DIR, 'rules.json')
ALERTS_FILE = os.path.join(DATA_DIR, 'alerts.json')
ALERTS_DIR = os.path.join(DATA_DIR, 'alerts_images')
os.makedirs(ALERTS_DIR, exist_ok=True)

def valid_alert_date(date):
    """?date= 参数校验：严格 YYYY-MM-DD（与 C++ 版 valid_alert_date 同口径）"""
    if not isinstance(date, str) or len(date) != 10:
        return False
    for i, c in enumerate(date):
        if i in (4, 7):
            if c != '-':
                return False
        elif not ('0' <= c <= '9'):
            return False
    return True

def alerts_file_for(date=''):
    """告警按天分文件；date 空 = 今天"""
    if not date:
        date = datetime.now().strftime('%Y-%m-%d')
    alerts_subdir = os.path.join(DATA_DIR, 'alerts_history')
    os.makedirs(alerts_subdir, exist_ok=True)
    return os.path.join(alerts_subdir, f'alerts_{date}.json')

def get_alerts_file():
    """获取当天日期的alerts文件路径"""
    return alerts_file_for('')

def _alerts_file_from_request():
    """按 ?date= 解析本次请求的告警文件；非法日期返回 (None, 400响应)"""
    date = request.args.get('date', '')
    if date and not valid_alert_date(date):
        return None, (jsonify({'error': 'date 必须为 YYYY-MM-DD 格式'}), 400)
    return alerts_file_for(date), None

# ==================== 设备管理API ====================

@app.route('/api/devices', methods=['GET'])
def get_devices():
    """获取所有设备"""
    if os.path.exists(DEVICES_FILE):
        with open(DEVICES_FILE, 'r', encoding='utf-8') as f:
            devices = json.load(f)
    else:
        devices = []
    return jsonify(devices)

@app.route('/api/devices', methods=['POST'])
def add_device():
    """添加设备"""
    data = request.json
    device = {
        'id': str(uuid.uuid4()),
        'name': data.get('name', ''),
        'ip': data.get('ip', ''),
        'port': data.get('port', '554'),
        'username': data.get('username', ''),
        'password': data.get('password', ''),
        'main_stream_path': data.get('main_stream_path', '/ch01'),
        'sub_stream_path': data.get('sub_stream_path', '/ch01_sub'),
        'stream_type': data.get('stream_type', 'main'),
        'status': 'offline',
        'created_at': datetime.now().isoformat()
    }

    # 测试RTSP连接
    rtsp_url = f"rtsp://{device['username']}:{device['password']}@{device['ip']}:{device['port']}{device['main_stream_path']}"
    print(f"📹 设备: {device['name']}")
    print(f"🔗 测试RTSP: {rtsp_url}")
    device['status'] = test_rtsp_connection(device)
    print(f"✅ 设备状态: {device['status']}")

    devices = []
    if os.path.exists(DEVICES_FILE):
        with open(DEVICES_FILE, 'r', encoding='utf-8') as f:
            devices = json.load(f)

    devices.append(device)

    # 打印RTSP URL用于调试
    rtsp_url = f"rtsp://{device['username']}:{device['password']}@{device['ip']}:{device['port']}{device['main_stream_path']}"
    print(f"📹 设备: {device['name']}")
    print(f"🔗 RTSP URL: {rtsp_url}")

    with open(DEVICES_FILE, 'w', encoding='utf-8') as f:
        json.dump(devices, f, ensure_ascii=False, indent=2)

    return jsonify(device)

# ---- 按设备过滤配置白名单清洗（/api/devices PUT ← /filters 页保存） ----

def _clamp01(v):
    return 0.0 if v < 0 else (1.0 if v > 1.0 else v)

def sanitize_roi_regions(in_val):
    """roi_regions: [{x,y,w,h} 归一化 0..1]，逐夹紧、非法项丢弃"""
    out = []
    if not isinstance(in_val, list):
        return out
    for r in in_val:
        if not isinstance(r, dict):
            continue
        try:
            x, y, w, h = (float(r['x']), float(r['y']), float(r['w']), float(r['h']))
        except (KeyError, TypeError, ValueError):
            continue
        x, y, w, h = _clamp01(x), _clamp01(y), _clamp01(w), _clamp01(h)
        if w <= 0 or h <= 0:
            continue
        out.append({'x': x, 'y': y, 'w': w, 'h': h})
    return out

def sanitize_roi_polys(in_val):
    """roi_polygons: [[{x,y},...] 归一化 0..1]，外层层数与每块顶点数均有上限
    （区域是每帧解析的，上限防手改 json 把设备线程拖慢）。上限与 parse_roi_polys 保持一致。"""
    out = []
    if not isinstance(in_val, list):
        return out
    for raw in in_val:
        if len(out) >= 16:
            break
        if not isinstance(raw, list):
            continue
        pts = []
        for pt in raw:
            if len(pts) >= 64:
                break
            if not isinstance(pt, dict) or 'x' not in pt or 'y' not in pt:
                continue
            try:
                x, y = _clamp01(float(pt['x'])), _clamp01(float(pt['y']))
            except (TypeError, ValueError):
                continue
            if pts and pts[-1] == [x, y]:   # 相邻重复点：射线法判定会退化
                continue
            pts.append({'x': x, 'y': y})
        if len(pts) > 1 and pts[0] == pts[-1]:   # 首尾同点 = 显式闭合，运行侧隐式闭合已覆盖
            pts.pop()
        if len(pts) < 3:
            continue
        area2 = 0.0                             # 鞋带公式：共线/面积极小的退化多边形丢弃
        n = len(pts)
        for i in range(n):
            area2 += pts[i - 1]['x'] * pts[i]['y'] - pts[i]['x'] * pts[i - 1]['y']
        if abs(area2) < 1e-6:
            continue
        out.append(pts)
    return out

def sanitize_orientation_filter(in_val):
    """orientation_filter: Front/Side/Back 子集，其余丢弃"""
    if not isinstance(in_val, list):
        return []
    return [a for a in in_val if a in ('Front', 'Side', 'Back')]

def _num_or(v, default):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default

@app.route('/api/devices/<device_id>', methods=['PUT'])
def update_device(device_id):
    """更新设备"""
    data = request.json
    devices = []
    if os.path.exists(DEVICES_FILE):
        with open(DEVICES_FILE, 'r', encoding='utf-8') as f:
            devices = json.load(f)

    for d in devices:
        if d['id'] == device_id:
            d.update({
                'name': data.get('name', d['name']),
                'ip': data.get('ip', d.get('ip', '')),
                'port': data.get('port', d.get('port', '554')),
                'username': data.get('username', d.get('username', '')),
                'password': data.get('password', d.get('password', '')),
                'main_stream_path': data.get('main_stream_path', d.get('main_stream_path', '/ch01')),
                'sub_stream_path': data.get('sub_stream_path', d.get('sub_stream_path', '/ch01_sub')),
                'stream_type': data.get('stream_type', d.get('stream_type', 'main'))
            })
            # 按设备过滤配置（/filters 页保存）：缺字段保持原值
            if 'roi_regions' in data:
                d['roi_regions'] = sanitize_roi_regions(data['roi_regions'])
            if 'roi_polygons' in data:
                d['roi_polygons'] = sanitize_roi_polys(data['roi_polygons'])
            if 'orientation_filter' in data:
                d['orientation_filter'] = sanitize_orientation_filter(data['orientation_filter'])
            for key in ('min_box_w', 'min_box_h'):
                if key in data:
                    v = _num_or(data[key], -1)
                    if 0 <= v <= 4096:
                        d[key] = int(v)
            break

    with open(DEVICES_FILE, 'w', encoding='utf-8') as f:
        json.dump(devices, f, ensure_ascii=False, indent=2)

    # 设备配置热更新走后台刷新链（ROI 区域/朝向过滤/低质量阈值免重启生效）
    from processor import get_monitor
    get_monitor().refresh_rules_async()
    return jsonify({'success': True})

@app.route('/api/devices/<device_id>', methods=['DELETE'])
def delete_device(device_id):
    """删除设备"""
    devices = []
    if os.path.exists(DEVICES_FILE):
        with open(DEVICES_FILE, 'r', encoding='utf-8') as f:
            devices = json.load(f)

    devices = [d for d in devices if d['id'] != device_id]

    with open(DEVICES_FILE, 'w', encoding='utf-8') as f:
        json.dump(devices, f, ensure_ascii=False, indent=2)

    return jsonify({'success': True})

# ==================== 规则管理API ====================
# 规则随任务走（/api/tasks），独立 /api/rules CRUD 已随框架同步移除（rknn 侧已删）

# ==================== 任务管理API ====================

TASKS_FILE = os.path.join(DATA_DIR, 'tasks.json')

@app.route('/api/tasks', methods=['GET'])
def get_tasks():
    """获取所有任务"""
    if os.path.exists(TASKS_FILE):
        with open(TASKS_FILE, 'r', encoding='utf-8') as f:
            tasks = json.load(f)
    else:
        tasks = []
    return jsonify(tasks)

@app.route('/api/tasks', methods=['POST'])
def create_task():
    """创建任务"""
    data = request.json
    task = {
        'id': data.get('id', str(uuid.uuid4())),
        'name': data.get('name', ''),
        'device_id': data.get('device_id', ''),
        'device_name': data.get('device_name', ''),
        'device': data.get('device', {}),
        'rules': data.get('rules', []),
        'created_at': datetime.now().isoformat()
    }

    tasks = []
    if os.path.exists(TASKS_FILE):
        with open(TASKS_FILE, 'r', encoding='utf-8') as f:
            tasks = json.load(f)

    tasks.append(task)

    with open(TASKS_FILE, 'w', encoding='utf-8') as f:
        json.dump(tasks, f, ensure_ascii=False, indent=2)

    # 后台刷新规则：不同步等 matcher 放锁（对齐 rknn refreshRulesAsync）
    from processor import get_monitor
    get_monitor().refresh_rules_async()

    return jsonify(task)

@app.route('/api/tasks/<task_id>', methods=['GET'])
def get_task(task_id):
    """获取单个任务"""
    tasks = []
    if os.path.exists(TASKS_FILE):
        with open(TASKS_FILE, 'r', encoding='utf-8') as f:
            tasks = json.load(f)

    task = next((t for t in tasks if t['id'] == task_id), None)
    if task:
        return jsonify(task)
    return jsonify({'error': 'Task not found'}), 404

@app.route('/api/tasks/<task_id>', methods=['PUT'])
def update_task(task_id):
    """更新任务"""
    data = request.json
    tasks = []
    if os.path.exists(TASKS_FILE):
        with open(TASKS_FILE, 'r', encoding='utf-8') as f:
            tasks = json.load(f)

    for t in tasks:
        if t['id'] == task_id:
            t['name'] = data.get('name', t['name'])
            t['device_id'] = data.get('device_id', t['device_id'])
            t['device_name'] = data.get('device_name', t['device_name'])
            t['device'] = data.get('device', t.get('device', {}))
            t['rules'] = data.get('rules', t.get('rules', []))
            break

    with open(TASKS_FILE, 'w', encoding='utf-8') as f:
        json.dump(tasks, f, ensure_ascii=False, indent=2)

    # 刷新监控器中的规则（后台线程+请求合并，HTTP 零等待）
    from processor import get_monitor
    get_monitor().refresh_rules_async()

    return jsonify({'success': True})

@app.route('/api/tasks/<task_id>', methods=['DELETE'])
def delete_task(task_id):
    """删除任务"""
    tasks = []
    if os.path.exists(TASKS_FILE):
        with open(TASKS_FILE, 'r', encoding='utf-8') as f:
            tasks = json.load(f)

    tasks = [t for t in tasks if t['id'] != task_id]

    with open(TASKS_FILE, 'w', encoding='utf-8') as f:
        json.dump(tasks, f, ensure_ascii=False, indent=2)

    # 此前删任务不刷新，规则残留到下次重启
    from processor import get_monitor
    get_monitor().refresh_rules_async()

    return jsonify({'success': True})

# ==================== 视频流API ====================

# 视频帧缓存（简化版，避免多线程问题）
video_frame_cache = {}

@app.route('/api/video/frame/<device_id>')
def video_frame(device_id):
    """获取单帧图片"""
    import threading
    lock = threading.Lock()

    # 如果有缓存且在1秒内，直接返回缓存
    if device_id in video_frame_cache:
        cached_time, cached_frame = video_frame_cache[device_id]
        if time.time() - cached_time < 1:
            return Response(cached_frame, mimetype='image/jpeg')

    # 获取设备信息
    devices = []
    if os.path.exists(DEVICES_FILE):
        with open(DEVICES_FILE, 'r', encoding='utf-8') as f:
            devices = json.load(f)

    device = next((d for d in devices if d['id'] == device_id), None)
    if not device:
        return serve_placeholder("Device not found")

    stream_type = device.get('stream_type', 'main')
    if stream_type == 'main':
        path = device.get('main_stream_path', '/ch01')
    else:
        path = device.get('sub_stream_path', '/ch01_sub')

    rtsp_url = f"rtsp://{device.get('username', '')}:{device.get('password', '')}@{device.get('ip', '')}:{device.get('port', '554')}{path}"

    try:
        # 使用 ffmpeg 子进程获取帧，兼容性更好
        cmd = [
            'ffmpeg', '-rtsp_transport', 'tcp',
            '-i', rtsp_url,
            '-vframes', '1',
            '-q:v', '5',
            '-f', 'image2pipe',
            '-vcodec', 'mjpeg',
            '-'
        ]
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        frame_data = process.stdout.read()
        process.terminate()

        if frame_data:
            with lock:
                video_frame_cache[device_id] = (time.time(), frame_data)
            return Response(frame_data, mimetype='image/jpeg')
    except Exception as e:
        print(f"视频帧获取失败: {e}")

    return serve_placeholder("No signal")

def serve_placeholder(text):
    """返回占位图"""
    blank = np.zeros((480, 640, 3), np.uint8)
    cv2.putText(blank, text, (180, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    _, buffer = cv2.imencode('.jpg', blank)
    return Response(buffer.tobytes(), mimetype='image/jpeg')

# ==================== 告警API ====================

# 告警按天分文件存储；以下接口均支持 ?date=YYYY-MM-DD（缺省 = 今天）

@app.route('/api/alerts/dates', methods=['GET'])
def get_alert_dates():
    """有告警文件的日期（新→旧），供前端日期下拉框用；今天永远排第一"""
    today = datetime.now().strftime('%Y-%m-%d')
    dates = [today]
    alerts_subdir = os.path.join(DATA_DIR, 'alerts_history')
    found = []
    if os.path.isdir(alerts_subdir):
        for name in os.listdir(alerts_subdir):
            if name.startswith('alerts_') and name.endswith('.json'):
                d = name[len('alerts_'):-len('.json')]
                if valid_alert_date(d) and d != today:
                    found.append(d)
    found.sort(reverse=True)
    return jsonify(dates + found)

@app.route('/api/alerts', methods=['GET'])
def get_alerts():
    """获取告警列表 - 默认当天，可 ?date= 指定"""
    alerts_file, err = _alerts_file_from_request()
    if err:
        return err
    if os.path.exists(alerts_file):
        with open(alerts_file, 'r', encoding='utf-8') as f:
            alerts = json.load(f)
    else:
        alerts = []

    return jsonify(alerts)

@app.route('/api/alerts/<alert_id>', methods=['DELETE'])
def delete_alert(alert_id):
    """删除告警"""
    alerts_file, err = _alerts_file_from_request()
    if err:
        return err
    temp_file = alerts_file + '.tmp'
    alerts = []
    try:
        if os.path.exists(alerts_file):
            with open(alerts_file, 'r', encoding='utf-8') as f:
                alerts = json.load(f)
    except (json.JSONDecodeError, Exception):
        alerts = []

    alerts = [a for a in alerts if a['id'] != alert_id]

    with open(temp_file, 'w', encoding='utf-8') as f:
        json.dump(alerts, f, ensure_ascii=False, indent=2)
    os.replace(temp_file, alerts_file)

    return jsonify({'success': True})

@app.route('/api/alerts/clear', methods=['POST'])
def clear_alerts():
    """清空告警（默认当天，可 ?date= 指定）"""
    alerts_file, err = _alerts_file_from_request()
    if err:
        return err
    with open(alerts_file, 'w', encoding='utf-8') as f:
        json.dump([], f)
    return jsonify({'success': True})

# ==================== 调试捕获 API（VLM_DEBUG_CAPTURE） ====================

@app.route('/api/debug', methods=['GET'])
def get_debug_log():
    """调试捕获日志（最新在前），可选 ?limit=N 截断"""
    from processor import DEBUG_LOG_FILE
    if os.path.exists(DEBUG_LOG_FILE):
        try:
            with open(DEBUG_LOG_FILE, 'r', encoding='utf-8') as f:
                logs = json.load(f)
            if not isinstance(logs, list):
                logs = []
        except (json.JSONDecodeError, Exception):
            logs = []
    else:
        logs = []

    limit = request.args.get('limit', type=int)
    if limit is not None and limit >= 0:
        logs = logs[:limit]
    return jsonify(logs)

@app.route('/api/debug/clear', methods=['POST'])
def clear_debug_log():
    """清空调试日志并删除调试图片目录（防磁盘堆积）"""
    import shutil
    from processor import DEBUG_DIR, DEBUG_LOG_FILE
    if os.path.isdir(DEBUG_DIR):
        shutil.rmtree(DEBUG_DIR, ignore_errors=True)
    os.makedirs(DEBUG_DIR, exist_ok=True)
    with open(DEBUG_LOG_FILE, 'w', encoding='utf-8') as f:
        json.dump([], f)
    return jsonify({'success': True})

@app.route('/data/debug_images/<path:filename>')
def serve_debug_image(filename):
    """提供调试图片（与告警图同款 1 小时缓存）"""
    from processor import DEBUG_DIR
    response = send_from_directory(DEBUG_DIR, filename)
    response.headers['Cache-Control'] = 'public, max-age=3600'
    return response

@app.route('/data/alerts_images/<path:filename>')
def serve_alert_image(filename):
    """提供告警图片"""
    response = send_from_directory(ALERTS_DIR, filename)
    # 添加缓存头，浏览器缓存1小时
    response.headers['Cache-Control'] = 'public, max-age=3600'
    return response

# ==================== 监控状态API ====================

def get_model_info():
    """返回当前VLM模型加载配置，用于前端展示。"""
    model_name = os.getenv("VLM_MODEL_NAME", "")
    model_format = os.getenv("VLM_MODEL_FORMAT", "").lower()
    if model_format == "gguf" or model_name.lower().endswith(".gguf"):
        model_type = "GGUF"
    else:
        model_type = "HuggingFace"

    return {
        "type": model_type,
        "match_mode": os.getenv("VLM_MATCH_MODE", "structured").lower(),
        "model_name": model_name,
        "model_basename": os.path.basename(model_name.rstrip(os.sep)) if model_name else "",
        "mmproj_path": os.getenv("VLM_MMPROJ_PATH", ""),
        "mmproj_basename": os.path.basename(os.getenv("VLM_MMPROJ_PATH", "")),
    }

@app.route('/api/monitoring/status', methods=['GET'])
def get_monitoring_status():
    """获取监控状态"""
    from processor import get_monitor, DEBUG_CAPTURE
    monitor = get_monitor()
    return jsonify({
        'running': monitor.is_running(),
        'device_count': len(monitor.processors),
        'debug_capture': DEBUG_CAPTURE,
        'model': get_model_info()
    })

@app.route('/api/monitoring/start', methods=['POST'])
def start_monitoring():
    """启动监控"""
    from processor import start_monitoring
    result = start_monitoring()
    return jsonify(result)

# 启动时预加载模型并自动启动监控
with app.app_context():
    print("🚀 预加载Qwen-VL + BGE模型...")
    from processor import get_monitor, start_monitoring
    monitor = get_monitor()
    print("✅ Qwen-VL + BGE模型加载完成")

    print("🚀 自动启动监控...")
    result = start_monitoring()
    print(f"✅ 监控已启动: {result.get('message', '')}")

@app.route('/api/monitoring/stop', methods=['POST'])
def stop_monitoring():
    """停止监控"""
    from processor import stop_monitoring
    result = stop_monitoring()
    return jsonify(result)

# ==================== 页面路由 ====================

def _inject_nav(html, href, label):
    """在公共导航最后一个链接后追加入口（同 C++ 版 inject*NavLink，模板零改动）"""
    nav = html.find('</nav>')
    if nav == -1:
        return html
    a = html.rfind('</a>', 0, nav)
    if a == -1:
        return html
    a += len('</a>')
    return html[:a] + f'\n            <a href="{href}" class="nav-link">{label}</a>' + html[a:]

def render_page(name):
    from processor import DEBUG_CAPTURE
    html = render_template(name + '.html')
    # 调试捕获开启时给公共导航注入 /debug 入口（模板与 C++ 版共用，源码不动）
    if DEBUG_CAPTURE and name != 'debug':
        html = _inject_nav(html, '/debug', '调试捕获')
    # 「区域与过滤」入口常驻注入；debug/filters 页自带完整导航，不重复注入
    if name not in ('debug', 'filters'):
        html = _inject_nav(html, '/filters', '区域与过滤')
    return html

@app.route('/')
def index():
    return render_page('index')

@app.route('/devices')
def devices_page():
    return render_page('devices')

@app.route('/rules')
def rules_page():
    return render_page('rules')

@app.route('/alerts')
def alerts_page():
    return render_page('alerts')

@app.route('/debug')
def debug_page():
    return render_page('debug')

@app.route('/filters')
def filters_page():
    return render_page('filters')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5003, debug=False, use_reloader=False)
