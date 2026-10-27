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

                updated = False
                for device in devices:
                    old_status = device.get('status', 'offline')
                    new_status = test_rtsp_connection(device)
                    if old_status != new_status:
                        device['status'] = new_status
                        updated = True
                        print(f"📡 设备 '{device['name']}' 状态变更: {old_status} -> {new_status}")

                if updated:
                    with open(DEVICES_FILE, 'w', encoding='utf-8') as f:
                        json.dump(devices, f, ensure_ascii=False, indent=2)
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

def get_alerts_file():
    """获取当天日期的alerts文件路径"""
    today = datetime.now().strftime('%Y-%m-%d')
    alerts_subdir = os.path.join(DATA_DIR, 'alerts_history')
    os.makedirs(alerts_subdir, exist_ok=True)
    return os.path.join(alerts_subdir, f'alerts_{today}.json')

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
            break

    with open(DEVICES_FILE, 'w', encoding='utf-8') as f:
        json.dump(devices, f, ensure_ascii=False, indent=2)

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

@app.route('/api/rules', methods=['GET'])
def get_rules():
    """获取所有规则"""
    if os.path.exists(RULES_FILE):
        with open(RULES_FILE, 'r', encoding='utf-8') as f:
            rules = json.load(f)
    else:
        rules = []
    return jsonify(rules)

@app.route('/api/rules', methods=['POST'])
def add_rule():
    """添加规则"""
    data = request.json
    rule = {
        'id': str(uuid.uuid4()),
        'name': data.get('name', ''),
        'text': data.get('text', ''),
        'positive': data.get('positive', []),
        'negative': data.get('negative', []),
        'threshold': float(data.get('threshold', 0.5)),
        'enabled': data.get('enabled', True),
        'created_at': datetime.now().isoformat()
    }

    rules = []
    if os.path.exists(RULES_FILE):
        with open(RULES_FILE, 'r', encoding='utf-8') as f:
            rules = json.load(f)

    rules.append(rule)

    with open(RULES_FILE, 'w', encoding='utf-8') as f:
        json.dump(rules, f, ensure_ascii=False, indent=2)

    # 同步更新tasks.json（添加到所有使用该规则的设备）
    if os.path.exists(TASKS_FILE):
        with open(TASKS_FILE, 'r', encoding='utf-8') as f:
            tasks = json.load(f)
        for task in tasks:
            task['rules'].append({
                'id': rule['id'],
                'name': rule['name'],
                'text': rule.get('text', ''),
                'positive': rule.get('positive', []),
                'negative': rule.get('negative', []),
                'threshold': rule.get('threshold', 0.5),
                'score_diff': rule.get('score_diff', 0),
                'enabled': rule.get('enabled', True)
            })
        with open(TASKS_FILE, 'w', encoding='utf-8') as f:
            json.dump(tasks, f, ensure_ascii=False, indent=2)

    # 刷新监控器中的规则
    from processor import get_monitor
    get_monitor().refresh_rules()

    return jsonify(rule)

@app.route('/api/rules/<rule_id>', methods=['PUT'])
def update_rule(rule_id):
    """更新规则"""
    data = request.json
    rules = []
    if os.path.exists(RULES_FILE):
        with open(RULES_FILE, 'r', encoding='utf-8') as f:
            rules = json.load(f)

    updated_rule = None
    for r in rules:
        if r['id'] == rule_id:
            r.update({
                'name': data.get('name', r['name']),
                'text': data.get('text', r['text']),
                'positive': data.get('positive', r.get('positive', [])),
                'negative': data.get('negative', r.get('negative', [])),
                'threshold': float(data.get('threshold', r['threshold'])),
                'enabled': data.get('enabled', r['enabled'])
            })
            updated_rule = r
            break

    with open(RULES_FILE, 'w', encoding='utf-8') as f:
        json.dump(rules, f, ensure_ascii=False, indent=2)

    # 同步更新tasks.json中的规则
    if os.path.exists(TASKS_FILE):
        with open(TASKS_FILE, 'r', encoding='utf-8') as f:
            tasks = json.load(f)
        for task in tasks:
            for rule in task.get('rules', []):
                if rule['id'] == rule_id:
                    rule.update({
                        'name': data.get('name', rule['name']),
                        'text': data.get('text', rule.get('text', '')),
                        'positive': data.get('positive', rule.get('positive', [])),
                        'negative': data.get('negative', rule.get('negative', [])),
                        'threshold': float(data.get('threshold', rule.get('threshold', 0.5))),
                        'score_diff': float(data.get('score_diff', rule.get('score_diff', 0))),
                        'enabled': data.get('enabled', rule.get('enabled', True))
                    })
        with open(TASKS_FILE, 'w', encoding='utf-8') as f:
            json.dump(tasks, f, ensure_ascii=False, indent=2)

    # 刷新监控器中的规则
    from processor import get_monitor
    get_monitor().refresh_rules()

    return jsonify({'success': True})

@app.route('/api/rules/<rule_id>', methods=['DELETE'])
def delete_rule(rule_id):
    """删除规则"""
    rules = []
    if os.path.exists(RULES_FILE):
        with open(RULES_FILE, 'r', encoding='utf-8') as f:
            rules = json.load(f)

    rules = [r for r in rules if r['id'] != rule_id]

    with open(RULES_FILE, 'w', encoding='utf-8') as f:
        json.dump(rules, f, ensure_ascii=False, indent=2)

    # 刷新监控器中的规则
    from processor import get_monitor
    get_monitor().refresh_rules()

    return jsonify({'success': True})

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

    # 刷新监控器中的规则
    from processor import get_monitor
    get_monitor().refresh_rules()

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

@app.route('/api/alerts', methods=['GET'])
def get_alerts():
    """获取告警列表 - 读取当天文件"""
    today_alerts_file = get_alerts_file()
    if os.path.exists(today_alerts_file):
        with open(today_alerts_file, 'r', encoding='utf-8') as f:
            alerts = json.load(f)
    else:
        alerts = []

    return jsonify(alerts)

@app.route('/api/alerts/<alert_id>', methods=['DELETE'])
def delete_alert(alert_id):
    """删除告警"""
    today_alerts_file = get_alerts_file()
    temp_file = today_alerts_file + '.tmp'
    alerts = []
    try:
        if os.path.exists(today_alerts_file):
            with open(today_alerts_file, 'r', encoding='utf-8') as f:
                alerts = json.load(f)
    except (json.JSONDecodeError, Exception):
        alerts = []

    alerts = [a for a in alerts if a['id'] != alert_id]

    with open(temp_file, 'w', encoding='utf-8') as f:
        json.dump(alerts, f, ensure_ascii=False, indent=2)
    os.replace(temp_file, today_alerts_file)

    return jsonify({'success': True})

@app.route('/api/alerts/clear', methods=['POST'])
def clear_alerts():
    """清空当天告警"""
    today_alerts_file = get_alerts_file()
    with open(today_alerts_file, 'w', encoding='utf-8') as f:
        json.dump([], f)
    return jsonify({'success': True})

@app.route('/api/logs', methods=['GET'])
def get_logs():
    """获取检测日志"""
    from processor import LOG_FILE
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, 'r', encoding='utf-8') as f:
            logs = json.load(f)
    else:
        logs = []
    return jsonify(logs)

@app.route('/api/logs/clear', methods=['POST'])
def clear_logs():
    """清空检测日志"""
    from processor import LOG_FILE
    with open(LOG_FILE, 'w', encoding='utf-8') as f:
        json.dump([], f)
    return jsonify({'success': True})

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
    from processor import get_monitor
    monitor = get_monitor()
    return jsonify({
        'running': monitor.is_running(),
        'device_count': len(monitor.processors),
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

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/devices')
def devices_page():
    return render_template('devices.html')

@app.route('/rules')
def rules_page():
    return render_template('rules.html')

@app.route('/alerts')
def alerts_page():
    return render_template('alerts.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5003, debug=False, use_reloader=False)
