"""
RTSP流处理器和FG-CLIP匹配引擎
"""
import os
import json
import cv2
import torch
import threading
import queue
import time
import subprocess
import numpy as np
import urllib.parse
import yaml

from datetime import datetime
from PIL import Image
from typing import List, Dict, Optional

# 导入FG-CLIP模型
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoImageProcessor, CLIPConfig
from ultralytics import YOLO
from fgclip.model.clip_strc.fgclip import FGCLIPModel

# 配置路径
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
DEVICES_FILE = os.path.join(DATA_DIR, 'devices.json')
RULES_FILE = os.path.join(DATA_DIR, 'rules.json')
ALERTS_DIR = os.path.join(DATA_DIR, 'alerts_images')
LOG_FILE = os.path.join(DATA_DIR, 'detection_log.json')

def get_alerts_file():
    """获取当天日期的alerts文件路径"""
    today = datetime.now().strftime('%Y-%m-%d')
    alerts_subdir = os.path.join(DATA_DIR, 'alerts_history')
    os.makedirs(alerts_subdir, exist_ok=True)
    return os.path.join(alerts_subdir, f'alerts_{today}.json')

os.makedirs(ALERTS_DIR, exist_ok=True)

# 任务文件路径
TASKS_FILE = os.path.join(DATA_DIR, 'tasks.json')

# 全局变量
_device_manager = None
_running = False
_processing_thread = None


# 日志写入锁，防止多线程并发写入导致文件损坏
_log_lock = threading.Lock()

def write_detection_log(entry: Dict):
    """写入检测日志"""
    with _log_lock:
        logs = []
        if os.path.exists(LOG_FILE):
            try:
                with open(LOG_FILE, 'r', encoding='utf-8') as f:
                    logs = json.load(f)
            except json.JSONDecodeError:
                logs = []  # 文件损坏时清空

        logs.insert(0, entry)  # 新日志放在前面
        logs = logs[:1000]  # 保留最近1000条

        with open(LOG_FILE, 'w', encoding='utf-8') as f:
            json.dump(logs, f, ensure_ascii=False, indent=2)


class DeviceManager:
    """设备管理器"""

    def __init__(self):
        self.devices = []
        self.streams = {}
        self.load_devices()

    def load_devices(self):
        """从文件加载设备"""
        if os.path.exists(DEVICES_FILE):
            with open(DEVICES_FILE, 'r', encoding='utf-8') as f:
                self.devices = json.load(f)

    def save_devices(self):
        """保存设备到文件"""
        with open(DEVICES_FILE, 'w', encoding='utf-8') as f:
            json.dump(self.devices, f, ensure_ascii=False, indent=2)

    def add_device(self, device):
        self.devices.append(device)
        self.save_devices()

    def remove_device(self, device_id):
        self.devices = [d for d in self.devices if d['id'] != device_id]
        self.save_devices()

    def get_device(self, device_id):
        for d in self.devices:
            if d['id'] == device_id:
                return d
        return None

    def get_rtsp_url(self, device):
        """构建RTSP URL"""
        username = urllib.parse.quote(device.get('username', ''), safe='')
        password = urllib.parse.quote(device.get('password', ''), safe='')
        ip = device.get('ip', '')
        port = device.get('port', '554')
        stream_type = device.get('stream_type', 'main')

        if stream_type == 'main':
            path = device.get('main_stream_path', '/ch01')
        else:
            path = device.get('sub_stream_path', '/ch01_sub')

        return f"rtsp://{username}:{password}@{ip}:{port}{path}"


class PersonDetector:
    """基于YOLO的行人检测器"""

    def __init__(self, model_path: str = "yolov8n.pt"):
        print(f"加载YOLO模型: {model_path}")
        self.model = YOLO(model_path)
        self.model.to('cuda:0')  # 使用5号GPU
        self.person_class_id = 0  # COCO中person的类别ID

    def detect(self, frame):
        """检测行人，返回检测到的行人区域列表"""
        # 缩放帧以提高处理速度
        h, w = frame.shape[:2]
        scale = min(640 / max(h, w), 1.0)
        if scale < 1.0:
            frame = cv2.resize(frame, None, fx=scale, fy=scale)

        # YOLO推理 - 禁用保存功能
        results = self.model(frame, verbose=False, conf=0.5, classes=[0], save=False, project='/tmp', name='')  # 只检测人

        detections = []
        if results and len(results) > 0:
            result = results[0]
            if result.boxes is not None:
                for box in result.boxes:
                    # 获取边界框坐标
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().item()

                    # 反向缩放坐标
                    if scale < 1.0:
                        x1, y1, x2, y2 = x1/scale, y1/scale, x2/scale, y2/scale

                    detections.append({
                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                        'confidence': conf,
                        'class': 'person'
                    })

        return detections


class FGCLIPMatcher:
    """FG-CLIP文本-图像匹配器 + 行人属性分类"""

    def __init__(self, model_path: str = "/data/yuesang/LLM/MARS/fg-clip",
                 classifier_path: str = "/data/yuesang/LLM/FG-CLIP/checkpoints/fgclip-classification/checkpoint-1060",
                 classifier_config_path: str = "/data/yuesang/LLM/FG-CLIP/configs/classifier_config.yaml"):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # 加载FG-CLIP匹配模型
        print(f"加载FG-CLIP匹配模型: {model_path}")
        self.model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True).to(self.device)
        self.model.eval()

        # 处理器
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.image_processor = AutoImageProcessor.from_pretrained(model_path)

        # 加载分类模型
        print(f"加载分类模型: {classifier_path}")
        try:
            # 加载分类配置
            with open(classifier_config_path) as f:
                cfg = yaml.safe_load(f)
            classifier_config = cfg['classiffer_config']

            # 加载模型权重
            config = CLIPConfig.from_pretrained(model_path)
            self.classifier = FGCLIPModel(config, classifier_config=classifier_config).to(self.device)
            self.classifier.eval()

            # 加载权重
            state = torch.load(f"{classifier_path}/global_step1060/mp_rank_00_model_states.pt",
                             map_location=self.device, weights_only=False)
            module_state = state['module']
            new_state = {k.replace('_forward_module.', ''): v for k, v in module_state.items()}
            self.classifier.load_state_dict(new_state, strict=False)
            print(f"分类头数量: {len(self.classifier.classification_heads)}")
        except Exception as e:
            print(f"分类模型加载失败: {e}")
            self.classifier = None

        # 预计算的规则嵌入
        self.rule_embeddings = {}
        self.negative_embeddings = {}  # 负样本嵌入
        self.rules = []
        self.last_trigger_time = {}  # 记录每条规则上次触发时间

    def update_rules(self, rules: List[Dict]):
        """更新规则并预计算文本嵌入"""
        self.rules = [r for r in rules if r.get('enabled', True)]

        if not self.rules:
            self.rule_embeddings = {}
            self.negative_embeddings = {}
            return

        with torch.no_grad():
            for rule in self.rules:
                rule_id = rule['id']

                # 正样本向量列表
                positive_texts = rule.get('positive', [rule.get('text', '')])
                if isinstance(positive_texts, str):
                    positive_texts = [positive_texts]

                positive_features = []
                for pos_text in positive_texts:
                    if pos_text:
                        caption_input = torch.tensor(
                            self.tokenizer([pos_text], max_length=77, padding="max_length", truncation=True).input_ids,
                            dtype=torch.long,
                            device=self.device
                        )
                        text_features = self.model.get_text_features(caption_input, walk_short_pos=True)
                        text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
                        positive_features.append(text_features.cpu())
                self.rule_embeddings[rule_id] = positive_features

                # 负样本向量
                negative_texts = rule.get('negative', [])
                if negative_texts:
                    negative_features = []
                    for neg_text in negative_texts:
                        caption_input = torch.tensor(
                            self.tokenizer([neg_text], max_length=77, padding="max_length", truncation=True).input_ids,
                            dtype=torch.long,
                            device=self.device
                        )
                        text_features = self.model.get_text_features(caption_input, walk_short_pos=True)
                        text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
                        negative_features.append(text_features.cpu())
                    self.negative_embeddings[rule_id] = negative_features
                else:
                    self.negative_embeddings[rule_id] = []

    def encode_image(self, image: Image.Image) -> torch.Tensor:
        """编码图像"""
        with torch.no_grad():
            image = image.convert('RGB')
            image = image.resize((224, 224))
            image_input = self.image_processor.preprocess(image, return_tensors='pt')['pixel_values'].to(self.device)
            image_features = self.model.get_image_features(image_input)
            image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
            return image_features.cpu()

    def classify_attributes(self, image: Image.Image) -> Dict:
        """使用分类模型识别行人属性"""
        if self.classifier is None:
            return {}

        with torch.no_grad():
            image = image.convert('RGB')
            image = image.resize((224, 224))
            image_input = self.image_processor.preprocess(image, return_tensors='pt')['pixel_values'].to(self.device)

            # 获取图像特征
            image_features = self.classifier.get_image_features(image_input)

            # 分类结果
            attributes = {}
            female_prob = 0
            for name, head in self.classifier.classification_heads.items():
                output = head(image_features)
                probs = torch.softmax(output, dim=1)[0]
                prob_yes = probs[1].item()
                attributes[name] = {
                    'prob': round(prob_yes, 4),  # 是/否的概率
                    'label': 'yes' if prob_yes > 0.5 else 'no'
                }
                if name == 'Female':
                    female_prob = prob_yes

            # 添加Male属性（Female的反面）
            attributes['Male'] = {
                'prob': round(1 - female_prob, 4),
                'label': 'no' if female_prob > 0.5 else 'yes'
            }

            return attributes

    def match(self, image: Image.Image) -> List[Dict]:
        """匹配图像与规则 - 多正样本策略：任一正样本超过阈值 且 大于所有负样本分数"""
        if not self.rule_embeddings:
            return []

        image_features = self.encode_image(image)
        current_time = time.time()

        matches = []
        for rule in self.rules:
            rule_id = rule['id']
            if rule_id not in self.rule_embeddings:
                continue

            # 检查触发间隔
            interval = rule.get('interval', 60)
            if rule_id in self.last_trigger_time:
                if current_time - self.last_trigger_time[rule_id] < interval:
                    continue

            # 正样本分数列表
            positive_features = self.rule_embeddings[rule_id]
            positive_texts = rule.get('positive', [rule.get('text', '')])

            positive_scores = []
            best_positive_score = 0
            best_positive_text = ''
            for i, pos_features in enumerate(positive_features):
                score = (image_features @ pos_features.T).item()
                positive_scores.append(round(score, 4))
                if score > best_positive_score:
                    best_positive_score = score
                    best_positive_text = positive_texts[i] if i < len(positive_texts) else ''

            # 负样本分数（取最高的）
            negative_features = self.negative_embeddings.get(rule_id, [])
            max_negative_score = 0
            negative_scores = []
            for neg_features in negative_features:
                neg_score = (image_features @ neg_features.T).item()
                negative_scores.append(round(neg_score, 4))
                max_negative_score = max(max_negative_score, neg_score)

            # 判断条件：任一正样本超过阈值 且 (正样本分数大于所有负样本 + 分数差值)
            # 或者：正负样本都超过阈值，但负样本更高（高风险告警）
            score_diff = rule.get('score_diff', 0)
            if best_positive_score >= rule['threshold'] and best_positive_score > max_negative_score + score_diff:
                self.last_trigger_time[rule_id] = current_time
                # 整理正负样本列表
                positive_list = rule.get('positive', [])
                if isinstance(positive_list, str):
                    positive_list = [positive_list]
                negative_list = rule.get('negative', [])

                # 检查是否负样本更高
                negative_higher = (max_negative_score >= rule['threshold'] and max_negative_score >= best_positive_score)

                # 结构化条件检查
                struct_conditions = rule.get('struct_conditions', {})
                struct_valid = True
                struct_failed_conditions = []
                if struct_conditions and hasattr(self, 'last_attributes') and self.last_attributes:
                    for attr_name, expected in struct_conditions.items():
                        actual = self.last_attributes.get(attr_name, {})
                        actual_label = actual.get('label', 'no')
                        # expected: "yes" 或 "no"
                        if actual_label != expected:
                            struct_valid = False
                            struct_failed_conditions.append(f"{attr_name}: 期望{expected}, 实际{actual_label}")

                matches.append({
                    'rule_id': rule_id,
                    'rule_name': rule['name'],
                    'positive': best_positive_text,
                    'positive_list': positive_list,
                    'negative_list': negative_list,
                    'positive_scores': positive_scores,
                    'negative_scores': negative_scores,
                    'final_score': round(best_positive_score, 4),
                    'negative_higher': negative_higher,  # 负样本更高标记
                    'struct_valid': struct_valid,
                    'struct_failed_conditions': struct_failed_conditions
                })

        return matches


class StreamProcessor:
    """流处理器"""

    def __init__(self, device: Dict, detector: PersonDetector, matcher: FGCLIPMatcher):
        self.device_info = device
        self.detector = detector
        self.matcher = matcher
        self.running = False
        self.cap = None

        # 告警回调
        self.on_alert = None

    def start(self):
        """启动处理 - 每个流独立运行，有错误隔离"""
        device_manager = DeviceManager()
        rtsp_url = device_manager.get_rtsp_url(self.device_info)
        device_name = self.device_info.get('name', 'Unknown')

        print(f"[{device_name}] 开始连接流: {rtsp_url}")
        self.running = True
        reconnect_delay = 1
        max_reconnect_delay = 30

        while self.running:
            try:
                # 使用 ffmpeg 子进程获取帧，避免 OpenCV H.264 解码问题
                cmd = [
                    'ffmpeg', '-rtsp_transport', 'tcp',
                    '-stimeout', '5000000',  # 5秒超时
                    '-i', rtsp_url,
                    '-vframes', '1',
                    '-q:v', '5',
                    '-f', 'image2pipe',
                    '-vcodec', 'mjpeg',
                    '-'
                ]
                process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
                frame_data = process.communicate(timeout=10)

                if not frame_data[0]:
                    raise Exception("No frame data")

                # 解码图片
                nparr = np.frombuffer(frame_data[0], np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                if frame is None:
                    raise Exception("Failed to decode frame")

                reconnect_delay = 1  # 成功后重置延迟
                self._process_frame(frame)

            except subprocess.TimeoutExpired:
                print(f"[{device_name}] 读取超时，等待重连...")
            except Exception as e:
                print(f"[{device_name}] 处理异常: {e}")
            finally:
                try:
                    process.terminate()
                except:
                    pass

            # 重连等待
            if self.running:
                time.sleep(reconnect_delay)
                reconnect_delay = min(reconnect_delay * 2, max_reconnect_delay)

    def _process_frame(self, frame):
        """处理单帧 - 独立的错误隔离"""
        try:
            detections = self.detector.detect(frame)

            if detections:
                device_name = self.device_info.get('name', 'Unknown')
                print(f"[{device_name}] 检测到 {len(detections)} 个行人")

            for det in detections:  # 处理所有检测到的目标
                try:
                    self._process_detection(det, frame)
                except Exception as e:
                    print(f"处理检测异常: {e}")
        except Exception as e:
            print(f"帧处理异常: {e}")

    def _process_detection(self, det, frame):
        """处理单个检测结果 - 独立的错误隔离"""
        bbox = det['bbox']
        x1, y1, x2, y2 = bbox
        person_img = frame[y1:y2, x1:x2]

        if person_img.size == 0:
            return

        # 质量检查：宽<64 或 高<32 标记为低质量
        h, w = person_img.shape[:2]
        is_low_quality = (w < 64 or h < 32)

        person_pil = Image.fromarray(cv2.cvtColor(person_img, cv2.COLOR_BGR2RGB))

        # 获取行人属性分类（用于结构化条件检查）
        attributes = self.matcher.classify_attributes(person_pil)
        self.matcher.last_attributes = attributes  # 传递给matcher用于结构化检查

        matches = self.matcher.match(person_pil)

        alert_count = int(time.time() * 1000) % 1000000

        for match in matches:
            timestamp = datetime.now().isoformat()
            today = datetime.now().strftime('%Y-%m-%d')

            safe_device_name = self.device_info['name'].replace('/', '_').replace('\\', '_')
            safe_rule_name = match['rule_name'].replace('/', '_').replace('\\', '_')
            roi_subdir = os.path.join(ALERTS_DIR, today, safe_device_name, safe_rule_name, 'roi')
            original_subdir = os.path.join(ALERTS_DIR, today, safe_device_name, safe_rule_name, 'original')
            os.makedirs(roi_subdir, exist_ok=True)
            os.makedirs(original_subdir, exist_ok=True)

            timestamp_str = f"{int(time.time() * 1000)}_{alert_count}"
            img_filename = f"alert_{timestamp_str}.jpg"

            cv2.imwrite(os.path.join(roi_subdir, img_filename), cv2.resize(person_img, (224, 224)))

            original_img = frame.copy()
            cv2.rectangle(original_img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
            cv2.imwrite(os.path.join(original_subdir, img_filename), original_img)

            print(f"🚨 告警: {match['rule_name']} 正样本分数: {match['final_score']} 负样本分数: {match.get('negative_scores', [])}")

            # 写入检测日志
            log_entry = {
                'timestamp': timestamp,
                'device_id': self.device_info['id'],
                'device_name': self.device_info['name'],
                'rule_id': match['rule_id'],
                'rule_name': match['rule_name'],
                'positive': match.get('positive', ''),
                'positive_list': match.get('positive_list', []),
                'negative_list': match.get('negative_list', []),
                'positive_scores': match.get('positive_scores', []),
                'negative_scores': match.get('negative_scores', []),
                'score': match.get('final_score', 0),
                'negative_higher': match.get('negative_higher', False),
                'image': f"{today}/{safe_device_name}/{safe_rule_name}/roi/{img_filename}",
                'original_image': f"{today}/{safe_device_name}/{safe_rule_name}/original/{img_filename}",
                'bbox': bbox
            }
            write_detection_log(log_entry)

            # 生成告警 - 保存完整规则信息
            alert = {
                'id': f"alert_{int(time.time())}_{alert_count}",
                'device_id': self.device_info['id'],
                'device_name': self.device_info['name'],
                'rule_id': match['rule_id'],
                'rule_name': match['rule_name'],
                'positive': match.get('positive', ''),
                'positive_list': match.get('positive_list', []),
                'negative_list': match.get('negative_list', []),
                'positive_scores': match.get('positive_scores', []),
                'negative_scores': match.get('negative_scores', []),
                'score': match.get('final_score', 0),
                'negative_higher': match.get('negative_higher', False),
                'timestamp': timestamp,
                'image': f"{today}/{safe_device_name}/{safe_rule_name}/roi/{img_filename}",
                'original_image': f"{today}/{safe_device_name}/{safe_rule_name}/original/{img_filename}",
                'bbox': bbox,
                'low_quality': is_low_quality,
                'img_size': {'width': w, 'height': h},
                'attributes': attributes,  # 行人属性分类结果
                'struct_valid': match.get('struct_valid', True),
                'struct_failed_conditions': match.get('struct_failed_conditions', [])
            }

            if self.on_alert:
                self.on_alert(alert)

    def stop(self):
        """停止处理"""
        self.running = False


class MonitoringManager:
    """监控管理器"""

    def __init__(self):
        self.device_manager = DeviceManager()
        self.detector = PersonDetector()  # YOLO行人检测
        self.processors = {}
        self.threads = {}
        self.alert_callback = None
        self.tasks = []  # 保存任务列表供后续使用

    def start(self):
        """启动监控"""
        # 从tasks.json加载任务
        if os.path.exists(TASKS_FILE):
            with open(TASKS_FILE, 'r', encoding='utf-8') as f:
                self.tasks = json.load(f)

        # 启动每个设备的处理
        for device in self.device_manager.devices:
            self.start_device(device['id'])

        return {'success': True, 'message': f'已启动 {len(self.processors)} 个设备监控'}

    def refresh_rules(self):
        """刷新所有设备的规则"""
        # 重新加载tasks.json
        if os.path.exists(TASKS_FILE):
            with open(TASKS_FILE, 'r', encoding='utf-8') as f:
                self.tasks = json.load(f)

        print("🔄 刷新监控规则...")
        total_updated = 0
        for device_id, processor in self.processors.items():
            # 重新加载该设备的规则
            device_rules = []
            for task in self.tasks:
                if task.get('device_id') == device_id:
                    for rule in task.get('rules', []):
                        rule['device_id'] = device_id
                        device_rules.append(rule)
            # 更新matcher的规则
            processor.matcher.update_rules(device_rules)
            rule_names = [r.get('name', '未知') for r in device_rules]
            print(f"  设备 {processor.device_info.get('name')} 更新了 {len(device_rules)} 条规则:")
            for name in rule_names:
                print(f"    - {name}")
            total_updated += len(device_rules)

        print(f"✅ 规则刷新完成，共 {total_updated} 条规则")

        return {'success': True, 'message': f'已刷新 {total_updated} 条规则'}

    def stop(self):
        """停止监控"""
        for device_id in list(self.processors.keys()):
            self.stop_device(device_id)

        return {'success': True, 'message': '已停止所有监控'}

    def start_device(self, device_id: str):
        """启动单个设备"""
        if device_id in self.processors:
            return

        device = self.device_manager.get_device(device_id)
        if not device:
            return

        # 只获取该设备对应的规则
        device_rules = []
        for task in self.tasks:
            if task.get('device_id') == device_id:
                for rule in task.get('rules', []):
                    rule['device_id'] = device_id
                    device_rules.append(rule)

        print(f"[{device.get('name')}] 加载了 {len(device_rules)} 条规则")

        # 每个设备独立的 matcher
        matcher = FGCLIPMatcher()
        matcher.update_rules(device_rules)

        processor = StreamProcessor(device, self.detector, matcher)
        processor.on_alert = self.handle_alert
        self.processors[device_id] = processor

        thread = threading.Thread(target=processor.start, daemon=True)
        self.threads[device_id] = thread
        thread.start()

    def stop_device(self, device_id: str):
        """停止单个设备"""
        if device_id in self.processors:
            self.processors[device_id].stop()
            del self.processors[device_id]

        if device_id in self.threads:
            del self.threads[device_id]

    def handle_alert(self, alert: Dict):
        """处理告警"""
        # 保存告警到文件（加锁防止并发写入）
        with _log_lock:
            alerts_file = get_alerts_file()
            temp_file = alerts_file + '.tmp'
            alerts = []
            if os.path.exists(alerts_file):
                try:
                    with open(alerts_file, 'r', encoding='utf-8') as f:
                        alerts = json.load(f)
                except (json.JSONDecodeError, Exception):
                    pass  # 读取失败时保留已初始化的空列表

            alerts.insert(0, alert)  # 新告警放在前面

            # 先写临时文件，再重命名，避免读到不完整数据
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(alerts, f, ensure_ascii=False, indent=2)
            os.replace(temp_file, alerts_file)  # 原子操作

        if self.alert_callback:
            self.alert_callback(alert)

    def is_running(self) -> bool:
        return len(self.processors) > 0

    @property
    def devices(self):
        return [p.device_info for p in self.processors.values()]


# 全局监控实例
monitor = None


def get_monitor() -> MonitoringManager:
    global monitor
    if monitor is None:
        monitor = MonitoringManager()
    return monitor


def start_monitoring():
    return get_monitor().start()


def stop_monitoring():
    return get_monitor().stop()