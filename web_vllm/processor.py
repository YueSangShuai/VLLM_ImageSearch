"""
RTSP流处理器和VLM文本匹配引擎
"""
import os
import json
import re
import cv2
import torch
import threading
import time
import subprocess
import numpy as np
import urllib.parse

from datetime import datetime
from PIL import Image
from typing import List, Dict

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ultralytics import YOLO

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

# ==================== 结构化属性识别配置 ====================

# 行人预定义属性列表（结构化模式输出仅允许出现这些属性名）
PEDESTRIAN_ATTRIBUTES = [
    "Female", "AgeOver60", "Age18-60", "AgeLess18", "Front", "Side", "Back",
    "Hat", "Glasses", "HandBag", "ShoulderBag", "Backpack", "HoldObjectsInFront",
    "ShortSleeve", "LongSleeve", "UpperStride", "UpperLogo", "UpperPlaid", "UpperSplice",
    "LowerStripe", "LowerPattern", "LongCoat", "Trousers", "Shorts", "Skirt&Dress", "boots",
    "ublack", "ugray", "ublue", "ugreen", "uwhite", "upurple", "ured", "ubrown",
    "uyellow", "upink", "uorange", "ubeige", "ustriped_color", "umulticolor",
    "lwhite", "lpink", "lred", "lgreen", "lyellow", "lpurple", "lbrown", "lblack",
    "lorange", "lblue", "lgray", "lbeige", "lstriped_color", "lmulticolor",
]

# 属性名查找表：小写 / 去分隔符两种写法都能命中
_ATTR_LOOKUP = {}
for _attr in PEDESTRIAN_ATTRIBUTES:
    _ATTR_LOOKUP[_attr.lower()] = _attr
    _ATTR_LOOKUP[re.sub(r"[-_ ]", "", _attr).lower()] = _attr
# Male 不在提示词列表里，由代码在 Female 缺失时派生；注册进查找表以便规则可符号匹配
_ATTR_LOOKUP["male"] = "Male"

# 结构化提示词：让VLM只输出值为1的属性名，逗号分隔
DEFAULT_STRUCTURED_PROMPT = (
    "请分析这张行人图像，从以下预定义属性中找出所有存在的属性（即值为1的属性），"
    "只输出这些属性名，用逗号分隔，不要输出任何其他文字或值为0的属性。\n"
    f"预定义属性列表：{','.join(PEDESTRIAN_ATTRIBUTES)}"
)

# 文本描述提示词（description模式）
DEFAULT_DESCRIPTION_PROMPT = (
    "Describe the pedestrian in the image in detail, including gender, age range, "
    "view direction, clothing, colors, carried objects, and other visible attributes."
)

# 规则直判提示词（rule_direct模式）：{rules} 处由代码填入规则列表
DEFAULT_RULE_JUDGE_PROMPT = (
    "请判断图像中的行人是否符合以下每一条规则。\n"
    "{rules}\n"
    "输出要求：只输出符合的规则名称，多条之间用顿号「、」分隔；"
    "如果没有任何规则符合，只输出：不匹配\n"
    "除规则名称或「不匹配」外，不要输出任何其他文字、编号、标点或解释；"
    "只依据图像中清晰可见的内容判断，不要依据规则名称猜测。"
)


def strip_image_placeholder(prompt: str) -> str:
    """去掉提示词里的 <image> 占位符；图片由 VLMDescriber 自行插入。"""
    prompt = re.sub(r"[ \t]*<\s*image\s*>[ \t]*", "\n", prompt, flags=re.IGNORECASE)
    return re.sub(r"[ \t]+", " ", prompt).strip()


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
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)
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


class VLMTextMatcher:
    """Qwen-VL图像描述 + BGE文本相似度匹配器"""
    _shared_vlm_describer = None
    _shared_vlm_key = None
    _shared_text_embedder = None
    _shared_embedder_key = None

    def __init__(
        self,
        vlm_model_path: str = None,
        embedding_model_path: str = None,
        vision_model_path: str = None
    ):
        device_name = os.getenv("VLM_DEVICE", 'cuda:0' if torch.cuda.is_available() else 'cpu')
        if device_name.startswith('cuda') and not torch.cuda.is_available():
            device_name = 'cpu'
        embedding_device_name = os.getenv("EMBEDDING_DEVICE", device_name)
        if embedding_device_name.startswith('cuda') and not torch.cuda.is_available():
            embedding_device_name = 'cpu'
        self.device = torch.device(device_name)
        self.embedding_device = torch.device(embedding_device_name)
        print(f"Using device: {self.device}")

        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.vlm_model_path = vlm_model_path or os.getenv(
            "VLM_MODEL_NAME",
            os.path.join(base_dir, "models", "qwen2.5-vl-3b-instruct")
        )
        self.embedding_model_path = embedding_model_path or os.getenv(
            "EMBEDDING_MODEL_NAME",
            os.path.join(base_dir, "models", "bge-small-zh-v1.5")
        )
        self.vision_model_path = vision_model_path or os.getenv(
            "VLM_VISION_MODEL_PATH",
            os.path.join(base_dir, "models", "siglip2-base-p32-256-ve")
        )
        os.environ.setdefault("VLM_VISION_MODEL_PATH", self.vision_model_path)

        vlm_search_dir = os.getenv("VLM_SEARCH_DIR", "/data/yuesang/LLM/vlm_search")
        if vlm_search_dir not in sys.path:
            sys.path.insert(0, vlm_search_dir)

        from vlm_describer import VLMDescriber
        from text_search import TextEmbedder

        # 匹配模式: structured=结构化属性输出 | description=自由文本描述
        # | combined=两者兼有 | rule_direct=规则直接代入提示词，由VLM判断并只输出命中的规则名
        self.match_mode = os.getenv("VLM_MATCH_MODE", "structured").strip().lower()
        if self.match_mode not in ("structured", "description", "combined", "rule_direct"):
            print(f"未知的VLM_MATCH_MODE: {self.match_mode}，回退到 structured")
            self.match_mode = "structured"
        # 主描述（英文）对应的中文翻译，describe_image 每次刷新，同线程内被 match 读取
        self.last_description_zh = ""
        # description 模式下与描述一并生成的结构化属性（校验后），同线程内被 match 读取
        self.last_attributes = []

        self.base_prompt = strip_image_placeholder(os.getenv(
            "VLM_DESCRIPTION_PROMPT",
            DEFAULT_DESCRIPTION_PROMPT
        ))
        self.structured_prompt = strip_image_placeholder(os.getenv(
            "VLM_STRUCTURED_PROMPT",
            DEFAULT_STRUCTURED_PROMPT
        ))
        self.structured_max_new_tokens = int(os.getenv("VLM_STRUCTURED_MAX_NEW_TOKENS", "128"))

        # rule_direct 模式：规则列表代入提示词，VLM 直接判断，只输出命中的规则名或「不匹配」
        self.rule_judge_prompt = strip_image_placeholder(os.getenv(
            "VLM_RULE_JUDGE_PROMPT",
            DEFAULT_RULE_JUDGE_PROMPT
        ))
        if "{rules}" not in self.rule_judge_prompt:
            self.rule_judge_prompt = DEFAULT_RULE_JUDGE_PROMPT
        self.rule_judge_max_new_tokens = int(os.getenv("VLM_RULE_JUDGE_MAX_NEW_TOKENS", "128"))

        # describer 的默认 template 按当前模式选择对应提示词
        template_prompt = (
            self.structured_prompt if self.match_mode == "structured" else self.base_prompt
        )

        self.rule_prompt_max_texts = int(os.getenv("VLM_RULE_PROMPT_MAX_TEXTS", "3"))
        self.vlm_max_new_tokens = int(os.getenv("VLM_MAX_NEW_TOKENS", "120"))
        self.rule_description_enabled = os.getenv("VLM_RULE_DESCRIPTION_ENABLED", "1").lower() not in ("0", "false", "no")
        self.rule_description_verify = os.getenv("VLM_RULE_DESCRIPTION_VERIFY", "1").lower() not in ("0", "false", "no")
        self.rule_description_max_new_tokens = int(os.getenv("VLM_RULE_DESCRIPTION_MAX_NEW_TOKENS", "60"))
        self.vlm_temperature = float(os.getenv("VLM_GENERATION_TEMPERATURE", "0.0"))
        self.vlm_repetition_penalty = float(os.getenv("VLM_REPETITION_PENALTY", "1.2"))

        vlm_key = (self.vlm_model_path, str(self.device), template_prompt, self.vision_model_path)
        if VLMTextMatcher._shared_vlm_describer is None or VLMTextMatcher._shared_vlm_key != vlm_key:
            print(f"加载Qwen-VL描述模型: {self.vlm_model_path} (模式: {self.match_mode})")
            VLMTextMatcher._shared_vlm_describer = VLMDescriber(
                model_name=self.vlm_model_path,
                device=str(self.device),
                template=template_prompt
            )
            VLMTextMatcher._shared_vlm_key = vlm_key
        else:
            print(f"复用Qwen-VL描述模型: {self.vlm_model_path}")
        self.vlm_describer = VLMTextMatcher._shared_vlm_describer

        embedder_key = (self.embedding_model_path, str(self.embedding_device))
        if VLMTextMatcher._shared_text_embedder is None or VLMTextMatcher._shared_embedder_key != embedder_key:
            print(f"加载BGE文本向量模型: {self.embedding_model_path}")
            VLMTextMatcher._shared_text_embedder = TextEmbedder(
                model_name=self.embedding_model_path,
                device=str(self.embedding_device)
            )
            VLMTextMatcher._shared_embedder_key = embedder_key
        else:
            print(f"复用BGE文本向量模型: {self.embedding_model_path}")
        self.text_embedder = VLMTextMatcher._shared_text_embedder

        self.rule_embeddings = {}
        self.negative_embeddings = {}
        self.rule_positive_texts = {}
        self.rule_negative_texts = {}
        # 结构化模式下，规则文本若全部由预定义属性名组成，则直接做符号匹配
        self.rule_attribute_positives = {}  # rule_id -> set(属性名) 或 None(需回退语义匹配)
        self.rule_attribute_negatives = {}  # rule_id -> set(属性名)
        self.rules = []
        self.last_trigger_time = {}
        self.description_prompt = (
            self.structured_prompt if self.match_mode == "structured" else self.base_prompt
        )

    def _normalize_texts(self, texts) -> List[str]:
        if isinstance(texts, str):
            texts = [texts]
        return [t.strip() for t in (texts or []) if isinstance(t, str) and t.strip()]

    def _lookup_attribute(self, token: str):
        """把单个token映射为预定义属性名（大小写/分隔符不敏感），未命中返回None"""
        cleaned = token.strip().strip(".。:：`*\"'")
        if not cleaned:
            return None
        return (
            _ATTR_LOOKUP.get(cleaned.lower())
            or _ATTR_LOOKUP.get(re.sub(r"[-_ ]", "", cleaned).lower())
        )

    def _extract_valid_attributes(self, text: str) -> List[str]:
        """从文本中解析预定义属性名，按出现顺序去重返回（忽略未知token）"""
        result = []
        seen = set()
        for token in re.split(r"[,，、;；/|。.！!？?\s]+", str(text or "")):
            attr = self._lookup_attribute(token)
            if attr and attr not in seen:
                seen.add(attr)
                result.append(attr)
        return result

    def _parse_rule_attributes(self, text: str):
        """规则文本若完全由预定义属性名组成，返回属性集合；否则返回None（回退语义匹配）"""
        attrs = set()
        found = False
        for chunk in re.split(r"[,，、;；/|]+", str(text or "")):
            for sub in chunk.split():
                cleaned = sub.strip().strip(".。:：`*\"'")
                if not cleaned:
                    continue
                attr = self._lookup_attribute(cleaned)
                if attr is None:
                    return None
                attrs.add(attr)
                found = True
        return attrs if found else None

    def extract_structured_attributes(self, image: Image.Image) -> List[str]:
        """用结构化提示词让VLM输出属性名列表，校验后返回合法属性；
        Female 未出现时自动派生 Male（互斥属性，模型只输出存在项，缺失即另一种性别）"""
        try:
            raw = self.vlm_describer.describe(
                image,
                prompt=self.structured_prompt,
                max_new_tokens=self.structured_max_new_tokens,
                temperature=self.vlm_temperature,
                repetition_penalty=self.vlm_repetition_penalty
            )
        except Exception as e:
            print(f"VLM结构化属性提取失败: {e}")
            return []
        attributes = self._extract_valid_attributes(raw)
        if "Female" not in attributes:
            attributes.append("Male")
        return attributes

    def _embed_texts(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, 384), dtype=np.float32)
        return self.text_embedder.encode(texts, normalize=True)

    def _sanitize_rule_text(self, text: str, max_len: int = 36) -> str:
        text = " ".join(str(text).replace("\n", " ").split())
        return text[:max_len]

    def _split_bilingual(self, text: str):
        """拆分 '英文描述 ... 中文: 中文翻译' 格式，返回 (英文, 中文)；无中文标记时中文为 ''"""
        text = str(text or "").strip()
        # 模型可能输出「中 文：」「中文描述:」等变体，允许字间空白
        m = re.search(r"中\s*文\s*(?:描\s*述)?\s*[：:]", text)
        if m:
            zh = text[m.end():].strip(" ：:，,。")
            en = text[:m.start()].strip(" ，。；;")
            return en, zh
        return text, ""

    def _clean_vlm_description(self, text: str, max_chars: int = 160) -> str:
        text = " ".join(str(text or "").split())
        for marker in ("请注意", "以上信息", "免责声明", "如果存在", "请理解", "在提供此类内容时", "需要注意", "最后需要注意", "此外，在此情况下"):
            if marker in text:
                text = text.split(marker, 1)[0].strip()
        text = text.strip(" ，。；;")
        if len(text) <= max_chars:
            return text
        clipped = text[:max_chars]
        for sep in ("。", "；", "，", ","):
            pos = clipped.rfind(sep)
            if pos >= max_chars // 2:
                return clipped[:pos].strip(" ，。；;")
        return clipped.strip(" ，。；;")

    def _is_no_match_description(self, text: str) -> bool:
        normalized = "".join(str(text or "").split())
        no_match_markers = (
            "不满足", "不符合", "未满足", "无法判断", "看不清", "不能确定", "无法确定",
            "不确定", "不存在", "没有符合", "无符合", "未看到", "没有看到", "没有穿",
            "并未", "不是", "可能符合"
        )
        return normalized.startswith(no_match_markers) or any(marker in normalized for marker in no_match_markers)

    def _extract_verified_rule_description(self, text: str) -> str:
        text = self._clean_vlm_description(text, max_chars=120)
        normalized = "".join(text.split())
        if self._is_no_match_description(text):
            return ""
        if normalized.startswith(("满足：", "满足:")):
            separator = "：" if "：" in text else ":"
            return text.split(separator, 1)[1].strip(" ，。；;")
        if normalized.startswith("满足"):
            return text.lstrip("满足:： ，。；;").strip()
        return text

    def _build_rule_description_prompt(
        self,
        rule: Dict,
        positive_texts: List[str],
        negative_texts: List[str]
    ) -> str:
        name = self._sanitize_rule_text(rule.get("name", "当前规则"), max_len=32)
        positives = [
            self._sanitize_rule_text(t, max_len=48)
            for t in positive_texts[:self.rule_prompt_max_texts]
        ]
        negatives = [
            self._sanitize_rule_text(t, max_len=48)
            for t in negative_texts[:self.rule_prompt_max_texts]
        ]

        prompt = (
            "请判断图像中的这个行人是否满足下面这条规则。"
            "输出格式只能二选一：\n"
            "满足：<只写一句与这条规则相关的行人描述>\n"
            "不满足\n"
            f"规则名称：{name}\n"
        )
        if positives:
            prompt += f"目标特征：{'、'.join(positives)}\n"
        if negatives:
            prompt += f"排除/区分特征：{'、'.join(negatives)}\n"
        prompt += (
            "输出要求：只写一句中文事实描述，重点说明与该规则相关的可见特征；"
            "可以包含性别估计、衣着颜色款式、头盔/帽子、携带物、动作或交通工具等与规则有关的信息；"
            "不要输出无关外观，不要照抄规则原文，不要输出匹配分数，不要输出“符合规则/不符合规则”等结论；"
            "不要因为规则名称或目标特征而编造图像中不存在的内容；"
            "不要输出请注意、免责声明、推理过程或额外解释。"
        )
        return prompt

    def update_rules(self, rules: List[Dict]):
        """更新规则并预计算正/负样本文本向量；结构化模式同时预解析属性规则"""
        self.rules = [r for r in rules if r.get('enabled', True)]
        self.rule_embeddings = {}
        self.negative_embeddings = {}
        self.rule_positive_texts = {}
        self.rule_negative_texts = {}
        self.rule_attribute_positives = {}
        self.rule_attribute_negatives = {}

        for rule in self.rules:
            rule_id = rule['id']
            positive_texts = self._normalize_texts(rule.get('positive') or [rule.get('text', '')])
            negative_texts = self._normalize_texts(rule.get('negative', []))

            self.rule_positive_texts[rule_id] = positive_texts
            self.rule_negative_texts[rule_id] = negative_texts
            if self.match_mode == "rule_direct":
                # 规则直判模式由VLM直接判断，无需向量匹配，跳过向量计算
                continue
            self.rule_embeddings[rule_id] = self._embed_texts(positive_texts)
            self.negative_embeddings[rule_id] = self._embed_texts(negative_texts)

            if self.match_mode in ("structured", "combined"):
                # 所有正样本文本都能解析成属性名时，该规则走符号匹配
                # （combined 模式：属性符号匹配 + 描述向量匹配同时可用）
                parsed_pos = [self._parse_rule_attributes(t) for t in positive_texts]
                if positive_texts and all(p is not None for p in parsed_pos):
                    self.rule_attribute_positives[rule_id] = set().union(*parsed_pos)
                else:
                    self.rule_attribute_positives[rule_id] = None

                neg_attrs = set()
                for t in negative_texts:
                    parsed_neg = self._parse_rule_attributes(t)
                    if parsed_neg:
                        neg_attrs |= parsed_neg
                self.rule_attribute_negatives[rule_id] = neg_attrs

        self.description_prompt = (
            self.structured_prompt if self.match_mode == "structured" else self.base_prompt
        )

    def describe_image(self, image: Image.Image) -> str:
        """按当前模式生成行人描述：structured返回属性串，description返回自由文本，
        combined两次分别生成结构化属性与自由描述（匹配用描述，属性用于符号匹配和展示）"""
        try:
            self.last_description_zh = ""
            if self.match_mode == "structured":
                attributes = self.extract_structured_attributes(image)
                return ",".join(attributes)
            if self.match_mode == "combined":
                # 先结构化属性（供符号匹配 + 展示），再自由描述（供语义匹配）
                self.last_attributes = self.extract_structured_attributes(image)
            description = self.vlm_describer.describe(
                image,
                prompt=self.description_prompt,
                max_new_tokens=self.vlm_max_new_tokens,
                temperature=self.vlm_temperature,
                repetition_penalty=self.vlm_repetition_penalty
            )
            description, zh = self._split_bilingual(description)
            self.last_description_zh = " ".join(zh.split()) if zh else ""
            return self._clean_vlm_description(description, max_chars=1200)
        except Exception as e:
            print(f"VLM描述生成失败: {e}")
            return ""

    def describe_rule_match(
        self,
        image: Image.Image,
        rule: Dict,
        positive_texts: List[str],
        negative_texts: List[str],
        fallback: str
    ) -> str:
        """命中规则后，生成该规则相关的行人描述"""
        if not self.rule_description_enabled:
            return fallback

        try:
            prompt = self._build_rule_description_prompt(rule, positive_texts, negative_texts)
            description = self.vlm_describer.describe(
                image,
                prompt=prompt,
                max_new_tokens=self.rule_description_max_new_tokens,
                temperature=self.vlm_temperature,
                repetition_penalty=self.vlm_repetition_penalty
            )
            description = self._extract_verified_rule_description(description)
            if self.rule_description_verify and not description:
                return ""
            return description or fallback
        except Exception as e:
            print(f"规则相关VLM描述生成失败: {e}")
            return fallback

    def _build_rule_judge_prompt(self, rules: List[Dict]) -> str:
        """rule_direct模式：把规则列表填入提示词的{rules}占位处"""
        lines = []
        for i, rule in enumerate(rules, 1):
            name = self._sanitize_rule_text(rule.get('name', ''), max_len=32)
            positives = [
                self._sanitize_rule_text(t, max_len=48)
                for t in self.rule_positive_texts.get(rule['id'], [])[:self.rule_prompt_max_texts]
            ]
            line = f"{i}. {name}"
            if positives:
                line += f"（目标特征：{'、'.join(positives)}）"
            lines.append(line)
        return self.rule_judge_prompt.replace("{rules}", "\n".join(lines))

    def _parse_matched_rule_names(self, output: str, candidates: List[Dict]) -> List[Dict]:
        """从模型输出中回认命中的规则名；输出「不匹配」或未命中任何规则名时返回空"""
        remaining = "".join(str(output or "").split())
        if not remaining:
            return []
        matched = []
        # 长名优先匹配，避免短规则名是长规则名子串时被重复命中
        for rule in sorted(candidates, key=lambda r: -len("".join(str(r.get('name', '')).split()))):
            name = "".join(self._sanitize_rule_text(rule.get('name', ''), max_len=32).split())
            if name and name in remaining:
                remaining = remaining.replace(name, "", 1)
                matched.append(rule)
        matched.sort(key=lambda r: candidates.index(r))
        return matched

    def match_rule_direct(self, image: Image.Image) -> List[Dict]:
        """rule_direct模式：规则代入提示词，VLM判断图像是否含该规则，
        命中只输出规则名，未命中输出「不匹配」"""
        current_time = time.time()
        eligible = []
        for rule in self.rules:
            rule_id = rule['id']
            interval = rule.get('interval', 60)
            if rule_id in self.last_trigger_time:
                if current_time - self.last_trigger_time[rule_id] < interval:
                    continue
            eligible.append(rule)
        if not eligible:
            return []

        prompt = self._build_rule_judge_prompt(eligible)
        try:
            output = self.vlm_describer.describe(
                image,
                prompt=prompt,
                max_new_tokens=self.rule_judge_max_new_tokens,
                temperature=self.vlm_temperature,
                repetition_penalty=self.vlm_repetition_penalty
            )
        except Exception as e:
            print(f"规则直判VLM调用失败: {e}")
            return []

        judge_output = " ".join(str(output or "").split())
        matched = self._parse_matched_rule_names(output, eligible)
        if not matched:
            if judge_output:
                print(f"规则直判未命中（模型输出: {judge_output}）")
            return []

        matches = []
        for rule in matched:
            self.last_trigger_time[rule['id']] = current_time
            positive_list = self.rule_positive_texts.get(rule['id'], [])
            negative_list = self.rule_negative_texts.get(rule['id'], [])
            rule_text = str(rule.get('text', '') or '').strip()
            matches.append({
                'rule_id': rule['id'],
                'rule_name': rule['name'],
                'positive': positive_list[0] if positive_list else rule_text or rule['name'],
                'positive_list': positive_list,
                'negative_list': negative_list,
                'positive_scores': [],
                'negative_scores': [],
                'final_score': 1.0,
                'negative_higher': False,
                'description': rule_text or (positive_list[0] if positive_list else rule['name']),
                'base_description': judge_output,
                'description_zh': '',
                'attributes': [],
            })
        return matches

    def match(self, image: Image.Image) -> List[Dict]:
        """结构化模式对属性规则做符号匹配，其余用BGE比较VLM图像描述与规则文本的相似度；
        rule_direct模式把规则代入提示词由VLM直接判断"""
        if not self.rules:
            return []
        if self.match_mode == "rule_direct":
            return self.match_rule_direct(image)
        if not self.rule_embeddings:
            return []

        description = self.describe_image(image)
        if not description:
            print("VLM描述为空，跳过本次匹配")
            return []

        detected_attributes = set()
        if self.match_mode == "structured":
            detected_attributes = {a for a in description.split(",") if a}
        elif self.match_mode == "combined":
            detected_attributes = set(getattr(self, 'last_attributes', []))

        # 结构化/混合模式下若所有规则都能符号匹配，则无需计算图像描述向量
        if self.match_mode in ("structured", "combined"):
            needs_embedding = any(
                self.rule_attribute_positives.get(rule['id']) is None for rule in self.rules
            )
        else:
            needs_embedding = True
        image_embedding = (
            self.text_embedder.encode([description], normalize=True)[0]
            if needs_embedding else None
        )
        current_time = time.time()
        matches = []

        for rule in self.rules:
            rule_id = rule['id']

            interval = rule.get('interval', 60)
            if rule_id in self.last_trigger_time:
                if current_time - self.last_trigger_time[rule_id] < interval:
                    continue

            symbolic_positive = self.rule_attribute_positives.get(rule_id)
            if symbolic_positive:
                # 符号匹配：得分为正样本属性覆盖率；任一负样本属性命中记为1.0
                pos_attrs = sorted(symbolic_positive)
                present = [a for a in pos_attrs if a in detected_attributes]
                positive_scores = [1.0 if a in detected_attributes else 0.0 for a in pos_attrs]
                best_positive_score = len(present) / len(pos_attrs)
                best_positive_text = ",".join(present) if present else pos_attrs[0]
                negative_scores = []
                max_negative_score = (
                    1.0 if self.rule_attribute_negatives.get(rule_id, set()) & detected_attributes
                    else 0.0
                )
            else:
                positive_embeddings = self.rule_embeddings.get(rule_id)
                if positive_embeddings is None or len(positive_embeddings) == 0:
                    continue

                positive_scores_raw = positive_embeddings @ image_embedding
                positive_scores = [round(float(s), 4) for s in positive_scores_raw]
                best_idx = int(np.argmax(positive_scores_raw))
                best_positive_score = float(positive_scores_raw[best_idx])
                positive_texts = self.rule_positive_texts.get(rule_id, [])
                best_positive_text = positive_texts[best_idx] if best_idx < len(positive_texts) else ''

                negative_embeddings = self.negative_embeddings.get(rule_id)
                negative_scores = []
                max_negative_score = 0.0
                if negative_embeddings is not None and len(negative_embeddings) > 0:
                    negative_scores_raw = negative_embeddings @ image_embedding
                    negative_scores = [round(float(s), 4) for s in negative_scores_raw]
                    max_negative_score = float(np.max(negative_scores_raw))

            score_diff = rule.get('score_diff', 0)
            threshold = float(rule.get('threshold', 0.5))
            if best_positive_score >= threshold and best_positive_score > max_negative_score + score_diff:
                self.last_trigger_time[rule_id] = current_time

                positive_list = self.rule_positive_texts.get(rule_id, [])
                negative_list = self.rule_negative_texts.get(rule_id, [])
                negative_higher = (max_negative_score >= threshold and max_negative_score >= best_positive_score)
                rule_description = self.describe_rule_match(
                    image,
                    rule,
                    positive_list,
                    negative_list,
                    description
                )
                if self.rule_description_enabled and self.rule_description_verify and not rule_description:
                    continue

                matches.append({
                    'rule_id': rule_id,
                    'rule_name': rule['name'],
                    'positive': best_positive_text,
                    'positive_list': positive_list,
                    'negative_list': negative_list,
                    'positive_scores': positive_scores,
                    'negative_scores': negative_scores,
                    'final_score': round(best_positive_score, 4),
                    'negative_higher': negative_higher,
                    'description': rule_description,
                    'base_description': description,
                    'description_zh': getattr(self, 'last_description_zh', ''),
                    'attributes': (
                        description.split(",")
                        if self.match_mode == "structured"
                        else list(getattr(self, 'last_attributes', []))
                    ) if detected_attributes else []
                })

        return matches


class StreamProcessor:
    """流处理器"""

    def __init__(self, device: Dict, detector: PersonDetector, matcher: VLMTextMatcher):
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
                'description': match.get('description', ''),
                'base_description': match.get('base_description', ''),
                'description_zh': match.get('description_zh', ''),
                'attributes': match.get('attributes', []),
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
                'description': match.get('description', ''),
                'base_description': match.get('base_description', ''),
                'description_zh': match.get('description_zh', ''),
                'attributes': match.get('attributes', []),
                'score': match.get('final_score', 0),
                'negative_higher': match.get('negative_higher', False),
                'timestamp': timestamp,
                'image': f"{today}/{safe_device_name}/{safe_rule_name}/roi/{img_filename}",
                'original_image': f"{today}/{safe_device_name}/{safe_rule_name}/original/{img_filename}",
                'bbox': bbox,
                'low_quality': is_low_quality,
                'img_size': {'width': w, 'height': h}
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
        matcher = VLMTextMatcher()
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
