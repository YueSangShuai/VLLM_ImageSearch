"""
将FG-CLIP模型拆分为两个独立模型:
1. ImageModel: 图像向量化 + 分类输出
2. TextModel: 文本向量化

分类输出可以是多个分类头的拼接向量
"""
import os
import torch
import torch.nn as nn
import json
from pathlib import Path
from safetensors.torch import save_file, load_file

# 添加项目路径
import sys
sys.path.insert(0, str(Path(__file__).parent))

from fgclip.model.clip_strc.configuration_clip import CLIPConfig
from fgclip.model.clip_strc.fgclip import FGCLIPModel, ClassificationHead


class ImageEncoderWithClassifier(nn.Module):
    """图像编码器 + 分类头"""
    def __init__(self, vision_model, visual_projection, classification_heads, projection_dim):
        super().__init__()
        self.vision_model = vision_model
        self.visual_projection = visual_projection
        self.classification_heads = classification_heads
        self.projection_dim = projection_dim
        self.classification_attr_names = list(classification_heads.keys()) if classification_heads else []

    def forward(self, pixel_values, return_classification=True):
        """
        Args:
            pixel_values: 输入图像
            return_classification: 是否返回分类结果
        Returns:
            image_features: 向量化结果 (normalized)
            classification_output: 分类结果拼接向量 (如果return_classification=True)
        """
        # 获取vision输出
        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )

        pooled_output = vision_outputs[1]
        image_features = self.visual_projection(pooled_output)
        # L2归一化
        image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)

        result = {"image_features": image_features}

        if return_classification and self.classification_heads:
            # 获取每个分类头的输出并拼接
            cls_outputs = []
            cls_details = {}
            for attr_name in self.classification_attr_names:
                logits = self.classification_heads[attr_name](image_features)
                cls_outputs.append(logits)
                cls_details[attr_name] = logits

            # 拼接所有分类头输出
            classification_output = torch.cat(cls_outputs, dim=-1)
            result["classification_output"] = classification_output
            result["classification_details"] = cls_details

        return result

    def get_embedding_only(self, pixel_values):
        """仅获取向量化结果"""
        result = self.forward(pixel_values, return_classification=False)
        return result["image_features"]

    def get_classification_only(self, pixel_values):
        """仅获取分类结果"""
        result = self.forward(pixel_values, return_classification=True)
        return result["classification_output"], result["classification_details"]


class TextEncoder(nn.Module):
    """文本编码器"""
    def __init__(self, text_model, text_projection, text_filip_projection, projection_dim):
        super().__init__()
        self.text_model = text_model
        self.text_projection = text_projection
        self.text_filip_projection = text_filip_projection
        self.projection_dim = projection_dim

    def forward(self, input_ids, attention_mask=None, position_ids=None, use_filip=False):
        """
        Args:
            input_ids: 文本token ids
            attention_mask: attention mask
            position_ids: 位置ids
            use_filip: 是否使用filip projection (用于长文本)
        Returns:
            text_features: 向量化结果 (normalized)
        """
        # walk_short_pos=True uses simple position_embedding
        # walk_short_pos=False uses extended position_embedding_ori/res with mask
        pos_flag = not use_filip

        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
            walk_short_pos=pos_flag,
        )

        pooled_output = text_outputs[1]

        if use_filip:
            text_features = self.text_filip_projection(pooled_output)
        else:
            text_features = self.text_projection(pooled_output)

        # 不做归一化 (与原始get_text_features保持一致)
        # 如需归一化可使用: text_features / text_features.norm(p=2, dim=-1, keepdim=True)

        return text_features


def load_fgclip_model(checkpoint_path, device="cpu"):
    """加载FG-CLIP模型"""
    config_path = Path(checkpoint_path) / "config.json"
    model_path = Path(checkpoint_path) / "model.safetensors"

    # 加载配置
    with open(config_path, "r") as f:
        config_dict = json.load(f)

    # 加载模型权重以检测分类头
    state_dict = load_file(model_path)

    # 从权重中检测分类头配置
    classification_heads_in_ckpt = {}
    for key in state_dict.keys():
        if key.startswith("classification_heads."):
            parts = key.split(".")
            attr_name = parts[1]
            param_name = ".".join(parts[2:])
            if "." not in param_name:  # 只检查weight来确定类别数
                continue
            # 获取输出层的权重来确定类别数
            if "net.3.weight" in key:  # 分类头的最后一层
                out_features = state_dict[key].shape[0]
                if attr_name not in classification_heads_in_ckpt:
                    classification_heads_in_ckpt[attr_name] = out_features

    print(f"Found {len(classification_heads_in_ckpt)} classification heads in checkpoint:")
    for name, nc in classification_heads_in_ckpt.items():
        print(f"  - {name}: {nc} classes")

    # 构建CLIPConfig
    config = CLIPConfig.from_dict(config_dict)

    # 如果有分类头，创建classifier_config
    classifier_config = None
    if classification_heads_in_ckpt:
        classifier_config = {
            "open": True,
            "classes": {name: {"nc": nc} for name, nc in classification_heads_in_ckpt.items()}
        }

    # 创建模型
    model = FGCLIPModel(config, classifier_config=classifier_config)

    # 加载权重
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

    if missing_keys:
        print(f"Missing keys: {missing_keys}")
    if unexpected_keys:
        pass  # 忽略分类头的unexpected keys

    model = model.to(device)
    model.eval()

    return model, config, config_dict


def split_and_save_models(
    checkpoint_path: str,
    output_dir: str,
    classifier_classes: dict = None,
    device: str = "cpu"
):
    """
    将FG-CLIP模型拆分为图像模型和文本模型

    Args:
        checkpoint_path: FG-CLIP checkpoint路径
        output_dir: 输出目录
        classifier_classes: 分类类别配置，如 {"gender": 2, "age": 10}
        device: 设备
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Loading FG-CLIP model from {checkpoint_path}...")
    model, config, config_dict = load_fgclip_model(checkpoint_path, device)

    # 创建图像模型
    print("Creating Image Model...")
    classification_heads = nn.ModuleDict()
    if classifier_classes:
        for attr_name, num_classes in classifier_classes.items():
            classification_heads[attr_name] = ClassificationHead(
                config.projection_dim, num_classes, dropout=0.0
            )
            # 尝试从原模型复制权重
            if attr_name in model.classification_heads:
                classification_heads[attr_name].load_state_dict(
                    model.classification_heads[attr_name].state_dict()
                )
                print(f"  Loaded classification head: {attr_name} ({num_classes} classes)")
    elif model.use_classification:
        classification_heads = model.classification_heads
        print(f"  Loaded {len(classification_heads)} classification heads from model")

    image_model = ImageEncoderWithClassifier(
        vision_model=model.vision_model,
        visual_projection=model.visual_projection,
        classification_heads=classification_heads,
        projection_dim=config.projection_dim
    )

    # 创建文本模型
    print("Creating Text Model...")
    text_model = TextEncoder(
        text_model=model.text_model,
        text_projection=model.text_projection,
        text_filip_projection=model.text_filip_projection,
        projection_dim=config.projection_dim
    )

    # 保存图像模型
    print(f"Saving Image Model to {output_path / 'image_model'}...")
    image_output_dir = output_path / "image_model"
    image_output_dir.mkdir(exist_ok=True)

    # 保存图像模型的state_dict
    image_state_dict = {
        "vision_model." + k: v for k, v in image_model.vision_model.state_dict().items()
    }
    image_state_dict["visual_projection.weight"] = image_model.visual_projection.weight.data

    for name, head in image_model.classification_heads.items():
        for k, v in head.state_dict().items():
            image_state_dict[f"classification_heads.{name}.{k}"] = v

    # 保存图像模型配置
    image_config = {
        "model_type": "fgclip_image",
        "vision_config": config_dict.get("vision_config", {}),
        "projection_dim": config.projection_dim,
        "logit_scale_init_value": config.logit_scale_init_value,
        "classification_classes": classifier_classes or (
            {k: v.net[-1].out_features for k, v in model.classification_heads.items()}
            if model.use_classification else {}
        )
    }
    with open(image_output_dir / "config.json", "w") as f:
        json.dump(image_config, f, indent=2)

    # 保存图像模型权重
    save_file(image_state_dict, image_output_dir / "model.safetensors")

    # 保存文本模型
    print(f"Saving Text Model to {output_path / 'text_model'}...")
    text_output_dir = output_path / "text_model"
    text_output_dir.mkdir(exist_ok=True)

    # 保存文本模型的state_dict
    text_state_dict = {
        "text_model." + k: v for k, v in text_model.text_model.state_dict().items()
    }
    text_state_dict["text_projection.weight"] = text_model.text_projection.weight.data
    text_state_dict["text_filip_projection.weight"] = text_model.text_filip_projection.weight.data

    # 保存文本模型配置
    text_config = {
        "model_type": "fgclip_text",
        "text_config": config_dict.get("text_config", {}),
        "projection_dim": config.projection_dim,
        "logit_scale_init_value": config.logit_scale_init_value,
    }
    with open(text_output_dir / "config.json", "w") as f:
        json.dump(text_config, f, indent=2)

    # 保存文本模型权重
    save_file(text_state_dict, text_output_dir / "model.safetensors")

    print("\nDone! Models saved to:")
    print(f"  - Image Model: {image_output_dir}")
    print(f"  - Text Model: {text_output_dir}")

    return image_model, text_model


def load_image_model(model_dir, device="cpu"):
    """加载图像模型"""
    from fgclip.model.clip_strc.modeling_clip import CLIPVisionTransformer

    config_path = Path(model_dir) / "config.json"
    model_path = Path(model_dir) / "model.safetensors"

    with open(config_path, "r") as f:
        config = json.load(f)

    # 创建vision模型
    from fgclip.model.clip_strc.configuration_clip import CLIPVisionConfig
    vision_config = CLIPVisionConfig(**config["vision_config"])

    vision_model = CLIPVisionTransformer(vision_config)
    # visual_projection: vision_embed_dim -> projection_dim
    vision_embed_dim = vision_config.hidden_size
    visual_projection = nn.Linear(vision_embed_dim, config["projection_dim"], bias=False)

    # 创建分类头
    classification_heads = nn.ModuleDict()
    cls_config = config.get("classification_classes", {})
    for attr_name, num_classes in cls_config.items():
        classification_heads[attr_name] = ClassificationHead(
            config["projection_dim"], num_classes, dropout=0.0
        )

    # 加载权重
    state_dict = load_file(model_path)

    # 分离并加载各部分权重
    vision_state_dict = {}
    proj_weight = None
    cls_state_dict = {}

    for k, v in state_dict.items():
        if k.startswith("vision_model."):
            vision_state_dict[k.replace("vision_model.", "")] = v
        elif k == "visual_projection.weight":
            proj_weight = v
        elif k.startswith("classification_heads."):
            parts = k.split(".")
            attr_name = parts[1]
            param_name = ".".join(parts[2:])
            if attr_name not in cls_state_dict:
                cls_state_dict[attr_name] = {}
            cls_state_dict[attr_name][param_name] = v

    vision_model.load_state_dict(vision_state_dict)
    visual_projection.weight.data = proj_weight

    for attr_name, params in cls_state_dict.items():
        classification_heads[attr_name].load_state_dict(params)

    model = ImageEncoderWithClassifier(
        vision_model=vision_model,
        visual_projection=visual_projection,
        classification_heads=classification_heads,
        projection_dim=config["projection_dim"]
    )

    model = model.to(device)
    model.eval()
    return model, config


def load_text_model(model_dir, device="cpu"):
    """加载文本模型"""
    from fgclip.model.clip_strc.modeling_clip import CLIPTextTransformer

    config_path = Path(model_dir) / "config.json"
    model_path = Path(model_dir) / "model.safetensors"

    with open(config_path, "r") as f:
        config = json.load(f)

    # 创建text模型
    from fgclip.model.clip_strc.configuration_clip import CLIPTextConfig
    text_config = CLIPTextConfig(**config["text_config"])

    text_model = CLIPTextTransformer(text_config)
    text_projection = nn.Linear(config["projection_dim"], config["projection_dim"], bias=False)
    text_filip_projection = nn.Linear(config["projection_dim"], config["projection_dim"], bias=False)

    # 加载权重
    state_dict = load_file(model_path)

    # 分离并加载各部分权重
    text_state_dict = {}
    proj_weight = None
    filip_weight = None

    for k, v in state_dict.items():
        if k.startswith("text_model."):
            text_state_dict[k.replace("text_model.", "")] = v
        elif k == "text_projection.weight":
            proj_weight = v
        elif k == "text_filip_projection.weight":
            filip_weight = v

    text_model.load_state_dict(text_state_dict)
    text_projection.weight.data = proj_weight
    text_filip_projection.weight.data = filip_weight

    model = TextEncoder(
        text_model=text_model,
        text_projection=text_projection,
        text_filip_projection=text_filip_projection,
        projection_dim=config["projection_dim"]
    )

    model = model.to(device)
    model.eval()
    return model, config


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Split FG-CLIP into Image and Text models")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to FG-CLIP checkpoint")
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu/cuda)")
    parser.add_argument(
        "--classifier-classes",
        type=str,
        default=None,
        help='Classification classes as JSON string, e.g., \'{"gender": 2, "age": 10}\''
    )

    args = parser.parse_args()

    classifier_classes = None
    if args.classifier_classes:
        classifier_classes = json.loads(args.classifier_classes)

    split_and_save_models(
        checkpoint_path=args.checkpoint,
        output_dir=args.output,
        classifier_classes=classifier_classes,
        device=args.device
    )

    print("\n\n--- Usage Example ---")
    print("""
# 加载使用示例
from split_fgclip_model import load_image_model, load_text_model
import torch

# 加载图像模型
image_model, image_config = load_image_model("output/image_model", device="cuda")
text_model, text_config = load_text_model("output/text_model", device="cuda")

# 图像向量化 + 分类
image_input = torch.randn(1, 3, 224, 224).to("cuda")
result = image_model(image_input)
embedding = result["image_features"]  # [1, projection_dim]
classification = result["classification_output"]  # 拼接的分类向量

# 文本向量化
text_input = torch.randint(0, 1000, (1, 20)).to("cuda")
text_embedding = text_model(text_input)  # [1, projection_dim]
""")