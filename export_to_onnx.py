"""
将拆分的FG-CLIP Image Model和Text Model导出为ONNX格式
"""
import os
import torch
import numpy as np
import json
from pathlib import Path
import onnx
import onnxruntime as ort
from split_fgclip_model import load_image_model, load_text_model


def export_image_model_to_onnx(model_dir, output_path, device="cpu"):
    """
    导出图像模型到ONNX

    输出两个输出节点:
    - image_features: 归一化图像向量 [batch, 512]
    - classification_output: 拼接分类向量 [batch, 108] (54个2分类)
    """
    print(f"Exporting Image Model to ONNX...")

    # 加载模型
    model, config = load_image_model(model_dir, device=device)
    model.eval()

    # 创建输出目录
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # 创建包装模型，合并两个输出
    class ImageModelWithClassification(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, pixel_values):
            result = self.model(pixel_values, return_classification=True)
            return result["image_features"], result["classification_output"]

    wrapper = ImageModelWithClassification(model)

    # 虚拟输入
    batch_size = 1
    dummy_input = torch.randn(batch_size, 3, 224, 224)

    # 导出ONNX
    torch.onnx.export(
        wrapper,
        dummy_input,
        output_path,
        input_names=["pixel_values"],
        output_names=["image_features", "classification_output"],
        dynamic_axes={
            "pixel_values": {0: "batch_size"},
            "image_features": {0: "batch_size"},
            "classification_output": {0: "batch_size"},
        },
        opset_version=14,
        do_constant_folding=True,
    )

    # 保存配置
    config_path = output_path.replace(".onnx", "_config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"Image Model exported to: {output_path}")
    print(f"Config saved to: {config_path}")

    return output_path


def export_text_model_to_onnx(model_dir, output_path, max_seq_len=77, device="cpu"):
    """
    导出文本模型到ONNX

    输出:
    - text_features: 文本向量 [batch, 512]
    - pooled_output: 池化后的向量 [batch, 512]
    """
    print(f"Exporting Text Model to ONNX...")
    print(f"Max sequence length: {max_seq_len}")

    # 加载模型
    model, config = load_text_model(model_dir, device=device)
    model.eval()

    # 创建包装模型
    class TextModelWithPooling(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, input_ids):
            # 使用简单的wrap_short_pos=True
            text_outputs = self.model.text_model(
                input_ids=input_ids,
                attention_mask=None,
                position_ids=None,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=True,
                walk_short_pos=True,
            )

            pooled_output = text_outputs[1]
            text_features = self.model.text_projection(pooled_output)

            return text_features, pooled_output

    wrapper = TextModelWithPooling(model)

    # 虚拟输入 - 使用max_seq_len
    batch_size = 1
    dummy_input_ids = torch.randint(0, 1000, (batch_size, max_seq_len))

    # 导出ONNX
    torch.onnx.export(
        wrapper,
        dummy_input_ids,
        output_path,
        input_names=["input_ids"],
        output_names=["text_features", "pooled_output"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "seq_len"},
            "text_features": {0: "batch_size"},
            "pooled_output": {0: "batch_size"},
        },
        opset_version=14,
        do_constant_folding=True,
    )

    # 保存配置
    config_path = output_path.replace(".onnx", "_config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"Text Model exported to: {output_path}")
    print(f"Config saved to: {config_path}")

    return output_path


def verify_onnx_model(onnx_path, torch_model_path, model_type="image", num_tests=5):
    """
    验证ONNX模型与PyTorch模型的精度一致性
    """
    print(f"\n{'='*50}")
    print(f"Verifying ONNX Model: {onnx_path}")
    print(f"Model type: {model_type}")
    print(f"{'='*50}")

    # 创建ONNX Runtime session
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(onnx_path, sess_options)

    # 加载PyTorch模型
    if model_type == "image":
        torch_model, _ = load_image_model(torch_model_path, device="cpu")
    else:
        torch_model, _ = load_text_model(torch_model_path, device="cpu")
    torch_model.eval()

    max_diffs = []

    for i in range(num_tests):
        torch.manual_seed(42 + i)

        if model_type == "image":
            # 图像输入
            pixel_values = torch.randn(2, 3, 224, 224)

            with torch.no_grad():
                if model_type == "image":
                    result = torch_model(pixel_values, return_classification=True)
                    torch_out1 = result["image_features"]
                    torch_out2 = result["classification_output"]

            onnx_out = session.run(
                None,
                {"pixel_values": pixel_values.numpy()}
            )
            onnx_out1, onnx_out2 = onnx_out

        else:
            # 文本输入 - 限制在77以内
            input_ids = torch.randint(0, 1000, (2, 77))

            with torch.no_grad():
                torch_out1, torch_out2 = torch_model(input_ids)

            onnx_out = session.run(
                None,
                {"input_ids": input_ids.numpy()}
            )
            onnx_out1, onnx_out2 = onnx_out

        # 计算差异
        diff1 = np.abs(torch_out1.numpy() - onnx_out1).max()
        diff2 = np.abs(torch_out2.numpy() - onnx_out2).max()
        max_diff = max(diff1, diff2)
        max_diffs.append(max_diff)

        print(f"Test {i+1}: Max diff = {max_diff:.2e}")

    avg_diff = np.mean(max_diffs)
    max_diff = np.max(max_diffs)

    print(f"\nAverage max diff: {avg_diff:.2e}")
    print(f"Overall max diff: {max_diff:.2e}")
    print(f"Precision match: {max_diff < 1e-4}")

    return max_diff < 1e-4


def verify_onnx_image_model(onnx_path, torch_model_path, num_tests=5):
    """专门验证图像ONNX模型"""
    print(f"\n{'='*50}")
    print(f"Verifying Image ONNX Model")
    print(f"{'='*50}")

    # 创建session
    sess = ort.InferenceSession(onnx_path, ort.SessionOptions())

    # 加载PyTorch模型
    torch_model, _ = load_image_model(torch_model_path, device="cpu")
    torch_model.eval()

    max_diffs_feat = []
    max_diffs_cls = []

    for i in range(num_tests):
        torch.manual_seed(42 + i)
        pixel_values = torch.randn(2, 3, 224, 224)

        with torch.no_grad():
            result = torch_model(pixel_values, return_classification=True)
            torch_feat = result["image_features"]
            torch_cls = result["classification_output"]

        onnx_out = sess.run(None, {"pixel_values": pixel_values.numpy()})
        onnx_feat, onnx_cls = onnx_out

        diff_feat = np.abs(torch_feat.numpy() - onnx_feat).max()
        diff_cls = np.abs(torch_cls.numpy() - onnx_cls).max()

        max_diffs_feat.append(diff_feat)
        max_diffs_cls.append(diff_cls)

        print(f"Test {i+1}: feature_diff={diff_feat:.2e}, cls_diff={diff_cls:.2e}")

    print(f"\nFeature embedding:")
    print(f"  Avg max diff: {np.mean(max_diffs_feat):.2e}")
    print(f"  Overall max: {np.max(max_diffs_feat):.2e}")
    print(f"  PASS: {np.max(max_diffs_feat) < 1e-4}")

    print(f"\nClassification:")
    print(f"  Avg max diff: {np.mean(max_diffs_cls):.2e}")
    print(f"  Overall max: {np.max(max_diffs_cls):.2e}")
    print(f"  PASS: {np.max(max_diffs_cls) < 1e-4}")

    return np.max(max_diffs_feat) < 1e-4 and np.max(max_diffs_cls) < 1e-4


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Export FG-CLIP models to ONNX")
    parser.add_argument("--split-models-dir", type=str, default="split_models",
                        help="Directory containing split models")
    parser.add_argument("--output-dir", type=str, default="onnx_exports",
                        help="Output directory for ONNX models")
    parser.add_argument("--max-seq-len", type=int, default=77,
                        help="Maximum sequence length for text model (max 77 for walk_short_pos)")
    parser.add_argument("--verify", action="store_true",
                        help="Verify ONNX models after export")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 导出图像模型
    image_onnx_path = os.path.join(args.output_dir, "image_model.onnx")
    export_image_model_to_onnx(
        model_dir=os.path.join(args.split_models_dir, "image_model"),
        output_path=image_onnx_path,
    )

    # 导出文本模型
    text_onnx_path = os.path.join(args.output_dir, "text_model.onnx")
    export_text_model_to_onnx(
        model_dir=os.path.join(args.split_models_dir, "text_model"),
        output_path=text_onnx_path,
        max_seq_len=args.max_seq_len,
    )

    # 验证
    if args.verify:
        print("\n" + "="*60)
        print("VERIFICATION")
        print("="*60)

        print("\n--- Verifying Image ONNX Model ---")
        img_pass = verify_onnx_image_model(
            onnx_path=image_onnx_path,
            torch_model_path=os.path.join(args.split_models_dir, "image_model"),
            num_tests=5
        )

        print("\n--- Verifying Text ONNX Model ---")
        txt_pass = verify_onnx_model(
            onnx_path=text_onnx_path,
            torch_model_path=os.path.join(args.split_models_dir, "text_model"),
            model_type="text",
            num_tests=5
        )

        print("\n" + "="*60)
        if img_pass and txt_pass:
            print("ALL ONNX MODELS VERIFIED SUCCESSFULLY!")
        else:
            print("VERIFICATION FAILED!")
        print("="*60)

    print(f"\nONNX models exported to: {args.output_dir}")
    print(f"  - Image: {image_onnx_path}")
    print(f"  - Text: {text_onnx_path}")