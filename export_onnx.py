"""
FG-CLIP to ONNX export and validation script.
"""
import sys
from pathlib import Path

# Add parent directory and import as package
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import torch
import numpy as np
import onnx
import onnxruntime as ort

# Import as package
from fg_clip.modeling_fgclip import FGCLIPModel, FGCLIPConfig

def export_to_onnx(
    model_path: str,
    output_dir: str = "./onnx_models",
    image_size: int = 224,
    max_text_len: int = 77,
    opset_version: int = 14,
    precision: str = "fp32"
):
    """Export FG-CLIP model to ONNX format."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    print("Loading model...")
    config = FGCLIPConfig.from_pretrained(model_path)
    model = FGCLIPModel.from_pretrained(model_path, config=config)
    model.eval()

    # Export image encoder
    print("Exporting image encoder...")
    dummy_pixel_values = torch.randn(1, 3, image_size, image_size)
    image_output_path = output_dir / f"image_encoder_{precision}.onnx"

    torch.onnx.export(
        model,
        (dummy_pixel_values,),
        str(image_output_path),
        input_names=["pixel_values"],
        output_names=["image_features"],
        dynamic_axes={
            "pixel_values": {0: "batch_size"},
            "image_features": {0: "batch_size"},
        },
        opset_version=opset_version,
        do_constant_folding=True,
    )
    print(f"Image encoder saved to {image_output_path}")

    # Export text encoder
    print("Exporting text encoder...")
    dummy_input_ids = torch.randint(0, 49408, (1, max_text_len))
    dummy_attention_mask = torch.ones(1, max_text_len, dtype=torch.long)
    text_output_path = output_dir / f"text_encoder_{precision}.onnx"

    torch.onnx.export(
        model,
        (dummy_input_ids, dummy_attention_mask),
        str(text_output_path),
        input_names=["input_ids", "attention_mask"],
        output_names=["text_features"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "seq_len"},
            "attention_mask": {0: "batch_size", 1: "seq_len"},
            "text_features": {0: "batch_size"},
        },
        opset_version=opset_version,
        do_constant_folding=True,
    )
    print(f"Text encoder saved to {text_output_path}")

    # Export full model (image + text)
    print("Exporting full model...")
    full_output_path = output_dir / f"fgclip_{precision}.onnx"

    torch.onnx.export(
        model,
        (dummy_pixel_values, dummy_input_ids, dummy_attention_mask),
        str(full_output_path),
        input_names=["pixel_values", "input_ids", "attention_mask"],
        output_names=["image_features", "text_features"],
        dynamic_axes={
            "pixel_values": {0: "batch_size"},
            "input_ids": {0: "batch_size", 1: "seq_len"},
            "attention_mask": {0: "batch_size", 1: "seq_len"},
            "image_features": {0: "batch_size"},
            "text_features": {0: "batch_size"},
        },
        opset_version=opset_version,
        do_constant_folding=True,
    )
    print(f"Full model saved to {full_output_path}")

    # Verify ONNX models
    print("\nVerifying ONNX models...")
    for onnx_path in output_dir.glob(f"*{precision}.onnx"):
        model_onnx = onnx.load(str(onnx_path))
        onnx.checker.check_model(model_onnx)
        print(f"  ✓ {onnx_path.name} is valid")

    return output_dir


def run_onnx_inference(onnx_path: str, inputs: dict) -> dict:
    """Run inference with ONNX model."""
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    outputs = session.run(None, inputs)
    return outputs


def validate_onnx(
    model_path: str,
    onnx_dir: str = "./onnx_models",
    image_size: int = 224,
    max_text_len: int = 77,
    num_tests: int = 10,
    atol: float = 1e-4,
    rtol: float = 1e-4
):
    """Validate ONNX model outputs match PyTorch model outputs."""
    print("=" * 60)
    print("Validating ONNX export accuracy")
    print("=" * 60)

    # Load PyTorch model
    print("\nLoading PyTorch model...")
    config = FGCLIPConfig.from_pretrained(model_path)
    model = FGCLIPModel.from_pretrained(model_path, config=config)
    model.eval()

    results = {"image_encoder": [], "text_encoder": [], "full_model": []}

    for i in range(num_tests):
        print(f"\nTest {i+1}/{num_tests}")

        # Generate random inputs
        batch_size = np.random.randint(1, 4)
        pixel_values = torch.randn(batch_size, 3, image_size, image_size)
        input_ids = torch.randint(0, 49408, (batch_size, max_text_len))
        attention_mask = torch.ones(batch_size, max_text_len, dtype=torch.long)

        # PyTorch inference
        with torch.no_grad():
            image_features_torch = model.get_image_features(pixel_values)
            text_features_torch = model.get_text_features(input_ids, attention_mask)

        # ONNX inference for image encoder
        onnx_image_path = Path(onnx_dir) / "image_encoder_fp32.onnx"
        if onnx_image_path.exists():
            onnx_image_out = run_onnx_inference(
                str(onnx_image_path),
                {"pixel_values": pixel_values.numpy().astype(np.float32)}
            )
            image_features_onnx = onnx_image_out[0]

            # Compare
            diff = np.abs(image_features_torch.numpy() - image_features_onnx)
            max_diff = np.max(diff)
            mean_diff = np.mean(diff)

            img_pass = np.allclose(image_features_torch.numpy(), image_features_onnx, atol=atol, rtol=rtol)
            results["image_encoder"].append({
                "max_diff": max_diff,
                "mean_diff": mean_diff,
                "passed": img_pass
            })
            status = "✓" if img_pass else "✗"
            print(f"  Image encoder: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f} {status}")

        # ONNX inference for text encoder
        onnx_text_path = Path(onnx_dir) / "text_encoder_fp32.onnx"
        if onnx_text_path.exists():
            onnx_text_out = run_onnx_inference(
                str(onnx_text_path),
                {
                    "input_ids": input_ids.numpy().astype(np.int64),
                    "attention_mask": attention_mask.numpy().astype(np.int64)
                }
            )
            text_features_onnx = onnx_text_out[0]

            # Compare
            diff = np.abs(text_features_torch.numpy() - text_features_onnx)
            max_diff = np.max(diff)
            mean_diff = np.mean(diff)

            txt_pass = np.allclose(text_features_torch.numpy(), text_features_onnx, atol=atol, rtol=rtol)
            results["text_encoder"].append({
                "max_diff": max_diff,
                "mean_diff": mean_diff,
                "passed": txt_pass
            })
            status = "✓" if txt_pass else "✗"
            print(f"  Text encoder: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f} {status}")

    # Summary
    print("\n" + "=" * 60)
    print("Validation Summary")
    print("=" * 60)

    for model_name, model_results in results.items():
        if model_results:
            passed = sum(1 for r in model_results if r["passed"])
            total = len(model_results)
            avg_max_diff = np.mean([r["max_diff"] for r in model_results])
            avg_mean_diff = np.mean([r["mean_diff"] for r in model_results])

            print(f"\n{model_name}:")
            print(f"  Passed: {passed}/{total}")
            print(f"  Avg max diff: {avg_max_diff:.6f}")
            print(f"  Avg mean diff: {avg_mean_diff:.6f}")

            if passed == total:
                print(f"  ✓ All tests passed!")
            else:
                print(f"  ✗ Some tests failed (atol={atol}, rtol={rtol})")

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Export FG-CLIP to ONNX and validate")
    parser.add_argument("--model_path", type=str, default="./fg-clip", help="Path to FG-CLIP model")
    parser.add_argument("--output_dir", type=str, default="./onnx_models", help="Output directory for ONNX models")
    parser.add_argument("--image_size", type=int, default=224, help="Image size")
    parser.add_argument("--max_text_len", type=int, default=77, help="Max text length")
    parser.add_argument("--opset_version", type=int, default=14, help="ONNX opset version")
    parser.add_argument("--validate_only", action="store_true", help="Skip export, only validate")
    parser.add_argument("--num_tests", type=int, default=10, help="Number of validation tests")
    parser.add_argument("--atol", type=float, default=1e-4, help="Absolute tolerance for validation")
    parser.add_argument("--rtol", type=float, default=1e-4, help="Relative tolerance for validation")

    args = parser.parse_args()

    if not args.validate_only:
        export_to_onnx(
            model_path=args.model_path,
            output_dir=args.output_dir,
            image_size=args.image_size,
            max_text_len=args.max_text_len,
            opset_version=args.opset_version,
        )

    validate_onnx(
        model_path=args.model_path,
        onnx_dir=args.output_dir,
        image_size=args.image_size,
        max_text_len=args.max_text_len,
        num_tests=args.num_tests,
        atol=args.atol,
        rtol=args.rtol,
    )