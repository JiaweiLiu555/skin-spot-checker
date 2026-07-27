from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnx import helper, numpy_helper

from src.data import IMAGENET_MEAN, IMAGENET_STD
from src.model import CONTOUR_ARCHITECTURE, load_checkpoint


def make_webgl_compatible(path: Path) -> None:
    """Replace activations unsupported by ONNX Runtime WebGL with equivalent basic ops."""
    model = onnx.load(path)
    replacements = []
    for index, node in enumerate(model.graph.node):
        if node.op_type not in {"HardSigmoid", "HardSwish"}:
            replacements.append(node)
            continue

        prefix = node.name or f"webgl_activation_{index}"
        alpha = next((attribute.f for attribute in node.attribute if attribute.name == "alpha"), 1.0 / 6.0)
        beta = next((attribute.f for attribute in node.attribute if attribute.name == "beta"), 0.5)
        alpha_name = f"{prefix}_alpha"
        beta_name = f"{prefix}_beta"
        min_name = f"{prefix}_min"
        max_name = f"{prefix}_max"
        scaled_name = f"{prefix}_scaled"
        shifted_name = f"{prefix}_shifted"
        clipped_name = node.output[0] if node.op_type == "HardSigmoid" else f"{prefix}_clipped"

        model.graph.initializer.extend(
            [
                numpy_helper.from_array(np.asarray(alpha, dtype=np.float32), alpha_name),
                numpy_helper.from_array(np.asarray(beta, dtype=np.float32), beta_name),
                numpy_helper.from_array(np.asarray(0.0, dtype=np.float32), min_name),
                numpy_helper.from_array(np.asarray(1.0, dtype=np.float32), max_name),
            ]
        )
        replacements.extend(
            [
                helper.make_node("Mul", [node.input[0], alpha_name], [scaled_name], name=f"{prefix}_scale"),
                helper.make_node("Add", [scaled_name, beta_name], [shifted_name], name=f"{prefix}_shift"),
                helper.make_node(
                    "Clip",
                    [shifted_name, min_name, max_name],
                    [clipped_name],
                    name=f"{prefix}_clip",
                ),
            ]
        )
        if node.op_type == "HardSwish":
            replacements.append(
                helper.make_node(
                    "Mul",
                    [node.input[0], clipped_name],
                    [node.output[0]],
                    name=f"{prefix}_multiply",
                )
            )

    del model.graph.node[:]
    model.graph.node.extend(replacements)
    onnx.checker.check_model(model)
    onnx.save(model, path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export and validate the model for browser inference.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("web/public/model/skin-lesion-classifier.onnx"))
    args = parser.parse_args()

    device = torch.device("cpu")
    model, checkpoint = load_checkpoint(args.checkpoint, device)
    image_size = int(checkpoint.get("image_size", 224))
    example = torch.randn(1, 3, image_size, image_size)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        example,
        args.output,
        input_names=["pixel_values"],
        output_names=["logit"],
        opset_version=17,
        dynamo=False,
    )
    make_webgl_compatible(args.output)

    with torch.inference_mode():
        expected = model(example).numpy()
    session = ort.InferenceSession(str(args.output), providers=["CPUExecutionProvider"])
    actual = session.run(["logit"], {"pixel_values": example.numpy()})[0]
    max_error = float(np.max(np.abs(expected - actual)))
    if max_error > 1e-4:
        raise RuntimeError(f"ONNX validation failed; maximum error was {max_error}")

    contour_aware = checkpoint.get("architecture") == CONTOUR_ARCHITECTURE
    metadata = {
        "model": (
            "Contour-aware MobileNetV3-Large (RGB + Sobel contour CNN)"
            if contour_aware
            else "MobileNetV3-Large phone-domain adapted"
        ),
        "architecture": checkpoint.get("architecture"),
        "contourAware": contour_aware,
        "contourMethod": (
            "Fixed Sobel edge maps encoded by a learned CNN branch and fused with RGB features"
            if contour_aware
            else None
        ),
        "browserRuntimes": {"iOS": "webgl", "default": "wasm"},
        "inputName": "pixel_values",
        "outputName": "logit",
        "imageSize": image_size,
        "mean": list(IMAGENET_MEAN),
        "std": list(IMAGENET_STD),
        "thresholds": checkpoint.get(
            "thresholds", {"higher_concern": 0.5, "melanoma": 0.5}
        ),
        "outputClasses": checkpoint.get("output_classes", ["higher_concern", "melanoma"]),
        "abstentionMargin": 0.0,
        "alwaysReturnsDecisionAfterQualityPass": True,
        "positiveLabel": "Higher concern",
        "negativeLabel": "Lower concern",
        "validationMetrics": checkpoint.get("validation_metrics", {}),
        "onnxMaxAbsoluteError": max_error,
        "intendedImageType": "phone or clinical close-up image of one skin lesion",
        "fairnessStatus": "not established",
        "trainingStrategy": checkpoint.get("training_strategy", {}),
        "readerFacingScore": {
            "scale": "1–10",
            "lowLabel": "Less similar to higher-concern training images",
            "highLabel": "More similar to higher-concern training images",
            "isCancerProbability": False,
        },
    }
    metadata_path = args.output.with_name("model-metadata.json")
    metadata_path.write_text(json.dumps(metadata, indent=2))
    print(f"Exported {args.output} ({args.output.stat().st_size / 1_000_000:.1f} MB)")
    print(f"Validated ONNX output; max absolute error={max_error:.8f}")


if __name__ == "__main__":
    main()
