"""Export and numerically verify the visible-lesion gate for browser inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from scripts.export_onnx import make_webgl_compatible
from scripts.train_lesion_presence import build_model
from src.data import IMAGENET_MEAN, IMAGENET_STD


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata-output", type=Path, required=True)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("architecture") != "mobilenet_v3_small_lesion_presence":
        raise ValueError("Unsupported visible-lesion checkpoint architecture")

    model = build_model(pretrained=False)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    image_size = int(checkpoint["image_size"])
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
    session = ort.InferenceSession(
        str(args.output), providers=["CPUExecutionProvider"]
    )
    actual = session.run(["logit"], {"pixel_values": example.numpy()})[0]
    max_error = float(np.max(np.abs(expected - actual)))
    if max_error > 1e-4:
        raise RuntimeError(f"Visible-lesion ONNX validation failed: {max_error}")

    metadata = {
        "version": "1.6",
        "architecture": checkpoint["architecture"],
        "url": f"/model/{args.output.name}",
        "inputName": "pixel_values",
        "outputName": "logit",
        "imageSize": image_size,
        "mean": list(IMAGENET_MEAN),
        "std": list(IMAGENET_STD),
        "threshold": float(checkpoint["threshold"]),
        "validationMetrics": checkpoint["validation_metrics"],
        "positiveMeaning": "a visible centered lesion or growth is present",
        "negativeMeaning": "no clear centered lesion was detected",
        "trainingSources": [
            "SCIN participant-submitted phone images",
            "PAD-UFES-20 phone lesion images",
            "SLICE-3D total-body lesion crops",
        ],
        "limitations": (
            "This is an input-routing model, not a cancer classifier. SCIN "
            "LOOKS_HEALTHY is a participant annotation, not pathology-confirmed normal skin."
        ),
        "onnxMaxAbsoluteError": max_error,
        "sizeBytes": args.output.stat().st_size,
    }
    args.metadata_output.write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
