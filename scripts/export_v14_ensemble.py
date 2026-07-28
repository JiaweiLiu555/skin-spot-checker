from __future__ import annotations

"""Export the RGB ensemble member and assemble browser metadata for v1.4."""

import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from scripts.export_onnx import make_webgl_compatible
from scripts.train_accuracy_candidate import build_model
from src.data import IMAGENET_MEAN, IMAGENET_STD


def export_efficientnet(checkpoint_path: Path, output_path: Path) -> tuple[dict, float]:
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    if checkpoint.get("architecture") != "efficientnet_b0_multitask_accuracy_candidate":
        raise ValueError(f"{checkpoint_path} is not an EfficientNet-B0 candidate.")
    model = build_model(pretrained=False)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    image_size = int(checkpoint.get("image_size", 224))
    example = torch.randn(1, 3, image_size, image_size)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        example,
        output_path,
        input_names=["pixel_values"],
        output_names=["logit"],
        opset_version=17,
        dynamo=False,
    )
    make_webgl_compatible(output_path)
    with torch.inference_mode():
        expected = model(example).numpy()
    session = ort.InferenceSession(
        str(output_path), providers=["CPUExecutionProvider"]
    )
    actual = session.run(["logit"], {"pixel_values": example.numpy()})[0]
    max_error = float(np.max(np.abs(expected - actual)))
    if max_error > 1e-4:
        raise RuntimeError(f"EfficientNet ONNX validation failed: {max_error}")
    return checkpoint, max_error


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clinical-efficientnet-checkpoint", type=Path, required=True)
    parser.add_argument("--efficientnet-checkpoint", type=Path, required=True)
    parser.add_argument("--ensemble", type=Path, required=True)
    parser.add_argument(
        "--contour-model",
        type=Path,
        default=Path("web/public/model/skin-lesion-classifier.onnx"),
    )
    parser.add_argument(
        "--clinical-efficientnet-output",
        type=Path,
        default=Path("web/public/model/efficientnet-b0-clinical.onnx"),
    )
    parser.add_argument(
        "--efficientnet-output",
        type=Path,
        default=Path("web/public/model/efficientnet-b0-clinical-phone.onnx"),
    )
    parser.add_argument(
        "--metadata-output",
        type=Path,
        default=Path("web/public/model/model-metadata.json"),
    )
    args = parser.parse_args()

    clinical_checkpoint, clinical_max_error = export_efficientnet(
        args.clinical_efficientnet_checkpoint,
        args.clinical_efficientnet_output,
    )
    checkpoint, max_error = export_efficientnet(
        args.efficientnet_checkpoint,
        args.efficientnet_output,
    )
    image_size = int(checkpoint.get("image_size", 224))
    if not args.contour_model.exists():
        raise FileNotFoundError(args.contour_model)

    ensemble = json.loads(args.ensemble.read_text())
    decision_policy = ensemble.get("decision_policy", {})
    melanoma_model_index = int(decision_policy.get("melanoma_model_index", 0))
    metadata = {
        "version": "1.4",
        "model": "Rank-fused contour MobileNetV3 + RGB EfficientNet-B0 ensemble",
        "architecture": "rank_logistic_ensemble",
        "browserRuntimes": {"iOS": "webgl", "default": "wasm"},
        "inputName": "pixel_values",
        "outputName": "logit",
        "imageSize": image_size,
        "mean": list(IMAGENET_MEAN),
        "std": list(IMAGENET_STD),
        "models": [
            {
                "name": "contour_mobilenet_v3",
                "role": "RGB, border, contour, and shape specialist",
                "url": "/model/skin-lesion-classifier.onnx",
                "sizeBytes": args.contour_model.stat().st_size,
            },
            {
                "name": "efficientnet_b0_clinical",
                "role": "Complementary clinical RGB color, texture, and morphology specialist",
                "url": "/model/efficientnet-b0-clinical.onnx",
                "sizeBytes": args.clinical_efficientnet_output.stat().st_size,
                "onnxMaxAbsoluteError": clinical_max_error,
            },
            {
                "name": "efficientnet_b0_clinical_phone",
                "role": "Complementary clinical-plus-phone RGB specialist",
                "url": "/model/efficientnet-b0-clinical-phone.onnx",
                "sizeBytes": args.efficientnet_output.stat().st_size,
                "onnxMaxAbsoluteError": max_error,
            },
        ],
        "fusion": {
            "method": "empirical-rank transform plus regularized logistic regression",
            "baseModels": ensemble["base_models"],
            "heads": ensemble["heads"],
        },
        "thresholds": ensemble["thresholds"],
        "decisionPolicy": {
            "higherConcern": "rank-logistic fusion of all three members",
            "melanoma": "dedicated contour-member melanoma head",
            "melanomaModelIndex": melanoma_model_index,
            "melanomaThreshold": ensemble["thresholds"]["melanoma"],
            "reason": "Preserves the dedicated melanoma safety head and its validation-selected threshold.",
        },
        "outputClasses": ["higher_concern", "melanoma"],
        "abstentionMargin": 0.0,
        "alwaysReturnsDecisionAfterQualityPass": True,
        "positiveLabel": "Higher concern",
        "negativeLabel": "Lower concern",
        "validationMetrics": ensemble["selection"],
        "intendedImageType": "phone or clinical close-up image of one skin lesion",
        "fairnessStatus": "not established",
        "trainingStrategy": {
            "contourMemberDatasets": ["MILK10k clinical", "PAD-UFES-20 phone"],
            "clinicalRgbMemberDatasets": ["MILK10k clinical"],
            "combinedRgbMemberDatasets": ["MILK10k clinical", "PAD-UFES-20 phone"],
            "fusionSelectionData": ensemble["selection"]["data"],
            "testDataUsedForFusionOrThresholdSelection": False,
            "sequentialBrowserInference": True,
        },
        "readerFacingScore": {
            "scale": "1–10",
            "lowLabel": "Less similar to higher-concern training images",
            "highLabel": "More similar to higher-concern training images",
            "isCancerProbability": False,
        },
    }
    args.metadata_output.write_text(json.dumps(metadata, indent=2))
    print(
        f"Exported RGB members: "
        f"{args.clinical_efficientnet_output.stat().st_size / 1_000_000:.1f} MB and "
        f"{args.efficientnet_output.stat().st_size / 1_000_000:.1f} MB"
    )
    print(f"Validated EfficientNet ONNX output; max error={max_error:.8f}")
    print(f"Wrote v1.4 metadata to {args.metadata_output}")


if __name__ == "__main__":
    main()
