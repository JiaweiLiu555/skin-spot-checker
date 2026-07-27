from __future__ import annotations

"""Evaluate a locked accuracy candidate on a named held-out manifest."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from scripts.train_accuracy_candidate import CandidateDataset, build_model
from src.metrics import multitask_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()

    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("architecture") != "efficientnet_b0_multitask_accuracy_candidate":
        raise ValueError("Unsupported candidate architecture")
    model = build_model(pretrained=False)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()
    dataset = CandidateDataset(
        args.manifest, training=False, image_size=int(checkpoint["image_size"])
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        persistent_workers=args.workers > 0,
    )
    labels, scores = [], []
    with torch.inference_mode():
        for images, targets in loader:
            labels.append(targets.numpy())
            scores.append(torch.sigmoid(model(images.to(device))).cpu().numpy())
    labels = np.concatenate(labels).astype(int)
    scores = np.concatenate(scores)
    thresholds = checkpoint["thresholds"]
    metrics = multitask_metrics(labels, scores, thresholds)
    predictions = (
        (scores[:, 0] >= thresholds["higher_concern"])
        | (scores[:, 1] >= thresholds["melanoma"])
    ).astype(int)
    output = pd.read_csv(args.manifest).copy()
    output["model_score_higher_concern"] = scores[:, 0]
    output["model_score_melanoma"] = scores[:, 1]
    output["prediction"] = predictions
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output_dir / "predictions.csv", index=False)
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
