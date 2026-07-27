from __future__ import annotations

"""Fast metric/prediction evaluation without plotting dependencies."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.data import LesionDataset
from src.metrics import metrics_from_predictions, multitask_metrics
from src.model import load_checkpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )
    model, checkpoint = load_checkpoint(args.checkpoint, device)
    dataset = LesionDataset(
        args.manifest, training=False, image_size=int(checkpoint.get("image_size", 224))
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
    subgroup_rows = []
    if "skin_tone_class" in output.columns:
        tones = output["skin_tone_class"].fillna("missing").astype(str)
        for tone in sorted(tones.unique()):
            mask = tones.eq(tone).to_numpy()
            group_labels = labels[mask, 0]
            positives = int(group_labels.sum())
            negatives = int(len(group_labels) - positives)
            sufficient = (
                len(group_labels) >= 20
                and positives >= 8
                and negatives >= 8
            )
            row = {
                "skin_tone_class": tone,
                "images": int(len(group_labels)),
                "higher_concern_images": positives,
                "lower_concern_images": negatives,
                "status": (
                    "descriptive only" if sufficient else "insufficient data"
                ),
            }
            if sufficient:
                row.update(
                    metrics_from_predictions(
                        group_labels,
                        predictions[mask],
                        scores[mask, 0],
                    )
                )
            subgroup_rows.append(row)
        pd.DataFrame(subgroup_rows).to_csv(
            args.output_dir / "skin_tone_metrics.csv", index=False
        )
    fairness_summary = {
        "status": "not established",
        "reason": (
            "Skin-tone subgroup estimates are descriptive only. Groups are uneven, "
            "some lack both outcome classes, and no prospective fairness study was run."
        ),
        "reported_groups": subgroup_rows,
    }
    (args.output_dir / "fairness_summary.json").write_text(
        json.dumps(fairness_summary, indent=2)
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
