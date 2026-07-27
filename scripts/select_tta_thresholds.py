from __future__ import annotations

"""Lock TTA operating thresholds using validation predictions only."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.metrics import multitask_metrics, select_multitask_thresholds


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--validation-predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-sensitivity", type=float, default=0.90)
    parser.add_argument("--min-melanoma-sensitivity", type=float, default=0.93)
    args = parser.parse_args()

    frame = pd.read_csv(args.validation_predictions)
    labels = np.column_stack(
        [frame["label"].astype(int), (frame["diagnosis"].str.upper() == "MEL").astype(int)]
    )
    scores = frame[["model_score_higher_concern", "model_score_melanoma"]].to_numpy()
    thresholds = select_multitask_thresholds(
        labels,
        scores,
        min_sensitivity=args.min_sensitivity,
        min_melanoma_sensitivity=args.min_melanoma_sensitivity,
    )
    metrics = multitask_metrics(labels, scores, thresholds)
    checkpoint = torch.load(args.source, map_location="cpu", weights_only=False)
    checkpoint["thresholds"] = thresholds
    checkpoint["validation_metrics"] = metrics
    checkpoint["inference_augmentations"] = ["original", "horizontal_flip", "vertical_flip", "both_flips"]
    checkpoint["threshold_selection"] = {
        "split": "validation",
        "minimum_overall_sensitivity": args.min_sensitivity,
        "minimum_melanoma_sensitivity": args.min_melanoma_sensitivity,
        "test_or_ood_used": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.output)
    print(json.dumps({"thresholds": thresholds, "validation_metrics": metrics}, indent=2))


if __name__ == "__main__":
    main()
