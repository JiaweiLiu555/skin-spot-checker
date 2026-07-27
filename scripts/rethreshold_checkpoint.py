from __future__ import annotations

"""Select new operating thresholds from validation predictions only."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.metrics import multitask_metrics, select_multitask_thresholds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--validation-predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-sensitivity", type=float, required=True)
    parser.add_argument("--min-melanoma-sensitivity", type=float, required=True)
    args = parser.parse_args()

    frame = pd.read_csv(args.validation_predictions)
    labels = np.column_stack(
        [
            frame["label"].astype(int).to_numpy(),
            frame["diagnosis"].astype(str).str.upper().eq("MEL").astype(int).to_numpy(),
        ]
    )
    scores = frame[
        ["model_score_higher_concern", "model_score_melanoma"]
    ].to_numpy()
    thresholds = select_multitask_thresholds(
        labels,
        scores,
        args.min_sensitivity,
        args.min_melanoma_sensitivity,
    )
    metrics = multitask_metrics(labels, scores, thresholds)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    checkpoint["thresholds"] = thresholds
    checkpoint["validation_metrics"] = {
        **checkpoint.get("validation_metrics", {}),
        **metrics,
        "thresholds": thresholds,
        "threshold_policy": {
            "minimum_overall_sensitivity": args.min_sensitivity,
            "minimum_melanoma_sensitivity": args.min_melanoma_sensitivity,
            "selection_data": str(args.validation_predictions),
            "test_or_ood_used_for_selection": False,
        },
    }
    checkpoint.setdefault("training_strategy", {})["release_threshold_policy"] = {
        "minimum_overall_sensitivity": args.min_sensitivity,
        "minimum_melanoma_sensitivity": args.min_melanoma_sensitivity,
        "selection_data": str(args.validation_predictions),
        "test_or_ood_used_for_selection": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.output)
    print({"thresholds": thresholds, "validation_metrics": metrics})


if __name__ == "__main__":
    main()
