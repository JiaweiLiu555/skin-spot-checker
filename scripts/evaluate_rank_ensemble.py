from __future__ import annotations

"""Evaluate a locked rank-logistic ensemble on aligned base-model predictions."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.fit_rank_ensemble import (
    IDENTITY_COLUMNS,
    SCORE_COLUMNS,
    labels_from_frame,
    read_aligned_predictions,
)
from src.ensemble import predict_rank_ensemble
from src.metrics import multitask_metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ensemble", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    payload = json.loads(args.ensemble.read_text())
    if len(args.predictions) != len(payload["base_models"]):
        raise ValueError("Base prediction count does not match the ensemble.")
    frames = read_aligned_predictions(args.predictions)
    labels = labels_from_frame(frames[0])
    base_scores = [
        frame[list(SCORE_COLUMNS)].astype(float).to_numpy() for frame in frames
    ]
    scores = predict_rank_ensemble(payload, base_scores)
    thresholds = payload["thresholds"]
    metrics = multitask_metrics(labels, scores, thresholds)
    decisions = (
        (scores[:, 0] >= thresholds["higher_concern"])
        | (scores[:, 1] >= thresholds["melanoma"])
    ).astype(int)

    output = frames[0].copy()
    output["ensemble_score_higher_concern"] = scores[:, 0]
    output["ensemble_score_melanoma"] = scores[:, 1]
    output["prediction"] = decisions
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output_dir / "predictions.csv", index=False)
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
