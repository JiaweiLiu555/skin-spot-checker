from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix, roc_curve
from torch.utils.data import DataLoader

from src.data import LesionDataset
from src.metrics import metrics_from_predictions, multitask_decision_scores, multitask_metrics
from src.model import load_checkpoint


def bootstrap_intervals(labels, scores, thresholds, repetitions: int = 500, seed: int = 2026):
    rng = np.random.default_rng(seed)
    collected = {
        name: []
        for name in (
            "sensitivity",
            "specificity",
            "balanced_accuracy",
            "roc_auc",
            "melanoma_sensitivity",
            "melanoma_head_roc_auc",
            "decision_margin_roc_auc",
        )
    }
    for _ in range(repetitions):
        indices = rng.integers(0, len(labels), size=len(labels))
        sample_labels = labels[indices]
        if np.unique(sample_labels[:, 0]).size < 2 or sample_labels[:, 1].sum() == 0:
            continue
        sample_metrics = multitask_metrics(sample_labels, scores[indices], thresholds)
        for name in collected:
            collected[name].append(sample_metrics[name])
    return {
        name: {
            "lower_95": float(np.percentile(values, 2.5)),
            "median": float(np.percentile(values, 50)),
            "upper_95": float(np.percentile(values, 97.5)),
        }
        for name, values in collected.items()
        if values
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a checkpoint on a held-out manifest.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("reports"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument(
        "--tta-flips",
        action="store_true",
        help="Average the original, horizontal, vertical, and both-flipped predictions.",
    )
    args = parser.parse_args()

    selected_device = args.device
    if selected_device == "auto":
        selected_device = (
            "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
        )
    device = torch.device(selected_device)
    model, checkpoint = load_checkpoint(args.checkpoint, device)
    dataset = LesionDataset(
        args.manifest, training=False, image_size=int(checkpoint.get("image_size", 224))
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    labels, scores = [], []
    with torch.inference_mode():
        for images, targets in loader:
            images = images.to(device)
            if args.tta_flips:
                views = (
                    images,
                    torch.flip(images, dims=(-1,)),
                    torch.flip(images, dims=(-2,)),
                    torch.flip(images, dims=(-2, -1)),
                )
                batch_scores = torch.stack(
                    [torch.sigmoid(model(view)) for view in views], dim=0
                ).mean(dim=0)
            else:
                batch_scores = torch.sigmoid(model(images))
            labels.extend(targets.numpy())
            scores.extend(batch_scores.cpu().numpy())
    labels, scores = np.asarray(labels, dtype=int), np.asarray(scores)
    thresholds = checkpoint.get("thresholds", {"higher_concern": 0.5, "melanoma": 0.5})
    metrics = multitask_metrics(labels, scores, thresholds)
    predictions = (
        (scores[:, 0] >= thresholds["higher_concern"])
        | (scores[:, 1] >= thresholds["melanoma"])
    ).astype(int)
    decision_margin = multitask_decision_scores(scores, thresholds)
    abstention_margin = 0.10
    abstained = np.abs(decision_margin) < abstention_margin
    covered = ~abstained
    followup_predictions = predictions | abstained
    metrics.update(
        {
            "abstention_margin": abstention_margin,
            "abstained_images": int(abstained.sum()),
            "coverage": float(covered.mean()),
            "sensitivity_when_abstention_routes_to_followup": float(
                followup_predictions[labels[:, 0] == 1].mean()
            ),
            "melanoma_abstained": int(abstained[labels[:, 1] == 1].sum()),
        }
    )
    if covered.any() and np.unique(labels[covered, 0]).size == 2:
        covered_metrics = metrics_from_predictions(
            labels[covered, 0], predictions[covered], scores[covered, 0]
        )
        metrics.update({f"covered_{name}": value for name, value in covered_metrics.items()})

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    intervals = bootstrap_intervals(labels, scores, thresholds)
    (args.output_dir / "bootstrap_intervals.json").write_text(json.dumps(intervals, indent=2))
    output = pd.read_csv(args.manifest).copy()
    output["model_score_higher_concern"] = scores[:, 0]
    output["model_score_melanoma"] = scores[:, 1]
    output["decision_margin"] = decision_margin
    output["prediction"] = predictions
    output["abstained"] = abstained
    output["result"] = np.where(
        abstained, "unable_to_assess", np.where(predictions == 1, "higher_concern", "lower_concern")
    )
    output.to_csv(args.output_dir / "predictions.csv", index=False)

    if "diagnosis" in output.columns:
        diagnosis_summary = (
            output.groupby(["diagnosis", "label"], as_index=False)
            .agg(
                images=("prediction", "size"),
                predicted_higher_concern_rate=("prediction", "mean"),
                mean_higher_concern_score=("model_score_higher_concern", "mean"),
                mean_melanoma_score=("model_score_melanoma", "mean"),
            )
            .sort_values("images", ascending=False)
        )
        diagnosis_summary.to_csv(args.output_dir / "diagnosis_summary.csv", index=False)

    subgroup_rows = []
    if "skin_tone_class" in output.columns:
        for group, subset in output.groupby("skin_tone_class", dropna=False):
            indices = subset.index.to_numpy()
            group_labels = labels[indices, 0]
            sufficient = len(subset) >= 20 and np.unique(group_labels).size == 2
            row = {
                "skin_tone_class": "missing" if pd.isna(group) else str(group),
                "images": int(len(subset)),
                "higher_concern_images": int(group_labels.sum()),
                "lower_concern_images": int((group_labels == 0).sum()),
                "status": "descriptive only" if sufficient else "insufficient data",
            }
            if sufficient:
                row.update(
                    metrics_from_predictions(
                        group_labels, predictions[indices], scores[indices, 0]
                    )
                )
            subgroup_rows.append(row)
    if subgroup_rows:
        pd.DataFrame(subgroup_rows).to_csv(args.output_dir / "skin_tone_metrics.csv", index=False)
    fairness_summary = {
        "status": "not established",
        "reason": (
            "Subgroup estimates are descriptive only. Skin-tone groups are uneven, some are absent or too small, "
            "and this retrospective dataset cannot establish fairness."
        ),
        "reported_groups": subgroup_rows,
    }
    (args.output_dir / "fairness_summary.json").write_text(json.dumps(fairness_summary, indent=2))

    matrix = confusion_matrix(labels[:, 0], predictions, labels=[0, 1])
    plt.figure(figsize=(5, 4))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Lower concern", "Higher concern"],
        yticklabels=["Lower concern", "Higher concern"],
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(args.output_dir / "confusion_matrix.png", dpi=180)
    plt.close()

    if np.unique(labels[:, 0]).size == 2:
        fpr, tpr, _ = roc_curve(labels[:, 0], scores[:, 0])
        plt.figure(figsize=(5, 4))
        plt.plot(fpr, tpr, label=f"ROC-AUC = {metrics['roc_auc']:.3f}")
        plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
        plt.xlabel("False-positive rate")
        plt.ylabel("True-positive rate")
        plt.legend()
        plt.tight_layout()
        plt.savefig(args.output_dir / "roc_curve.png", dpi=180)
        plt.close()
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
