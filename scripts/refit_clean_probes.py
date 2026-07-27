from __future__ import annotations

"""Refit both output heads on frozen features from the clean lesion-level model.

Only training images fit the linear probes. Validation images select regularization
and operating thresholds. The script never reads the test or phone-photo splits.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader

from src.data import LesionDataset
from src.metrics import multitask_metrics, select_multitask_thresholds
from src.model import build_model, load_checkpoint, save_checkpoint


class FrozenFeatureModel(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.features = model.features
        self.avgpool = model.avgpool
        self.embedding = model.classifier[:-1]

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        values = self.features(images)
        values = self.avgpool(values)
        values = torch.flatten(values, 1)
        return self.embedding(values)


def extract(model, manifest, image_size, batch_size, workers, device):
    dataset = LesionDataset(manifest, training=False, image_size=image_size)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers)
    feature_rows, label_rows = [], []
    with torch.inference_mode():
        for images, labels in loader:
            feature_rows.append(model(images.to(device)).cpu().numpy())
            label_rows.append(labels.numpy())
    return np.concatenate(feature_rows), np.concatenate(label_rows).astype(int)


def select_probe(train_x, train_y, val_x, val_y, seed):
    candidates = []
    for c_value in (0.0001, 0.001, 0.01, 0.1, 1.0):
        probe = LogisticRegression(
            C=c_value,
            class_weight="balanced",
            max_iter=2_000,
            random_state=seed,
        ).fit(train_x, train_y)
        scores = probe.predict_proba(val_x)[:, 1]
        candidates.append((roc_auc_score(val_y, scores), c_value, probe, scores))
    return max(candidates, key=lambda item: (item[0], -item[1]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("models/skin_lesion_mobilenet_v3.pt"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--output", type=Path, default=Path("models/skin_lesion_mobilenet_v3_v2.pt"))
    parser.add_argument("--report", type=Path, default=Path("reports/v2_training_summary.json"))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--min-sensitivity", type=float, default=0.90)
    parser.add_argument("--min-melanoma-sensitivity", type=float, default=0.98)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    )
    source_model, source_checkpoint = load_checkpoint(args.source, device)
    image_size = int(source_checkpoint.get("image_size", 224))
    frozen = FrozenFeatureModel(source_model).to(device).eval()
    train_features, train_labels = extract(
        frozen, args.data_dir / "train.csv", image_size, args.batch_size, args.workers, device
    )
    val_features, val_labels = extract(
        frozen, args.data_dir / "val.csv", image_size, args.batch_size, args.workers, device
    )
    scaler = StandardScaler().fit(train_features)
    train_scaled, val_scaled = scaler.transform(train_features), scaler.transform(val_features)

    selected = [
        select_probe(train_scaled, train_labels[:, output], val_scaled, val_labels[:, output], args.seed)
        for output in range(2)
    ]
    val_scores = np.column_stack([item[3] for item in selected])
    thresholds = select_multitask_thresholds(
        val_labels,
        val_scores,
        min_sensitivity=args.min_sensitivity,
        min_melanoma_sensitivity=args.min_melanoma_sensitivity,
    )

    model = build_model(pretrained=False)
    model.features.load_state_dict(source_model.features.state_dict())
    model.classifier[:-1].load_state_dict(source_model.classifier[:-1].state_dict())
    with torch.no_grad():
        for output, (_, _, probe, _) in enumerate(selected):
            scaled_weights = probe.coef_[0]
            raw_weights = scaled_weights / scaler.scale_
            raw_bias = float(probe.intercept_[0] - np.sum(scaled_weights * scaler.mean_ / scaler.scale_))
            model.classifier[-1].weight[output].copy_(torch.from_numpy(raw_weights).float())
            model.classifier[-1].bias[output].fill_(raw_bias)
    model.eval()

    metrics = multitask_metrics(val_labels, val_scores, thresholds)
    for output, name in enumerate(("higher_concern", "melanoma")):
        predictions = val_scores[:, output] >= thresholds[name]
        positive = val_labels[:, output] == 1
        metrics[f"{name}_head_sensitivity"] = float(predictions[positive].mean())
        metrics[f"{name}_head_specificity"] = float((~predictions[~positive]).mean())
    metrics["selected_c_higher_concern"] = float(selected[0][1])
    metrics["selected_c_melanoma"] = float(selected[1][1])
    save_checkpoint(args.output, model, thresholds, image_size, metrics)

    summary = {
        "method": "two frozen linear probes on clean lesion-level training features",
        "source_checkpoint": str(args.source),
        "training_images": int(len(train_labels)),
        "validation_images": int(len(val_labels)),
        "test_or_ood_used_for_selection": False,
        "selected_c": {"higher_concern": selected[0][1], "melanoma": selected[1][1]},
        "validation_head_roc_auc": {
            "higher_concern": selected[0][0],
            "melanoma": selected[1][0],
        },
        "thresholds": thresholds,
        "validation_metrics": metrics,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
