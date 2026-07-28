"""Train a small on-device visible-lesion gate from phone and clinical images."""

from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import confusion_matrix, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

from src.data import IMAGENET_MEAN, IMAGENET_STD


class PresenceDataset(Dataset):
    def __init__(self, manifest: Path, training: bool):
        self.manifest = manifest
        self.frame = pd.read_csv(manifest)
        operations: list[object]
        if training:
            operations = [
                transforms.RandomResizedCrop(192, scale=(0.72, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(18),
                transforms.RandomApply(
                    [transforms.ColorJitter(0.2, 0.2, 0.18, 0.03)], p=0.75
                ),
            ]
        else:
            operations = [transforms.Resize((192, 192))]
        operations += [
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
        self.transform = transforms.Compose(operations)

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        row = self.frame.iloc[index]
        path = Path(row.image_path)
        if not path.is_absolute():
            path = (self.manifest.parent / path).resolve()
        with Image.open(path) as source:
            image = self.transform(source.convert("RGB"))
        return image, torch.tensor(float(row.label), dtype=torch.float32)


def build_model(pretrained: bool = True) -> nn.Module:
    weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
    model = mobilenet_v3_small(weights=weights)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 1)
    return model


def select_threshold(labels: np.ndarray, scores: np.ndarray) -> tuple[float, dict]:
    best: tuple[float, float, float, float] | None = None
    for threshold in np.unique(scores):
        predictions = scores >= threshold
        sensitivity = float(predictions[labels == 1].mean())
        specificity = float((~predictions[labels == 0]).mean())
        if sensitivity < 0.95:
            continue
        candidate = (specificity, (sensitivity + specificity) / 2, sensitivity, float(threshold))
        if best is None or candidate > best:
            best = candidate
    if best is None:
        raise RuntimeError("No lesion-presence threshold met the sensitivity floor")
    specificity, balanced_accuracy, sensitivity, threshold = best
    return threshold, {
        "sensitivity": sensitivity,
        "specificity": specificity,
        "balanced_accuracy": balanced_accuracy,
    }


def evaluate(model, loader, device):
    labels, scores = [], []
    model.eval()
    with torch.inference_mode():
        for images, targets in loader:
            logits = model(images.to(device)).squeeze(1)
            labels.append(targets.numpy())
            scores.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(labels).astype(int), np.concatenate(scores)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=2052)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    train_data = PresenceDataset(args.data_dir / "train.csv", True)
    val_data = PresenceDataset(args.data_dir / "val.csv", False)
    class_counts = train_data.frame.label.value_counts()
    weights = train_data.frame.label.map(lambda value: 1 / class_counts[value]).to_numpy(copy=True)
    sampler = WeightedRandomSampler(
        weights, len(weights), replacement=True, generator=torch.Generator().manual_seed(args.seed)
    )
    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.workers,
        persistent_workers=args.workers > 0,
    )
    val_loader = DataLoader(
        val_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        persistent_workers=args.workers > 0,
    )
    model = build_model().to(device)
    for parameter in model.features.parameters():
        parameter.requires_grad = False
    optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=8e-4)
    loss_fn = nn.BCEWithLogitsLoss()
    best = None
    history = []

    for epoch in range(1, args.epochs + 1):
        if epoch == 3:
            for stage in model.features[-3:]:
                for parameter in stage.parameters():
                    parameter.requires_grad = True
            optimizer = torch.optim.AdamW(
                [
                    {"params": model.features.parameters(), "lr": 3e-5},
                    {"params": model.classifier.parameters(), "lr": 2e-4},
                ],
                weight_decay=2e-4,
            )
        model.train()
        losses = []
        for images, targets in train_loader:
            logits = model(images.to(device)).squeeze(1)
            loss = loss_fn(logits, targets.to(device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        labels, scores = evaluate(model, val_loader, device)
        threshold, metrics = select_threshold(labels, scores)
        metrics.update(
            {
                "epoch": epoch,
                "loss": float(np.mean(losses)),
                "roc_auc": float(roc_auc_score(labels, scores)),
                "threshold": threshold,
            }
        )
        history.append(metrics)
        print(json.dumps(metrics), flush=True)
        objective = metrics["balanced_accuracy"] + 0.1 * metrics["roc_auc"]
        if best is None or objective > best[0]:
            best = (objective, copy.deepcopy(model.state_dict()), copy.deepcopy(metrics))

    if best is None:
        raise RuntimeError("No lesion-presence checkpoint was produced")
    _, state_dict, metrics = best
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "architecture": "mobilenet_v3_small_lesion_presence",
            "state_dict": state_dict,
            "image_size": 192,
            "threshold": metrics["threshold"],
            "validation_metrics": metrics,
            "seed": args.seed,
        },
        args.output,
    )
    args.output.with_suffix(".history.json").write_text(json.dumps(history, indent=2))
    print(json.dumps({"checkpoint": str(args.output), "best": metrics}, indent=2))


if __name__ == "__main__":
    main()
