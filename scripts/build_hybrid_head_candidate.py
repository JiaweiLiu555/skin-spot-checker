from __future__ import annotations

"""Combine the deployed concern head with a stronger refit melanoma head.

Only validation data is used to select operating thresholds. Test and phone
manifests are not opened by this script.
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data import LesionDataset
from src.metrics import multitask_metrics, select_multitask_thresholds
from src.model import load_checkpoint, save_checkpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--min-sensitivity", type=float, default=0.90)
    parser.add_argument("--min-melanoma-sensitivity", type=float, default=0.90)
    args = parser.parse_args()

    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )
    model, base_checkpoint = load_checkpoint(args.base, device)
    probe, _ = load_checkpoint(args.probe, device)
    with torch.no_grad():
        model.classifier[-1].weight[1].copy_(probe.classifier[-1].weight[1])
        model.classifier[-1].bias[1].copy_(probe.classifier[-1].bias[1])
    model.eval()
    dataset = LesionDataset(
        args.validation,
        training=False,
        image_size=int(base_checkpoint.get("image_size", 224)),
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
    thresholds = select_multitask_thresholds(
        labels,
        scores,
        args.min_sensitivity,
        args.min_melanoma_sensitivity,
    )
    metrics = multitask_metrics(labels, scores, thresholds)
    metrics["candidate"] = "deployed concern head plus clean refit melanoma head"
    metrics["test_or_ood_used_for_selection"] = False
    save_checkpoint(
        args.output,
        model,
        thresholds,
        int(base_checkpoint.get("image_size", 224)),
        metrics,
    )
    print(metrics)


if __name__ == "__main__":
    main()
