from __future__ import annotations

"""Train a contour-aware, tone-reweighted phone-domain candidate.

Checkpoint selection uses validation data only. Clinical and phone test manifests
are intentionally never opened by this script.
"""

import argparse
import copy
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
from torchvision.transforms import functional as TF

from src.data import IMAGENET_MEAN, IMAGENET_STD
from src.metrics import multitask_metrics, select_multitask_thresholds
from src.model import build_contour_model, load_checkpoint, save_checkpoint


class RandomGamma:
    """Simulate camera/lighting changes without assigning a synthetic skin tone."""

    def __init__(self, minimum: float = 0.72, maximum: float = 1.38):
        self.minimum = minimum
        self.maximum = maximum

    def __call__(self, image: Image.Image) -> Image.Image:
        gamma = random.uniform(self.minimum, self.maximum)
        gain = random.uniform(0.92, 1.08)
        return TF.adjust_gamma(image, gamma=gamma, gain=gain)


class RobustContourDataset(Dataset):
    def __init__(self, manifest: Path, training: bool, image_size: int):
        self.manifest = manifest
        self.frame = pd.read_csv(manifest)
        self.transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    image_size, scale=(0.66, 1.0), ratio=(0.80, 1.25)
                ),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(30),
                transforms.RandomApply(
                    [transforms.ColorJitter(0.34, 0.30, 0.24, 0.05)], p=0.84
                ),
                transforms.RandomApply([RandomGamma()], p=0.36),
                transforms.RandomAutocontrast(p=0.18),
                transforms.RandomGrayscale(p=0.06),
                transforms.RandomApply(
                    [transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.2))],
                    p=0.12,
                ),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
            if training
            else [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        path = Path(row.image_path)
        if not path.is_absolute():
            path = (self.manifest.parent / path).resolve()
        with Image.open(path) as source:
            image = self.transform(source.convert("RGB"))
        target = torch.tensor(
            [float(row.label), float(str(row.diagnosis).upper() == "MEL")],
            dtype=torch.float32,
        )
        return image, target


def run_epoch(model, loader, loss_fn, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    losses, labels, scores = [], [], []
    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        for images, targets in loader:
            images, targets = images.to(device), targets.to(device)
            logits = model(images)
            loss = loss_fn(logits, targets)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
            losses.append(float(loss.detach().cpu()))
            labels.append(targets.detach().cpu().numpy())
            scores.append(torch.sigmoid(logits).detach().cpu().numpy())
    return float(np.mean(losses)), np.concatenate(labels), np.concatenate(scores)


def tone_auc_summary(frame: pd.DataFrame, labels: np.ndarray, scores: np.ndarray):
    rows = []
    tones = frame["skin_tone_class"].fillna("missing").astype(str)
    for tone in sorted(tones.unique()):
        mask = tones.eq(tone).to_numpy()
        group_labels = labels[mask, 0].astype(int)
        positives = int(group_labels.sum())
        negatives = int(len(group_labels) - positives)
        eligible = len(group_labels) >= 20 and positives >= 8 and negatives >= 8
        rows.append(
            {
                "skin_tone_class": tone,
                "images": int(len(group_labels)),
                "positives": positives,
                "negatives": negatives,
                "roc_auc": (
                    float(roc_auc_score(group_labels, scores[mask, 0]))
                    if eligible
                    else None
                ),
                "selection_eligible": eligible,
            }
        )
    eligible_aucs = [row["roc_auc"] for row in rows if row["selection_eligible"]]
    return {
        "groups": rows,
        "eligible_group_count": len(eligible_aucs),
        "mean_eligible_tone_auc": (
            float(np.mean(eligible_aucs)) if eligible_aucs else float("nan")
        ),
        "minimum_eligible_tone_auc": (
            float(np.min(eligible_aucs)) if eligible_aucs else float("nan")
        ),
        "eligible_tone_auc_range": (
            float(np.max(eligible_aucs) - np.min(eligible_aucs))
            if eligible_aucs
            else float("nan")
        ),
    }


def tone_aware_sample_weights(
    frame: pd.DataFrame,
    cap: float,
) -> np.ndarray:
    grouped = frame.groupby(["source_domain", "diagnosis"]).size()
    source_counts = frame.source_domain.value_counts()
    tones = frame["skin_tone_class"].fillna("missing").astype(str)
    tone_counts = tones.value_counts()
    largest_tone_group = float(tone_counts.max())
    weights = []
    for index, row in enumerate(frame.itertuples(index=False)):
        base = 1.0 / np.sqrt(
            float(source_counts[row.source_domain])
            * float(grouped[(row.source_domain, row.diagnosis)])
        )
        tone_factor = min(
            cap,
            np.sqrt(
                largest_tone_group
                / float(tone_counts.loc[tones.iloc[index]])
            ),
        )
        weights.append(base * tone_factor)
    return np.asarray(weights, dtype=np.float64)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data/combined_v2"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=2031)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--tone-weight-cap", type=float, default=3.0)
    parser.add_argument("--sensitivity-target", type=float, default=0.98)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(
        "mps"
        if torch.backends.mps.is_available()
        else "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )
    source_model, source_checkpoint = load_checkpoint(args.source, device)
    image_size = int(source_checkpoint.get("image_size", 224))
    model = build_contour_model(pretrained=False, source_model=source_model).to(device)

    for parameter in model.features.parameters():
        parameter.requires_grad = False
    for stage in model.features[-6:]:
        for parameter in stage.parameters():
            parameter.requires_grad = True

    train_data = RobustContourDataset(
        args.data_dir / "train.csv", True, image_size
    )
    val_data = RobustContourDataset(args.data_dir / "val.csv", False, image_size)
    sample_weights = tone_aware_sample_weights(
        train_data.frame, cap=args.tone_weight_cap
    )
    sampler = WeightedRandomSampler(
        sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
        generator=torch.Generator().manual_seed(args.seed),
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

    train_labels = np.column_stack(
        [
            train_data.frame.label.astype(int).to_numpy(),
            (train_data.frame.diagnosis.str.upper() == "MEL")
            .astype(int)
            .to_numpy(),
        ]
    )
    positives = train_labels.sum(axis=0)
    negatives = len(train_labels) - positives
    pos_weight = torch.tensor(
        np.maximum(1.0, np.sqrt(negatives / positives)),
        dtype=torch.float32,
        device=device,
    )
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(
        [
            {
                "params": [
                    p for p in model.features.parameters() if p.requires_grad
                ],
                "lr": 1.2e-5,
            },
            {"params": model.contour_encoder.parameters(), "lr": 2.5e-4},
            {"params": model.classifier.parameters(), "lr": 1.0e-4},
        ],
        weight_decay=2e-4,
    )

    phone_mask = val_data.frame.source_domain.eq("PAD_UFES_phone").to_numpy()
    best_objective, best_state, best_record = -np.inf, None, None
    history, stale = [], 0
    for epoch in range(1, args.epochs + 1):
        train_loss, _, _ = run_epoch(
            model, train_loader, loss_fn, device, optimizer
        )
        val_loss, labels, scores = run_epoch(model, val_loader, loss_fn, device)
        thresholds = select_multitask_thresholds(
            labels,
            scores,
            args.sensitivity_target,
            args.sensitivity_target,
        )
        metrics = multitask_metrics(labels, scores, thresholds)
        phone_auc = float(
            roc_auc_score(labels[phone_mask, 0], scores[phone_mask, 0])
        )
        overall_auc = float(roc_auc_score(labels[:, 0], scores[:, 0]))
        melanoma_auc = float(roc_auc_score(labels[:, 1], scores[:, 1]))
        tone_summary = tone_auc_summary(val_data.frame, labels, scores)
        tone_mean = tone_summary["mean_eligible_tone_auc"]
        tone_minimum = tone_summary["minimum_eligible_tone_auc"]
        objective = (
            0.25 * overall_auc
            + 0.25 * phone_auc
            + 0.25 * melanoma_auc
            + 0.15 * tone_mean
            + 0.10 * tone_minimum
        )
        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "selection_objective": objective,
            "overall_concern_roc_auc": overall_auc,
            "phone_concern_roc_auc": phone_auc,
            "melanoma_roc_auc": melanoma_auc,
            **metrics,
            **{
                name: value
                for name, value in tone_summary.items()
                if name != "groups"
            },
            "tone_groups": tone_summary["groups"],
        }
        history.append(record)
        print(
            f"epoch={epoch:02d} loss={train_loss:.4f}/{val_loss:.4f} "
            f"auc_all/phone/mel={overall_auc:.3f}/{phone_auc:.3f}/"
            f"{melanoma_auc:.3f} tone_mean/min={tone_mean:.3f}/{tone_minimum:.3f} "
            f"sens/spec/mel={metrics['sensitivity']:.3f}/"
            f"{metrics['specificity']:.3f}/"
            f"{metrics['melanoma_sensitivity']:.3f}",
            flush=True,
        )
        if objective > best_objective + 1e-4:
            best_objective = objective
            best_state = copy.deepcopy(model.state_dict())
            best_record = copy.deepcopy(record)
            best_record["thresholds"] = thresholds
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                break

    if best_state is None:
        raise RuntimeError("No contour-aware checkpoint was produced")
    model.load_state_dict(best_state)
    save_checkpoint(
        args.output,
        model,
        best_record["thresholds"],
        image_size,
        best_record,
        extra_metadata={
            "training_strategy": {
                "source_checkpoint": str(args.source),
                "contour_method": "fixed Sobel maps plus learned CNN fusion",
                "tone_reweighting": True,
                "tone_weight_cap": args.tone_weight_cap,
                "camera_and_lighting_augmentation": True,
                "validation_sensitivity_target": args.sensitivity_target,
                "selection_data": str(args.data_dir / "val.csv"),
                "test_data_used_for_selection": False,
            }
        },
    )
    args.output.with_suffix(".history.json").write_text(
        json.dumps(history, indent=2)
    )
    print(json.dumps(best_record, indent=2))


if __name__ == "__main__":
    main()
