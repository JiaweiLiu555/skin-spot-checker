from __future__ import annotations

"""Train accuracy-focused, leak-free skin-lesion model candidates.

The script fits on ``train.csv`` only, uses ``val.csv`` for epoch/threshold
selection, and never opens the clinical test or phone-photo manifests.
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
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

from src.data import IMAGENET_MEAN, IMAGENET_STD
from src.metrics import multitask_metrics, select_multitask_thresholds


class CandidateDataset(Dataset):
    def __init__(self, manifest: Path, training: bool, image_size: int):
        self.manifest = manifest
        self.frame = pd.read_csv(manifest)
        if training:
            self.transform = transforms.Compose(
                [
                    transforms.RandomResizedCrop(
                        image_size, scale=(0.68, 1.0), ratio=(0.85, 1.18)
                    ),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip(),
                    transforms.RandomRotation(25),
                    transforms.RandomApply(
                        [transforms.ColorJitter(0.22, 0.22, 0.16, 0.04)], p=0.8
                    ),
                    transforms.RandomAutocontrast(p=0.15),
                    transforms.ToTensor(),
                    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
                    transforms.RandomErasing(
                        p=0.12, scale=(0.01, 0.06), ratio=(0.4, 2.5), value="random"
                    ),
                ]
            )
        else:
            self.transform = transforms.Compose(
                [
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


def build_model(pretrained: bool = True):
    weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
    model = efficientnet_b0(weights=weights)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, 2)
    return model


def set_backbone_trainable(model: nn.Module, trainable: bool):
    for parameter in model.features.parameters():
        parameter.requires_grad = False
    if trainable:
        # Fine-tuning the final three EfficientNet stages retains the useful
        # ImageNet representation while making each epoch practical on a Mac.
        for stage in model.features[-3:]:
            for parameter in stage.parameters():
                parameter.requires_grad = True


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


def validation_objective(labels, scores):
    concern_auc = roc_auc_score(labels[:, 0], scores[:, 0])
    melanoma_auc = roc_auc_score(labels[:, 1], scores[:, 1])
    return float(0.65 * concern_auc + 0.35 * melanoma_auc), concern_auc, melanoma_auc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=18)
    parser.add_argument("--frozen-epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--min-sensitivity", type=float, default=0.90)
    parser.add_argument("--min-melanoma-sensitivity", type=float, default=0.90)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )
    print(f"device={device} seed={args.seed}", flush=True)

    train_manifest = args.data_dir / "train.csv"
    val_manifest = args.data_dir / "val.csv"
    train_data = CandidateDataset(train_manifest, True, args.image_size)
    val_data = CandidateDataset(val_manifest, False, args.image_size)

    diagnosis_counts = train_data.frame.diagnosis.value_counts()
    sample_weights = train_data.frame.diagnosis.map(
        lambda value: 1.0 / np.sqrt(float(diagnosis_counts[value]))
    ).to_numpy()
    sampler_generator = torch.Generator().manual_seed(args.seed)
    sampler = WeightedRandomSampler(
        sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
        generator=sampler_generator,
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

    labels = np.column_stack(
        [
            train_data.frame.label.astype(int).to_numpy(),
            (train_data.frame.diagnosis.str.upper() == "MEL").astype(int).to_numpy(),
        ]
    )
    positives = labels.sum(axis=0)
    negatives = len(labels) - positives
    # A square-root weight improves the rare melanoma head without allowing the
    # loss scale to be dominated by it.
    pos_weight = torch.tensor(
        np.maximum(1.0, np.sqrt(negatives / positives)),
        dtype=torch.float32,
        device=device,
    )
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    model = build_model(pretrained=True).to(device)
    best_state = None
    best_objective = -np.inf
    best_record = None
    history = []
    stale = 0

    for epoch in range(1, args.epochs + 1):
        frozen = epoch <= args.frozen_epochs
        set_backbone_trainable(model, not frozen)
        if frozen:
            optimizer = torch.optim.AdamW(
                model.classifier.parameters(), lr=8e-4, weight_decay=1e-4
            )
        elif epoch == args.frozen_epochs + 1:
            optimizer = torch.optim.AdamW(
                [
                    {"params": model.features.parameters(), "lr": 3e-5},
                    {"params": model.classifier.parameters(), "lr": 2e-4},
                ],
                weight_decay=2e-4,
            )
        # Reuse the unfrozen optimizer so momentum is preserved.
        train_loss, _, _ = run_epoch(model, train_loader, loss_fn, device, optimizer)
        val_loss, val_labels, val_scores = run_epoch(model, val_loader, loss_fn, device)
        objective, concern_auc, melanoma_auc = validation_objective(val_labels, val_scores)
        thresholds = select_multitask_thresholds(
            val_labels,
            val_scores,
            min_sensitivity=args.min_sensitivity,
            min_melanoma_sensitivity=args.min_melanoma_sensitivity,
        )
        metrics = multitask_metrics(val_labels, val_scores, thresholds)
        record = {
            "epoch": epoch,
            "frozen_backbone": frozen,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "selection_objective": objective,
            "concern_roc_auc": concern_auc,
            "melanoma_roc_auc": melanoma_auc,
            **metrics,
        }
        history.append(record)
        print(
            f"epoch={epoch:02d} loss={train_loss:.4f}/{val_loss:.4f} "
            f"auc={concern_auc:.4f}/{melanoma_auc:.4f} "
            f"sens={metrics['sensitivity']:.3f} spec={metrics['specificity']:.3f} "
            f"mel_sens={metrics['melanoma_sensitivity']:.3f}",
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
            if epoch > args.frozen_epochs + 2 and stale >= args.patience:
                print("early_stopping", flush=True)
                break

    if best_state is None or best_record is None:
        raise RuntimeError("Training produced no candidate checkpoint")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "architecture": "efficientnet_b0_multitask_accuracy_candidate",
            "state_dict": best_state,
            "thresholds": best_record["thresholds"],
            "image_size": args.image_size,
            "validation_metrics": best_record,
            "output_classes": ["higher_concern", "melanoma"],
            "seed": args.seed,
            "test_or_ood_used_for_selection": False,
        },
        args.output,
    )
    history_path = args.output.with_suffix(".history.json")
    history_path.write_text(json.dumps(history, indent=2))
    print(json.dumps({"checkpoint": str(args.output), "best": best_record}, indent=2))


if __name__ == "__main__":
    main()
