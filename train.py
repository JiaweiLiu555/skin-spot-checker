from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.data import IMAGE_SIZE, LesionDataset
from src.metrics import multitask_metrics, select_multitask_thresholds
from src.model import build_model, save_checkpoint


def run_epoch(model, loader, device, loss_fn, optimizer=None):
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
                optimizer.step()
            losses.append(loss.item())
            labels.extend(targets.detach().cpu().numpy())
            scores.extend(torch.sigmoid(logits).detach().cpu().numpy())
    return float(np.mean(losses)), np.asarray(labels), np.asarray(scores)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the multitask skin-lesion classifier.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--output", type=Path, default=Path("models/skin_lesion_mobilenet_v3.pt"))
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--image-size", type=int, default=IMAGE_SIZE)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-sensitivity", type=float, default=0.90)
    parser.add_argument("--min-melanoma-sensitivity", type=float, default=0.90)
    parser.add_argument("--no-pretrained", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Using device: {device}", flush=True)

    train_manifest = args.data_dir / "train.csv"
    val_manifest = args.data_dir / "val.csv"
    train_data = LesionDataset(train_manifest, training=True, image_size=args.image_size)
    val_data = LesionDataset(val_manifest, training=False, image_size=args.image_size)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    train_frame = pd.read_csv(train_manifest)
    higher_labels = train_frame["label"].astype(int).to_numpy()
    melanoma_labels = (
        train_frame["diagnosis"].astype(str).str.upper() == "MEL"
    ).astype(int).to_numpy()
    label_matrix = np.column_stack([higher_labels, melanoma_labels])
    positives = label_matrix.sum(axis=0)
    negatives = len(label_matrix) - positives
    if (positives == 0).any() or (negatives == 0).any():
        raise ValueError("Training data needs positive and negative examples for both outputs.")
    pos_weight = torch.tensor(negatives / positives, dtype=torch.float32, device=device)

    model = build_model(pretrained=not args.no_pretrained).to(device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)

    best_state, best_score, best_thresholds, best_metrics = None, -1.0, {}, {}
    history, stale_epochs = [], 0
    for epoch in range(1, args.epochs + 1):
        train_loss, _, _ = run_epoch(model, train_loader, device, loss_fn, optimizer)
        val_loss, val_labels, val_scores = run_epoch(model, val_loader, device, loss_fn)
        thresholds = select_multitask_thresholds(
            val_labels,
            val_scores,
            min_sensitivity=args.min_sensitivity,
            min_melanoma_sensitivity=args.min_melanoma_sensitivity,
        )
        metrics = multitask_metrics(val_labels, val_scores, thresholds)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, **metrics})
        print(
            f"epoch={epoch:02d} train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"balanced_accuracy={metrics['balanced_accuracy']:.3f} "
            f"sensitivity={metrics['sensitivity']:.3f} "
            f"melanoma_sensitivity={metrics['melanoma_sensitivity']:.3f}",
            flush=True,
        )
        if metrics["balanced_accuracy"] > best_score:
            best_score = metrics["balanced_accuracy"]
            best_state = copy.deepcopy(model.state_dict())
            best_thresholds, best_metrics = thresholds, metrics
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                print("Early stopping.", flush=True)
                break

    if best_state is None:
        raise RuntimeError("Training did not produce a checkpoint.")
    model.load_state_dict(best_state)
    save_checkpoint(args.output, model, best_thresholds, args.image_size, best_metrics)
    history_path = args.output.with_name("training_history.json")
    history_path.write_text(json.dumps(history, indent=2))
    print(f"Saved checkpoint to {args.output}", flush=True)
    print(f"Saved history to {history_path}", flush=True)


if __name__ == "__main__":
    main()
