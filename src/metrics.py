from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def select_threshold(labels, scores, min_sensitivity: float = 0.90) -> float:
    """Choose the most specific validation threshold meeting a sensitivity target."""
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    if np.unique(labels).size < 2:
        return 0.5
    false_positive_rate, true_positive_rate, thresholds = roc_curve(labels, scores)
    feasible = np.isfinite(thresholds) & (true_positive_rate >= min_sensitivity)
    if feasible.any():
        feasible_indices = np.flatnonzero(feasible)
        best = feasible_indices[int(np.argmin(false_positive_rate[feasible]))]
    else:
        finite = np.isfinite(thresholds)
        candidates = np.where(finite, true_positive_rate - false_positive_rate, -np.inf)
        best = int(np.argmax(candidates))
    return float(np.clip(thresholds[best], 0.0, 1.0))


def select_multitask_thresholds(
    labels,
    scores,
    min_sensitivity: float = 0.90,
    min_melanoma_sensitivity: float = 0.90,
) -> dict[str, float]:
    """Jointly select two thresholds without double-counting false positives."""
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    quantiles = np.linspace(0.0, 1.0, 101)
    candidates_higher = np.unique(np.append(np.quantile(scores[:, 0], quantiles), 1.0))
    candidates_melanoma = np.unique(np.append(np.quantile(scores[:, 1], quantiles), 1.0))
    positives = labels[:, 0] == 1
    negatives = ~positives
    melanoma = labels[:, 1] == 1
    best = None
    for threshold_higher in candidates_higher:
        higher_predictions = scores[:, 0] >= threshold_higher
        for threshold_melanoma in candidates_melanoma:
            predictions = higher_predictions | (scores[:, 1] >= threshold_melanoma)
            sensitivity = float(predictions[positives].mean())
            melanoma_sensitivity = float(predictions[melanoma].mean())
            specificity = float((~predictions[negatives]).mean())
            feasible = (
                sensitivity >= min_sensitivity
                and melanoma_sensitivity >= min_melanoma_sensitivity
            )
            objective = (
                int(feasible),
                specificity if feasible else min(sensitivity / min_sensitivity, melanoma_sensitivity / min_melanoma_sensitivity),
                (sensitivity + specificity) / 2,
            )
            if best is None or objective > best[0]:
                best = (objective, float(threshold_higher), float(threshold_melanoma))
    return {"higher_concern": best[1], "melanoma": best[2]}


def binary_metrics(labels, scores, threshold: float) -> dict[str, float]:
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    predictions = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    result = {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "sensitivity": float(recall_score(labels, predictions, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if tn + fp else 0.0,
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }
    result["roc_auc"] = (
        float(roc_auc_score(labels, scores)) if np.unique(labels).size == 2 else float("nan")
    )
    return result


def metrics_from_predictions(labels, predictions, scores=None) -> dict[str, float]:
    labels = np.asarray(labels, dtype=int)
    predictions = np.asarray(predictions, dtype=int)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    result = {
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "sensitivity": float(recall_score(labels, predictions, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if tn + fp else 0.0,
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }
    result["roc_auc"] = (
        float(roc_auc_score(labels, np.asarray(scores, dtype=float)))
        if scores is not None and np.unique(labels).size == 2
        else float("nan")
    )
    return result


def multitask_decision_scores(scores, thresholds: dict[str, float]) -> np.ndarray:
    """Return normalized signed distances whose zero boundary matches the two-head OR rule."""
    scores = np.asarray(scores, dtype=float)
    threshold_values = np.asarray(
        [thresholds["higher_concern"], thresholds["melanoma"]], dtype=float
    )
    threshold_values = np.clip(threshold_values, 1e-6, 1 - 1e-6)
    distances = np.where(
        scores >= threshold_values,
        (scores - threshold_values) / (1 - threshold_values),
        (scores - threshold_values) / threshold_values,
    )
    return distances.max(axis=1)


def multitask_metrics(labels, scores, thresholds: dict[str, float]) -> dict[str, float]:
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    predictions = (
        (scores[:, 0] >= thresholds["higher_concern"])
        | (scores[:, 1] >= thresholds["melanoma"])
    ).astype(int)
    decision_scores = multitask_decision_scores(scores, thresholds)
    result = metrics_from_predictions(labels[:, 0], predictions, scores[:, 0])
    melanoma_mask = labels[:, 1] == 1
    result.update(
        {
            "threshold_higher_concern": float(thresholds["higher_concern"]),
            "threshold_melanoma": float(thresholds["melanoma"]),
            "melanoma_images": int(melanoma_mask.sum()),
            "melanoma_sensitivity": float(predictions[melanoma_mask].mean())
            if melanoma_mask.any()
            else float("nan"),
            "melanoma_head_roc_auc": float(roc_auc_score(labels[:, 1], scores[:, 1]))
            if np.unique(labels[:, 1]).size == 2
            else float("nan"),
            "decision_margin_roc_auc": float(roc_auc_score(labels[:, 0], decision_scores))
            if np.unique(labels[:, 0]).size == 2
            else float("nan"),
        }
    )
    return result
