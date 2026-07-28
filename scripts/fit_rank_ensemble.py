from __future__ import annotations

"""Fit a small rank-logistic fusion layer using validation predictions only."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from src.ensemble import OUTPUT_NAMES, empirical_rank, sigmoid
from src.metrics import multitask_metrics, select_multitask_thresholds


SCORE_COLUMNS = (
    "model_score_higher_concern",
    "model_score_melanoma",
)
IDENTITY_COLUMNS = ("image_path", "label", "diagnosis")


def read_aligned_predictions(paths: list[Path]) -> list[pd.DataFrame]:
    frames = [pd.read_csv(path) for path in paths]
    reference = frames[0]
    for path, frame in zip(paths[1:], frames[1:]):
        if len(frame) != len(reference):
            raise ValueError(f"Prediction row count differs in {path}.")
        if not frame[list(IDENTITY_COLUMNS)].equals(reference[list(IDENTITY_COLUMNS)]):
            raise ValueError(f"Prediction rows are not aligned in {path}.")
    return frames


def group_ids(frame: pd.DataFrame) -> np.ndarray:
    patient = frame.get("patient_id", pd.Series(index=frame.index, dtype=object))
    lesion = frame.get("lesion_id", pd.Series(index=frame.index, dtype=object))
    fallback = frame["image_path"].astype(str)
    groups = patient.where(patient.notna() & patient.astype(str).ne(""), lesion)
    return groups.where(groups.notna() & groups.astype(str).ne(""), fallback).astype(str).to_numpy()


def labels_from_frame(frame: pd.DataFrame) -> np.ndarray:
    return np.column_stack(
        [
            frame["label"].astype(int).to_numpy(),
            frame["diagnosis"].astype(str).str.upper().eq("MEL").astype(int).to_numpy(),
        ]
    )


def rank_features(score_vectors: list[np.ndarray], references=None) -> tuple[np.ndarray, list[np.ndarray]]:
    if references is None:
        references = [np.sort(np.asarray(scores, dtype=float)) for scores in score_vectors]
    features = np.column_stack(
        [
            empirical_rank(scores, reference)
            for scores, reference in zip(score_vectors, references)
        ]
    )
    return features, references


def fit_head_oof(
    score_vectors: list[np.ndarray],
    labels: np.ndarray,
    groups: np.ndarray,
    folds: int,
    seed: int,
) -> tuple[np.ndarray, LogisticRegression, list[np.ndarray]]:
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    oof = np.zeros(len(labels), dtype=float)
    for train_indices, holdout_indices in splitter.split(
        np.zeros(len(labels)), labels, groups
    ):
        train_vectors = [scores[train_indices] for scores in score_vectors]
        train_features, references = rank_features(train_vectors)
        holdout_features, _ = rank_features(
            [scores[holdout_indices] for scores in score_vectors],
            references,
        )
        model = LogisticRegression(
            C=0.35,
            class_weight="balanced",
            solver="liblinear",
            random_state=seed,
        )
        model.fit(train_features, labels[train_indices])
        oof[holdout_indices] = model.predict_proba(holdout_features)[:, 1]

    full_features, full_references = rank_features(score_vectors)
    final_model = LogisticRegression(
        C=0.35,
        class_weight="balanced",
        solver="liblinear",
        random_state=seed,
    )
    final_model.fit(full_features, labels)
    return oof, final_model, full_references


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, action="append", required=True)
    parser.add_argument("--model-name", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--oof-output", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2041)
    parser.add_argument("--sensitivity-target", type=float, default=0.95)
    parser.add_argument("--melanoma-sensitivity-target", type=float, default=0.98)
    parser.add_argument("--dedicated-melanoma-model-index", type=int)
    args = parser.parse_args()

    if len(args.predictions) != len(args.model_name):
        raise ValueError("Provide one --model-name for every --predictions file.")
    frames = read_aligned_predictions(args.predictions)
    frame = frames[0].copy()
    labels = labels_from_frame(frame)
    groups = group_ids(frame)

    oof_scores = np.zeros_like(labels, dtype=float)
    full_scores = np.zeros_like(labels, dtype=float)
    heads = {}
    for output_index, output_name in enumerate(OUTPUT_NAMES):
        score_vectors = [
            candidate[SCORE_COLUMNS[output_index]].astype(float).to_numpy()
            for candidate in frames
        ]
        oof, model, references = fit_head_oof(
            score_vectors,
            labels[:, output_index],
            groups,
            args.folds,
            args.seed + output_index,
        )
        oof_scores[:, output_index] = oof
        full_features, _ = rank_features(score_vectors, references)
        full_scores[:, output_index] = sigmoid(
            full_features @ model.coef_[0] + model.intercept_[0]
        )
        heads[output_name] = {
            "coefficients": model.coef_[0].astype(float).tolist(),
            "intercept": float(model.intercept_[0]),
            "rank_references": [
                reference.astype(float).tolist() for reference in references
            ],
        }

    decision_policy = {
        "higher_concern_source": "rank_logistic_fusion",
        "melanoma_source": "rank_logistic_fusion",
    }
    if args.dedicated_melanoma_model_index is not None:
        index = args.dedicated_melanoma_model_index
        if index < 0 or index >= len(frames):
            raise ValueError("Dedicated melanoma model index is out of range.")
        dedicated_scores = (
            frames[index][SCORE_COLUMNS[1]].astype(float).to_numpy()
        )
        full_scores[:, 1] = dedicated_scores
        oof_scores[:, 1] = dedicated_scores
        decision_policy = {
            "higher_concern_source": "rank_logistic_fusion",
            "melanoma_source": "base_model",
            "melanoma_model_index": index,
            "melanoma_model_name": args.model_name[index],
            "reason": "Preserve the dedicated melanoma safety head.",
        }

    thresholds = select_multitask_thresholds(
        labels,
        full_scores,
        min_sensitivity=args.sensitivity_target,
        min_melanoma_sensitivity=args.melanoma_sensitivity_target,
    )
    validation_metrics = multitask_metrics(labels, full_scores, thresholds)
    oof_thresholds = select_multitask_thresholds(
        labels,
        oof_scores,
        min_sensitivity=args.sensitivity_target,
        min_melanoma_sensitivity=args.melanoma_sensitivity_target,
    )
    oof_metrics = multitask_metrics(labels, oof_scores, oof_thresholds)

    payload = {
        "version": "1.4",
        "architecture": "rank_logistic_ensemble",
        "base_models": args.model_name,
        "score_columns": list(SCORE_COLUMNS),
        "heads": heads,
        "decision_policy": decision_policy,
        "thresholds": thresholds,
        "selection": {
            "data": [str(path) for path in args.predictions],
            "folds": args.folds,
            "seed": args.seed,
            "sensitivity_target": args.sensitivity_target,
            "melanoma_sensitivity_target": args.melanoma_sensitivity_target,
            "test_data_used": False,
            "validation_metrics_full_fit": validation_metrics,
            "validation_metrics_oof": oof_metrics,
            "oof_higher_concern_roc_auc": float(
                roc_auc_score(labels[:, 0], oof_scores[:, 0])
            ),
            "oof_melanoma_roc_auc": float(
                roc_auc_score(labels[:, 1], oof_scores[:, 1])
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))

    frame["ensemble_score_higher_concern"] = oof_scores[:, 0]
    frame["ensemble_score_melanoma"] = oof_scores[:, 1]
    frame["ensemble_full_fit_score_higher_concern"] = full_scores[:, 0]
    frame["ensemble_full_fit_score_melanoma"] = full_scores[:, 1]
    args.oof_output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.oof_output, index=False)
    print(json.dumps(payload["selection"], indent=2))


if __name__ == "__main__":
    main()
