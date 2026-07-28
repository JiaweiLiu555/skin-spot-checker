from __future__ import annotations

from dataclasses import dataclass

import numpy as np


OUTPUT_NAMES = ("higher_concern", "melanoma")


def empirical_rank(scores: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Map scores to validation-reference percentiles without assuming calibration."""
    reference = np.sort(np.asarray(reference, dtype=float))
    scores = np.asarray(scores, dtype=float)
    if reference.size == 0:
        raise ValueError("Rank reference cannot be empty.")
    return np.searchsorted(reference, scores, side="right") / reference.size


def sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    positive = values >= 0
    output = np.empty_like(values)
    output[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exponential = np.exp(values[~positive])
    output[~positive] = exponential / (1.0 + exponential)
    return output


@dataclass(frozen=True)
class RankFusionHead:
    coefficients: np.ndarray
    intercept: float
    references: tuple[np.ndarray, ...]

    def predict(self, model_scores: list[np.ndarray]) -> np.ndarray:
        if len(model_scores) != len(self.references):
            raise ValueError("Model-score count does not match fusion references.")
        features = np.column_stack(
            [
                empirical_rank(scores, reference)
                for scores, reference in zip(model_scores, self.references)
            ]
        )
        return sigmoid(features @ self.coefficients + self.intercept)


def head_from_dict(payload: dict) -> RankFusionHead:
    return RankFusionHead(
        coefficients=np.asarray(payload["coefficients"], dtype=float),
        intercept=float(payload["intercept"]),
        references=tuple(
            np.asarray(reference, dtype=float)
            for reference in payload["rank_references"]
        ),
    )


def predict_rank_ensemble(
    payload: dict,
    base_scores: list[np.ndarray],
) -> np.ndarray:
    if len(base_scores) != len(payload["base_models"]):
        raise ValueError("Base prediction count does not match ensemble metadata.")
    outputs = []
    for output_index, output_name in enumerate(OUTPUT_NAMES):
        head = head_from_dict(payload["heads"][output_name])
        outputs.append(
            head.predict([scores[:, output_index] for scores in base_scores])
        )
    scores = np.column_stack(outputs)
    policy = payload.get("decision_policy", {})
    if policy.get("melanoma_source") == "base_model":
        scores[:, 1] = base_scores[int(policy["melanoma_model_index"])][:, 1]
    return scores
