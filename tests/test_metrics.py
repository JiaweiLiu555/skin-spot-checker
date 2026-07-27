import pytest

import numpy as np

from src.metrics import binary_metrics, multitask_metrics, select_multitask_thresholds, select_threshold


def test_perfect_scores_produce_perfect_metrics():
    labels = [0, 0, 1, 1]
    scores = [0.05, 0.2, 0.8, 0.95]
    metrics = binary_metrics(labels, scores, threshold=0.5)
    assert metrics["sensitivity"] == pytest.approx(1.0)
    assert metrics["specificity"] == pytest.approx(1.0)
    assert metrics["roc_auc"] == pytest.approx(1.0)


def test_selected_threshold_is_valid():
    threshold = select_threshold([0, 0, 1, 1], [0.1, 0.3, 0.7, 0.9])
    assert 0.0 <= threshold <= 1.0


def test_selected_threshold_meets_sensitivity_target():
    labels = [0, 0, 0, 1, 1, 1, 1]
    scores = [0.1, 0.2, 0.8, 0.45, 0.6, 0.7, 0.9]
    threshold = select_threshold(labels, scores, min_sensitivity=0.75)
    metrics = binary_metrics(labels, scores, threshold)
    assert metrics["sensitivity"] >= 0.75


def test_joint_thresholds_meet_both_sensitivity_targets():
    labels = np.array([[0, 0], [0, 0], [0, 0], [1, 0], [1, 0], [1, 1], [1, 1]])
    scores = np.array(
        [[0.05, 0.05], [0.15, 0.10], [0.55, 0.15], [0.60, 0.10], [0.75, 0.15], [0.35, 0.80], [0.40, 0.90]]
    )
    thresholds = select_multitask_thresholds(labels, scores, 0.75, 1.0)
    metrics = multitask_metrics(labels, scores, thresholds)
    assert metrics["sensitivity"] >= 0.75
    assert metrics["melanoma_sensitivity"] == 1.0
