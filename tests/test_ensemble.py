import numpy as np

from src.ensemble import empirical_rank, predict_rank_ensemble


def test_empirical_rank_uses_reference_distribution():
    reference = np.array([0.1, 0.2, 0.4, 0.8])
    scores = np.array([0.05, 0.2, 0.5, 0.9])
    assert np.allclose(empirical_rank(scores, reference), [0.0, 0.5, 0.75, 1.0])


def test_predict_rank_ensemble_returns_two_bounded_outputs():
    payload = {
        "base_models": ["a", "b"],
        "heads": {
            "higher_concern": {
                "coefficients": [1.0, 0.5],
                "intercept": -0.4,
                "rank_references": [[0.1, 0.5], [0.2, 0.6]],
            },
            "melanoma": {
                "coefficients": [0.25, 1.0],
                "intercept": -0.7,
                "rank_references": [[0.1, 0.5], [0.2, 0.6]],
            },
        },
    }
    first = np.array([[0.2, 0.3], [0.7, 0.8]])
    second = np.array([[0.3, 0.4], [0.9, 0.9]])
    scores = predict_rank_ensemble(payload, [first, second])
    assert scores.shape == (2, 2)
    assert np.all((scores > 0) & (scores < 1))


def test_predict_rank_ensemble_can_preserve_dedicated_melanoma_head():
    payload = {
        "base_models": ["contour", "rgb"],
        "decision_policy": {
            "melanoma_source": "base_model",
            "melanoma_model_index": 0,
        },
        "heads": {
            "higher_concern": {
                "coefficients": [1.0, 1.0],
                "intercept": 0.0,
                "rank_references": [[0.1, 0.5], [0.2, 0.6]],
            },
            "melanoma": {
                "coefficients": [0.0, 4.0],
                "intercept": 0.0,
                "rank_references": [[0.1, 0.5], [0.2, 0.6]],
            },
        },
    }
    contour = np.array([[0.2, 0.03], [0.7, 0.08]])
    rgb = np.array([[0.3, 0.9], [0.9, 0.95]])
    scores = predict_rank_ensemble(payload, [contour, rgb])
    assert np.allclose(scores[:, 1], contour[:, 1])
