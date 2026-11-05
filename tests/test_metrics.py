import numpy as np
import pytest

from catch_em_all.metrics import prediction_metrics


def test_uniform_logits_have_maximum_entropy_and_zero_margin() -> None:
    result = prediction_metrics(np.zeros(4), top_k=2)
    assert result.normalized_entropy == pytest.approx(1.0)
    assert result.top2_margin == pytest.approx(0.0)
    assert result.top_probabilities.tolist() == pytest.approx([0.25, 0.25])


def test_confident_logits_have_low_entropy_and_positive_margin() -> None:
    result = prediction_metrics(np.array([12.0, 0.0, -2.0]), top_k=3)
    assert result.top_indices[0] == 0
    assert result.top_probabilities[0] > 0.99
    assert result.normalized_entropy < 0.01
    assert result.top2_margin > 0.99


def test_invalid_logits_are_rejected() -> None:
    with pytest.raises(ValueError):
        prediction_metrics(np.array([1.0]))
