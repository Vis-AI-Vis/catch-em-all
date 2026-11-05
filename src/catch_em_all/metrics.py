"""Metrics derived from a closed-set classifier's logits."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PredictionMetrics:
    """Per-sample confidence and uncertainty statistics."""

    probabilities: np.ndarray
    top_indices: np.ndarray
    top_probabilities: np.ndarray
    normalized_entropy: float
    top2_margin: float


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)
    exp = np.exp(shifted)
    return exp / exp.sum()


def prediction_metrics(logits: np.ndarray, top_k: int = 5) -> PredictionMetrics:
    """Compute top-k predictions, normalized entropy, and the top-2 margin."""

    values = np.asarray(logits, dtype=np.float64)
    if values.ndim != 1 or values.size < 2:
        raise ValueError("logits must be a one-dimensional array with at least two classes")
    if not 1 <= top_k <= values.size:
        raise ValueError("top_k must be between 1 and the number of classes")

    probabilities = _softmax(values)
    top_indices = np.argsort(probabilities)[::-1][:top_k]
    top_probabilities = probabilities[top_indices]
    entropy = -np.sum(probabilities * np.log(np.clip(probabilities, 1e-12, None)))
    normalized_entropy = float(entropy / np.log(probabilities.size))
    top2 = np.partition(probabilities, -2)[-2:]
    top2_margin = float(np.max(top2) - np.min(top2))

    return PredictionMetrics(
        probabilities=probabilities,
        top_indices=top_indices,
        top_probabilities=top_probabilities,
        normalized_entropy=normalized_entropy,
        top2_margin=top2_margin,
    )
