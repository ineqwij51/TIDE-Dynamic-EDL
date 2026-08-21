"""Nine-dimensional label-distribution metrics."""

from __future__ import annotations

import numpy as np


EPSILON = 1e-12
DISTRIBUTION_METRICS = (
    "Chebyshev",
    "Clark",
    "Canberra",
    "KL",
    "Cosine",
    "Intersection",
)


def normalize_rows(value: np.ndarray) -> np.ndarray:
    array = np.clip(np.asarray(value, dtype=np.float64), 0.0, None)
    if array.ndim != 2 or array.shape[1] != 9:
        raise ValueError("distribution metrics require [N,9] arrays")
    row_sum = array.sum(axis=-1, keepdims=True)
    normalized = array / np.maximum(row_sum, EPSILON)
    normalized[row_sum.squeeze(-1) <= EPSILON] = 1.0 / 9.0
    return normalized


def batch_distribution_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
) -> dict[str, np.ndarray]:
    pred = normalize_rows(prediction)
    truth = normalize_rows(target)
    if pred.shape != truth.shape:
        raise ValueError("prediction and target distributions must align")
    difference = pred - truth
    return {
        "Chebyshev": np.abs(difference).max(axis=1),
        "Clark": np.sqrt(
            (np.square(difference) / (np.square(pred + truth) + EPSILON)).sum(axis=1)
        ),
        "Canberra": (
            np.abs(difference) / (pred + truth + EPSILON)
        ).sum(axis=1),
        "KL": (
            truth * (np.log(truth + EPSILON) - np.log(pred + EPSILON))
        ).sum(axis=1),
        "Cosine": (pred * truth).sum(axis=1)
        / (np.linalg.norm(pred, axis=1) * np.linalg.norm(truth, axis=1) + EPSILON),
        "Intersection": np.minimum(pred, truth).sum(axis=1),
    }


__all__ = [
    "DISTRIBUTION_METRICS",
    "EPSILON",
    "batch_distribution_metrics",
    "normalize_rows",
]

