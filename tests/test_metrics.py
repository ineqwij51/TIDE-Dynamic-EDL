from __future__ import annotations

import numpy as np

from tide.metrics.distribution import batch_distribution_metrics
from tide.metrics.dynamic import (
    batch_dynamic_metrics,
    static_aware_pcc,
    static_aware_srcc,
)


def test_static_aware_association_rules() -> None:
    assert static_aware_pcc([1, 1, 1], [2, 2, 2]) == 1.0
    assert static_aware_srcc([1, 1, 1], [1, 2, 3]) == 0.0
    assert np.isclose(static_aware_pcc([1, 2, 3], [3, 2, 1]), -1.0)
    assert np.isclose(static_aware_srcc([1, 2, 3], [3, 2, 1]), -1.0)


def test_dynamic_metric_identity() -> None:
    curve = np.linspace(0.1, 0.9, 20, dtype=np.float64)[None, :]
    metrics = batch_dynamic_metrics(curve, curve)
    assert np.isclose(metrics["StaticAware_SRCC"][0], 1.0)
    assert np.isclose(metrics["StaticAware_PCC"][0], 1.0)
    assert np.isclose(metrics["ZMAE"][0], 0.0)
    assert np.isclose(metrics["TimeAugmented_Discrete_Frechet"][0], 0.0)


def test_distribution_metric_identity() -> None:
    distribution = np.array([[0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]])
    metrics = batch_distribution_metrics(distribution, distribution)
    for name in ("Chebyshev", "Clark", "Canberra", "KL"):
        assert np.isclose(metrics[name][0], 0.0)
    assert np.isclose(metrics["Cosine"][0], 1.0)
    assert np.isclose(metrics["Intersection"][0], 1.0)

