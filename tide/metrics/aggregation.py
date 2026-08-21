"""Subject-trial, fold, and seed aggregation."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd


IDENTITY_COLUMNS = ("method", "fold", "seed", "subject_group", "trial_group")


def subject_trial_means(
    window_metrics: pd.DataFrame,
    metric_columns: Sequence[str],
) -> pd.DataFrame:
    required = set(IDENTITY_COLUMNS) | set(metric_columns)
    missing = sorted(required - set(window_metrics.columns))
    if missing:
        raise ValueError(f"window metrics are missing columns: {missing}")
    return (
        window_metrics.groupby(list(IDENTITY_COLUMNS), sort=False)[list(metric_columns)]
        .mean()
        .reset_index()
    )


def fold_seed_means(
    subject_trial_metrics: pd.DataFrame,
    metric_columns: Sequence[str],
) -> pd.DataFrame:
    return (
        subject_trial_metrics.groupby(["method", "fold", "seed"], sort=False)[
            list(metric_columns)
        ]
        .mean()
        .reset_index()
    )


def three_seed_summary(
    subject_trial_metrics: pd.DataFrame,
    metric_columns: Sequence[str],
) -> pd.DataFrame:
    per_seed = (
        subject_trial_metrics.groupby(["method", "seed"], sort=False)[
            list(metric_columns)
        ]
        .mean()
        .reset_index()
    )
    rows: list[dict[str, object]] = []
    for method, group in per_seed.groupby("method", sort=False):
        for metric in metric_columns:
            rows.append(
                {
                    "method": method,
                    "metric": metric,
                    "mean": float(group[metric].mean()),
                    "standard_deviation": float(group[metric].std(ddof=1)),
                    "n_seeds": int(group["seed"].nunique()),
                }
            )
    return pd.DataFrame(rows)


__all__ = ["fold_seed_means", "subject_trial_means", "three_seed_summary"]

