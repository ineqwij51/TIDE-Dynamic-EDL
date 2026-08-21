from __future__ import annotations

import numpy as np

from tide.data import make_fold_datasets
from tide.data.splits import load_split_plan


def test_grouped_schedule_and_normalization_scope(synthetic_data_root) -> None:
    plan = load_split_plan(synthetic_data_root)
    assert len(plan) == 5
    for fold in plan:
        groups = [
            set(fold[name])
            for name in ("train_groups", "validation_groups", "test_groups")
        ]
        assert not (groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])
    datasets, (mean, std), audit = make_fold_datasets(synthetic_data_root, 0)
    assert datasets["train"][0]["features"].shape == (20, 30, 5)
    assert mean.shape == (30, 5)
    assert std.shape == (30, 5)
    assert np.all(std > 0)
    assert audit["test_data_used_for_normalization"] is False

