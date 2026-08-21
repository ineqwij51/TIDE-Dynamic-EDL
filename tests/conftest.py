from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tide.data.preprocessing import prepare_processed_data


@pytest.fixture()
def synthetic_data_root(tmp_path: Path) -> Path:
    random = np.random.default_rng(11)
    trials = 10
    seconds = 25
    features = random.normal(size=(trials, seconds, 30, 5)).astype(np.float32)
    intensity = random.uniform(0.0, 1.0, size=(trials, seconds)).astype(np.float32)
    distribution = random.dirichlet(np.ones(9), size=trials).astype(np.float32)
    feature_path = tmp_path / "features.npy"
    label_path = tmp_path / "labels.npz"
    np.save(feature_path, features)
    np.savez_compressed(
        label_path,
        target_intensity=intensity,
        distribution=distribution,
        target_emotion=np.arange(trials, dtype=np.int64) % 9,
        subject_group=np.arange(trials, dtype=np.int64),
        dyad_group=np.arange(trials, dtype=np.int64),
        trial_group=np.arange(trials, dtype=np.int64),
        group_stratum=np.arange(trials, dtype=np.int64) % 2,
        lengths=np.full(trials, seconds, dtype=np.int64),
    )
    output = tmp_path / "processed"
    prepare_processed_data(feature_path, label_path, output)
    return output

