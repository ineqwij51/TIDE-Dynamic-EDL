"""Audited 30-electrode geometry used by TIDE."""

from __future__ import annotations

from functools import lru_cache
import math
from typing import Sequence

import numpy as np


ELECTRODE_ORDER: tuple[str, ...] = (
    "FP1", "FP2", "F7", "F3", "FZ", "F4", "F8", "FT7", "FC3", "FCZ",
    "FC4", "FT8", "T3", "C3", "CZ", "C4", "T4", "TP7", "CP3", "CPZ",
    "CP4", "TP8", "T5", "P3", "PZ", "P4", "T6", "O1", "OZ", "O2",
)


@lru_cache(maxsize=2)
def _montage_lookup(name: str) -> dict[str, np.ndarray]:
    try:
        import mne
    except ImportError as exc:  # pragma: no cover - exercised by installation checks
        raise RuntimeError("MNE is required to construct the TIDE electrode prior") from exc
    montage = mne.channels.make_standard_montage(name)
    positions = montage.get_positions()["ch_pos"]
    return {key.strip().upper(): np.asarray(value, dtype=np.float64) for key, value in positions.items()}


def electrode_coordinates(
    channel_order: Sequence[str] = ELECTRODE_ORDER,
) -> np.ndarray:
    standard_1020 = _montage_lookup("standard_1020")
    standard_1005 = _montage_lookup("standard_1005")
    coordinates: list[np.ndarray] = []
    missing: list[str] = []
    for channel in channel_order:
        normalized = str(channel).strip().upper()
        coordinate = standard_1020.get(normalized)
        if coordinate is None:
            coordinate = standard_1005.get(normalized)
        if coordinate is None:
            missing.append(str(channel))
            coordinates.append(np.full(3, math.nan, dtype=np.float64))
        else:
            coordinates.append(coordinate)
    if missing:
        raise ValueError(f"unmatched electrodes: {missing}")
    return np.asarray(coordinates, dtype=np.float64)


def electrode_adjacency(
    channel_order: Sequence[str] = ELECTRODE_ORDER,
    *,
    neighbours: int = 6,
) -> np.ndarray:
    """Return the frozen row-normalized distance-RBF graph prior."""

    coordinates = electrode_coordinates(channel_order)
    difference = coordinates[:, None, :] - coordinates[None, :, :]
    distance = np.sqrt(np.sum(difference * difference, axis=-1, dtype=np.float64))
    finite = distance[np.isfinite(distance) & (distance > 0)]
    sigma = max(float(np.median(finite)) if finite.size else 1.0, 1e-6)
    adjacency = np.exp(-np.square(distance) / (2.0 * sigma * sigma))
    k = max(1, min(int(neighbours), adjacency.shape[0] - 1))
    keep = np.zeros_like(adjacency, dtype=bool)
    for index in range(adjacency.shape[0]):
        nearest = np.argsort(distance[index])[: k + 1]
        keep[index, nearest] = True
    adjacency = np.where(np.logical_or(keep, keep.T), adjacency, 0.0)
    np.fill_diagonal(adjacency, 1.0)
    row_sum = adjacency.sum(axis=1, keepdims=True)
    return (adjacency / np.clip(row_sum, 1e-12, None)).astype(np.float32)


__all__ = ["ELECTRODE_ORDER", "electrode_adjacency", "electrode_coordinates"]

