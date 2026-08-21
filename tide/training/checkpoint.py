"""Minimal checkpoint serialization with strict state loading."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from tide.models import TIDE


def save_checkpoint(
    path: str | Path,
    model: TIDE,
    *,
    config: Mapping[str, Any],
    fold: int,
    seed: int,
    epoch: int,
    validation_loss: float,
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": "tide.checkpoint.v1",
            "model_state_dict": model.state_dict(),
            "config": dict(config),
            "fold": int(fold),
            "seed": int(seed),
            "epoch": int(epoch),
            "validation_loss": float(validation_loss),
            "feature_mean": torch.as_tensor(feature_mean, dtype=torch.float32),
            "feature_std": torch.as_tensor(feature_std, dtype=torch.float32),
            "ablation": model.ablation,
        },
        target,
    )


def load_checkpoint(
    path: str | Path,
    *,
    device: str | torch.device = "cpu",
    model: TIDE | None = None,
) -> tuple[TIDE, dict[str, Any]]:
    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")
    payload = torch.load(
        checkpoint_path,
        map_location=torch.device(device),
        weights_only=False,
    )
    if not isinstance(payload, Mapping):
        raise ValueError("checkpoint payload must be a mapping")
    state = payload.get("model_state_dict", payload.get("state_dict"))
    if not isinstance(state, Mapping):
        raise ValueError("checkpoint does not contain a model state dictionary")
    resolved_model = model or TIDE(ablation=payload.get("ablation"))
    cleaned_state = {
        str(key).removeprefix("module."): value for key, value in state.items()
    }
    resolved_model.load_state_dict(cleaned_state, strict=True)
    resolved_model.to(device)
    return resolved_model, dict(payload)


__all__ = ["load_checkpoint", "save_checkpoint"]

