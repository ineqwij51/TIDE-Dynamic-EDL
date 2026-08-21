"""Checkpoint evaluation on public processed splits."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from tide.data import make_fold_datasets
from tide.metrics import (
    DISTRIBUTION_METRICS,
    DYNAMIC_METRICS,
    batch_distribution_metrics,
    batch_dynamic_metrics,
)
from tide.training.checkpoint import load_checkpoint


def _state_checksum(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        array = value.detach().cpu().contiguous().numpy()
        digest.update(name.encode("utf-8"))
        digest.update(array.tobytes())
    return digest.hexdigest()


@torch.inference_mode()
def evaluate_loader(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    method: str,
    fold: int,
    seed: int,
) -> pd.DataFrame:
    model.eval()
    predictions: list[np.ndarray] = []
    sequence_distributions: list[np.ndarray] = []
    target_intensity: list[np.ndarray] = []
    target_distributions: list[np.ndarray] = []
    target_emotions: list[np.ndarray] = []
    metadata = {name: [] for name in ("subject_group", "dyad_group", "trial_group", "window_start")}
    for batch in loader:
        features = batch["features"].to(device, non_blocking=True)
        output = model(features)
        predictions.append(output["pred_dist_seq"].detach().float().cpu().numpy())
        sequence_distributions.append(output["pred_dist_T"].detach().float().cpu().numpy())
        target_intensity.append(batch["target_intensity"].numpy())
        target_distributions.append(batch["target_distribution"].numpy())
        target_emotions.append(batch["target_emotion"].numpy())
        for name in metadata:
            metadata[name].append(batch[name].numpy())
    pred_seq = np.concatenate(predictions)
    pred_dist = np.concatenate(sequence_distributions)
    intensity = np.concatenate(target_intensity)
    distribution = np.concatenate(target_distributions)
    emotion = np.concatenate(target_emotions).astype(np.int64)
    target_share = pred_seq[
        np.arange(pred_seq.shape[0])[:, None],
        np.arange(pred_seq.shape[1])[None, :],
        emotion[:, None],
    ]
    dynamic = batch_dynamic_metrics(target_share, intensity)
    label_distribution = batch_distribution_metrics(pred_dist, distribution)
    frame = pd.DataFrame(
        {
            "method": method,
            "fold": int(fold),
            "seed": int(seed),
            **{name: np.concatenate(values) for name, values in metadata.items()},
            "target_emotion": emotion,
            **dynamic,
            **label_distribution,
        }
    )
    return frame


def evaluate_checkpoint(
    checkpoint_path: str | Path,
    *,
    data_root: str | Path,
    split: str = "test",
    device_name: str = "cpu",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if split not in {"validation", "test"}:
        raise ValueError("evaluation split must be validation or test")
    device = torch.device(device_name)
    model, payload = load_checkpoint(checkpoint_path, device=device)
    fold = int(payload.get("fold", -1))
    seed = int(payload.get("seed", -1))
    if fold not in range(5) or seed < 0:
        raise ValueError("checkpoint must record fold and seed")
    config = payload.get("config")
    if not isinstance(config, dict) or "data" not in config or "training" not in config:
        raise ValueError("checkpoint does not contain a public training configuration")
    mean = payload.get("feature_mean")
    std = payload.get("feature_std")
    if not torch.is_tensor(mean) or not torch.is_tensor(std):
        raise ValueError("checkpoint is missing train-fold normalization")
    datasets, _, data_audit = make_fold_datasets(
        data_root,
        fold,
        window_seconds=int(config["data"]["window_seconds"]),
        train_stride=int(config["data"]["train_stride"]),
        evaluation_stride=int(config["data"]["evaluation_stride"]),
        normalization=(mean.cpu().numpy(), std.cpu().numpy()),
    )
    loader = DataLoader(
        datasets[split],
        batch_size=int(config["training"]["evaluation_batch_size"]),
        shuffle=False,
        num_workers=int(config["training"].get("workers", 0)),
        pin_memory=device.type == "cuda",
    )
    before = _state_checksum(model)
    frame = evaluate_loader(
        model,
        loader,
        device,
        method="TIDE" if payload.get("ablation") is None else str(payload["ablation"]),
        fold=fold,
        seed=seed,
    )
    after = _state_checksum(model)
    if before != after:
        raise RuntimeError("model state changed during inference")
    return frame, {
        "checkpoint": str(checkpoint_path),
        "fold": fold,
        "seed": seed,
        "split": split,
        "n_windows": len(frame),
        "state_unchanged": True,
        "normalization_scope": "unique training-fold seconds",
        "test_data_used_for_normalization": False,
        "data": data_audit,
        "metrics": [*DYNAMIC_METRICS, *DISTRIBUTION_METRICS],
    }


__all__ = ["evaluate_checkpoint", "evaluate_loader"]

