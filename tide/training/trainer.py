"""One-fold TIDE training loop."""

from __future__ import annotations

from pathlib import Path
import time
from typing import Any, Mapping

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from tide.data import make_fold_datasets
from tide.losses import TIDELoss
from tide.models import TIDE
from tide.training.checkpoint import save_checkpoint
from tide.utils.io import write_json
from tide.utils.seed import dataloader_generator, seed_everything


def _move_batch(
    batch: Mapping[str, torch.Tensor],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def run_epoch(
    model: TIDE,
    loader: DataLoader,
    criterion: TIDELoss,
    device: torch.device,
    *,
    optimizer: torch.optim.Optimizer | None,
    max_gradient_norm: float,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals: dict[str, float] = {}
    sample_count = 0
    context = torch.enable_grad if training else torch.no_grad
    with context():
        for batch in loader:
            moved = _move_batch(batch, device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            output = model(moved["features"])
            loss, components = criterion(
                output,
                target_emotion=moved["target_emotion"],
                target_intensity=moved["target_intensity"],
                target_distribution=moved["target_distribution"],
            )
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_gradient_norm,
                )
                optimizer.step()
            batch_size = int(moved["features"].shape[0])
            sample_count += batch_size
            for key, value in components.items():
                totals[key] = totals.get(key, 0.0) + float(value) * batch_size
    if sample_count == 0:
        raise RuntimeError("empty training or validation loader")
    return {key: value / sample_count for key, value in totals.items()}


def run_training(
    config: Mapping[str, Any],
    *,
    data_root: str | Path,
    fold: int,
    seed: int,
    output_dir: str | Path,
    device_name: str = "cpu",
    ablation: str | None = None,
) -> dict[str, Any]:
    """Train one fold/seed and select by minimum validation total loss."""

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    seed_everything(int(seed), deterministic=bool(config["training"].get("deterministic", True)))
    torch.set_num_threads(int(config["training"].get("torch_threads", 4)))
    device = torch.device(device_name)
    data_config = config["data"]
    datasets, normalization, data_audit = make_fold_datasets(
        data_root,
        int(fold),
        window_seconds=int(data_config["window_seconds"]),
        train_stride=int(data_config["train_stride"]),
        evaluation_stride=int(data_config["evaluation_stride"]),
    )
    feature_mean, feature_std = normalization
    np.savez_compressed(
        destination / "normalization.npz",
        feature_mean=feature_mean,
        feature_std=feature_std,
    )
    batch_size = int(config["training"]["batch_size"])
    evaluation_batch_size = int(config["training"]["evaluation_batch_size"])
    train_loader = DataLoader(
        datasets["train"],
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=int(config["training"].get("workers", 0)),
        generator=dataloader_generator(int(seed)),
        pin_memory=device.type == "cuda",
    )
    validation_loader = DataLoader(
        datasets["validation"],
        batch_size=evaluation_batch_size,
        shuffle=False,
        num_workers=int(config["training"].get("workers", 0)),
        pin_memory=device.type == "cuda",
    )
    model = TIDE(
        hidden_dim=int(config["model"]["hidden_dim"]),
        residual_scale=float(config["model"]["residual_scale"]),
        ablation=ablation,
    ).to(device)
    criterion = TIDELoss(
        config["loss"],
        partial_emotion_learning=ablation != "no_partial_emotion_learning",
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)
    max_gradient_norm = float(config["training"]["max_gradient_norm"])
    epochs = int(config["training"]["epochs"])
    best_loss = float("inf")
    best_epoch = -1
    rows: list[dict[str, Any]] = []
    started = time.time()
    for epoch in range(epochs):
        train_log = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer=optimizer,
            max_gradient_norm=max_gradient_norm,
        )
        validation_log = run_epoch(
            model,
            validation_loader,
            criterion,
            device,
            optimizer=None,
            max_gradient_norm=max_gradient_norm,
        )
        score = float(validation_log["loss_total"])
        if not np.isfinite(score):
            raise FloatingPointError("validation loss is not finite")
        improved = score < best_loss - 1e-8
        if improved:
            best_loss = score
            best_epoch = epoch
            save_checkpoint(
                destination / "best_model.pt",
                model,
                config=config,
                fold=fold,
                seed=seed,
                epoch=epoch,
                validation_loss=score,
                feature_mean=feature_mean,
                feature_std=feature_std,
            )
        rows.append(
            {
                "epoch": epoch,
                "improved": improved,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                **{f"train_{key}": value for key, value in train_log.items()},
                **{
                    f"validation_{key}": value
                    for key, value in validation_log.items()
                },
            }
        )
        scheduler.step()
    if best_epoch < 0:
        raise RuntimeError("training did not produce a finite checkpoint")
    save_checkpoint(
        destination / "last_model.pt",
        model,
        config=config,
        fold=fold,
        seed=seed,
        epoch=epochs - 1,
        validation_loss=float(rows[-1]["validation_loss_total"]),
        feature_mean=feature_mean,
        feature_std=feature_std,
    )
    pd.DataFrame(rows).to_csv(destination / "epoch_metrics.csv", index=False)
    summary = {
        "status": "complete",
        "method": "TIDE",
        "ablation": ablation,
        "fold": int(fold),
        "seed": int(seed),
        "epochs": epochs,
        "best_epoch": int(best_epoch),
        "best_validation_loss": float(best_loss),
        "checkpoint_selection": "minimum validation loss_total",
        "runtime_seconds": float(time.time() - started),
        "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
        "data": data_audit,
    }
    write_json(destination / "run_summary.json", summary)
    return summary


__all__ = ["run_epoch", "run_training"]

