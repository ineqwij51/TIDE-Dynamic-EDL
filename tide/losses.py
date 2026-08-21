"""Core TIDE objective under partial emotion supervision."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class TIDELossConfig:
    """Weights of the frozen public objective."""

    lambda_instant_energy: float = 1.0
    lambda_final_energy: float = 1.0
    lambda_shape: float = 0.0
    lambda_bridge: float = 0.10
    lambda_macro: float = 0.50
    lambda_center: float = 0.0
    smooth_l1_beta: float = 0.10

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, object] | None,
    ) -> "TIDELossConfig":
        data = dict(value or {})
        unknown = sorted(set(data) - set(cls.__dataclass_fields__))
        if unknown:
            raise ValueError(f"unknown TIDE loss settings: {unknown}")
        result = cls(**data)
        for name, number in asdict(result).items():
            scalar = float(number)
            if not torch.isfinite(torch.tensor(scalar)) or scalar < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if result.lambda_shape != 0.0 or result.lambda_center != 0.0:
            raise ValueError("the public TIDE objective fixes shape and center weights at zero")
        return result


DEFAULT_LOSS_WEIGHTS = asdict(TIDELossConfig())


def gather_target(sequence: torch.Tensor, target_emotion: torch.Tensor) -> torch.Tensor:
    if sequence.ndim != 3 or sequence.shape[-1] != 9:
        raise ValueError("target gathering expects [batch, seconds, 9]")
    index = target_emotion.long().view(-1, 1, 1).expand(
        -1,
        sequence.shape[1],
        1,
    )
    return sequence.gather(-1, index).squeeze(-1)


def per_window_correlation_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    pred = prediction - prediction.mean(dim=1, keepdim=True)
    truth = target - target.mean(dim=1, keepdim=True)
    pred_variance = pred.square().mean(dim=1)
    truth_variance = truth.square().mean(dim=1)
    correlation = (pred * truth).mean(dim=1) / torch.sqrt(
        (pred_variance + epsilon) * (truth_variance + epsilon)
    )
    dynamic = 1.0 - correlation.clamp(-1.0, 1.0)
    static = pred_variance / (pred_variance.detach() + 1.0)
    return torch.where(truth_variance > epsilon, dynamic, static)


def distribution_kl(
    target: torch.Tensor,
    prediction: torch.Tensor,
    epsilon: float = 1e-8,
) -> torch.Tensor:
    truth = target.clamp_min(epsilon)
    pred = prediction.clamp_min(epsilon)
    truth = truth / truth.sum(-1, keepdim=True).clamp_min(epsilon)
    pred = pred / pred.sum(-1, keepdim=True).clamp_min(epsilon)
    return (truth * (truth.log() - pred.log())).sum(dim=-1).mean()


class TIDELoss(nn.Module):
    """Two intensity losses, partial-emotion bridge, and sequence-level KL."""

    def __init__(
        self,
        config: Mapping[str, object] | TIDELossConfig | None = None,
        *,
        partial_emotion_learning: bool = True,
    ) -> None:
        super().__init__()
        resolved = (
            config
            if isinstance(config, TIDELossConfig)
            else TIDELossConfig.from_mapping(config)
        )
        if not partial_emotion_learning:
            resolved = TIDELossConfig(
                **{
                    **asdict(resolved),
                    "lambda_bridge": 0.0,
                }
            )
        self.config = resolved

    def forward(
        self,
        output: Mapping[str, object],
        *,
        target_emotion: torch.Tensor,
        target_intensity: torch.Tensor,
        target_distribution: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        required = (
            "instant_energy_seq",
            "energy_seq",
            "pred_dist_seq",
            "pred_dist_T",
            "dynamic_residual_seq",
            "aggregation_weights",
        )
        tensors = {key: output[key] for key in required}
        if not all(torch.is_tensor(value) for value in tensors.values()):
            raise TypeError("TIDE loss requires tensor model outputs")

        target = target_intensity.float()
        instant_target = gather_target(
            tensors["instant_energy_seq"],
            target_emotion,
        )
        final_target = gather_target(tensors["energy_seq"], target_emotion)
        final_share = gather_target(tensors["pred_dist_seq"], target_emotion)
        loss_instant = F.smooth_l1_loss(
            instant_target,
            target,
            beta=self.config.smooth_l1_beta,
        )
        loss_final = F.smooth_l1_loss(
            final_target,
            target,
            beta=self.config.smooth_l1_beta,
        )
        loss_shape = per_window_correlation_loss(final_share, target).mean()
        target_vs_rest = torch.log(final_share.clamp_min(1e-8)) - torch.log(
            (1.0 - final_share).clamp_min(1e-8)
        )
        loss_bridge = per_window_correlation_loss(target_vs_rest, target).mean()
        loss_kl = distribution_kl(
            target_distribution.float(),
            tensors["pred_dist_T"],
        )
        weighted_residual = torch.einsum(
            "bt,btk->bk",
            tensors["aggregation_weights"],
            tensors["dynamic_residual_seq"],
        )
        loss_center = weighted_residual.square().mean()

        weighted = {
            "loss_energy_instant": self.config.lambda_instant_energy
            * loss_instant,
            "loss_energy_final": self.config.lambda_final_energy * loss_final,
            "loss_shape": self.config.lambda_shape * loss_shape,
            "loss_bridge": self.config.lambda_bridge * loss_bridge,
            "loss_kl": self.config.lambda_macro * loss_kl,
            "loss_center": self.config.lambda_center * loss_center,
        }
        total = sum(weighted.values())
        components = {
            "loss_total": float(total.detach()),
            "loss_energy": float((loss_instant + loss_final).detach()),
            "loss_energy_instant": float(loss_instant.detach()),
            "loss_energy_final": float(loss_final.detach()),
            "loss_shape": float(loss_shape.detach()),
            "loss_bridge": float(loss_bridge.detach()),
            "loss_kl": float(loss_kl.detach()),
            "loss_center": float(loss_center.detach()),
            **{
                f"weighted_{name}": float(value.detach())
                for name, value in weighted.items()
            },
        }
        if not torch.isfinite(total):
            raise FloatingPointError(f"non-finite TIDE loss: {components}")
        return total, components


__all__ = [
    "DEFAULT_LOSS_WEIGHTS",
    "TIDELoss",
    "TIDELossConfig",
    "distribution_kl",
    "gather_target",
    "per_window_correlation_loss",
]

