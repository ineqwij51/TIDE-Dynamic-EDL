"""Causal residual decoder for second-wise emotion distributions."""

from __future__ import annotations

import torch
from torch import nn

from tide.models.emotion_query import EPSILON


def normalize_energy(energy: torch.Tensor) -> torch.Tensor:
    return energy / energy.sum(dim=-1, keepdim=True).clamp_min(EPSILON)


def compose_distribution(
    stable: torch.Tensor,
    residual: torch.Tensor,
) -> torch.Tensor:
    return torch.softmax(
        torch.log(stable.clamp_min(EPSILON)) + residual,
        dim=-1,
    )


class DistributionDecoder(nn.Module):
    """Decode a bounded, zero-mean causal residual over nine emotions."""

    def __init__(self, hidden_dim: int, *, residual_scale: float = 0.25) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(3 * hidden_dim),
            nn.Linear(3 * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 9),
        )
        self.residual_scale = float(residual_scale)

    def forward(
        self,
        context: torch.Tensor,
        context_innovation: torch.Tensor,
        observation_innovation: torch.Tensor,
    ) -> torch.Tensor:
        raw = self.residual_scale * torch.tanh(
            self.network(
                torch.cat(
                    (context, context_innovation, observation_innovation),
                    dim=-1,
                )
            )
        )
        return raw - raw.mean(dim=-1, keepdim=True)


class ZeroContextInnovationDecoder(nn.Module):
    def __init__(self, decoder: DistributionDecoder) -> None:
        super().__init__()
        self.decoder = decoder

    def forward(
        self,
        context: torch.Tensor,
        context_innovation: torch.Tensor,
        observation_innovation: torch.Tensor,
    ) -> torch.Tensor:
        del context_innovation
        return self.decoder(
            context,
            torch.zeros_like(context),
            observation_innovation,
        )


class ZeroDistributionDecoder(nn.Module):
    def forward(
        self,
        context: torch.Tensor,
        context_innovation: torch.Tensor,
        observation_innovation: torch.Tensor,
    ) -> torch.Tensor:
        del context_innovation, observation_innovation
        return context.new_zeros((*context.shape[:-1], 9))


__all__ = [
    "DistributionDecoder",
    "ZeroContextInnovationDecoder",
    "ZeroDistributionDecoder",
    "compose_distribution",
    "normalize_energy",
]

