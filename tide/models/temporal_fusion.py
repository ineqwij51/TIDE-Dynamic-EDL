"""Temporal Intensity Fusion over stable class-energy trajectories."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from tide.models.distribution_decoder import normalize_energy


def _previous(value: torch.Tensor) -> torch.Tensor:
    return torch.cat((value[:, :1], value[:, :-1]), dim=1)


class _SingleAttentionPool(nn.Module):
    def __init__(self, hidden_dim: int, *, uniform_floor: float = 0.20) -> None:
        super().__init__()
        self.uniform_floor = float(uniform_floor)
        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim + 18, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        state: torch.Tensor,
        energy: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        seconds = energy.shape[1]
        log_energy = torch.log(energy.clamp_min(1e-8))
        features = torch.cat(
            (state, log_energy, (log_energy - _previous(log_energy)).abs()),
            dim=-1,
        ).detach()
        learned = torch.softmax(self.scorer(features).squeeze(-1), dim=1)
        weights = self.uniform_floor / seconds + (
            1.0 - self.uniform_floor
        ) * learned
        pooled_energy = torch.einsum("bt,btk->bk", weights, energy)
        return normalize_energy(pooled_energy), weights


class TemporalIntensityFusion(nn.Module):
    """Fuse uniform, sustained-salience, and change-salience energy views."""

    mode_names = (
        "uniform",
        "sustained_salience",
        "composition_change_salience",
    )

    def __init__(
        self,
        hidden_dim: int,
        *,
        temporal_uniform_floor: float = 0.20,
        gate_uniform_floor: float = 0.15,
    ) -> None:
        super().__init__()
        self.single = _SingleAttentionPool(
            hidden_dim,
            uniform_floor=temporal_uniform_floor,
        )
        self.sustained_scorer = nn.Sequential(
            nn.Linear(hidden_dim + 10, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.change_scorer = nn.Sequential(
            nn.Linear(19, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.mode_gate = nn.Sequential(
            nn.Linear(6, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 3),
        )
        nn.init.zeros_(self.mode_gate[-1].weight)
        nn.init.zeros_(self.mode_gate[-1].bias)
        self.temporal_uniform_floor = float(temporal_uniform_floor)
        self.gate_uniform_floor = float(gate_uniform_floor)

    def _temporal_weights(
        self,
        state: torch.Tensor,
        base_energy: torch.Tensor,
        stable_composition: torch.Tensor,
    ) -> torch.Tensor:
        batch, seconds = base_energy.shape[:2]
        uniform = base_energy.new_full((batch, seconds), 1.0 / seconds)
        arousal = base_energy.sum(dim=-1, keepdim=True)
        log_energy = torch.log(base_energy.clamp_min(1e-8))
        sustained_features = torch.cat(
            (state, log_energy, arousal), dim=-1
        ).detach()
        sustained_learned = torch.softmax(
            self.sustained_scorer(sustained_features).squeeze(-1), dim=1
        )
        sustained = self.temporal_uniform_floor / seconds + (
            1.0 - self.temporal_uniform_floor
        ) * sustained_learned
        change_features = torch.cat(
            (
                (log_energy - _previous(log_energy)).abs(),
                (stable_composition - _previous(stable_composition)).abs(),
                (arousal - _previous(arousal)).abs(),
            ),
            dim=-1,
        ).detach()
        change_learned = torch.softmax(
            self.change_scorer(change_features).squeeze(-1), dim=1
        )
        change = self.temporal_uniform_floor / seconds + (
            1.0 - self.temporal_uniform_floor
        ) * change_learned
        return torch.stack((uniform, sustained, change), dim=1)

    def _gate(
        self,
        base_energy: torch.Tensor,
        stable_composition: torch.Tensor,
    ) -> torch.Tensor:
        arousal = base_energy.sum(dim=-1)
        entropy = -(
            stable_composition.clamp_min(1e-8)
            * torch.log(stable_composition.clamp_min(1e-8))
        ).sum(-1)
        statistics = torch.stack(
            (
                arousal.mean(1),
                arousal.max(1).values,
                arousal.max(1).values / arousal.mean(1).clamp_min(1e-8),
                (base_energy - _previous(base_energy)).abs().mean(dim=(1, 2)),
                entropy.mean(1),
                (
                    entropy
                    - _previous(entropy.unsqueeze(-1)).squeeze(-1)
                ).abs().mean(1),
            ),
            dim=-1,
        ).detach()
        learned = torch.softmax(self.mode_gate(statistics), dim=-1)
        return self.gate_uniform_floor / 3.0 + (
            1.0 - self.gate_uniform_floor
        ) * learned

    def forward(
        self,
        state: torch.Tensor,
        base_energy: torch.Tensor,
        stable_composition: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        temporal = self._temporal_weights(
            state,
            base_energy,
            stable_composition,
        )
        mode_energy = torch.einsum("bmt,btk->bmk", temporal, base_energy)
        mode_distribution = normalize_energy(mode_energy)
        gate = self._gate(base_energy, stable_composition)
        pooled_energy = torch.einsum("bm,bmk->bk", gate, mode_energy)
        distribution = normalize_energy(pooled_energy)
        effective_weights = torch.einsum("bm,bmt->bt", gate, temporal)
        return distribution, effective_weights, {
            "mode_probabilities": gate,
            "mode_temporal_weights": temporal,
            "mode_distributions": mode_distribution,
            "mode_energies": mode_energy,
        }


class TemporalIntensityReadout(nn.Module):
    """Checkpoint-compatible public wrapper around Temporal Intensity Fusion."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.m3 = TemporalIntensityFusion(hidden_dim)
        # Retained only to make the frozen state dictionary a strict mapping.
        self.unified_scorer = nn.Sequential(
            nn.Linear(hidden_dim + 31, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.uniform_floor = 0.20

    def forward(
        self,
        state: torch.Tensor,
        base_energy: torch.Tensor,
        stable_composition: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        return self.m3(state, base_energy, stable_composition)

    def instant(
        self,
        state: torch.Tensor,
        energy: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        distribution, weights = self.m3.single(state, energy)
        pooled_energy = torch.einsum("bt,btk->bk", weights, energy)
        return distribution, weights, {
            "mode_probabilities": energy.new_ones((energy.shape[0], 1)),
            "mode_temporal_weights": weights.unsqueeze(1),
            "mode_distributions": distribution.unsqueeze(1),
            "mode_energies": pooled_energy.unsqueeze(1),
        }


class MeanSequenceReadout(nn.Module):
    """Arithmetic mean of base class intensity for the readout ablation."""

    def forward(
        self,
        state: torch.Tensor,
        base_energy: torch.Tensor,
        stable_composition: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        del state, stable_composition
        pooled_energy = base_energy.mean(dim=1)
        distribution = normalize_energy(pooled_energy)
        weights = base_energy.new_full(
            base_energy.shape[:2], 1.0 / base_energy.shape[1]
        )
        return distribution, weights, {
            "mode_probabilities": base_energy.new_ones((base_energy.shape[0], 1)),
            "mode_temporal_weights": weights.unsqueeze(1),
            "mode_distributions": distribution.unsqueeze(1),
            "mode_energies": pooled_energy.unsqueeze(1),
            "mean_sequence_readout": base_energy.new_ones(()),
        }

    def instant(
        self,
        state: torch.Tensor,
        energy: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        del state
        pooled_energy = energy.mean(dim=1)
        distribution = normalize_energy(pooled_energy)
        weights = energy.new_full(energy.shape[:2], 1.0 / energy.shape[1])
        return distribution, weights, {
            "mode_probabilities": energy.new_ones((energy.shape[0], 1)),
            "mode_temporal_weights": weights.unsqueeze(1),
            "mode_distributions": distribution.unsqueeze(1),
            "mode_energies": pooled_energy.unsqueeze(1),
        }


__all__ = [
    "MeanSequenceReadout",
    "TemporalIntensityFusion",
    "TemporalIntensityReadout",
]

