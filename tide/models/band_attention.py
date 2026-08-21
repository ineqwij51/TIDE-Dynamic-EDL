"""Band attention, adaptive electrode graph, and electrode readout."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from tide.models.electrode_graph import ELECTRODE_ORDER, electrode_adjacency


class AdaptiveElectrodeGraph(nn.Module):
    """Geometry-prior graph with a learned symmetric adjacency residual."""

    def __init__(self, hidden_dim: int, *, enabled: bool = True) -> None:
        super().__init__()
        prior = torch.as_tensor(
            electrode_adjacency(ELECTRODE_ORDER),
            dtype=torch.float32,
        )
        self.register_buffer("geometry_prior", prior, persistent=True)
        self.delta_adjacency = nn.Parameter(torch.zeros_like(prior))
        self.layers = nn.ModuleList((nn.Linear(hidden_dim, hidden_dim),))
        self.norms = nn.ModuleList((nn.LayerNorm(hidden_dim),))
        self.enabled = bool(enabled)
        self.epsilon = 1e-12

    def adjacency(self) -> torch.Tensor:
        residual = 0.5 * (
            self.delta_adjacency + self.delta_adjacency.transpose(0, 1)
        )
        logits = torch.log(self.geometry_prior.clamp_min(self.epsilon)) + residual
        return torch.softmax(logits, dim=-1)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 3 or tokens.shape[1] != 30:
            raise ValueError("electrode graph expects [batch, 30, hidden]")
        if not self.enabled:
            return tokens
        value = tokens
        adjacency = self.adjacency().to(tokens)
        for projection, normalization in zip(self.layers, self.norms):
            message = torch.einsum("cd,bdf->bcf", adjacency, value)
            value = normalization(value + F.gelu(projection(message)))
        return value

    def drift_l2(self) -> torch.Tensor:
        return (self.adjacency() - self.geometry_prior).square().mean().sqrt()


class ElectrodeAttention(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.score = nn.Linear(hidden_dim, 1)

    def forward(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        weights = torch.softmax(self.score(tokens).squeeze(-1), dim=-1)
        pooled = torch.einsum("bc,bcd->bd", weights, tokens)
        return pooled, weights


class BandAttentionEncoder(nn.Module):
    """Encode the five DE bands of each electrode into one spatial token."""

    def __init__(self, hidden_dim: int = 48, *, graph_enabled: bool = True) -> None:
        super().__init__()
        self.descriptor_projection = nn.Linear(1, hidden_dim)
        self.band_embedding = nn.Parameter(torch.empty(5, hidden_dim))
        # Two rows are retained so frozen scientific checkpoints load strictly.
        self.type_embedding = nn.Parameter(torch.empty(2, hidden_dim))
        nn.init.normal_(self.band_embedding, std=0.02)
        nn.init.normal_(self.type_embedding, std=0.02)
        self.band_score = nn.Linear(hidden_dim, 1)
        self.graph = AdaptiveElectrodeGraph(hidden_dim, enabled=graph_enabled)
        self.pool = ElectrodeAttention(hidden_dim)

    @staticmethod
    def _canonical_descriptors(features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 3 or features.shape[1] != 30:
            raise ValueError("band encoder expects [batch, 30, bands]")
        if features.shape[-1] == 5:
            differential_entropy = features
        elif features.shape[-1] == 10:
            differential_entropy = features[..., 5:]
        else:
            raise ValueError("TIDE expects five DE bands or canonical ten-column features")
        band_power = torch.zeros_like(differential_entropy)
        return torch.stack((band_power, differential_entropy), dim=-1)

    def forward(
        self,
        features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        descriptor = self._canonical_descriptors(features)
        tokens = self.descriptor_projection(descriptor.unsqueeze(-1))
        tokens = tokens + self.band_embedding.view(1, 1, 5, 1, -1)
        tokens = tokens + self.type_embedding.view(1, 1, 1, 2, -1)
        tokens = F.gelu(tokens)
        descriptor_scores = self.band_score(tokens).squeeze(-1)
        descriptor_scores = descriptor_scores.masked_fill(
            torch.tensor([True, False], device=features.device).view(1, 1, 1, 2),
            -torch.inf,
        )
        descriptor_weights = torch.softmax(
            descriptor_scores.flatten(-2), dim=-1
        ).reshape_as(descriptor_scores)
        electrode_tokens = torch.einsum(
            "bckt,bcktd->bcd", descriptor_weights, tokens
        )
        electrode_tokens = self.graph(electrode_tokens)
        pooled, electrode_weights = self.pool(electrode_tokens)
        return electrode_tokens, pooled, {
            "band_descriptor_weights": descriptor_weights,
            "electrode_weights": electrode_weights,
            "electrode_adjacency_drift_l2": self.graph.drift_l2(),
        }


class BypassElectrodeGraph(nn.Module):
    """A parameter-free graph bypass used by the public graph ablation."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("_zero", torch.zeros(()), persistent=False)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return tokens

    def drift_l2(self) -> torch.Tensor:
        return self._zero


__all__ = [
    "AdaptiveElectrodeGraph",
    "BandAttentionEncoder",
    "BypassElectrodeGraph",
    "ElectrodeAttention",
]

