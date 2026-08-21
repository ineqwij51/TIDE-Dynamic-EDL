"""Emotion-query readout for global intensity and stable composition."""

from __future__ import annotations

import math

import torch
from torch import nn


EPSILON = 1e-6


class EmotionQueryReadout(nn.Module):
    """Read nine emotion compositions from electrode tokens."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.arousal = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1),
            nn.Softplus(),
        )
        # Retained for strict compatibility with the frozen state dictionary.
        self.simple = nn.Linear(hidden_dim, 9)
        self.queries = nn.Parameter(torch.empty(9, hidden_dim))
        nn.init.normal_(self.queries, std=0.02)
        self.key = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.value = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.query_out = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1),
        )
        self.relation = None

    def forward(
        self,
        electrode_tokens: torch.Tensor,
        pooled: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        if electrode_tokens.ndim != 3 or electrode_tokens.shape[1] != 30:
            raise ValueError("emotion readout expects [batch, 30, hidden] tokens")
        if pooled.ndim != 2:
            raise ValueError("emotion readout expects [batch, hidden] pooled state")
        arousal = self.arousal(pooled).squeeze(-1) + EPSILON
        keys = self.key(electrode_tokens)
        values = self.value(electrode_tokens)
        scores = torch.einsum("kd,bcd->bkc", self.queries, keys) / math.sqrt(
            keys.shape[-1]
        )
        weights = torch.softmax(scores, dim=-1)
        emotion_tokens = (
            torch.einsum("bkc,bcd->bkd", weights, values)
            + self.queries.unsqueeze(0)
        )
        logits = self.query_out(emotion_tokens).squeeze(-1)
        composition = torch.softmax(logits, dim=-1)
        energy = arousal.unsqueeze(-1) * composition
        return arousal, composition, energy, {
            "emotion_tokens": emotion_tokens,
            "emotion_electrode_attention": weights,
        }


__all__ = ["EPSILON", "EmotionQueryReadout"]

