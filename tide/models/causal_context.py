"""Strictly causal dilated temporal context encoder."""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F
from torch import nn


ContextCache = tuple[torch.Tensor, ...]


class _DilatedBlock(nn.Module):
    def __init__(self, hidden_dim: int, dilation: int) -> None:
        super().__init__()
        self.dilation = int(dilation)
        self.conv = nn.Conv1d(
            hidden_dim,
            hidden_dim,
            kernel_size=2,
            dilation=self.dilation,
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def initial_cache(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        return torch.zeros(
            batch_size,
            self.conv.in_channels,
            self.dilation,
            device=device,
            dtype=dtype,
        )

    def step(
        self,
        current: torch.Tensor,
        cache: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        expected = (current.shape[0], current.shape[1], self.dilation)
        if cache.shape != expected:
            raise ValueError(f"invalid causal cache shape {tuple(cache.shape)}; expected {expected}")
        previous = cache[:, :, 0]
        update = F.linear(previous, self.conv.weight[:, :, 0])
        update = update + F.linear(
            current,
            self.conv.weight[:, :, 1],
            self.conv.bias,
        )
        output = self.norm(current + F.gelu(update))
        next_cache = torch.cat((cache[:, :, 1:], current.unsqueeze(-1)), dim=-1)
        return output, next_cache


class CausalContextEncoder(nn.Module):
    """Kernel-2 TCN with dilations 1, 2, and 4 and a gated correction."""

    def __init__(
        self,
        hidden_dim: int,
        dilations: Sequence[int] = (1, 2, 4),
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            _DilatedBlock(hidden_dim, dilation) for dilation in dilations
        )
        self.correction = nn.Linear(hidden_dim, hidden_dim)
        nn.init.zeros_(self.correction.weight)
        nn.init.zeros_(self.correction.bias)
        self.gate_logit = nn.Parameter(torch.tensor(-4.0))

    def initial_state(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> ContextCache:
        return tuple(
            block.initial_cache(batch_size, device, dtype) for block in self.blocks
        )

    def step(
        self,
        current: torch.Tensor,
        state: ContextCache,
    ) -> tuple[torch.Tensor, ContextCache, dict[str, torch.Tensor]]:
        value = current
        next_state: list[torch.Tensor] = []
        for block, cache in zip(self.blocks, state):
            value, next_cache = block.step(value, cache)
            next_state.append(next_cache)
        correction = torch.tanh(self.correction(value))
        gate = torch.sigmoid(self.gate_logit)
        return current + gate * correction, tuple(next_state), {
            "context_gate": gate,
            "context_correction_l2": correction.square().mean().sqrt(),
        }


class NoContextEncoder(nn.Module):
    """Identity context used by the public context ablation."""

    def initial_state(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[()]:
        del batch_size, device, dtype
        return ()

    def step(
        self,
        current: torch.Tensor,
        state: tuple[()],
    ) -> tuple[torch.Tensor, tuple[()], dict[str, torch.Tensor]]:
        del state
        zero = current.new_zeros(())
        return current, (), {
            "context_gate": zero,
            "context_correction_l2": zero,
        }


__all__ = ["CausalContextEncoder", "ContextCache", "NoContextEncoder"]

