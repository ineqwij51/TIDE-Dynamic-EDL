"""Stable public TIDE model."""

from __future__ import annotations

from typing import Any, Mapping, TypeAlias

import torch
from torch import nn

from tide.models.band_attention import BandAttentionEncoder, BypassElectrodeGraph
from tide.models.causal_context import CausalContextEncoder, NoContextEncoder
from tide.models.distribution_decoder import (
    DistributionDecoder,
    ZeroContextInnovationDecoder,
    ZeroDistributionDecoder,
    compose_distribution,
    normalize_energy,
)
from tide.models.emotion_query import EmotionQueryReadout
from tide.models.temporal_fusion import MeanSequenceReadout, TemporalIntensityReadout


ABLATION_NAMES = (
    "no_context",
    "no_distribution_decoder",
    "no_partial_emotion_learning",
    "no_adaptive_graph",
    "mean_sequence_readout",
)

TIDEState: TypeAlias = dict[str, Any]


class TIDE(nn.Module):
    """Causal dynamic emotion distribution learning from five-band DE features.

    Parameters
    ----------
    hidden_dim:
        Frozen paper width. The released configuration uses 48.
    residual_scale:
        Bound applied to the causal distribution residual.
    ablation:
        Optional public ablation name from :data:`ABLATION_NAMES`.
    """

    required_inputs = ("feat_seq",)

    def __init__(
        self,
        hidden_dim: int = 48,
        *,
        residual_scale: float = 0.25,
        ablation: str | None = None,
    ) -> None:
        super().__init__()
        if hidden_dim < 16 or hidden_dim % 4:
            raise ValueError("hidden_dim must be a multiple of four and at least 16")
        if ablation is not None and ablation not in ABLATION_NAMES:
            raise ValueError(f"unknown TIDE ablation {ablation!r}")
        self.hidden_dim = int(hidden_dim)
        self.ablation = ablation

        self.encoder = BandAttentionEncoder(hidden_dim, graph_enabled=True)
        self.context: nn.Module = (
            NoContextEncoder()
            if ablation == "no_context"
            else CausalContextEncoder(hidden_dim)
        )
        self.head = EmotionQueryReadout(hidden_dim)
        self.context_token_norm = nn.LayerNorm(hidden_dim)
        # This inactive tensor block is retained solely for strict checkpoint loading.
        self.composition_adapter = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.fusion = None
        self.predictive = None
        self.residual: nn.Module = DistributionDecoder(
            hidden_dim,
            residual_scale=residual_scale,
        )
        self.emotion_evidence = None
        self.macro_attention: nn.Module = TemporalIntensityReadout(hidden_dim)

        if ablation == "no_context":
            self.residual = ZeroContextInnovationDecoder(self.residual)
        elif ablation == "no_distribution_decoder":
            self.residual = ZeroDistributionDecoder()
        elif ablation == "no_adaptive_graph":
            self.encoder.graph = BypassElectrodeGraph()
        elif ablation == "mean_sequence_readout":
            self.macro_attention = MeanSequenceReadout()

    @staticmethod
    def _validate_features(features: torch.Tensor, *, sequence: bool) -> None:
        rank = 4 if sequence else 3
        if features.ndim != rank:
            raise ValueError(f"TIDE expects a rank-{rank} feature tensor")
        if features.shape[-2] != 30 or features.shape[-1] not in {5, 10}:
            shape = "[batch, seconds, 30, 5]" if sequence else "[batch, 30, 5]"
            raise ValueError(f"TIDE expects {shape}; canonical ten-column input is also accepted")
        if not torch.is_floating_point(features):
            raise TypeError("TIDE features must be floating point")

    def _encode_current(
        self,
        features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        tokens, global_token, auxiliary = self.encoder(features)
        token_map = tokens.unsqueeze(2).expand(-1, -1, 5, -1)
        composition_token = global_token
        dynamic_token = global_token
        auxiliary = {
            **auxiliary,
            "band_electrode_tokens": token_map,
            "band_tokens": token_map.mean(1),
            "composition_token": composition_token,
            "dynamic_token": dynamic_token,
        }
        return tokens, observation_or_global(global_token), composition_token, auxiliary

    def initial_state(
        self,
        batch_size: int,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> TIDEState:
        device = torch.device(device)
        zeros = torch.zeros(
            batch_size,
            self.hidden_dim,
            device=device,
            dtype=dtype,
        )
        return {
            "context": self.context.initial_state(batch_size, device, dtype),
            "previous_context": zeros,
            "previous_observation": zeros.clone(),
            "step_index": 0,
        }

    def step(
        self,
        feat_1s: torch.Tensor,
        state: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Run one causal second and return the updated streaming state."""

        self._validate_features(feat_1s, sequence=False)
        tokens, observation, _composition_token, encoder_aux = self._encode_current(
            feat_1s
        )
        dynamic_token = encoder_aux["dynamic_token"]
        _, instant_composition, instant_energy, instant_aux = self.head(
            tokens,
            observation,
        )
        first = int(state["step_index"]) == 0
        prediction = torch.zeros_like(dynamic_token)
        innovation = (
            torch.zeros_like(dynamic_token)
            if first
            else dynamic_token - state["previous_observation"]
        )
        context, context_state, context_aux = self.context.step(
            observation,
            state["context"],
        )
        stable_state = context
        context_tokens = self.context_token_norm(
            tokens + (stable_state - observation).unsqueeze(1)
        )
        stable_arousal, stable_composition, base_energy, stable_aux = self.head(
            context_tokens,
            stable_state,
        )
        context_innovation = (
            torch.zeros_like(context)
            if first
            else context - state["previous_context"]
        )
        residual = self.residual(context, context_innovation, innovation)
        final_composition = compose_distribution(stable_composition, residual)
        final_energy = stable_arousal.unsqueeze(-1) * final_composition
        next_state: TIDEState = {
            "context": context_state,
            "previous_context": context,
            "previous_observation": dynamic_token,
            "step_index": int(state["step_index"]) + 1,
        }
        emotion_weights = tokens.new_zeros((tokens.shape[0], 9, 150))
        return {
            "instant_energy": instant_energy,
            "instant_pred_dist": instant_composition,
            "energy": final_energy,
            "pred_dist": final_composition,
            "base_energy": base_energy,
            "stable_composition": stable_composition,
            "dynamic_residual": residual,
            "second_repr": context,
            "prediction": prediction,
            "prediction_target": dynamic_token,
            "innovation": innovation,
            "state": next_state,
            "aux": {
                **encoder_aux,
                **{f"instant_{key}": value for key, value in instant_aux.items()},
                **stable_aux,
                **context_aux,
                "emotion_query_evidence_weights": emotion_weights,
            },
        }

    @staticmethod
    def _stack_auxiliary(
        step_outputs: list[dict[str, Any]],
    ) -> dict[str, torch.Tensor]:
        if not step_outputs:
            return {}
        keys = set.intersection(*(set(step["aux"]) for step in step_outputs))
        result: dict[str, torch.Tensor] = {}
        batch_size = step_outputs[0]["energy"].shape[0]
        for key in keys:
            values = [step["aux"][key] for step in step_outputs]
            if not all(torch.is_tensor(value) for value in values):
                continue
            first = values[0]
            if first.ndim > 0 and first.shape[0] == batch_size:
                result[key] = torch.stack(values, dim=1)
            elif first.numel() == 1:
                result[key] = torch.stack(
                    [value.reshape(()) for value in values]
                ).mean()
        return result

    def forward_sequence(self, feat_seq: torch.Tensor) -> dict[str, Any]:
        """Predict second-wise and sequence-level emotion distributions."""

        self._validate_features(feat_seq, sequence=True)
        if feat_seq.shape[1] < 1:
            raise ValueError("TIDE requires at least one second")
        state = self.initial_state(
            feat_seq.shape[0],
            feat_seq.device,
            feat_seq.dtype,
        )
        steps: list[dict[str, Any]] = []
        for second in range(feat_seq.shape[1]):
            output = self.step(feat_seq[:, second], state)
            state = output["state"]
            steps.append(output)

        def stack(name: str) -> torch.Tensor:
            return torch.stack([step[name] for step in steps], dim=1)

        instant_energy = stack("instant_energy")
        final_energy = stack("energy")
        base_energy = stack("base_energy")
        stable_composition = stack("stable_composition")
        context = stack("second_repr")
        pred_dist_t, weights, pool_aux = self.macro_attention(
            context,
            base_energy,
            stable_composition,
        )
        instant_pred_dist_t, instant_weights, _ = self.macro_attention.instant(
            stack("prediction_target"),
            instant_energy,
        )
        return {
            "instant_energy_seq": instant_energy,
            "instant_pred_dist_seq": normalize_energy(instant_energy),
            "energy_seq": final_energy,
            "pred_dist_seq": normalize_energy(final_energy),
            "pred_dist_T": pred_dist_t,
            "aggregation_weights": weights,
            "stable_composition_seq": stable_composition,
            "dynamic_residual_seq": stack("dynamic_residual"),
            "base_energy_seq": base_energy,
            "macro_source_energy_seq": base_energy,
            "second_repr": context,
            "prediction_seq": stack("prediction"),
            "prediction_target_seq": stack("prediction_target"),
            "innovation_seq": stack("innovation"),
            "aux": {
                **self._stack_auxiliary(steps),
                **pool_aux,
                "instant_pred_dist_T": instant_pred_dist_t,
                "instant_aggregation_weights": instant_weights,
            },
        }

    def forward(self, feat_seq: torch.Tensor) -> dict[str, Any]:
        return self.forward_sequence(feat_seq)


def observation_or_global(global_token: torch.Tensor) -> torch.Tensor:
    """Name the frozen single-token observation path explicitly."""

    return global_token


__all__ = ["ABLATION_NAMES", "TIDE", "TIDEState"]

