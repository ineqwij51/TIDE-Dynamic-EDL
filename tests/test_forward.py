from __future__ import annotations

import inspect

import torch

from tide.models import TIDE


def test_synthetic_forward_and_public_signature() -> None:
    torch.manual_seed(3)
    model = TIDE().eval()
    features = torch.randn(2, 6, 30, 5)
    with torch.inference_mode():
        output = model(features)
    assert output["pred_dist_seq"].shape == (2, 6, 9)
    assert output["pred_dist_T"].shape == (2, 9)
    assert output["energy_seq"].shape == (2, 6, 9)
    assert output["aggregation_weights"].shape == (2, 6)
    assert torch.all(output["energy_seq"] > 0)
    torch.testing.assert_close(
        output["pred_dist_seq"].sum(-1),
        torch.ones(2, 6),
        atol=1e-6,
        rtol=0.0,
    )
    torch.testing.assert_close(
        output["pred_dist_T"].sum(-1),
        torch.ones(2),
        atol=1e-6,
        rtol=0.0,
    )
    assert sum(parameter.numel() for parameter in model.parameters()) == 47778
    forbidden = {"target_emotion", "target_intensity", "subject", "trial", "dyad"}
    assert not (set(inspect.signature(TIDE.forward).parameters) & forbidden)


def test_canonical_ten_column_input_selects_same_de_features() -> None:
    torch.manual_seed(4)
    model = TIDE().eval()
    differential_entropy = torch.randn(1, 4, 30, 5)
    full = torch.cat((torch.randn_like(differential_entropy), differential_entropy), dim=-1)
    with torch.inference_mode():
        public = model(differential_entropy)
        canonical = model(full)
    for key in ("energy_seq", "pred_dist_seq", "pred_dist_T", "aggregation_weights"):
        torch.testing.assert_close(public[key], canonical[key], atol=0.0, rtol=0.0)

