from __future__ import annotations

import torch

from tide.losses import TIDELoss
from tide.models import ABLATION_NAMES, TIDE


def test_all_public_ablations_preserve_output_contract() -> None:
    torch.manual_seed(21)
    features = torch.randn(1, 4, 30, 5)
    for name in ABLATION_NAMES:
        model = TIDE(ablation=name).eval()
        with torch.inference_mode():
            output = model(features)
        assert output["pred_dist_seq"].shape == (1, 4, 9)
        assert output["pred_dist_T"].shape == (1, 9)
        torch.testing.assert_close(
            output["pred_dist_T"].sum(-1),
            torch.ones(1),
            atol=1e-6,
            rtol=0.0,
        )


def test_decoder_and_mean_readout_definitions() -> None:
    features = torch.randn(1, 4, 30, 5)
    decoder_ablation = TIDE(ablation="no_distribution_decoder").eval()
    mean_ablation = TIDE(ablation="mean_sequence_readout").eval()
    with torch.inference_mode():
        decoder_output = decoder_ablation(features)
        mean_output = mean_ablation(features)
    assert torch.count_nonzero(decoder_output["dynamic_residual_seq"]) == 0
    torch.testing.assert_close(
        decoder_output["pred_dist_seq"],
        decoder_output["stable_composition_seq"],
        atol=1e-7,
        rtol=0.0,
    )
    expected = mean_output["base_energy_seq"].mean(dim=1)
    expected = expected / expected.sum(dim=-1, keepdim=True)
    torch.testing.assert_close(
        mean_output["pred_dist_T"],
        expected,
        atol=1e-7,
        rtol=0.0,
    )


def test_partial_emotion_learning_ablation_changes_only_bridge_weight() -> None:
    criterion = TIDELoss(partial_emotion_learning=False)
    assert criterion.config.lambda_bridge == 0.0
    assert criterion.config.lambda_instant_energy == 1.0
    assert criterion.config.lambda_final_energy == 1.0
    assert criterion.config.lambda_macro == 0.5

