from __future__ import annotations

import torch

from tide.losses import TIDELoss, distribution_kl
from tide.models import TIDE


def _labels(batch: int, seconds: int) -> dict[str, torch.Tensor]:
    return {
        "target_emotion": torch.arange(batch) % 9,
        "target_intensity": torch.rand(batch, seconds),
        "target_distribution": torch.softmax(torch.randn(batch, 9), dim=-1),
    }


def test_loss_is_finite_and_removed_terms_are_exactly_zero() -> None:
    torch.manual_seed(12)
    model = TIDE().eval()
    output = model(torch.randn(2, 5, 30, 5))
    loss, components = TIDELoss()(output, **_labels(2, 5))
    assert torch.isfinite(loss)
    assert components["weighted_loss_shape"] == 0.0
    assert components["weighted_loss_center"] == 0.0
    assert components["weighted_loss_bridge"] >= 0.0
    assert components["weighted_loss_kl"] >= 0.0


def test_macro_kl_is_blocked_from_dynamic_decoder() -> None:
    torch.manual_seed(13)
    model = TIDE()
    output = model(torch.randn(2, 5, 30, 5))
    macro = distribution_kl(
        torch.softmax(torch.randn(2, 9), dim=-1),
        output["pred_dist_T"],
    )
    decoder_parameters = list(model.residual.parameters())
    gradients = torch.autograd.grad(
        macro,
        decoder_parameters,
        allow_unused=True,
    )
    assert all(gradient is None or torch.count_nonzero(gradient) == 0 for gradient in gradients)


def test_micro_objective_reaches_dynamic_decoder() -> None:
    torch.manual_seed(14)
    model = TIDE()
    output = model(torch.randn(2, 5, 30, 5))
    loss, _ = TIDELoss()(output, **_labels(2, 5))
    loss.backward()
    total = sum(
        float(parameter.grad.abs().sum())
        for parameter in model.residual.parameters()
        if parameter.grad is not None
    )
    assert total > 0.0

