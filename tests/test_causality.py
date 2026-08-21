from __future__ import annotations

import torch

from tide.models import TIDE


def test_step_matches_sequence() -> None:
    torch.manual_seed(8)
    model = TIDE().eval()
    features = torch.randn(2, 8, 30, 5)
    with torch.inference_mode():
        sequence = model.forward_sequence(features)
        state = model.initial_state(2, features.device, features.dtype)
        steps = []
        for second in range(features.shape[1]):
            output = model.step(features[:, second], state)
            state = output["state"]
            steps.append(output["pred_dist"])
        streamed = torch.stack(steps, dim=1)
    assert float((streamed - sequence["pred_dist_seq"]).abs().max()) <= 1e-7


def test_future_perturbation_cannot_change_past() -> None:
    torch.manual_seed(9)
    model = TIDE().eval()
    features = torch.randn(1, 9, 30, 5)
    changed = features.clone()
    changed[:, 5:] += 100.0 * torch.randn_like(changed[:, 5:])
    with torch.inference_mode():
        original = model(features)["pred_dist_seq"]
        altered = model(changed)["pred_dist_seq"]
    assert float((original[:, :5] - altered[:, :5]).abs().max()) <= 1e-7

