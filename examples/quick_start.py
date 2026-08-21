#!/usr/bin/env python
"""Synthetic sequence and streaming inference with TIDE."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from tide.models import TIDE  # noqa: E402


def main() -> None:
    torch.manual_seed(7)
    torch.set_num_threads(2)
    features = torch.randn(2, 20, 30, 5)
    model = TIDE().eval()
    with torch.inference_mode():
        output = model.forward_sequence(features)
        state = model.initial_state(features.shape[0], features.device, features.dtype)
        steps = []
        for second in range(features.shape[1]):
            step_output = model.step(features[:, second], state)
            state = step_output["state"]
            steps.append(step_output["pred_dist"])
        streamed = torch.stack(steps, dim=1)
    sequence_error = float((streamed - output["pred_dist_seq"]).abs().max())
    second_simplex_error = float(
        (output["pred_dist_seq"].sum(-1) - 1.0).abs().max()
    )
    sequence_simplex_error = float(
        (output["pred_dist_T"].sum(-1) - 1.0).abs().max()
    )
    print(f"features:      {tuple(features.shape)}")
    print(f"pred_dist_seq: {tuple(output['pred_dist_seq'].shape)}")
    print(f"pred_dist_T:   {tuple(output['pred_dist_T'].shape)}")
    print(f"step_sequence_max_abs_error: {sequence_error:.3e}")
    print(f"second_simplex_max_abs_error: {second_simplex_error:.3e}")
    print(f"sequence_simplex_max_abs_error: {sequence_simplex_error:.3e}")
    if sequence_error > 1e-7 or max(
        second_simplex_error,
        sequence_simplex_error,
    ) > 1e-6:
        raise RuntimeError("synthetic quick-start checks failed")


if __name__ == "__main__":
    main()
