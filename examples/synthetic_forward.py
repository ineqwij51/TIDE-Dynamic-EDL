#!/usr/bin/env python
"""Minimal model-use example."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from tide.models import TIDE  # noqa: E402


model = TIDE().eval()
features = torch.randn(1, 20, 30, 5)
with torch.inference_mode():
    prediction = model(features)
print(prediction["pred_dist_seq"].shape, prediction["pred_dist_T"].shape)

