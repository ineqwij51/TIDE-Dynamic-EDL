#!/usr/bin/env python
"""Train one TIDE fold and seed."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from tide.config import load_config  # noqa: E402
from tide.models import ABLATION_NAMES  # noqa: E402
from tide.training import run_training  # noqa: E402


def resolve_device(value: str) -> str:
    if value != "auto":
        return value
    return "cuda" if torch.cuda.is_available() else "cpu"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train one TIDE fold/seed run.")
    parser.add_argument("--config", default="configs/tide.yaml")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--fold", required=True, type=int, choices=range(5))
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, default=None, help="explicit smoke/debug override")
    parser.add_argument("--ablation", choices=ABLATION_NAMES, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = deepcopy(load_config(args.config))
    if args.epochs is not None:
        if args.epochs < 1:
            raise ValueError("--epochs must be positive")
        config["training"]["epochs"] = args.epochs
    summary = run_training(
        config,
        data_root=args.data_root,
        fold=args.fold,
        seed=args.seed,
        output_dir=args.output_dir,
        device_name=resolve_device(args.device),
        ablation=args.ablation,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

