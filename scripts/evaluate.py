#!/usr/bin/env python
"""Evaluate one checkpoint or a fold/seed run tree."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from tide.metrics import DISTRIBUTION_METRICS, DYNAMIC_METRICS  # noqa: E402
from tide.metrics.aggregation import (  # noqa: E402
    fold_seed_means,
    subject_trial_means,
    three_seed_summary,
)
from tide.training import evaluate_checkpoint  # noqa: E402
from tide.utils.io import write_json  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate TIDE checkpoints without training.")
    parser.add_argument("--run-root", required=True, help="checkpoint file or tree containing best_model.pt")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cpu")
    return parser


def find_checkpoints(path: str | Path) -> list[Path]:
    root = Path(path)
    if root.is_file():
        return [root]
    checkpoints = sorted(root.rglob("best_model.pt"))
    if not checkpoints:
        raise FileNotFoundError(f"no best_model.pt files found under {root}")
    return checkpoints


def main() -> None:
    args = build_parser().parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    frames: list[pd.DataFrame] = []
    audits = []
    for checkpoint in find_checkpoints(args.run_root):
        frame, audit = evaluate_checkpoint(
            checkpoint,
            data_root=args.data_root,
            split=args.split,
            device_name=args.device,
        )
        frames.append(frame)
        audits.append(audit)
        print(f"evaluated {checkpoint}: windows={len(frame)}")
    windows = pd.concat(frames, ignore_index=True)
    metrics = [*DYNAMIC_METRICS, *DISTRIBUTION_METRICS]
    subject_trials = subject_trial_means(windows, metrics)
    per_fold_seed = fold_seed_means(subject_trials, metrics)
    summary = three_seed_summary(subject_trials, metrics)
    windows.to_csv(output / "window_metrics.csv", index=False)
    subject_trials.to_csv(output / "subject_trial_metrics.csv", index=False)
    per_fold_seed.to_csv(output / "per_fold_seed.csv", index=False)
    summary.to_csv(output / "summary.csv", index=False)
    write_json(
        output / "evaluation_audit.json",
        {
            "status": "complete",
            "split": args.split,
            "checkpoint_count": len(audits),
            "window_count": len(windows),
            "state_unchanged": all(audit["state_unchanged"] for audit in audits),
            "test_fitted_normalization": False,
            "runs": audits,
        },
    )
    print(f"wrote evaluation to {output}")


if __name__ == "__main__":
    main()

