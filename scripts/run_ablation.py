#!/usr/bin/env python
"""Launch a named public TIDE ablation matrix."""

from __future__ import annotations

import argparse
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = ROOT / "scripts" / "train.py"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tide.models import ABLATION_NAMES  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one of the five formal TIDE ablations.")
    parser.add_argument("--name", required=True, choices=ABLATION_NAMES)
    parser.add_argument("--config", default="configs/tide.yaml")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--seeds", nargs="+", type=int, default=[17, 42, 97])
    parser.add_argument("--output-root", default="outputs/ablations")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    commands: list[list[str]] = []
    for seed in args.seeds:
        for fold in args.folds:
            output = Path(args.output_root) / args.name / f"fold{fold}" / f"seed{seed}"
            commands.append(
                [
                    sys.executable,
                    str(TRAIN_SCRIPT),
                    "--config",
                    args.config,
                    "--data-root",
                    args.data_root,
                    "--fold",
                    str(fold),
                    "--seed",
                    str(seed),
                    "--output-dir",
                    str(output),
                    "--device",
                    args.device,
                    "--ablation",
                    args.name,
                ]
            )
    for number, command in enumerate(commands, start=1):
        print(f"[{number:02d}/{len(commands):02d}] {shlex.join(command)}")
    print(f"ablation={args.name} runs={len(commands)}")
    if args.dry_run:
        return
    for command in commands:
        subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()

