#!/usr/bin/env python
"""Launch or print the five-fold, three-seed TIDE matrix."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = ROOT / "scripts" / "train.py"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the public TIDE cross-validation matrix.")
    parser.add_argument("--config", default="configs/tide.yaml")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--seeds", nargs="+", type=int, default=[17, 42, 97])
    parser.add_argument("--gpus", nargs="*", default=None, help="visible GPU ids, e.g. --gpus 0")
    parser.add_argument("--device", default=None, help="explicit torch device; overrides --gpus")
    parser.add_argument("--output-root", default="outputs/tide")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def matrix_commands(args: argparse.Namespace) -> list[tuple[list[str], str | None]]:
    if not args.folds or any(fold not in range(5) for fold in args.folds):
        raise ValueError("--folds must be a nonempty subset of 0,1,2,3,4")
    if not args.seeds:
        raise ValueError("--seeds cannot be empty")
    commands: list[tuple[list[str], str | None]] = []
    for index, (fold, seed) in enumerate(
        (fold, seed) for seed in args.seeds for fold in args.folds
    ):
        gpu = args.gpus[index % len(args.gpus)] if args.gpus else None
        device = args.device or ("cuda:0" if gpu is not None else "auto")
        output = Path(args.output_root) / f"fold{fold}" / f"seed{seed}"
        command = [
            sys.executable,
            str(TRAIN_SCRIPT),
            "--config",
            str(args.config),
            "--data-root",
            str(args.data_root),
            "--fold",
            str(fold),
            "--seed",
            str(seed),
            "--output-dir",
            str(output),
            "--device",
            device,
        ]
        commands.append((command, gpu))
    return commands


def main() -> None:
    args = build_parser().parse_args()
    commands = matrix_commands(args)
    for number, (command, gpu) in enumerate(commands, start=1):
        prefix = f"CUDA_VISIBLE_DEVICES={gpu} " if gpu is not None else ""
        print(f"[{number:02d}/{len(commands):02d}] {prefix}{shlex.join(command)}")
    print(f"runs={len(commands)} folds={len(args.folds)} seeds={len(args.seeds)}")
    if args.dry_run:
        return
    for command, gpu in commands:
        environment = os.environ.copy()
        if gpu is not None:
            environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
        subprocess.run(command, cwd=ROOT, env=environment, check=True)


if __name__ == "__main__":
    main()

