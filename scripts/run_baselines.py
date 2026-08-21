#!/usr/bin/env python
"""Audit and dispatch separately installed official baseline adapters."""

from __future__ import annotations

import argparse
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tide.baselines.adapters import external_adapter_command, validate_runner  # noqa: E402
from tide.baselines.registry import (  # noqa: E402
    BASELINE_NAMES,
    audit_baseline_sources,
    load_baseline_registry,
    require_baseline_sources,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run pinned official baselines through an external licensed adapter."
    )
    parser.add_argument("--models", nargs="+", choices=BASELINE_NAMES, required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--seeds", nargs="+", type=int, default=[17, 42, 97])
    parser.add_argument("--config", default="configs/baselines.yaml")
    parser.add_argument("--external-root", default=None)
    parser.add_argument("--runner", default=None, help="licensed local adapter runner implementing the documented CLI")
    parser.add_argument("--output-root", default="outputs/baselines")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    registry = load_baseline_registry(args.config)
    rows = audit_baseline_sources(
        registry,
        args.models,
        external_root=args.external_root,
    )
    for row in rows:
        status = "ready" if row["available"] else "missing"
        print(
            f"{row['model']}: {status}; source={row['source']}; "
            f"revision={row['expected_revision']}; repository={row['repository']}"
        )
    if args.dry_run:
        print(f"runs={len(args.models) * len(args.folds) * len(args.seeds)}")
        return
    require_baseline_sources(rows)
    if args.runner is None:
        raise RuntimeError(
            "official sources are present, but no licensed adapter runner was supplied; "
            "pass --runner and follow docs/baselines.md"
        )
    runner = validate_runner(args.runner)
    for model in args.models:
        command = external_adapter_command(
            runner,
            model=model,
            data_root=args.data_root,
            folds=args.folds,
            seeds=args.seeds,
            output_root=Path(args.output_root) / model,
        )
        if runner.suffix == ".py":
            command = [sys.executable, *command]
        print(shlex.join(command))
        subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()

