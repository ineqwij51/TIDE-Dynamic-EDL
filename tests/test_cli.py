from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "script",
    (
        "prepare_data.py",
        "train.py",
        "run_cross_validation.py",
        "evaluate.py",
        "run_ablation.py",
        "run_baselines.py",
        "reproduce_tables.py",
        "reproduce_figures.py",
    ),
)
def test_cli_help(script: str) -> None:
    completed = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / script), "--help"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "usage:" in completed.stdout


def test_cross_validation_dry_run_has_fifteen_units() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "run_cross_validation.py"),
            "--config",
            "configs/tide.yaml",
            "--data-root",
            "DUMMY_DATA_ROOT",
            "--folds",
            "0",
            "1",
            "2",
            "3",
            "4",
            "--seeds",
            "17",
            "42",
            "97",
            "--dry-run",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.count("/15]") == 15
    assert "runs=15 folds=5 seeds=3" in completed.stdout


def test_quick_start() -> None:
    completed = subprocess.run(
        [sys.executable, str(ROOT / "examples" / "quick_start.py")],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "pred_dist_seq: (2, 20, 9)" in completed.stdout

