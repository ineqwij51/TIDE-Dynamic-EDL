"""Stable boundary for separately installed official baseline adapters."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
import shlex


def external_adapter_command(
    runner: str | Path,
    *,
    model: str,
    data_root: str | Path,
    folds: Sequence[int],
    seeds: Sequence[int],
    output_root: str | Path,
) -> list[str]:
    """Build the documented subprocess contract for an external adapter runner."""

    command = [
        str(runner),
        "--model",
        model,
        "--data-root",
        str(data_root),
        "--folds",
        *map(str, folds),
        "--seeds",
        *map(str, seeds),
        "--output-root",
        str(output_root),
    ]
    return command


def format_command(command: Sequence[str]) -> str:
    return shlex.join(map(str, command))


def validate_runner(path: str | Path) -> Path:
    runner = Path(path)
    if not runner.is_file():
        raise FileNotFoundError(
            "external baseline adapter runner not found; see docs/baselines.md "
            f"for the required interface: {runner}"
        )
    return runner


__all__ = ["external_adapter_command", "format_command", "validate_runner"]

