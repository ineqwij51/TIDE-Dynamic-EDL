"""Pinned official baseline source registry."""

from __future__ import annotations

from pathlib import Path
import subprocess
from typing import Any, Mapping, Sequence

import yaml


BASELINE_NAMES = (
    "emod",
    "eegnet",
    "emt",
    "tsception",
    "eeg_deformer",
    "cbramod",
    "helo",
)


class BaselineSourceError(RuntimeError):
    """Raised when a pinned third-party source is absent or mismatched."""


def load_baseline_registry(path: str | Path) -> dict[str, Any]:
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, Mapping) or not isinstance(value.get("models"), Mapping):
        raise ValueError("baseline configuration must define a models mapping")
    models = dict(value["models"])
    if set(models) != set(BASELINE_NAMES):
        raise ValueError("baseline configuration does not contain the frozen public set")
    return {**dict(value), "models": models}


def audit_baseline_sources(
    registry: Mapping[str, Any],
    models: Sequence[str],
    *,
    external_root: str | Path | None = None,
) -> list[dict[str, Any]]:
    root = Path(external_root or registry["external_root"])
    rows: list[dict[str, Any]] = []
    for name in models:
        if name not in BASELINE_NAMES:
            raise ValueError(f"unknown baseline {name!r}; choose from {BASELINE_NAMES}")
        spec = dict(registry["models"][name])
        source = root / str(spec["directory"])
        missing = [
            str(relative)
            for relative in spec["required_files"]
            if not (source / str(relative)).is_file()
        ]
        revision = None
        revision_matches = False
        if source.is_dir() and (source / ".git").exists():
            completed = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=source,
                text=True,
                capture_output=True,
                check=False,
            )
            if completed.returncode == 0:
                revision = completed.stdout.strip()
                revision_matches = revision == str(spec["revision"])
        rows.append(
            {
                "model": name,
                "display_name": spec["display_name"],
                "source": str(source),
                "repository": spec["repository"],
                "expected_revision": spec["revision"],
                "actual_revision": revision,
                "revision_matches": revision_matches,
                "missing_files": missing,
                "available": source.is_dir() and not missing and revision_matches,
            }
        )
    return rows


def require_baseline_sources(rows: Sequence[Mapping[str, Any]]) -> None:
    failures = [row for row in rows if not row["available"]]
    if not failures:
        return
    details = []
    for row in failures:
        details.append(
            f"{row['model']}: clone {row['repository']} at {row['expected_revision']} "
            f"into {row['source']} (missing={row['missing_files']}, actual_revision={row['actual_revision']})"
        )
    raise BaselineSourceError("baseline source audit failed:\n" + "\n".join(details))


__all__ = [
    "BASELINE_NAMES",
    "BaselineSourceError",
    "audit_baseline_sources",
    "load_baseline_registry",
    "require_baseline_sources",
]

