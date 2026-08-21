"""YAML configuration loading and validation."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"configuration not found: {config_path}")
    value = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError("configuration root must be a mapping")
    config = deepcopy(dict(value))
    required = {"model", "loss", "data", "training"}
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"configuration is missing sections: {missing}")
    if int(config["data"].get("window_seconds", 0)) < 2:
        raise ValueError("data.window_seconds must be at least two")
    if int(config["training"].get("epochs", 0)) < 1:
        raise ValueError("training.epochs must be positive")
    return config


def merge_config(
    base: Mapping[str, Any],
    override: Mapping[str, Any],
) -> dict[str, Any]:
    """Recursively merge a small public override into a base configuration."""

    result = deepcopy(dict(base))
    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], Mapping)
            and isinstance(value, Mapping)
        ):
            result[key] = merge_config(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


__all__ = ["load_config", "merge_config"]

