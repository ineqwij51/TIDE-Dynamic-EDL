#!/usr/bin/env python
"""Render aggregate result figures from compact public CSVs."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Reproduce aggregate TIDE result figures.")
    parser.add_argument("--results-root", default="results")
    parser.add_argument("--output-dir", default="results/figures")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    root = Path(args.results_root)
    source = root / "main_dynamic.csv"
    if not source.is_file():
        source = root / "summary.csv"
    frame = pd.read_csv(source).rename(columns={"standard_deviation": "std"})
    selected = frame[frame["metric"] == "StaticAware_SRCC"].copy()
    if selected.empty:
        raise ValueError("result file has no StaticAware_SRCC rows")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(8.0, 4.2))
    axis.bar(
        selected["method"],
        selected["mean"],
        yerr=selected["std"],
        color=["#2563eb" if method == "TIDE" else "#94a3b8" for method in selected["method"]],
        capsize=2,
    )
    axis.set_ylabel("Static-aware SRCC")
    axis.set_xlabel("")
    axis.tick_params(axis="x", rotation=45)
    axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    figure.savefig(output / "dynamic_comparison.png", dpi=200)
    figure.savefig(output / "dynamic_comparison.pdf")
    plt.close(figure)
    print(f"wrote figures to {output}")


if __name__ == "__main__":
    main()

