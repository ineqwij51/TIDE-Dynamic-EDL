#!/usr/bin/env python
"""Build compact paper tables from evaluation or archived result CSVs."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402


MAIN_METRICS = (
    "StaticAware_SRCC",
    "StaticAware_PCC",
    "TimeAugmented_Discrete_Frechet",
    "ZMAE",
    "KL",
)


def markdown_table(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(map(str, columns)) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(map(str, row)) + " |")
    return "\n".join(lines) + "\n"


def load_summary(results_root: Path) -> pd.DataFrame:
    evaluation = results_root / "summary.csv"
    if evaluation.is_file():
        frame = pd.read_csv(evaluation)
        return frame.rename(columns={"standard_deviation": "std"})
    dynamic = results_root / "main_dynamic.csv"
    distribution = results_root / "main_distribution.csv"
    if not dynamic.is_file() or not distribution.is_file():
        raise FileNotFoundError(
            "results root must contain summary.csv or main_dynamic.csv plus main_distribution.csv"
        )
    return pd.concat(
        [pd.read_csv(dynamic), pd.read_csv(distribution)],
        ignore_index=True,
    ).rename(columns={"standard_deviation": "std"})


def formatted_pivot(frame: pd.DataFrame, identity: str) -> pd.DataFrame:
    selected = frame[frame["metric"].isin(MAIN_METRICS)].copy()
    selected["value"] = selected.apply(
        lambda row: f"{float(row['mean']):.4f} ± {float(row['std']):.4f}",
        axis=1,
    )
    result = selected.pivot(index=identity, columns="metric", values="value")
    ordered = [metric for metric in MAIN_METRICS if metric in result.columns]
    return result[ordered].reset_index()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Reproduce public paper result tables.")
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    results_root = Path(args.results_root)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    summary = load_summary(results_root)
    main_table = formatted_pivot(summary, "method")
    main_table.to_csv(output / "main_results.csv", index=False)
    (output / "main_results.md").write_text(
        markdown_table(main_table), encoding="utf-8"
    )
    ablation_path = results_root / "ablation.csv"
    if ablation_path.is_file():
        ablation = pd.read_csv(ablation_path).rename(
            columns={"standard_deviation": "std"}
        )
        identity = "variant" if "variant" in ablation else "method"
        ablation_table = formatted_pivot(ablation, identity)
        ablation_table.to_csv(output / "ablation_results.csv", index=False)
        (output / "ablation_results.md").write_text(
            markdown_table(ablation_table), encoding="utf-8"
        )
    print(f"wrote tables to {output}")


if __name__ == "__main__":
    main()

