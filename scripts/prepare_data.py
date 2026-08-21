#!/usr/bin/env python
"""Validate prepared DE features and create the public TIDE index."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tide.data.preprocessing import prepare_processed_data  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare TIDE inputs from released DE features and label arrays."
    )
    parser.add_argument("--features", required=True, help=".npy/.npz/.h5 prepared feature file")
    parser.add_argument("--labels", required=True, help=".npz labels and grouping metadata")
    parser.add_argument("--output-root", required=True, help="destination for dataset.npz and splits.json")
    parser.add_argument("--feature-key", default=None, help="optional dataset key for NPZ/HDF5 features")
    parser.add_argument("--split-seed", type=int, default=20260610)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    metadata = prepare_processed_data(
        args.features,
        args.labels,
        args.output_root,
        feature_key=args.feature_key,
        split_seed=args.split_seed,
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

