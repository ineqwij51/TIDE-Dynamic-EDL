# TIDE

*Dynamic Emotion Distribution Learning from EEG under Partial Target Emotion Intensity Supervision*

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)
[![PyTorch 2.3+](https://img.shields.io/badge/PyTorch-2.3%2B-ee4c2c.svg)](https://pytorch.org/)

TIDE is a strictly causal EEG model that predicts a nine-emotion distribution every second and a sequence-level distribution from 20 one-second differential-entropy (DE) observations. Only the target emotion has continuous intensity supervision; the remaining second-wise components are latent and constrained by the post-trial distribution.

> The exact final manuscript framework asset is not available in the scientific source, so this repository does not substitute an approximate figure. The verified implementation flow is documented in [data format](docs/data_format.md) and [reproduction](docs/reproduction.md); see [limitations](docs/limitations.md).

## Highlights

- Five-band, 30-electrode Band Attention Encoder with an Adaptive Electrode Graph.
- Strictly causal dilated context encoder with equivalent sequence and streaming APIs.
- Emotion Query Readout and causal residual Distribution Decoder.
- Temporal Intensity Fusion of uniform, sustained, and change-sensitive views.
- Dyad-grouped five-fold evaluation with three fixed model seeds.

## Installation

```bash
conda env create -f environment.yml
conda activate tide
pip install -e .
```

The tested environment is recorded in [reproduction](docs/reproduction.md). GPU users may install the appropriate PyTorch build before `pip install -e .`.

## Data format

Raw EEG preprocessing is not released as a complete, audited pipeline. The supported boundary is prepared one-second DE features plus a public NPZ label schema:

```bash
python scripts/prepare_data.py \
  --features <FEATURE_FILE> \
  --labels <LABEL_FILE> \
  --output-root <PROCESSED_DATA_ROOT>
```

The model consumes `features [B,20,30,5]` and returns `pred_dist_seq [B,20,9]` plus `pred_dist_T [B,9]`. See [docs/data_format.md](docs/data_format.md) for keys, orders, grouping fields, normalization, and split rules.

## Quick start

```bash
python examples/quick_start.py
```

This creates a synthetic batch, runs sequence and repeated-step inference, prints output shapes, and checks streaming and simplex constraints.

## Train one fold

```bash
python scripts/train.py \
  --config configs/tide.yaml \
  --data-root <DATA_ROOT> \
  --fold 0 \
  --seed 17 \
  --output-dir outputs/tide/fold0/seed17
```

## Five folds × three seeds

Inspect the 15-run matrix without launching training:

```bash
python scripts/run_cross_validation.py \
  --config configs/tide.yaml \
  --data-root <DATA_ROOT> \
  --folds 0 1 2 3 4 \
  --seeds 17 42 97 \
  --gpus 0 \
  --dry-run
```

Remove `--dry-run` to execute the matrix. This refactor did not rerun the full experiment.

## Evaluate checkpoints

```bash
python scripts/evaluate.py \
  --run-root outputs/tide \
  --data-root <DATA_ROOT> \
  --split test \
  --output-dir results/evaluation
```

## Reproduce paper tables

```bash
python scripts/reproduce_tables.py \
  --results-root results/evaluation \
  --output-dir results/tables
```

To rebuild tables from the included final evidence, use `--results-root results`.

## Ablations and baselines

```bash
python scripts/run_ablation.py \
  --name no_context \
  --data-root <DATA_ROOT> \
  --folds 0 1 2 3 4 \
  --seeds 17 42 97

python scripts/run_baselines.py \
  --models emod eegnet emt tsception eeg_deformer cbramod helo \
  --data-root <DATA_ROOT> \
  --folds 0 1 2 3 4 \
  --seeds 17 42 97 \
  --dry-run
```

Baseline bodies and weights are not redistributed. The wrapper audits pinned official sources and requires a separately licensed adapter runner for execution; see [docs/baselines.md](docs/baselines.md).

## Results snapshot

Outer-test values are mean ± sample standard deviation over seeds 17, 42, and 97 after subject-trial aggregation and five-fold pooling.

| Method | Static-aware SRCC ↑ | Fréchet ↓ | ZMAE ↓ | KL ↓ |
|---|---:|---:|---:|---:|
| TIDE | **0.1414 ± 0.0117** | **2.0075 ± 0.0368** | **0.9429 ± 0.0053** | **0.2275 ± 0.0001** |
| EMOD pretrained finetune | 0.0031 ± 0.0003 | 2.4143 ± 0.0064 | 1.0289 ± 0.0010 | 0.2304 ± 0.0004 |
| EEG Deformer | 0.0033 ± 0.0027 | 2.3860 ± 0.0021 | 1.0327 ± 0.0014 | 0.2316 ± 0.0038 |
| TSception | 0.0018 ± 0.0011 | 2.4095 ± 0.0082 | 1.0308 ± 0.0019 | 0.2280 ± 0.0008 |

Complete summaries, ablations, paired intervals, and fold/seed evidence are in [results](results) and described in [docs/results.md](docs/results.md).

## Repository structure

```text
configs/       final method, cross-validation, ablation, and baseline configs
tide/          public model, data, loss, metrics, training, and adapter package
scripts/       preparation, training, evaluation, and reproduction CLIs
examples/      synthetic sequence and streaming inference
tests/         public synthetic and protocol tests
results/       compact final aggregate and fold/seed evidence
docs/          data, reproduction, results, provenance, and release audits
third_party/   installation boundary; no third-party code is vendored
```

## Data availability

Raw EEG, private annotations, identity-bearing metadata, normalization snapshots, predictions, and checkpoints are not distributed. Access remains subject to the original study's ethics, consent, and data agreements.

## Citation

The manuscript author list, order, affiliations, and persistent paper identifier are not confirmed in the available source. `CITATION.cff` is intentionally absent until the authors provide those fields; do not cite placeholder metadata.

## License

No project license has been selected. All rights are reserved by default, and the repository must remain private until the authors add a license and complete the third-party rights review.

