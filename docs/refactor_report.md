# Refactor report

## Outcome

The standalone archive is now a stable public `tide` package with version-free model, data, training, evaluation, metric, ablation, baseline, and reproduction interfaces. No full experiment was launched, no commit was created, and nothing was pushed.

## Size reduction

Counts below use regular working-tree files and exclude `.git` and runtime caches.

| Measure | Before | After | Reduction |
|---|---:|---:|---:|
| Files | 2,755 | 79 | 2,676 (97.1%) |
| Bytes | 14,773,245 | 611,308 | 14,161,937 (95.9%) |
| Python source files | at least 200 in model/script trees | 49 | at least 75.5% |
| Public result files | 44 plus 2,460 run-output files | 6 | compact evidence only |

Removed from the public tree:

- the historical `src` namespace and candidate-family registries;
- 13 versioned script directories plus an internal external-source generator;
- six versioned result archive trees and all per-run output trees;
- internal workflow documentation, completion artifacts, and draft table copies;
- empty manuscript/figure placeholders and private-data path documentation;
- the refactor prompt and plan themselves.

The retained result evidence is 2,077 rows across six CSVs (about 439 KB): main dynamic/distribution summaries, five formal ablations, sensitivity metrics, selected per-fold/per-seed rows, and paired intervals.

## Public commands

```bash
conda env create -f environment.yml
conda activate tide
pip install -e .

python examples/quick_start.py

python scripts/prepare_data.py \
  --features <FEATURE_FILE> \
  --labels <LABEL_FILE> \
  --output-root <PROCESSED_DATA_ROOT>

python scripts/train.py \
  --config configs/tide.yaml \
  --data-root <DATA_ROOT> \
  --fold 0 \
  --seed 17 \
  --output-dir outputs/tide/fold0/seed17

python scripts/run_cross_validation.py \
  --config configs/tide.yaml \
  --data-root <DATA_ROOT> \
  --folds 0 1 2 3 4 \
  --seeds 17 42 97 \
  --gpus 0 \
  --dry-run

python scripts/evaluate.py \
  --run-root outputs/tide \
  --data-root <DATA_ROOT> \
  --split test \
  --output-dir results/evaluation

python scripts/reproduce_tables.py \
  --results-root results/evaluation \
  --output-dir results/tables
```

The five ablation names are `no_context`, `no_distribution_decoder`, `no_partial_emotion_learning`, `no_adaptive_graph`, and `mean_sequence_readout`. The baseline wrapper accepts `emod`, `eegnet`, `emt`, `tsception`, `eeg_deformer`, `cbramod`, and `helo`.

## Verification

- Compile: passed.
- Synthetic quick start: passed.
- Cross-validation dry run: 15/15 units printed.
- One-ablation dry run: 15/15 units printed.
- Baseline dry run: 105 units represented.
- Public tests: 25 passed.
- Synthetic end-to-end path: data preparation, one epoch, checkpoint reload, evaluation, and table generation passed on CPU.
- Editable-install smoke in a temporary environment: passed.
- Source parity: all numerical gates passed; see `docs/parity_report.md`.
- Diff whitespace check: passed.

## Clean export

The no-history export is created at:

```text
<workspace-parent>/TIDE-Dynamic-EDL-public
```

It contains only the audited public tree and no `.git` directory.

## Remaining blockers

The code/package refactor is complete, but public release is blocked by the absent project license, unconfirmed citation metadata, missing exact manuscript framework figure, private-data release constraints, and pending third-party rights review. The full training matrix was deliberately not rerun.
