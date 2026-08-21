# Reproduction

## Tested environment

| Component | Tested version |
|---|---|
| Python | 3.13.9 |
| PyTorch | 2.10.0+cu128 |
| CUDA runtime | 12.8 |
| NumPy | 2.4.4 |
| SciPy | 1.17.1 |
| pandas | 3.0.1 |
| scikit-learn | 1.8.0 (environment audit; not a runtime import) |
| h5py | 3.16.0 |
| MNE | 1.11.0 |
| matplotlib | 3.10.8 |
| PyYAML | 6.0.3 |
| pytest | 9.0.3 |

The dependency files specify supported lower bounds rather than a server-wide freeze.

## Commands

```bash
conda env create -f environment.yml
conda activate tide
pip install -e .

python examples/quick_start.py

python scripts/prepare_data.py \
  --features <FEATURE_FILE> \
  --labels <LABEL_FILE> \
  --output-root <DATA_ROOT>

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

Remove `--dry-run` only when the complete 15-run computation is intended. Training is 20 fixed epochs with AdamW (`5e-4`, weight decay `1e-4`), batch size 64, constant learning rate, gradient norm cap 5, and checkpoint selection by minimum validation `loss_total`.

The output of each training unit is deliberately small: best/last checkpoints, normalization, epoch metrics, and one run summary. Evaluation writes generated window and subject-trial metrics under an ignored directory. Public result evidence is aggregated as window → subject-trial arithmetic mean → pooled five-fold result per seed → mean and sample standard deviation over three seeds.

