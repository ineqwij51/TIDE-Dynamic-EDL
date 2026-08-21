# Results

The six CSV files in `results/` are the complete public evidence retained from the frozen outer-test evaluation:

| File | Contents |
|---|---|
| `main_dynamic.csv` | Nine methods × complete dynamic metric suite |
| `main_distribution.csv` | Nine methods × six distribution metrics |
| `ablation.csv` | TIDE plus five formal component ablations |
| `metric_sensitivity.csv` | Additional correlation and lag sensitivity summaries |
| `per_fold_seed.csv` | Five paper-facing metrics for every method/variant, fold, and seed |
| `paired_statistics.csv` | Exact-window dyad- and subject-clustered paired bootstrap intervals |

Each summary row records the evaluation split, aggregation definition, folds, seeds, and scientific source commit. Per-fold/per-seed rows retain anonymized counts only; no window predictions or identity-bearing metadata are included.

The primary aggregation is:

```text
window metric
→ arithmetic mean within subject-trial
→ pooled five-fold mean within seed
→ mean and sample standard deviation over seeds 17, 42, and 97
```

Paired comparisons preserve exact fold/seed/window pairing and report 5,000-draw dyad-clustered intervals, with subject clustering as a sensitivity analysis. `ZMAE` is an independently standardized trajectory error, not raw-scale MAE.

The included numbers are evidence from the existing frozen runs. This repository refactor did not launch or selectively rerun the experiment matrix.

