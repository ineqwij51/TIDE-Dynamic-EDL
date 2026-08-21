# Data format

TIDE starts from prepared one-second differential-entropy (DE) features. The repository does not claim raw-signal reproducibility because the complete filtering, resampling, artifact-rejection, and annotation-release pipeline is not available as a public, audited unit.

## Input files

`scripts/prepare_data.py` accepts a feature file and an NPZ label file.

Features may be:

- `de_features`, `features`, or `bandpower_de` in `.npz`/`.h5`, or one array in `.npy`;
- trial-shaped `[N,L,30,5]` DE features; or
- canonical `[N,L,30,10]` BP+DE features, from which columns 5–9 are selected;
- per-second `[S,30,5|10]` features when labels provide `feature_rows [N,L]`.

The label NPZ contains:

| Key | Shape | Type | Meaning |
|---|---:|---|---|
| `target_intensity` | `[N,L]` | float32 | Continuous target-emotion intensity, expected in `[0,1]` |
| `distribution` | `[N,9]` | float32 | Nonnegative post-trial distribution; normalized during preparation |
| `target_emotion` | `[N]` | int64 | Target component index in `[0,8]` |
| `subject_group` | `[N]` | integer/string | Pseudonymous grouping value; never a model input |
| `dyad_group` | `[N]` | integer/string | Pseudonymous split group; never a model input |
| `trial_group` | `[N]` | integer/string | Optional within-subject trial key |
| `group_stratum` | `[N]` | integer/string | Optional split-balancing stratum; defaults to one stratum |
| `lengths` | `[N]` | int64 | Optional valid seconds before padding |
| `feature_rows` | `[N,L]` | int64 | Required only for per-second feature arrays |

Preparation factorizes group values and does not retain the source identifiers or a reverse mapping.

## Fixed orders

DE bands:

```text
delta [1,4), theta [4,8), alpha [8,13), beta [13,30), gamma [30,45) Hz
```

Emotions:

```text
YQ Friendship, JW Awe, QQ Family affection, ZZ Respect, GJ Gratitude,
GX Joy/Happiness, KW Desire/Longing, ZH Pride, AQ Romantic love
```

Electrodes:

```text
FP1 FP2 F7 F3 FZ F4 F8 FT7 FC3 FCZ FC4 FT8 T3 C3 CZ C4 T4
TP7 CP3 CPZ CP4 TP8 T5 P3 PZ P4 T6 O1 OZ O2
```

## Generated dataset

Preparation writes:

```text
<DATA_ROOT>/
├── dataset.npz
├── metadata.json
└── splits.json
```

`dataset.npz` stores only five-band DE features and anonymized group indices. Runtime windows have:

```text
features             float32 [B,20,30,5]
target_intensity     float32 [B,20]
target_distribution  float32 [B,9]
target_emotion       int64   [B]
```

## Splits and normalization

Dyads are assigned to five deterministic folds with split seed `20260610`. For run `i`, test is fold `i`, validation is fold `(i+1) mod 5`, and the remaining three folds train the model. Subject and dyad disjointness is asserted.

Training uses 20-second windows with stride 5. Validation and test use non-overlapping stride-20 windows. Per-electrode, per-band mean and standard deviation are fitted once on unique seconds from the three training folds, stored with the checkpoint, and reused unchanged for validation and test. Labels and grouping metadata never enter `TIDE.forward`, `forward_sequence`, or `step`.

