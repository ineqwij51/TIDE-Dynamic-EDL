# Public release audit

Audit date: 2026-08-21

## Scope and result

The complete public working tree and clean no-history export were scanned. Internal mappings are permitted only in `docs/provenance.md`.

| Gate | Result |
|---|---|
| Total files below 250 | PASS (`79`) |
| Python files below 60 | PASS (49) |
| Public result files below 20 | PASS (6) |
| No file above 10 MB | PASS (largest: `results/per_fold_seed.csv`, 177,096 bytes) |
| No versioned public paths | PASS |
| No internal version labels outside provenance | PASS |
| No internal workflow terminology outside provenance | PASS |
| No absolute server paths outside provenance | PASS |
| No username or email address | PASS |
| No token, credential, or secret payload | PASS |
| No private subject/dyad identifier fields | PASS; only anonymized `subject_group`/`dyad_group` interfaces |
| No checkpoints or model-weight payloads | PASS |
| No prediction arrays, raw EEG, HDF5, NPZ, or normalization payloads | PASS |
| No cache directory or compiled Python file | PASS |
| No vendored third-party source | PASS |
| Clean export has no `.git` | PASS |

Protective ignore patterns containing words such as `credentials` and `secrets` are policy controls, not detected payloads. Documentation may discuss withheld checkpoints/private data without containing those artifacts.

## Release blockers

| Blocker | Status |
|---|---|
| Project license | BLOCKED: no author-selected license; `LICENSE` intentionally absent |
| Citation metadata | BLOCKED: author list/order/affiliations and persistent identifier unconfirmed; `CITATION.cff` intentionally absent |
| Framework figure | BLOCKED: exact final manuscript asset unavailable; no approximation substituted |
| Study data | BLOCKED for redistribution: ethics, consent, and data agreements apply |
| Baseline rights | REVIEW REQUIRED: official source and weight licenses must be checked independently |

## Decision

The tree is technically clean and reproducible at the released prepared-feature boundary, but it is **not authorized for public release** until the license, citation, figure, data statement, and third-party review blockers are resolved. No push or publication action was performed.
