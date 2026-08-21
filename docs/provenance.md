# Scientific provenance

This is the only public document that retains internal source mappings. They are provided for audit, not as public API names.

## Read-only source

- Repository: `/home/jwq/DyadEEG`
- Inspected branch: `codex/p44-tif-and-correlation-audit`
- Inspected working-tree commit: `7b68182eaca181d31391b7ee5a4c681220bcb27e`
- Core result/model source commit recorded by the formal evidence: `e7ad79fe6ad5b5884ccb5d60d35891ef7d308976`
- Baseline source commit recorded by the formal evidence: `9455b7b67cdf3829316a276deae37f6832efd85f`
- Standalone staging base: `5b2052b8c1f6f28838deff81d80eba1460f31e1e`

The source repository was read only. Its existing untracked working-tree files were not staged, edited, moved, or deleted.

## Code lineage

| Public implementation | Canonical source symbol/path | Migration |
|---|---|---|
| `tide.models.TIDE` | `src/models/dynamic_edl/p33/model.py::P33DynamicEDL` | Locked E0/A1 path renamed and narrowed to five DE features |
| Band Attention Encoder | `src/models/dynamic_edl/p31/bpde_encoder.py::BPDEEncoder` | DE-only path retained; raw/BP alternatives removed |
| Adaptive Electrode Graph | `src/models/dynamic_edl/p31/bpde_encoder.py::SingleSelfLoopElectrodeGraph` and `src/models/dynamic_edl/electrode_graph.py` | Exact montage/RBF/kNN prior and residual graph retained |
| Causal Context Encoder | `src/models/dynamic_edl/p31/causal_context.py::CausalTCNContext` | TCN-only streaming path retained |
| Emotion Query Readout | `src/models/dynamic_edl/p31/scdr_head.py::ArousalCompositionHead` | Query composition path retained |
| Distribution Decoder | `src/models/dynamic_edl/p31/scdr_head.py::DynamicResidualHead` | Exact bounded zero-mean residual retained |
| Temporal Intensity Fusion | `src/models/dynamic_edl/p32/multimode_pool.py` and `src/models/dynamic_edl/p33/macro_attention.py` | Exact stable-energy three-view path retained |
| `tide.losses.TIDELoss` | `src/losses/dynamic_edl/p41/loss_variants.py` and `src/losses/dynamic_edl/p42/core_objective.py` | Frozen Core weights and calculations retained |
| Dynamic metrics | `src/metrics/dynamic_edl/p43/metrics.py` | Public metric names and formulas retained |
| Distribution metrics | `src/losses/dynamic_edl/metrics.py` | Six formal metrics retained |
| Public ablations | `src/models/dynamic_edl/p43/ablations.py` | Five manuscript ablations renamed to stable public names |

The canonical class constructs a few inactive tensors even on the locked path (`head.simple`, `composition_adapter`, and `macro_attention.unified_scorer`). These tensors remain in the public class solely so all 15 frozen checkpoints load strictly with identical parameter/state schemas. No public configuration exposes the inactive alternatives.

## Checkpoint and sample audit

Parity used the existing frozen checkpoint at:

```text
/home/jwq/DyadEEG/outputs/p42/training/core_energy_perl_kl/fold0/seed17/best_model.pt
```

Its SHA-256 is `df6d9aa4f7ef6af474dd1e2907e48565169866161ba09571251c63ac870cc860`. All 15 checkpoints in the P42 Core fold/seed matrix were checked for strict state mapping. Four already prepared feature windows were read in memory for forward, loss, streaming, causality, simplex, and metric checks. Neither checkpoints nor feature/label samples were copied into this repository.

## Result lineage

The six public CSVs are mechanical, value-preserving consolidations of aggregate evidence under `results/p43/` and `results/p44/` in the source repository. Internal round fields and paths were replaced by stable method/variant names plus the recorded `source_commit`. Private predictions, identity-bearing columns, checkpoint paths, protocol locks, completion capsules, and run logs were excluded.

