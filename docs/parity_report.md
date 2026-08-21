# Source parity report

## Result

The stable public implementation meets every required numerical gate against the read-only scientific source.

| Check | Result | Gate |
|---|---:|---:|
| Canonical/public parameters | 47,778 / 47,778 | equal |
| State tensors | 74 | exact schema |
| Frozen checkpoints mapped | 15 / 15 | all |
| Unmapped or shape-mismatched tensors | 0 | 0 |
| Forward max absolute error, canonical ten-column input | 0.0 | ≤ 1e-7 |
| Forward max absolute error, public five-band input | 0.0 | ≤ 1e-7 |
| Source/public repeated-step max absolute error | 0.0 | ≤ 1e-7 |
| Public step vs sequence max absolute error | 2.9802322387695312e-08 | ≤ 1e-7 |
| Total loss absolute error | 0.0 | ≤ 1e-7 |
| Loss-component max absolute error | 0.0 | ≤ 1e-7 |
| Future-perturbation prefix error | 0.0 | 0 |
| Dynamic metric max absolute error | 0.0 | exact |
| Distribution metric max absolute error | 0.0 | exact |
| Maximum float32 simplex deviation | 2.384185791015625e-07 | ≤ 1e-6 |

## Procedure

The comparison loaded the same frozen state into the canonical and public classes, used four existing prepared 20-second feature windows normalized with the frozen training-fold statistics, and compared all public loss-relevant and diagnostic tensors. The public five-band path and canonical ten-column path were both evaluated. Streaming states were initialized independently and advanced one second at a time. A large perturbation applied only after second 10 produced zero change in outputs through second 10.

The public Core objective was compared term by term using the existing target intensity, target-emotion index, and sequence-distribution label. The public dynamic/distribution metrics were compared to the canonical functions on the same predictions. No private sample, prediction, normalization array, or checkpoint was written into the public tree.

## Public path validation

A separate synthetic smoke completed:

```text
prepare data → grouped split → one training epoch → best checkpoint
→ strict checkpoint reload → outer-split evaluation → paper-table generation
```

The smoke used 10 synthetic trials, completed in under 10 seconds end to end on CPU, and produced the documented minimal artifacts. It is a software-path check, not scientific evidence.

