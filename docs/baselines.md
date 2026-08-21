# Third-party baselines

No third-party model body or pretrained weight is redistributed. `configs/baselines.yaml` records the tested source revisions, and `scripts/run_baselines.py --dry-run` audits the intended matrix without cloning or executing anything.

| Public name | Paper / source | Tested revision | Input adaptation | Public task readout |
|---|---|---|---|---|
| EMOD | [official repository](https://github.com/cyn4396/EMOD.git) | `3a219f51ba4f90b9e2d22824cfb90dde77a9875c` | Official embedding and axial transformer per one-second raw EEG window; released pretrained body for the main row | Per-second nonnegative latent intensities, normalized to nine components; mean sequence readout |
| EEGNet | [public PyTorch reference](https://github.com/aliasvishnu/EEGNet.git) | `cccd2f87de8c6b500fd1bffde034ee674aa4966a` | 30×200 one-second EEG adaptation | Same two-output task interface |
| EmT | [official repository](https://github.com/yi-ding-cs/EmT.git) | `3ea05dc90a915182ab3407191c287456f912316d` | Graph encoder and temporal transformer over prepared features | Original classifier replaced by the task readout |
| TSception | [official repository](https://github.com/yi-ding-cs/TSception.git) | `9efd666b618d006e32e6da1d30dbc79b1d190604` | Official temporal/spatial/fusion blocks per one-second raw EEG window | Original classifier replaced by the task readout |
| EEG Deformer | [official repository](https://github.com/yi-ding-cs/EEG-Deformer.git) | `47df1c4d8ce375ae21117f84e81e969041d907f6` | Official CNN/transformer body per one-second raw EEG window | Original MLP head replaced by the task readout |
| CBraMod | [official repository](https://github.com/wjq-learning/CBraMod.git) | `0ff6be918985689e7df679bc731ffb70e6c6224f` | Official patch embedding and criss-cross transformer trained from scratch | Dataset classifier replaced by the task readout |
| HeLo | [official repository](https://github.com/kaio-99/HeLo) | `396e307f83ed09ad5ed54be218b46dbb389a65e9` | EEG-only feature path; unavailable modalities, multimodal fusion, and ground-truth-label branches disabled | Task-adapted two-output readout |

All rows use target-emotion intensity supervision during training and a sequence-level distribution target. Target emotion and metadata are labels only, never predictive inputs. Checkpoint selection uses validation loss, and the test split is inference-only.

## Installation boundary

Clone each repository independently into `third_party/sources/<directory>` and check out the exact revision in `configs/baselines.yaml`. Obtain pretrained weights only from their authorized upstream source. Then provide a local licensed runner implementing:

```text
<runner> --model <name> --data-root <root> --folds ... --seeds ... --output-root <root>
```

The wrapper verifies required files and Git revisions before dispatch. It intentionally has no fallback implementation. Upstream license terms must be reviewed individually before release; a source revision record is not a grant of redistribution rights.

