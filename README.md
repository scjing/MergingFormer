<div align="center">

# MergingFormer

**Distributionally consistent two-dimensional merging behavior model for
autonomous vehicle simulation test at highway on-ramps**

Shoucai Jing, Wanpeng Zhu, Aohua Wang, Xiangmo Zhao, Xiaolong Ma, and
Asad J. Khattak

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.9-blue.svg)](environment.yml)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12.0-ee4c2c.svg)](requirements.txt)

</div>

MergingFormer is a PyTorch research project for multivariate vehicle trajectory
prediction. The main model combines an LSTM temporal branch, adaptive sparse
window attention, data embeddings, and a Transformer encoder-decoder. The
repository also contains recurrent, Transformer, Performer, Informer,
Autoformer, and DLinear comparison models.

> **Status:** the original environment has been reconstructed from the recorded
> package versions, but installation, training, and evaluation have not yet been
> revalidated from a clean checkout.

## 1. Model overview

The current implementation in [`models/MyTransformer.py`](models/MyTransformer.py)
uses four main stages:

1. embed the multivariate input sequence;
2. extract temporal features with a three-layer LSTM;
3. refine the LSTM features with adaptive sparse window attention, which learns
   a mixture of softmax attention and squared-ReLU attention; and
4. fuse the embedded and recurrent features before Transformer encoding and
   decoding.

The default training configuration predicts 10 future steps from 15 input
features and produces 2 target variables: lateral speed (`speedY`) and speed
(`speed`). Some experimental scripts use different feature and target counts;
review their configuration before use.

## 2. Repository structure

```text
.
|-- data/                 # Local datasets; contents are excluded from Git
|-- layers/               # Attention, embedding, and encoder-decoder layers
|-- models/               # MergingFormer and comparison models
|-- src/                  # Data loading, training, and evaluation scripts
|-- tools/                # Small environment inspection utilities
|-- utils/                # Masks, time features, metrics, and data helpers
|-- environment.yml       # Recorded Conda environment
`-- requirements.txt      # Pinned direct Python dependencies
```

## 3. Environment setup

The recorded training environment used Python 3.9.23 and PyTorch 1.12.0 with
CUDA 11.3. Create it with Conda:

```bash
git clone https://github.com/scjing/MergingFormer.git
cd MergingFormer
conda env create -f environment.yml
conda activate mergingformer
```

Alternatively, install the pinned packages into an existing Python 3.9
environment:

```bash
pip install -r requirements.txt
```

The PyTorch packages in these files target CUDA 11.3. Users with another CUDA
version or CPU-only systems should install a compatible PyTorch build first and
then install the remaining dependencies.

## 4. Data preparation

The original dataset is not open-sourced and is not distributed in this
repository. The Apache License 2.0 for the source code grants no rights to the
dataset. Users must provide their own lawfully obtained CSV trajectories in a
compatible schema.

Suggested local layout:

```text
data/
`-- your_dataset/
    |-- trajectory_001.csv
    `-- trajectory_002.csv
```

See [`data/README.md`](data/README.md) for the 15 required input columns, the 2
prediction targets, preprocessing behaviour, and known schema differences.

## 5. Paths and configurations to review

Most experiment settings are currently defined inside individual scripts.
Replace the example paths and confirm the dimensions before running them.

| Location | Setting | Purpose |
| --- | --- | --- |
| `src/train.py` | `data_folder` | Directory containing training CSV files |
| `src/train.py` | `save_dir` | Directory for checkpoints, loss logs, and plots |
| `src/evaluate.py` | `--data_folder` | Evaluation dataset directory |
| `src/evaluate.py` | `--config_name` | Checkpoint experiment directory |
| `src/evaluate.py` | `--specific_file` | One trajectory used for visualization |
| each `Config` class | `enc_in`, `dec_in`, `c_out` | Input and output dimensions |

Paths are interpreted relative to the process working directory. Dataset and
checkpoint paths are excluded from Git by default.

## 6. Training and evaluation

The current main entry points are:

- `src/train.py` for MergingFormer training;
- `src/evaluate.py` for aggregate evaluation and trajectory visualization;
- `src/train_*.py` for comparison-model training; and
- `src/test_*.py` for comparison-model evaluation.

Training writes the best checkpoint, epoch weights, loss values, and a loss
curve under the selected checkpoint directory. Evaluation writes numerical
metrics and figures to the corresponding experiment directory.

Commands are not presented as validated recipes yet because the hard-coded
paths and experimental feature dimensions still need to be unified. Any result
reported from this code should record the exact script, data schema,
configuration, checkpoint, and dependency environment used.

## 7. Known limitations

- The pinned environment has not yet been recreated from a clean checkout.
- The dataset and pretrained weights are not included.
- Some training and evaluation scripts use 23 inputs and 7 outputs, while the
  main loader currently provides 15 inputs and 2 targets.
- Hyperparameters and paths are duplicated across standalone scripts.
- The loader concatenates CSV files before generating sliding windows and then
  randomly splits windows. Review this procedure for trajectory-boundary and
  train/test leakage before reporting experimental results.

## 8. License and attribution

The source code in this repository is licensed under the
[Apache License 2.0](LICENSE). This license does not cover the private dataset.

Parts of the time-series layers, utilities, and comparison models are derived
from or based on
[THUML Time-Series-Library](https://github.com/thuml/Time-Series-Library), which
is distributed under the MIT License. See
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) and the retained upstream
license for details.

## 9. Citation

This repository accompanies the following unpublished manuscript:

> Shoucai Jing, Wanpeng Zhu, Aohua Wang, Xiangmo Zhao, Xiaolong Ma, and Asad J.
> Khattak. *Mergingformer: distributionally consistent two-dimensional merging
> behavior model for autonomous vehicle simulation test at highway on-ramps*.

GitHub can also export the metadata in [`CITATION.cff`](CITATION.cff). The
publication venue, year, DOI, and paper URL will be added after publication.

## 10. Contributing

Focused issues and pull requests are welcome. Please read
[`CONTRIBUTING.md`](CONTRIBUTING.md) and do not upload private or
redistribution-restricted trajectory data.
