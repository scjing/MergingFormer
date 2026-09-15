# MergingFormer

MergingFormer is a PyTorch research project for sequence modelling and vehicle
trajectory prediction in lane-changing scenarios. The repository contains the
main MergingFormer implementation together with several recurrent and
Transformer-based comparison models.

> **Project status:** this repository is being prepared for its first public
> release. The installation, training, and evaluation workflow has not yet been
> validated in a clean environment. Interfaces and file paths may change.

## Repository structure

```text
.
├── data/       # Local datasets (excluded from version control)
├── layers/     # Attention, embedding, correlation, and encoder/decoder layers
├── models/     # MergingFormer and comparison model definitions
├── src/        # Data loading, training, evaluation, and test scripts
├── tools/      # Small environment inspection utilities
└── utils/      # Masks, time features, metrics, and preprocessing helpers
```

The current main model is defined in `models/MyTransformer.py`, and the
corresponding training entry point is `src/train.py`. Additional scripts cover
LSTM, BiLSTM, Transformer, Performer, Informer, Autoformer, DLinear, and hybrid
variants.

## Requirements

- Python 3.9 or a compatible version
- PyTorch
- The packages listed in `requirements.txt`
- A CUDA-capable GPU is optional; the scripts select CUDA when it is available

Create a virtual environment and install the inferred dependencies:

```bash
python -m venv .venv

# Windows PowerShell
.venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

Exact dependency versions have not yet been validated or locked.

## Data preparation

The dataset is not open-sourced or distributed with this repository. Users must
provide their own lawfully obtained, compatible CSV data. Place local CSV files
under a subdirectory of `data/` and update the `data_folder` value in the script
you intend to use.

See [`data/README.md`](data/README.md) for the CSV columns expected by the
current main data loader and for known schema differences between scripts.

## Training and evaluation

The repository currently contains standalone research scripts rather than a
single stable command-line interface:

- `src/train.py` trains the model from `models/MyTransformer.py`.
- `src/evaluate.py` evaluates a saved model and generates metrics and plots.
- Files named `src/train_*.py` and `src/test_*.py` cover comparison models and
  experimental variants.

Before running a script, review its data path, checkpoint path, feature count,
target count, and model configuration. Several values are currently defined
inside the scripts. Commands are intentionally omitted until the workflow has
been validated from a clean checkout.

Training outputs are written to a checkpoint directory and include model
weights, per-epoch losses, and a loss curve. These generated files are excluded
from version control.

## Known limitations

- The project has not yet been tested from a clean installation.
- Dependency versions are not pinned.
- Data and pretrained weights are not included.
- Training and evaluation defaults currently use different feature dimensions
  in some scripts.
- Data paths and most hyperparameters are hard-coded in individual scripts.
- The current loader concatenates CSV files before creating sliding windows and
  randomly splitting samples. This behaviour should be reviewed before using
  the code for reported experiments.

## Contributing

Bug reports and focused pull requests are welcome once the first public release
is available. When reporting an issue, include the operating system, Python and
PyTorch versions, the script used, and the full error message. Do not attach
private or restricted trajectory data.

See [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) for the pending source
and license audit.

## Acknowledgements

Parts of the reusable time-series layers are derived from or based on
[THUML Time-Series-Library](https://github.com/thuml/Time-Series-Library), which
is distributed under the MIT License. See
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) for details and the retained
upstream license.

## Citation

If this repository accompanies a paper, citation metadata will be added before
the first stable release.

## License

The source code in this repository is licensed under the Apache License 2.0.
See [`LICENSE`](LICENSE). This license does not apply to the dataset, which is
not included or open-sourced as part of this project.
