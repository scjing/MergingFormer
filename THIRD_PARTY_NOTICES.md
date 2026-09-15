# Third-party notices

No third-party source notices were present in the codebase when this repository
was prepared for public release. The following source relationship was
identified by comparing the files with their upstream counterparts.

## THUML Time-Series-Library

Portions of the `layers/`, `models/`, and `utils/` directories are derived from
or based on:

- Project: Time-Series-Library
- Copyright: THUML and Time-Series-Library contributors
- Source: <https://github.com/thuml/Time-Series-Library>
- License: MIT License
- Local license copy:
  [`third_party/licenses/Time-Series-Library-LICENSE`](third_party/licenses/Time-Series-Library-LICENSE)

The affected files include the AutoCorrelation, Autoformer encoder/decoder,
convolution, Crossformer encoder/decoder, ETSformer encoder/decoder, embedding,
Fourier correlation, multi-wavelet correlation, Pyraformer encoder/decoder,
self-attention, normalization, and Transformer encoder/decoder implementations.
They also include the masking, time-feature, and training utility modules, and
the Autoformer, DLinear, and Informer comparison-model implementations. The
local copies differ from the current upstream versions and have been adapted for
this project's model interfaces and experiments.

Before publishing the repository, review at least these areas:

- attention, embedding, encoder/decoder, correlation, and normalization code in
  `layers/`;
- Autoformer, Informer, DLinear, Performer, Transformer, LSTM, and BiLSTM
  comparison implementations in `models/`;
- training and evaluation scripts adapted from experiments, papers, notebooks,
  or other repositories; and
- preprocessing and metric utilities in `utils/`.

For each adapted component, record:

| Component or path | Upstream project | Source URL | Upstream license | Local modifications |
| --- | --- | --- | --- | --- |
| `layers/*.py` components listed above | THUML Time-Series-Library | <https://github.com/thuml/Time-Series-Library> | MIT | Local interfaces and code differ from current upstream versions |
| `models/db_Autoformer.py`, `models/db_DLinear.py`, `models/db_informer.py` | THUML Time-Series-Library | <https://github.com/thuml/Time-Series-Library> | MIT | Adapted to local model interfaces |
| `utils/masking.py`, `utils/timefeatures.py`, `utils/tools.py` | THUML Time-Series-Library | <https://github.com/thuml/Time-Series-Library> | MIT | Local code differs from current upstream versions |
| Remaining `models/`, `src/`, and `utils/` files | _Pending audit_ |  |  |  |

Copy all notices required by upstream licenses into this repository. If an
upstream component has no license, obtain permission or replace it before the
public release.
