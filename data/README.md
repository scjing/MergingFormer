# Data

This directory is reserved for datasets used by MergingFormer. Dataset files are
excluded from version control by default.

## Expected format

The current main data loader (`src/data_processing.py`) reads every `.csv` file
from one directory. It expects the following input columns:

- `carCenterXft`
- `carCenterYft`
- `speedY`
- `speed`
- `laneId`
- `pressure`
- `leadCarCenterXft`
- `leadCarCenterYft`
- `leadSpeed`
- `leftLeadCarCenterXft`
- `leftLeadCarCenterYft`
- `leftLeadSpeed`
- `leftFollowCarCenterXft`
- `leftFollowCarCenterYft`
- `leftFollowSpeed`

The prediction targets are:

- `speedY`
- `speed`

Missing values are currently replaced with `-1`, and the loader applies
min-max scaling before generating fixed-length sequences.

## Directory layout

Place CSV files in a local subdirectory, for example:

```text
data/
└── cleaned_test01/
    ├── trajectory_001.csv
    └── trajectory_002.csv
```

Update the `data_folder` value in the selected training or evaluation script to
match the local path.

## Data availability

The original dataset is not open-sourced and is not included in this
repository. The Apache License 2.0 used for the repository source code grants no
rights to the dataset. Users must supply their own lawfully obtained data in a
compatible format.

Do not commit original records, derived records, samples, plots that expose
records, or dataset metadata unless the data owner separately authorizes their
publication.

## Current limitation

Some evaluation scripts define 23 input features and 7 output features, while
the current main data loader defines 15 input features and 2 targets. These
scripts and their associated data schema have not yet been reconciled or
validated.
