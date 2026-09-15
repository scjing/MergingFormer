# Data

This directory is reserved for datasets used by MergingFormer. Dataset files are
excluded from version control by default.

## Provenance and citation

The experimental dataset used by this project was produced by processing the
**Freeway C Merge/Diverge Segment** of the
[UCF-SST CitySim Dataset](https://github.com/UCF-SST-Lab/UCF-SST-CitySim1-Dataset).
CitySim and the processed trajectories are not included in this repository.
Users must obtain the source data from the official provider and comply with
the terms attached to their access.

Research using these data should cite the CitySim publication:

```bibtex
@article{zheng2024citysim,
  author  = {Ou Zheng and Mohamed Abdel-Aty and Lishengsa Yue and
             Amr Abdelraouf and Zijin Wang and Nada Mahmoud},
  title   = {CitySim: A Drone-Based Vehicle Trajectory Dataset for
             Safety-Oriented Research and Digital Twins},
  journal = {Transportation Research Record},
  volume  = {2678},
  number  = {4},
  pages   = {606--621},
  year    = {2024},
  doi     = {10.1177/03611981231185768}
}
```

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

The CitySim source data and the processed dataset are not included in this
repository. The Apache License 2.0 used for the repository source code grants no
rights to either dataset. Users must supply their own lawfully obtained data in
a compatible format.

Do not commit original records, derived records, samples, plots that expose
records, or dataset metadata unless the data owner separately authorizes their
publication.

## Current limitation

Some evaluation scripts define 23 input features and 7 output features, while
the current main data loader defines 15 input features and 2 targets. These
scripts and their associated data schema have not yet been reconciled or
validated.
