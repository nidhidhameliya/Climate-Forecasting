# Change Log and Agent Handoff

## Purpose

This file summarizes the reviewer-related changes made in this repository so another coding agent can continue without repeating the audit.

## Verified Raw Dataset

The raw ERA5 directory contains:

- `data/raw/era5/era5_2019.nc`
- `data/raw/era5/era5_2020.nc`
- `data/raw/era5/era5_2021.nc`
- `data/raw/era5/era5_2022.nc`
- `data/raw/era5/era5_2023.nc`
- `data/raw/era5/era5_2024.nc`
- `data/raw/era5/era5_2025.nc`

Each file contains only `t2m`, hourly observations, Kelvin units, a global `721 x 1440` grid, and complete yearly coverage. The raw coverage is exactly `2019-01-01 00:00` through `2025-12-31 23:00`.

The raw files use the coordinate name `valid_time`, not `time`.

## Code Changes

### Model compatibility

- `models/cnn_lstm.py`
  - Changed the output from a scalar `1 x 1` map to a full `121 x 141` spatial prediction.
  - Added a spatial decoder using a `1 x 1` convolution.
- `models/transformer.py`
  - Replaced the invalid linear projection of all `17,061` grid values.
  - Added a `1 x 1` spatial input projection.
  - Configured the Transformer encoder with `batch_first=True`.
  - Added a full-grid spatial decoder.
- `models/lstm_baseline.py`
  - Added a per-grid-cell shared LSTM baseline.
- `models/model_utils.py`
  - Added the `lstm` model option.

All four learned models now accept `(batch, time, channel, height, width)` and return `(batch, 1, height, width)`.

### Tensor layout

- `data_loader/data_loader.py`
  - Removed an incorrect transpose. The saved sequence tensors are already channel-first.
  - Added shape validation.
- `training/train.py`
  - Removed the incorrect training-time permutation.
- `training/validate.py`
  - Removed the incorrect validation-time permutation.

### Baseline experiment runner

- Added `run_baseline_experiments.py`.
- Supports:
  - Persistence
  - LSTM
  - CNN-LSTM
  - Conv3D model currently named `ConvLSTMModel`
  - Transformer
- Uses the same saved train/validation/test tensors, sequence length, one-day horizon, optimizer, loss, and flattened metrics.
- Saves checkpoints and JSON/CSV comparison results under `experiments/reviewer_baselines/`.
- Uses seed `42`, CPU in the current environment, and supports resuming/merging completed model results.

### Reviewer analysis

- Added `reviewer_analysis.py`.
- Generates:
  - Dataset and architecture metadata
  - Persistence/checkpoint metrics
  - Seasonal metrics
  - Regional metrics
  - Spatial RMSE, MAE, and bias arrays/maps
- Outputs are under `outputs/reviewer_revision/`.

### Pipeline entry point

- `main.py`
  - Fixed import from the missing `climate_forecasting.utils` package to the existing root `pipeline_utils.py`.
- `pipeline_utils.py`
  - Fixed the project-root calculation.

### Raw-data preprocessing changes

- `preprocessing/merge_years.py`
  - Normalizes `valid_time` to `time` before concatenation.
  - Concatenates yearly files explicitly and sorts by time.
  - This was intended to prevent the old merge path from creating all-NaN values.
- `preprocessing/subset_region.py`
  - Now subsets each raw file to latitude `5..35` and longitude `65..100` before concatenation.
  - Writes `data/interim/region_subset_raw_verified.nc` instead of the locked stale `region_subset.nc`.
- `preprocessing/split_by_year.py`
  - Reads `region_subset_raw_verified.nc`.
  - Configured split ranges remain train `2019-2023`, validation `2024`, test `2025`.

## Existing Reviewer Documentation Changes

`README.md` was updated with:

- Study bounds
- t2m-only rationale and limitation
- Architecture facts
- Leakage discussion
- Corrected metric interpretation
- Seasonal/regional/spatial output references
- More conservative claims about model superiority

## Existing Experiment Results

A previous one-epoch comparison was recorded in `experiments/reviewer_baselines/comparison_results.json` and `.csv` using the then-existing processed tensors:

| Model | Test RMSE | Test MAE | Test R2 | Correlation |
|---|---:|---:|---:|---:|
| Persistence | 1.1277 | 0.9304 | 0.9794 | 0.9897 |
| LSTM | 3.9517 | 3.1145 | 0.7466 | 0.9497 |
| CNN-LSTM | 1.2654 | 1.0115 | 0.9740 | 0.9884 |
| Conv3D/ConvLSTM-named model | 0.9376 | 0.7025 | 0.9857 | 0.9935 |
| Transformer | 2.6350 | 2.1977 | 0.8874 | 0.9453 |

These are one-epoch smoke/comparison experiments, not final tuned research results. They must be regenerated after the raw-derived preprocessing artifacts are repaired and verified.

## Important Current Blocker

The latest rebuild attempt found that the old processed artifacts were stale and that the old global merge path was too large and produced all-NaN output. The raw files themselves are finite and valid.

During the latest rebuild:

- `train.nc`, `val.nc`, and `test.nc` were correctly regenerated for 2019-2023, 2024, and 2025.
- Daily files had the expected counts: train `1826`, validation `366`, test `365`.
- Normalization then produced `NaN` mean/std and zero-valued arrays because the regional source artifact was still affected by a failed/locked NetCDF write.
- Do not use the current `data/processed/*.npy`, `mean_std.json`, or dependent reviewer metrics as final scientific results until preprocessing is rerun successfully from raw files and finite-value checks pass.

## Recommended Next Agent Sequence

1. Close any VS Code/NetCDF viewer holding files in `data/interim` open.
2. Run `python -m preprocessing.subset_region` and verify `region_subset_raw_verified.nc` contains finite `t2m` values.
3. Run `python -m preprocessing.split_by_year`.
4. Run `python -m preprocessing.resample_time`.
5. Run `python -m preprocessing.normalize` and verify `mean_std.json` contains finite numbers and all normalized arrays are finite.
6. Run `python -m preprocessing.create_sequences`.
7. Run `python reviewer_analysis.py`.
8. Run `python run_baseline_experiments.py --epochs 1 --batch-size 1 --models persistence,lstm,cnn_lstm,convlstm,transformer`.
9. Update README metrics only from the regenerated artifacts.

Do not claim that the full baseline comparison is final until this sequence succeeds on the raw-derived tensors.
