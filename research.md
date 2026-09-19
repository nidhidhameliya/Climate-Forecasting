# Climate Forecasting Paper — Reviewer Revision Audit

## Scope of this audit

This document is a repository-based audit of the climate forecasting project as it exists in this workspace. It follows the project evidence in the checked-in code, data files, and output artifacts rather than the README narrative alone.

The following files were inspected and used as primary evidence:

- [README.md](README.md)
- [config.yaml](config.yaml)
- [preprocessing/subset_region.py](preprocessing/subset_region.py)
- [preprocessing/split_by_year.py](preprocessing/split_by_year.py)
- [preprocessing/resample_time.py](preprocessing/resample_time.py)
- [preprocessing/normalize.py](preprocessing/normalize.py)
- [preprocessing/create_sequences.py](preprocessing/create_sequences.py)
- [models/convlstm.py](models/convlstm.py)
- [models/cnn_lstm.py](models/cnn_lstm.py)
- [models/transformer.py](models/transformer.py)
- [models/lstm_baseline.py](models/lstm_baseline.py)
- [training/train.py](training/train.py)
- [training/test.py](training/test.py)
- [training/metrics.py](training/metrics.py)
- [training/losses.py](training/losses.py)
- [run_baseline_experiments.py](run_baseline_experiments.py)
- [reviewer_analysis.py](reviewer_analysis.py)
- [change.md](change.md)
- [outputs/reviewer_revision/dataset_and_architecture_metadata.json](outputs/reviewer_revision/dataset_and_architecture_metadata.json)
- [outputs/reviewer_revision/test_model_comparison.json](outputs/reviewer_revision/test_model_comparison.json)
- [experiments/reviewer_baselines/comparison_results.json](experiments/reviewer_baselines/comparison_results.json)

## Verified repository facts

### Dataset and source

The raw ERA5 directory contains 7 yearly NetCDF files, and the repository documents that the raw coverage is 2019-01-01 00:00 through 2025-12-31 23:00. The raw global grid in the source files is 721 latitude rows by 1440 longitude columns, consistent with ERA5.

This is confirmed by the checked-in project notes in [change.md](change.md) and by the actual raw file metadata read from the repository.

### Study region

The configured study region is explicitly defined in [config.yaml](config.yaml) as:

- lat_min = 5
- lat_max = 35
- lon_min = 65
- lon_max = 100

The region extraction script [preprocessing/subset_region.py](preprocessing/subset_region.py) subsets the raw data with:

- latitude=slice(35, 5)
- longitude=slice(65, 100)

The synthetic dataset generator [generate_synthetic_data.py](generate_synthetic_data.py) confirms the intended grid size:

- latitude start = 5, end = 35, step = 0.25
- longitude start = 65, end = 100, step = 0.25

This gives:

- latitude count = ((35 - 5) / 0.25) + 1 = 121
- longitude count = ((100 - 65) / 0.25) + 1 = 141

Therefore the study grid is verified as 121 × 141. The 0.25° spatial resolution is also consistent with the project’s synthetic grid specification and with the actual region subset.

### Actual daily split files currently on disk

The daily split files in [data/interim](data/interim) currently show:

- train_daily.nc: 2019-01-01 through 2023-12-31, 1826 daily observations
- val_daily.nc: 2024-01-01 through 2024-12-31, 366 daily observations
- test_daily.nc: 2025-01-01 through 2025-12-31, 365 daily observations

The repository therefore currently contains a 2019-2023 / 2024 / 2025 temporal split on disk, even though the configured split script and some stale reviewer artifacts still describe different historical ranges.

### Model implementation

The primary model is implemented in [models/convlstm.py](models/convlstm.py) and is a Conv3D encoder-decoder, not a true LSTM-cell implementation. The architecture is:

- encoder.0: Conv3d, in_channels = 1, out_channels = 32, kernel_size = (3, 3, 3), padding = 1
- ReLU
- encoder.2: Conv3d, in_channels = 32, out_channels = 32, kernel_size = (3, 3, 3), padding = 1
- ReLU
- decoder: Conv3d, in_channels = 32, out_channels = 1, kernel_size = (3, 3, 3), padding = 1

The output is created by taking the last time step from the decoder result: `out[:, :, -1, :, :]`.

This model has 29,441 trainable parameters, as reported in [outputs/reviewer_revision/dataset_and_architecture_metadata.json](outputs/reviewer_revision/dataset_and_architecture_metadata.json).

### Normalization and missing values

The current normalization path in [preprocessing/normalize.py](preprocessing/normalize.py) does the following:

- loads train_daily.nc, val_daily.nc, and test_daily.nc
- converts Kelvin to Celsius with `value - 273.15`
- computes mean and std from the training split only
- computes `train_norm = (train - mean) / std`, and similarly for validation and test
- applies `np.nan_to_num(..., nan=0.0)` after normalization

The saved [data/processed/mean_std.json](data/processed/mean_std.json) is currently invalid: it contains `NaN` values. This means the processed arrays are not publication-ready because the current normalization artifact is not finite.

The project also contains a fix script [fix_normalization.py](fix_normalization.py), which is evidence that the current normalization artifacts were known to be stale or corrupted and that they should be regenerated before use in a paper.

### Baselines and comparison scripts

The repository contains baselines in:

- [models/lstm_baseline.py](models/lstm_baseline.py)
- [models/cnn_lstm.py](models/cnn_lstm.py)
- [models/transformer.py](models/transformer.py)
- [run_baseline_experiments.py](run_baseline_experiments.py)

The baseline runner evaluates:

- Persistence
- LSTM
- CNN-LSTM
- ConvLSTM-equivalent current model
- Transformer

However, the checked-in comparison results are one-epoch smoke results and are not a final tuned research benchmark. The project notes in [change.md](change.md) explicitly say these results must be regenerated after the raw-derived preprocessing artifacts are repaired and verified.

## Conflicts in the repository

The repository contains unresolved or stale inconsistencies that must be reported instead of silently choosing one value.

### Conflict 1: configured split ranges vs actual current split files

In [preprocessing/split_by_year.py](preprocessing/split_by_year.py), the split script is configured to create:

- train: 2019-01-01 to 2023-12-31
- val: 2024-01-01 to 2024-12-31
- test: 2025-01-01 to 2025-12-31

However, some reviewer metadata files still describe older ranges, for example the stale output in [outputs/reviewer_revision/dataset_and_architecture_metadata.json](outputs/reviewer_revision/dataset_and_architecture_metadata.json) says:

- train: 2019-01-01 to 2021-12-31
- val: 2022-01-01 to 2022-12-31
- test: 2023-01-01 to 2023-12-31

The actual current split files on disk are the 2019-2023 / 2024 / 2025 ranges. The stale reviewer metadata is not the actual current experiment provenance.

### Conflict 2: normalization artifact values

The file [preprocessing/normalize.py](preprocessing/normalize.py) states that it converts Hz Kelvin to Celsius before normalization and computes the statistics from the training split only. The saved file [data/processed/mean_std.json](data/processed/mean_std.json) currently contains `NaN` values. The reviewer metadata in [outputs/reviewer_revision/dataset_and_architecture_metadata.json](outputs/reviewer_revision/dataset_and_architecture_metadata.json) lists a mean of 296.4881571400821 and std of 7.789291077828382, but those values are inconsistent with the current code output because the current processed arrays are invalid. The repository explicitly documents that the saved normalization statistics should be regenerated before publication.

### Conflict 3: README claims vs current evidence

The README and other documents state a 1,089-sequence train split and a 358/358 validation/test split. These values are not confirmed by the current daily split files because the processed tensor files are stale or invalid and should be regenerated. The project’s own reviewer notes state that the checked-in daily files and processed outputs must be regenerated before publication.

## Current evidence from saved benchmark artifacts

The checked-in comparison artifacts in [outputs/reviewer_revision/test_model_comparison.json](outputs/reviewer_revision/test_model_comparison.json) and [experiments/reviewer_baselines/comparison_results.json](experiments/reviewer_baselines/comparison_results.json) report the following values on the repository’s saved test arrays:

| Model | RMSE | MAE | R² | Correlation |
|---|---:|---:|---:|---:|
| Persistence | 1.1277 | 0.9304 | 0.9794 | 0.9897 |
| LSTM | 3.9517 | 3.1145 | 0.7466 | 0.9497 |
| CNN-LSTM | 1.2654 | 1.0115 | 0.9740 | 0.9884 |
| Conv3D/ConvLSTM-named model | 0.9376 | 0.7025 | 0.9857 | 0.9935 |
| Transformer | 2.6350 | 2.1977 | 0.8874 | 0.9453 |

These are actual project outputs, but the project notes explicitly state that they are not final scientific results because the preprocessing artifacts have not yet been regenerated and verified properly.

## Reviewer-by-reviewer status

### Reviewer 1

[1] Study region — DONE

Verified from [config.yaml](config.yaml), [preprocessing/subset_region.py](preprocessing/subset_region.py), and the synthetic grid definition: 5°N to 35°N, 65°E to 100°E, 121 × 141 grid, 0.25° resolution. The repository does not provide a separate scientific rationale for the region beyond the fact that it is the configured India study region.

[2] Exact dates — PARTIAL

The daily split files currently on disk are verified as:

- train_daily.nc: 2019-01-01 to 2023-12-31
- val_daily.nc: 2024-01-01 to 2024-12-31
- test_daily.nc: 2025-01-01 to 2025-12-31

The older reviewer metadata still refers to a different date range and must be corrected before publication. The processed tensor counts are not yet trustworthy because the normalization artifact is invalid.

[3] ConvLSTM architecture — DONE

The actual implementation is a Conv3D encoder-decoder with two encoder convolutions and one decoder convolution. Parameter count: 29,441.

[4] Data split / leakage — PARTIAL

There is no evidence of split leakage in the window generation itself because [preprocessing/create_sequences.py](preprocessing/create_sequences.py) creates windows within each split independently. However, the repository currently has conflicting split metadata and stale processed artifacts, so the final chronology is not yet publication-ready.

[5] Missing values / normalization — PARTIAL

Missing-value handling is not a verified scientific preprocessing method. The current pipeline does not remove NaNs before training in a robust way; it applies `np.nan_to_num` after normalization. The `mean_std.json` artifact is invalid (`NaN`), so the final normalization statement cannot be published without regeneration.

[6] Figures — PARTIAL

The repository contains reviewer-specific outputs under [outputs/reviewer_revision](outputs/reviewer_revision), including spatial maps and metrics, but the project notes explicitly say those artifacts must be regenerated after preprocessing is repaired and verified. The figures are therefore provisional, not final publication figures.

### Reviewer 2

[1] Baselines — PARTIAL

The repository contains a fair baseline framework, but the saved baseline comparison is a one-epoch smoke benchmark rather than a final research benchmark. The project notes indicate that all baseline comparisons must be regenerated after preprocessing is fixed.

[2] Why t2m — DONE

The project states that the study focuses on 2-meter air temperature as a controlled single-variable forecasting task and explicitly notes that this is a scope decision, not a claim that additional variables are unnecessary.

[3] Seasonal / regional / spatial analysis — PARTIAL

The reviewer analysis script generates seasonal, regional, and spatial artifacts, but the repository’s own evidence says these outputs must be regenerated by a repaired preprocessing pipeline before publication.

[4] R² / correlation — DONE

The actual script [reviewer_analysis.py](reviewer_analysis.py) recomputes both metrics from the same flattened arrays and documents that R² and correlation are distinct statistics. The discrepancy in the paper is not a code bug; it is the consequence of using different metrics that are mathematically different.

### Reviewer 3

[1] Introduction / citations — PARTIAL

The introduction and citation quality in the repo are not fully verified as publication-quality. The project contains dataset and software acknowledgements, but no thorough reviewer-level evidence audit is yet complete.

[2] Figure 3 explanation — PARTIAL

The repository contains reviewer baselines and comparison artifacts, but the project notes explicitly say these must be regenerated and the figure explanation should be tied to a reproducible experiment. The current state is not final.

[3] Independent validation — NOT DONE

There are no verified station observations or operational forecast comparisons available in the repository.

[4] Literature comparison — PARTIAL

The repository has a general references and dataset citation, but the comparison table is not yet fully verified against comparable studies and task definitions.

[5] Conclusion moderation — PARTIAL

The README and reviewer notes are more conservative than the older paper claims, but a final moderated conclusion cannot yet be fully verified because the dataset and normalized outputs remain unstable.

## Verified final results

The following are the only values that can be reported without invention from the current repository evidence:

- Dataset period (raw ERA5): 2019-01-01 00:00 through 2025-12-31 23:00
- Study region: latitude 5°N to 35°N; longitude 65°E to 100°E
- Grid: 121 × 141
- Spatial resolution: 0.25° latitude × 0.25° longitude
- Primary model: Conv3D encoder-decoder named `ConvLSTMModel` in [models/convlstm.py](models/convlstm.py)
- Parameter count: 29,441
- Observed current daily files: train 1826 / val 366 / test 365
- Current normalization artifact: invalid (`NaN` mean/std in [data/processed/mean_std.json](data/processed/mean_std.json))
- Current saved benchmark values from the artifact JSON: Persistence RMSE 1.1277, MAE 0.9304, R² 0.9794, correlation 0.9897; Conv3D checkpoint RMSE 1.8823, MAE 1.6127, R² 0.9425, correlation 0.9720

## Not verified / still required

- A final publication-grade normalization pass from valid training statistics
- Final processed train/validation/test arrays and sequence-count verification
- Final tuned baseline comparison under the same preprocessing and split protocol
- Independent external validation
- Final publication-ready figures and tables
- A final corrected paper conclusion and literature comparison table

## Files changed

- [research.md](research.md) created as the audit and revision-status record.

## Files generated

- [outputs/reviewer_revision/dataset_and_architecture_metadata.json](outputs/reviewer_revision/dataset_and_architecture_metadata.json)
- [outputs/reviewer_revision/test_model_comparison.json](outputs/reviewer_revision/test_model_comparison.json)
- [experiments/reviewer_baselines/comparison_results.json](experiments/reviewer_baselines/comparison_results.json)
- [outputs/reviewer_revision/seasonal_metrics.csv](outputs/reviewer_revision/seasonal_metrics.csv)
- [outputs/reviewer_revision/regional_metrics.csv](outputs/reviewer_revision/regional_metrics.csv)
- [outputs/reviewer_revision/spatial_rmse.png](outputs/reviewer_revision/spatial_rmse.png)
- [outputs/reviewer_revision/spatial_mae.png](outputs/reviewer_revision/spatial_mae.png)
- [outputs/reviewer_revision/spatial_bias.png](outputs/reviewer_revision/spatial_bias.png)

# Reviewer Revision Status

## Reviewer 1
[1] Study region — DONE
[2] Exact dates — PARTIAL
[3] ConvLSTM architecture — DONE
[4] Data split/leakage — PARTIAL
[5] Missing values/normalization — PARTIAL
[6] Figures — PARTIAL

## Reviewer 2
[1] Baselines — PARTIAL
[2] Why t2m — DONE
[3] Seasonal/regional/spatial analysis — PARTIAL
[4] R²/correlation — DONE

## Reviewer 3
[1] Introduction/citations — PARTIAL
[2] Figure 3 explanation — PARTIAL
[3] Independent validation — NOT DONE
[4] Literature comparison — PARTIAL
[5] Conclusion moderation — PARTIAL

## Verified Final Results
- Dataset period: Raw ERA5 data covers 2019-01-01 00:00 to 2025-12-31 23:00; current daily split files are 2019-2023 train, 2024 validation, and 2025 test.
- Study region: 5°N to 35°N, 65°E to 100°E.
- Grid: 121 × 141.
- Samples: current daily split files are 1826 / 366 / 365 for train / validation / test.
- Input window: 7 days (verified in [config.yaml](config.yaml) and [preprocessing/create_sequences.py](preprocessing/create_sequences.py)).
- Forecast horizon: 1 day.
- Train/validation/test: current on-disk chronology is 2019-2023 / 2024 / 2025, but the processed tensor pipeline is not final because the normalization artifact is invalid.
- ConvLSTM architecture: two Conv3D encoder layers with 32 channels and one Conv3D decoder, ReLU activations, padding 1, 29,441 parameters.
- Parameters: 29,441.
- RMSE: current saved checkpoint artifact reported RMSE 1.8823 on the repository’s saved test arrays; the persistence baseline was 1.1277.
- MAE: current saved checkpoint artifact reported MAE 1.6127; persistence was 0.9304.
- R²: current saved checkpoint artifact reported R² 0.9425; persistence was 0.9794.
- Correlation: current saved checkpoint artifact reported correlation 0.9720; persistence was 0.9897.

## NOT VERIFIED / STILL REQUIRED
- Final publication-grade preprocessing output
- Final regenerated benchmark metrics under a valid normalized dataset
- Independent external validation
- Fully revised literature and conclusion sections
- Final figure package for a camera-ready manuscript

## Files Changed
- [research.md](research.md)

## Files Generated
- [outputs/reviewer_revision/dataset_and_architecture_metadata.json](outputs/reviewer_revision/dataset_and_architecture_metadata.json)
- [outputs/reviewer_revision/test_model_comparison.json](outputs/reviewer_revision/test_model_comparison.json)
- [outputs/reviewer_revision/seasonal_metrics.csv](outputs/reviewer_revision/seasonal_metrics.csv)
- [outputs/reviewer_revision/regional_metrics.csv](outputs/reviewer_revision/regional_metrics.csv)
- [outputs/reviewer_revision/spatial_rmse.png](outputs/reviewer_revision/spatial_rmse.png)
- [outputs/reviewer_revision/spatial_mae.png](outputs/reviewer_revision/spatial_mae.png)
- [outputs/reviewer_revision/spatial_bias.png](outputs/reviewer_revision/spatial_bias.png)
