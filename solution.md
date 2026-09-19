# Reviewer Revision Solution

## What We Found

The project already had raw ERA5 data for seven complete years, from 2019 through 2025. The files contain hourly 2-meter air temperature (`t2m`) in Kelvin.

The raw data is valid:

- Time coverage: `2019-01-01 00:00` to `2025-12-31 23:00`
- Variable: `t2m`
- Units: Kelvin
- Raw grid: `721 x 1440`
- Coordinate used by the files: `valid_time`

The main problems were in the derived pipeline and model comparison code, not in the raw dataset.

## Problem 1: Processed Dates Did Not Match Raw Dates

The raw files contain:

- 2019
- 2020
- 2021
- 2022
- 2023
- 2024
- 2025

However, the old processed daily files represented only:

- Training: 2019-2021
- Validation: 2022
- Test: 2023

This happened because the processed artifacts were stale. The configuration and split code intended to use:

- Training: 2019-01-01 through 2023-12-31
- Validation: 2024-01-01 through 2024-12-31
- Test: 2025-01-01 through 2025-12-31

The split implementation itself is chronological and creates sequences separately inside each split, preventing windows from crossing split boundaries.

## Problem 2: The Merge Produced NaN Values

The raw files use `valid_time`, while the old merge path expected `time`. The old processing flow could align files incorrectly and create an all-NaN merged dataset.

The merge code was changed to:

1. Open each raw file.
2. Rename `valid_time` to `time`.
3. Keep only `t2m`.
4. Concatenate explicitly along `time`.
5. Sort the result chronologically.

The region extraction was also changed to select the study region from each raw yearly file before concatenating. This avoids creating a huge global seven-year intermediate file.

## Problem 3: The Study Region

The configured study region is:

- Latitude: `5°N` to `35°N`
- Longitude: `65°E` to `100°E`
- Regional grid: `121 x 141`

This is the India/South Asia study rectangle used by the code. The code documents the bounds, but the paper still needs a scientific explanation for why this region was selected.

## Problem 4: The Main Model Was Not Actually ConvLSTM

The class was named `ConvLSTMModel`, but its implementation is a Conv3D encoder-decoder:

- Conv3D: 1 input channel to 32 channels
- ReLU
- Conv3D: 32 channels to 32 channels
- ReLU
- Conv3D decoder: 32 channels to 1 channel
- Kernel size: `3 x 3 x 3`
- No dropout
- Trainable parameters: `29,441`
- Input: `(batch, 7, 1, 121, 141)`
- Output: `(batch, 1, 121, 141)`

The documentation was corrected to avoid falsely describing this as an LSTM-cell architecture.

## Problem 5: Baseline Models Were Not Comparable

The existing CNN-LSTM returned only a `1 x 1` prediction. The Transformer also returned a scalar and initially failed because it tried to project all spatial grid values through a layer configured for only 32 features.

The fixes were:

- CNN-LSTM now combines temporal LSTM context with the final spatial CNN feature map and decodes a full grid.
- Transformer now uses a `1 x 1` convolution to create spatial features, applies temporal attention to pooled features, and decodes a full grid.
- A new shared per-grid LSTM baseline was added.
- All models now return the same `121 x 141` output shape.
- Persistence predicts the next day as the final day in the seven-day input sequence.

## Problem 6: Tensor Axes Were Being Permuted Incorrectly

The sequence generator saves tensors in channel-first format:

- Inputs: `(N, T, C, H, W)`
- Targets: `(N, C, H, W)`

The data loader and training/validation code were transposing these tensors again, corrupting their dimensions. Those extra permutations were removed and shape validation was added.

## Problem 7: R2 and Correlation Were Confused

The values `R2 = 0.9720` and correlation `0.2477` do not appear in the current repository artifacts as a valid pair.

When the saved test prediction and target arrays were flattened identically:

- Correlation was approximately `0.9720`.
- R2 was approximately `0.9425`.

The values are different because R2 and correlation are different statistics. The corrected analysis calculates both from the same prediction and target arrays.

The old value `0.2477` was not found in the main current metric code or saved JSON outputs. It should not be reported unless an older notebook, figure, log, or paper draft identifies its origin.

## Problem 8: Seasonal, Regional, and Spatial Analysis

`reviewer_analysis.py` now produces:

- Seasonal metrics for Winter, Pre-monsoon/Summer, Monsoon, and Post-monsoon.
- Regional metrics for North, Central, and South latitude bands.
- Grid-cell RMSE map.
- Grid-cell MAE map.
- Grid-cell bias map.

Outputs are stored under `outputs/reviewer_revision/`.

## Baseline Results Already Produced

A one-epoch comparison was run before the final raw-data rebuild issue was discovered:

| Model | Test RMSE | Test MAE | Test R2 | Correlation |
|---|---:|---:|---:|---:|
| Persistence | 1.1277 | 0.9304 | 0.9794 | 0.9897 |
| LSTM | 3.9517 | 3.1145 | 0.7466 | 0.9497 |
| CNN-LSTM | 1.2654 | 1.0115 | 0.9740 | 0.9884 |
| Conv3D model named ConvLSTM | 0.9376 | 0.7025 | 0.9857 | 0.9935 |
| Transformer | 2.6350 | 2.1977 | 0.8874 | 0.9453 |

These results prove that the comparison code works, but they are not final paper results because the source tensors were generated before the raw-data pipeline was fully repaired.

## Why Only t2m Was Used

The repository is designed around one target variable: 2-meter air temperature. Using only `t2m` keeps the current study focused on next-day spatial temperature forecasting and avoids introducing unsupported assumptions about how wind, humidity, pressure, or precipitation should be encoded.

This is a scope limitation, not a claim that other variables are unimportant. Future work should evaluate multivariable inputs and independent observations.

## Current Status

Completed:

- Raw ERA5 coverage inspection
- Region bounds identification
- Correct chronological split design
- Model output compatibility
- LSTM baseline implementation
- Persistence baseline
- Shared comparison runner
- Seasonal analysis code
- Regional analysis code
- Spatial error maps
- Same-array R2/correlation recalculation
- README documentation updates

Still required before the paper can use final numbers:

- Successfully regenerate regional and processed arrays from raw ERA5 files.
- Verify finite normalization statistics.
- Verify all sequence arrays contain finite values.
- Rerun all five baselines on the regenerated tensors.
- Rerun seasonal, regional, and spatial analyses.
- Update the paper with the regenerated results.
- Add literature references and any independent-observation comparison if available.

The most recent failure was a Windows NetCDF file-lock/HDF issue involving the old derived files, not a problem with the raw ERA5 files. The next agent should close any open NetCDF viewer or VS Code data preview, then run the region-first preprocessing path described in `change.md`.
