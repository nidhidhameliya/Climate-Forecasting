"""Generate reproducible evidence for the reviewer revision requirements.

This script only uses committed data, configuration, model code, and checkpoints.
It writes provenance, comparable metrics for the saved ConvLSTM checkpoint and a
persistence baseline, seasonal/regional tables, and spatial error maps under
``outputs/reviewer_revision``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import xarray as xr
import yaml
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from models.convlstm import ConvLSTMModel


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs" / "reviewer_revision"


def load_config() -> dict[str, Any]:
    with (ROOT / "config.yaml").open() as handle:
        return yaml.safe_load(handle)


def metrics(pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    pred_flat = pred.reshape(-1)
    target_flat = target.reshape(-1)
    return {
        "rmse": float(np.sqrt(mean_squared_error(target_flat, pred_flat))),
        "mae": float(mean_absolute_error(target_flat, pred_flat)),
        "r2": float(r2_score(target_flat, pred_flat)),
        "correlation": float(np.corrcoef(pred_flat, target_flat)[0, 1]),
    }


def date_text(value: Any) -> str:
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def collect_provenance(config: dict[str, Any], arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    daily_paths = {
        split: ROOT / "data" / "interim" / f"{split}_daily.nc"
        for split in ("train", "val", "test")
    }
    split_info: dict[str, Any] = {}
    for split, path in daily_paths.items():
        with xr.open_dataset(path) as dataset:
            dates = pd.DatetimeIndex(dataset.time.values)
            split_info[split] = {
                "file": str(path.relative_to(ROOT)),
                "date_start": date_text(dates[0]),
                "date_end": date_text(dates[-1]),
                "daily_observations": len(dates),
                "latitude_start": float(dataset.latitude.values[0]),
                "latitude_end": float(dataset.latitude.values[-1]),
                "longitude_start": float(dataset.longitude.values[0]),
                "longitude_end": float(dataset.longitude.values[-1]),
                "spatial_shape": [int(dataset.sizes["latitude"]), int(dataset.sizes["longitude"])],
                "resampling_operation": "daily maximum (resample_time.py)",
                "sequence_samples": int(arrays[f"{split}_X"].shape[0]),
                "sequence_length": int(arrays[f"{split}_X"].shape[1]),
                "first_target_date": date_text(dates[arrays[f"{split}_X"].shape[1]]),
                "last_target_date": date_text(dates[-1]),
            }

    with (ROOT / "data" / "processed" / "mean_std.json").open() as handle:
        normalization = json.load(handle)

    model = ConvLSTMModel(config)
    architecture = {
        "class": "models.convlstm.ConvLSTMModel",
        "implementation_note": "Despite its historical name, this class is a Conv3D encoder-decoder, not an LSTM cell.",
        "input_shape": "(batch, time=7, channels=1, height=121, width=141)",
        "output_shape": "(batch, channels=1, height=121, width=141)",
        "layers": [
            {"name": "encoder.0", "type": "Conv3d", "in_channels": 1, "out_channels": config["model"]["hidden_dim"], "kernel_size": [3, 3, 3], "padding": 1},
            {"name": "encoder.1", "type": "ReLU"},
            {"name": "encoder.2", "type": "Conv3d", "in_channels": config["model"]["hidden_dim"], "out_channels": config["model"]["hidden_dim"], "kernel_size": [3, 3, 3], "padding": 1},
            {"name": "encoder.3", "type": "ReLU"},
            {"name": "decoder", "type": "Conv3d", "in_channels": config["model"]["hidden_dim"], "out_channels": 1, "kernel_size": [3, 3, 3], "padding": 1},
        ],
        "hidden_dim": config["model"]["hidden_dim"],
        "dropout": 0.0,
        "activation": "ReLU after both encoder convolutions",
        "trainable_parameters": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
    }
    return {
        "variable": config["variable"],
        "region": config["region"],
        "region_selection_note": "The configured rectangle covers the India study region and surrounding grid cells; the repository does not provide a separate scientific selection rationale.",
        "normalization": {
            "method_in_current_code": "training-split scalar z-score after Kelvin-to-Celsius conversion",
            "formula_in_current_code": "(value_celsius - train_mean_celsius) / train_std_celsius",
            "statistics": normalization,
            "missing_values": "nan_to_num(...), replacing remaining NaN/inf values with zero after normalization",
            "statistics_source": "data/interim/train_daily.nc only, as implemented in preprocessing/normalize.py",
            "artifact_consistency": "The saved mean 296.488... is Kelvin-scale, whereas current normalize.py computes Celsius statistics after subtracting 273.15. Regenerate preprocessing artifacts before publication and do not describe the checked-in arrays as verified Celsius-normalized outputs.",
        },
        "split_and_leakage": {
            "split_method": "independent chronological files created by year ranges in preprocessing/split_by_year.py",
            "configured_ranges": {"train": "2019-01-01..2023-12-31", "val": "2024-01-01..2024-12-31", "test": "2025-01-01..2025-12-31"},
            "sequence_method": "each split creates windows internally; no window crosses split boundaries",
            "caveat": "The checked-in daily NetCDF files do not match those configured year ranges; their actual dates are reported below and must be resolved before publication.",
        },
        "training": {
            "optimizer": "Adam",
            "learning_rate": config["training"]["learning_rate"],
            "batch_size": config["training"]["batch_size"],
            "configured_epochs": config["training"]["epochs"],
            "early_stopping": False,
            "loss": "ExtremeWeightedMSE (default extreme_weight=5.0; threshold file absent falls back to normalized threshold 0.0)",
        },
        "architecture": architecture,
        "splits": split_info,
    }


def load_arrays() -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    for split in ("train", "val", "test"):
        arrays[f"{split}_X"] = np.load(ROOT / "data" / "processed" / "tensors" / f"{split}_X.npy")
        arrays[f"{split}_y"] = np.load(ROOT / "data" / "processed" / "tensors" / f"{split}_y.npy")
    return arrays


def evaluate_test(config: dict[str, Any], arrays: dict[str, np.ndarray]) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    stats = json.loads((ROOT / "data" / "processed" / "mean_std.json").read_text())
    mean, std = float(stats["mean"]), float(stats["std"])
    target = arrays["test_y"][:, 0] * std + mean
    persistence = arrays["test_X"][:, -1, 0] * std + mean

    model = ConvLSTMModel(config)
    checkpoint = ROOT / "experiments" / "latest" / "model.pth"
    model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    model.eval()
    torch.set_num_threads(min(2, torch.get_num_threads()))
    predictions = []
    with torch.no_grad():
        for start in range(0, len(arrays["test_X"]), 2):
            batch = torch.from_numpy(arrays["test_X"][start : start + 2]).float()
            predictions.append(model(batch).numpy()[:, 0])
    prediction = np.concatenate(predictions, axis=0) * std + mean

    result = {
        "dataset": "test",
        "num_samples": int(target.shape[0]),
        "spatial_shape": list(target.shape[1:]),
        "models": {
            "Persistence": metrics(persistence, target),
            "Conv3D_encoder_decoder_checkpoint": metrics(prediction, target),
        },
        "metric_definition": "All metrics are computed over the same flattened denormalized test predictions and targets; the unresolved Kelvin/Celsius offset does not change RMSE, MAE, R2, or correlation.",
        "r2_correlation_note": "R2 and correlation are different statistics; both are recomputed from the identical arrays here.",
    }
    return result, prediction, target, persistence, np.arange(target.shape[0])


def seasonal_table(prediction: np.ndarray, target: np.ndarray) -> pd.DataFrame:
    with xr.open_dataset(ROOT / "data" / "interim" / "test_daily.nc") as dataset:
        dates = pd.DatetimeIndex(dataset.time.values)[7 : 7 + len(target)]
    season_by_month = {
        1: "Winter", 2: "Winter", 3: "Pre-monsoon/Summer", 4: "Pre-monsoon/Summer",
        5: "Pre-monsoon/Summer", 6: "Monsoon", 7: "Monsoon", 8: "Monsoon",
        9: "Monsoon", 10: "Post-monsoon", 11: "Post-monsoon", 12: "Winter",
    }
    seasons = np.array([season_by_month[month] for month in dates.month])
    frame = []
    for season in ["Winter", "Pre-monsoon/Summer", "Monsoon", "Post-monsoon"]:
        indices = np.flatnonzero(seasons == season)
        if len(indices):
            frame.append({"season": season, "samples": len(indices), **metrics(prediction[indices], target[indices])})
    return pd.DataFrame(frame)


def regional_table(prediction: np.ndarray, target: np.ndarray) -> pd.DataFrame:
    with xr.open_dataset(ROOT / "data" / "interim" / "test_daily.nc") as dataset:
        lat = dataset.latitude.values
        lon = dataset.longitude.values
    latitude, longitude = np.meshgrid(lat, lon, indexing="ij")
    masks = {
        "North": latitude >= 25,
        "Central": (latitude >= 15) & (latitude < 25),
        "South": latitude < 15,
    }
    rows = []
    for name, mask in masks.items():
        rows.append({"region": name, "latitude_definition": {"North": ">=25", "Central": "15 to <25", "South": "<15"}[name], "longitude_range": f"{lon.min():g} to {lon.max():g}", "grid_cells": int(mask.sum()), **metrics(prediction[:, mask], target[:, mask])})
    return pd.DataFrame(rows)


def save_spatial_outputs(prediction: np.ndarray, target: np.ndarray) -> None:
    with xr.open_dataset(ROOT / "data" / "interim" / "test_daily.nc") as dataset:
        lat, lon = dataset.latitude.values, dataset.longitude.values
    errors = prediction - target
    fields = {
        "rmse": np.sqrt(np.mean(errors**2, axis=0)),
        "mae": np.mean(np.abs(errors), axis=0),
        "bias": np.mean(errors, axis=0),
    }
    for name, field in fields.items():
        np.save(OUTPUT_DIR / f"spatial_{name}.npy", field)
        fig, axis = plt.subplots(figsize=(8, 5))
        image = axis.pcolormesh(lon, lat, field, shading="auto", cmap="coolwarm")
        axis.set(xlabel="Longitude (degrees east)", ylabel="Latitude (degrees north)", title=f"Test-set spatial {name.upper()} (stored temperature scale)")
        fig.colorbar(image, ax=axis, label="Stored temperature scale")
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / f"spatial_{name}.png", dpi=300)
        plt.close(fig)


def main() -> None:
    global OUTPUT_DIR
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    args = parser.parse_args()
    OUTPUT_DIR = Path(args.output_dir)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    config = load_config()
    arrays = load_arrays()
    provenance = collect_provenance(config, arrays)
    results, prediction, target, persistence, _ = evaluate_test(config, arrays)
    provenance_path = OUTPUT_DIR / "dataset_and_architecture_metadata.json"
    provenance_path.write_text(json.dumps(provenance, indent=2))
    (OUTPUT_DIR / "test_model_comparison.json").write_text(json.dumps(results, indent=2))
    seasonal_table(prediction, target).to_csv(OUTPUT_DIR / "seasonal_metrics.csv", index=False)
    regional_table(prediction, target).to_csv(OUTPUT_DIR / "regional_metrics.csv", index=False)
    save_spatial_outputs(prediction, target)
    print(f"Wrote reviewer evidence to {OUTPUT_DIR}")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()