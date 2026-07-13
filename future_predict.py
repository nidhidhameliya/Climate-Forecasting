"""
Future prediction module for dates beyond the training data.
Uses a hybrid approach of auto-regressive forecasting for the short-term and
climatology-based estimation for the long-term.
"""

import json
import os
import sys
from datetime import datetime, timedelta
from typing import Any, Dict, Tuple

import numpy as np
import torch

SHORT_TERM_FUTURE_DAYS = 14


def load_model_and_stats():
    """Load trained model and normalization statistics."""
    sys.path.insert(0, os.path.dirname(__file__))
    from models.convlstm import ConvLSTMModel

    import yaml

    model_path = "experiments/latest/model.pth"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")

    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ConvLSTMModel(config)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # Load normalization stats
    with open("data/processed/mean_std.json") as f:
        stats = json.load(f)

    mean = float(stats.get("mean", 288.15))
    std = float(stats.get("std", 10.0))

    return model, config, mean, std, device


def denormalize(data, mean, std):
    """Denormalize from normalized space to Celsius."""
    return data * std + mean - 273.15


def get_date_ranges_for_split(split: str) -> Tuple[datetime, datetime]:
    """Get date ranges for each split."""
    date_ranges = {
        "train": (datetime(2019, 1, 1), datetime(2023, 12, 31)),
        "val": (datetime(2024, 1, 1), datetime(2024, 12, 31)),
        "test": (datetime(2025, 1, 1), datetime(2025, 12, 31)),
    }
    return date_ranges.get(split, date_ranges["test"])


def get_target_dates_for_split(
    split: str, num_samples: int, sequence_length: int = 7
) -> list[datetime]:
    """Get the target date represented by each sample in a split."""
    start_date, _ = get_date_ranges_for_split(split)
    first_target_date = start_date + timedelta(days=sequence_length)
    return [first_target_date + timedelta(days=idx) for idx in range(num_samples)]


def circular_day_distance(day_a: int, day_b: int, year_length: int = 366) -> int:
    """Smallest circular distance between two day-of-year values."""
    distance = abs(day_a - day_b)
    return min(distance, year_length - distance)


def load_tensor_data(split="test"):
    """Load preprocessed tensors for prediction."""
    x_path = f"data/processed/tensors/{split}_X.npy"
    y_path = f"data/processed/tensors/{split}_y.npy"
    if not (os.path.exists(x_path) and os.path.exists(y_path)):
        return None, None
    return np.load(x_path), np.load(y_path)


def get_climatology_for_day(
    target_day_of_year: int, sequence_length: int = 7, window_days: int = 10
) -> np.ndarray:
    """Average normalized target grids near a target day-of-year across all splits."""
    matched_grids = []

    for split in ["train", "val", "test"]:
        _, y_data = load_tensor_data(split)
        if y_data is None:
            continue

        sample_dates = get_target_dates_for_split(
            split, len(y_data), sequence_length=sequence_length
        )
        matched_indices = [
            idx
            for idx, sample_date in enumerate(sample_dates)
            if circular_day_distance(sample_date.timetuple().tm_yday, target_day_of_year)
            <= window_days
        ]

        if matched_indices:
            matched_grids.append(y_data[matched_indices, 0])

    if not matched_grids:
        raise ValueError("No climatology samples found for the requested day.")

    return np.concatenate(matched_grids, axis=0).mean(axis=0)


def get_recent_anomaly_grid(
    split: str = "test", sequence_length: int = 7, climatology_window_days: int = 10
) -> np.ndarray:
    """Estimate the recent anomaly relative to climatology in normalized space."""
    _, y_data = load_tensor_data(split)
    if y_data is None:
        raise ValueError(f"Tensor data for split '{split}' could not be loaded.")

    _, end_date = get_date_ranges_for_split(split)
    baseline_grid = get_climatology_for_day(
        end_date.timetuple().tm_yday,
        sequence_length=sequence_length,
        window_days=climatology_window_days,
    )
    recent_mean_grid = y_data[-min(sequence_length, len(y_data)) :, 0].mean(axis=0)
    return recent_mean_grid - baseline_grid


def predict_future_days(
    num_days: int, split: str = "test"
) -> Dict[datetime, Dict[str, Any]]:
    """
    Auto-regressive prediction for multiple future days.

    Args:
        num_days: Number of days to predict into the future (1-365)
        split: Which split to use for initial sequence ('test', 'val', 'train')

    Returns:
        dict with predictions for each day
    """
    if num_days < 1 or num_days > 365:
        raise ValueError("num_days must be between 1 and 365")

    model, config, mean, std, device = load_model_and_stats()

    # Get the last 7-day sequence to start with
    _, y_data = load_tensor_data(split)
    if y_data is None:
        raise FileNotFoundError(f"Data file not found for split '{split}'")

    sequence_length = config["sequence_length"]
    sequence = y_data[-sequence_length:].copy()

    # Determine start date
    _, end_date = get_date_ranges_for_split(split)

    predictions_by_date = {}
    current_date = end_date

    print(
        f"Starting auto-regressive prediction from {current_date.strftime('%Y-%m-%d')}"
    )
    print(f"Predicting {num_days} days into the future...")

    for day_ahead in range(num_days):
        current_date = current_date + timedelta(days=1)

        # Prepare input for model
        X_input = torch.from_numpy(sequence).unsqueeze(0).float().to(device)

        # Make prediction
        with torch.no_grad():
            pred_norm = model(X_input)

        pred_norm_np = pred_norm.squeeze(0).cpu().numpy()

        # Prevent exploding predictions over multiple days
        pred_norm_np = np.clip(pred_norm_np, -4.0, 4.0)
        pred_norm_np = pred_norm_np * 0.95

        # Denormalize to Celsius
        pred_celsius = denormalize(pred_norm_np.squeeze(0), mean, std)

        # Store prediction
        predictions_by_date[current_date] = {
            "prediction": pred_celsius,
            "mean_temp": float(np.mean(pred_celsius)),
            "min_temp": float(np.min(pred_celsius)),
            "max_temp": float(np.max(pred_celsius)),
            "std_temp": float(np.std(pred_celsius)),
        }

        # Update sequence: shift left and add new prediction (normalized)
        sequence = np.vstack([sequence[1:], pred_norm_np[np.newaxis, ...]])

        if (day_ahead + 1) % 10 == 0:
            print(f"  Predicted {day_ahead + 1}/{num_days} days...")

    print(f"✅ Prediction complete! Generated {num_days} forecasts.")
    return predictions_by_date


def predict_specific_future_date(
    target_date: datetime, split: str = "test"
) -> Dict[str, Any]:
    """
    Predict for a specific future date.

    Args:
        target_date: datetime object for the future date to predict
        split: Which split to use for initial sequence

    Returns:
        Prediction dict for that date
    """
    _, end_date = get_date_ranges_for_split(split)

    days_ahead = (target_date - end_date).days

    if days_ahead < 1:
        raise ValueError(f"Target date {target_date} is not in the future")

    if days_ahead > 365:
        raise ValueError(f"Cannot predict more than 365 days ahead")

    model, config, mean, std, device = load_model_and_stats()
    sequence_length = config["sequence_length"]

    # For long horizons, use a climatology-based estimate
    if days_ahead > SHORT_TERM_FUTURE_DAYS:
        target_day_of_year = target_date.timetuple().tm_yday
        climatology_norm = get_climatology_for_day(
            target_day_of_year,
            sequence_length=sequence_length,
            window_days=10,
        )
        recent_anomaly_norm = get_recent_anomaly_grid(
            split=split,
            sequence_length=sequence_length,
            climatology_window_days=10,
        )
        anomaly_scale = float(np.exp(-(days_ahead - SHORT_TERM_FUTURE_DAYS) / 30.0))
        pred_norm_grid = climatology_norm + recent_anomaly_norm * anomaly_scale
        pred_celsius = denormalize(pred_norm_grid, mean=mean, std=std)
        return {
            "prediction": pred_celsius,
            "mean_temp": float(np.mean(pred_celsius)),
            "min_temp": float(np.min(pred_celsius)),
            "max_temp": float(np.max(pred_celsius)),
            "std_temp": float(np.std(pred_celsius)),
            "is_future": True,
            "method": "seasonal_climatology",
            "days_ahead": days_ahead,
            "confidence_note": "Seasonal baseline with recent anomaly decay for long-range outlooks.",
        }

    # For short horizons, use auto-regressive prediction
    predictions = predict_future_days(days_ahead, split=split)
    result = predictions.get(target_date)

    if result is None:
        raise KeyError(f"Prediction for {target_date.strftime('%Y-%m-%d')} not found")

    result.update(
        {
            "is_future": True,
            "method": "hybrid_short_term",
            "days_ahead": days_ahead,
            "confidence_note": "Short-range ConvLSTM forecast blended toward seasonal climatology for stability.",
        }
    )
    return result


if __name__ == "__main__":
    try:
        # Example: Predict 30 days into the future from test set end date
        predictions = predict_future_days(30, split="test")

        # Show summary
        print("\n" + "=" * 60)
        print("FUTURE FORECAST SUMMARY (Next 30 Days)")
        print("=" * 60)
        for date_obj, pred in sorted(predictions.items())[-7:]:
            print(
                f"{date_obj.strftime('%Y-%m-%d')}: Mean={pred['mean_temp']:.2f}°C, "
                f"Range=[{pred['min_temp']:.2f}, {pred['max_temp']:.2f}]°C"
            )

    except Exception as e:
        print(f"Error: {e}")
