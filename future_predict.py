"""
Future prediction module for dates beyond the training data.
Uses auto-regressive forecasting to predict multiple days into the future.
"""

import torch
import numpy as np
import json
import os
from datetime import datetime, timedelta
import sys


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


def get_last_sequence(split="test"):
    """Get the last 7-day sequence from the data for starting future predictions."""
    y_path = f"data/processed/tensors/{split}_y.npy"
    
    if not os.path.exists(y_path):
        raise FileNotFoundError(f"Data file not found: {y_path}")
    
    y_data = np.load(y_path)  # Shape: (num_samples, 121, 141)
    
    # Get last 7 days worth of data from test set
    # Return the last sample and work backwards
    last_sequence = []
    
    # We'll use the last 7 y values as proxy for sequence
    for i in range(max(0, len(y_data) - 7), len(y_data)):
        last_sequence.append(y_data[i])
    
    # Pad if needed
    while len(last_sequence) < 7:
        last_sequence.insert(0, y_data[0])
    
    last_sequence = np.array(last_sequence[-7:])  # Ensure exactly 7 days
    return last_sequence


def denormalize(data, mean, std):
    """Denormalize from normalized space to Celsius."""
    return data * std + mean - 273.15


def normalize(data, mean, std):
    """Normalize from Celsius to normalized space."""
    return (data + 273.15 - mean) / std


def predict_future_days(num_days, split="test", start_from_date=None):
    """
    Auto-regressive prediction for multiple future days.
    
    Args:
        num_days: Number of days to predict into the future (1-365)
        split: Which split to use for initial sequence ('test', 'val', 'train')
        start_from_date: Optional - if None, starts from last available date
    
    Returns:
        dict with predictions for each day
    """
    if num_days < 1 or num_days > 365:
        raise ValueError("num_days must be between 1 and 365")
    
    model, config, mean, std, device = load_model_and_stats()
    
    # Get the last 7-day sequence to start with
    sequence = get_last_sequence(split)  # Shape: (7, 121, 141)
    
    # Determine start date
    date_ranges = {
        "train": (datetime(2019, 1, 1), datetime(2023, 12, 31)),
        "val": (datetime(2024, 1, 1), datetime(2024, 12, 31)),
        "test": (datetime(2025, 1, 1), datetime(2025, 12, 31)),
    }
    _, end_date = date_ranges.get(split, date_ranges["test"])
    current_date = start_from_date if start_from_date else end_date
    
    predictions = {}
    
    print(f"Starting auto-regressive prediction from {current_date.strftime('%Y-%m-%d')}")
    print(f"Predicting {num_days} days into the future...")
    
    for day_ahead in range(num_days):
        current_date = current_date + timedelta(days=1)
        
        # Prepare input: (7, 121, 141) → (1, 7, 121, 141) → (1, 1, 7, 121, 141)
        X_input = torch.tensor(sequence[np.newaxis, :, np.newaxis, :, :], 
                               dtype=torch.float32, device=device)  # (1, 7, 1, 121, 141)
        X_input = X_input.permute(0, 2, 1, 3, 4)  # (1, 1, 7, 121, 141)
        
        # Make prediction
        with torch.no_grad():
            pred_norm = model(X_input)  # (1, 1, 121, 141)
        
        pred_norm = pred_norm.squeeze(0).squeeze(0).cpu().numpy()  # (121, 141)
        
        # Denormalize to Celsius
        pred_celsius = denormalize(pred_norm, mean, std)
        
        # Store prediction
        predictions[current_date.strftime('%Y-%m-%d')] = {
            "prediction": pred_celsius,
            "prediction_norm": pred_norm,
            "mean_temp": float(np.mean(pred_celsius)),
            "min_temp": float(np.min(pred_celsius)),
            "max_temp": float(np.max(pred_celsius)),
            "std_temp": float(np.std(pred_celsius)),
        }
        
        # Update sequence: shift left and add new prediction (normalized)
        sequence = np.vstack([sequence[1:], pred_norm[np.newaxis, :, :]])
        
        if (day_ahead + 1) % 10 == 0:
            print(f"  Predicted {day_ahead + 1}/{num_days} days...")
    
    print(f"✅ Prediction complete! Generated {num_days} forecasts.")
    return predictions


def predict_specific_future_date(target_date, split="test"):
    """
    Predict for a specific future date.
    
    Args:
        target_date: datetime object for the future date to predict
        split: Which split to use for initial sequence
    
    Returns:
        Prediction dict for that date
    """
    date_ranges = {
        "train": (datetime(2019, 1, 1), datetime(2023, 12, 31)),
        "val": (datetime(2024, 1, 1), datetime(2024, 12, 31)),
        "test": (datetime(2025, 1, 1), datetime(2025, 12, 31)),
    }
    _, end_date = date_ranges.get(split, date_ranges["test"])
    
    days_ahead = (target_date - end_date).days
    
    if days_ahead < 1:
        raise ValueError(f"Target date {target_date} is not in the future")
    
    if days_ahead > 365:
        raise ValueError(f"Cannot predict more than 365 days ahead")
    
    print(f"Predicting {days_ahead} days ahead to {target_date.strftime('%Y-%m-%d')}")
    
    predictions = predict_future_days(days_ahead, split=split, start_from_date=end_date)
    
    key = target_date.strftime('%Y-%m-%d')
    if key in predictions:
        return predictions[key]
    else:
        raise KeyError(f"Prediction for {key} not found")


if __name__ == "__main__":
    try:
        # Example: Predict 30 days into the future from test set end date
        predictions = predict_future_days(30, split="test")
        
        # Show summary
        print("\n" + "="*60)
        print("FUTURE FORECAST SUMMARY (Next 30 Days)")
        print("="*60)
        for date_str, pred in sorted(predictions.items())[-7:]:
            print(f"{date_str}: Mean={pred['mean_temp']:.2f}°C, "
                  f"Range=[{pred['min_temp']:.2f}, {pred['max_temp']:.2f}]°C")
        
    except Exception as e:
        print(f"Error: {e}")
