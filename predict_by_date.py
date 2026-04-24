"""
Day-wise prediction helper for Streamlit dashboard.
Makes predictions for specific dates using the trained model.
"""

import torch
import numpy as np
import json
import os
from datetime import datetime, timedelta
import sys

def load_model_and_data():
    """Load the trained model and normalization statistics."""
    # Load model
    model_path = "experiments/latest/model.pth"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}. Run training first.")
    
    # Add models to path
    sys.path.insert(0, os.path.dirname(__file__))
    from models.convlstm import ConvLSTM
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ConvLSTM(input_channels=1, output_channels=1, device=device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    # Load normalization statistics
    with open("data/processed/mean_std.json", "r") as f:
        norm_stats = json.load(f)
    
    return model, norm_stats, device

def load_processed_data(split="test"):
    """Load processed sequences with dates."""
    X_path = f"data/processed/tensors/{split}_X.npy"
    y_path = f"data/processed/tensors/{split}_y.npy"
    
    if not os.path.exists(X_path) or not os.path.exists(y_path):
        raise FileNotFoundError(f"Data files not found for split '{split}'")
    
    X = np.load(X_path)  # Shape: (num_samples, 7, 121, 141)
    y = np.load(y_path)  # Shape: (num_samples, 121, 141)
    
    return X, y

def get_date_range(split="test"):
    """Get approximate date range for the split."""
    # Based on preprocessing: train(2019-2023), val(2024), test(2025)
    date_ranges = {
        "train": ("2019-01-01", "2023-12-31"),
        "val": ("2024-01-01", "2024-12-31"),
        "test": ("2025-01-01", "2025-12-31"),
    }
    
    if split not in date_ranges:
        split = "test"
    
    start, end = date_ranges[split]
    return datetime.strptime(start, "%Y-%m-%d"), datetime.strptime(end, "%Y-%m-%d")

def predict_for_sample(sample_idx, split="test", denormalize=True):
    """
    Predict for a specific sample index.
    
    Args:
        sample_idx: Index of the sample in the processed data
        split: 'train', 'val', or 'test'
        denormalize: Whether to denormalize to Celsius
    
    Returns:
        dict with prediction, actual, metrics
    """
    model, norm_stats, device = load_model_and_data()
    X, y = load_processed_data(split)
    
    if sample_idx >= len(X):
        raise ValueError(f"Sample index {sample_idx} out of range (max: {len(X)-1})")
    
    # Extract single sample and prepare for model
    X_sample = X[sample_idx:sample_idx+1]  # (1, 7, 121, 141)
    y_sample = y[sample_idx:sample_idx+1]  # (1, 121, 141)
    
    # Reshape for ConvLSTM: (B, T, C, H, W)
    X_tensor = torch.tensor(X_sample, dtype=torch.float32).unsqueeze(2)  # (1, 7, 1, 121, 141)
    X_tensor = X_tensor.permute(0, 2, 1, 3, 4).to(device)  # (1, 1, 7, 121, 141)
    
    # Make prediction
    with torch.no_grad():
        pred_norm = model(X_tensor)  # (1, 1, 121, 141)
    
    pred_norm = pred_norm.squeeze(0).squeeze(0).cpu().numpy()  # (121, 141)
    y_sample = y_sample.squeeze(0)  # (121, 141)
    
    # Denormalize
    mean = norm_stats["mean"]
    std = norm_stats["std"]
    
    pred_kelvin = (pred_norm * std) + mean
    y_kelvin = (y_sample * std) + mean
    
    pred_celsius = pred_kelvin - 273.15
    y_celsius = y_kelvin - 273.15
    
    # Calculate metrics
    rmse = np.sqrt(np.mean((pred_celsius - y_celsius) ** 2))
    mae = np.mean(np.abs(pred_celsius - y_celsius))
    correlation = np.corrcoef(pred_celsius.flatten(), y_celsius.flatten())[0, 1]
    
    result = {
        "sample_idx": sample_idx,
        "split": split,
        "prediction_celsius": pred_celsius,
        "actual_celsius": y_celsius,
        "prediction_kelvin": pred_kelvin,
        "actual_kelvin": y_kelvin,
        "metrics": {
            "rmse_celsius": float(rmse),
            "mae_celsius": float(mae),
            "correlation": float(correlation),
            "rmse_kelvin": float(np.sqrt(np.mean((pred_kelvin - y_kelvin) ** 2))),
            "mae_kelvin": float(np.mean(np.abs(pred_kelvin - y_kelvin))),
        },
        "statistics": {
            "pred_mean_celsius": float(np.mean(pred_celsius)),
            "actual_mean_celsius": float(np.mean(y_celsius)),
            "pred_min_celsius": float(np.min(pred_celsius)),
            "pred_max_celsius": float(np.max(pred_celsius)),
            "actual_min_celsius": float(np.min(y_celsius)),
            "actual_max_celsius": float(np.max(y_celsius)),
        }
    }
    
    return result

def predict_for_date_range(start_date, end_date, split="test"):
    """
    Predict for a range of dates.
    
    Args:
        start_date: datetime object or string "YYYY-MM-DD"
        end_date: datetime object or string "YYYY-MM-DD"
        split: 'train', 'val', or 'test'
    
    Returns:
        list of predictions for each date
    """
    if isinstance(start_date, str):
        start_date = datetime.strptime(start_date, "%Y-%m-%d")
    if isinstance(end_date, str):
        end_date = datetime.strptime(end_date, "%Y-%m-%d")
    
    # Get date range for split
    split_start, split_end = get_date_range(split)
    
    # Clamp to split range
    start_date = max(start_date, split_start)
    end_date = min(end_date, split_end)
    
    # Approximate sample indices (assuming daily data)
    # Each year has ~365 days, but this is approximate
    days_diff_start = (start_date - split_start).days
    days_diff_end = (end_date - split_start).days
    
    X, y = load_processed_data(split)
    num_samples = len(X)
    
    # Scale to available samples
    start_idx = int((days_diff_start / (split_end - split_start).days) * num_samples)
    end_idx = int((days_diff_end / (split_end - split_start).days) * num_samples)
    
    start_idx = max(0, start_idx)
    end_idx = min(num_samples - 1, end_idx)
    
    predictions = []
    for idx in range(start_idx, end_idx + 1):
        try:
            pred = predict_for_sample(idx, split=split)
            predictions.append(pred)
        except Exception as e:
            print(f"Error predicting for sample {idx}: {e}")
    
    return predictions

if __name__ == "__main__":
    # Test: predict for first sample in test set
    try:
        result = predict_for_sample(0, split="test")
        print(f"Prediction for sample 0:")
        print(f"RMSE: {result['metrics']['rmse_celsius']:.4f}°C")
        print(f"MAE: {result['metrics']['mae_celsius']:.4f}°C")
        print(f"Mean Prediction: {result['statistics']['pred_mean_celsius']:.2f}°C")
        print(f"Mean Actual: {result['statistics']['actual_mean_celsius']:.2f}°C")
    except Exception as e:
        print(f"Error: {e}")
