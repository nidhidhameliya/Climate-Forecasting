import os
import json
import logging
import time
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from pathlib import Path
from typing import List, Dict, Any, Tuple

from models.convlstm import ConvLSTMModel
from config import load_config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def plot_loss_curve(log_file: Path, output_path: Path) -> None:
    """
    Plots and saves the training and validation loss curves from a log file.

    Args:
        log_file: Path to the training log file.
        output_path: Path to save the resulting plot.
    """
    train_losses = []
    val_losses = []
    with open(log_file, 'r') as f:
        for line in f:
            if "Train Loss:" in line and "Val Loss:" in line:
                parts = line.split(',')
                train_loss_part = parts[1].split(': ')[1]
                val_loss_part = parts[2].split(': ')[1]
                train_losses.append(float(train_loss_part))
                val_losses.append(float(val_loss_part))

    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(train_losses, label='Training Loss', color='royalblue', linewidth=2)
    ax.plot(val_losses, label='Validation Loss', color='darkorange', linestyle='--', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss (MSE)', fontsize=12)
    ax.set_title('Training and Validation Loss Curve', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.set_facecolor('white')
    fig.set_facecolor('white')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Loss curve saved to {output_path}")
    logging.info(f"Loss curve saved to {output_path}")

def plot_actual_vs_predicted(y_true: np.ndarray, y_pred: np.ndarray, output_path: Path) -> None:
    """
    Plots a scatter plot of actual vs. predicted values.

    Args:
        y_true: Ground truth values.
        y_pred: Predicted values.
        output_path: Path to save the resulting plot.
    """
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(8, 8))

    # Flatten and sample to avoid overplotting
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    
    # Use a random sample of 5000 points for clarity
    if len(y_true_flat) > 5000:
        indices = np.random.choice(len(y_true_flat), 5000, replace=False)
        y_true_flat = y_true_flat[indices]
        y_pred_flat = y_pred_flat[indices]

    ax.scatter(y_true_flat, y_pred_flat, alpha=0.5, color='royalblue', edgecolors='w', s=50)
    
    # Add a 1:1 line
    lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),
        np.max([ax.get_xlim(), ax.get_ylim()]),
    ]
    ax.plot(lims, lims, 'k--', alpha=0.75, zorder=0, label='Ideal Fit')
    
    ax.set_xlabel('Actual Temperature (°C)', fontsize=12)
    ax.set_ylabel('Predicted Temperature (°C)', fontsize=12)
    ax.set_title('Actual vs. Predicted Temperature', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.set_aspect('equal', adjustable='box')
    ax.set_facecolor('white')
    fig.set_facecolor('white')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Actual vs. Predicted graph saved to {output_path}")
    logging.info(f"Actual vs. Predicted graph saved to {output_path}")

def plot_spatial_maps(y_true: np.ndarray, y_pred: np.ndarray, output_path: Path) -> None:
    """
    Plots spatial maps of ground truth and predicted temperature for a sample.

    Args:
        y_true: Ground truth temperature maps.
        y_pred: Predicted temperature maps.
        output_path: Path to save the resulting plot.
    """
    true_map = y_true[0, 0, :, :]
    pred_map = y_pred[0, 0, :, :]

    vmin = min(true_map.min(), pred_map.min())
    vmax = max(true_map.max(), pred_map.max())
    norm = colors.Normalize(vmin=vmin, vmax=vmax)

    plt.style.use('default')
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor='white')
    fig.suptitle('Spatial Temperature Prediction (Sample)', fontsize=16, fontweight='bold')

    # Ground Truth
    im1 = axes[0].imshow(true_map, cmap='coolwarm', norm=norm)
    axes[0].set_title('Ground Truth', fontsize=12)
    axes[0].set_xlabel('Longitude Index')
    axes[0].set_ylabel('Latitude Index')

    # Predicted Temperature
    im2 = axes[1].imshow(pred_map, cmap='coolwarm', norm=norm)
    axes[1].set_title('Predicted Temperature', fontsize=12)
    axes[1].set_xlabel('Longitude Index')
    axes[1].set_ylabel('') # Hide y-label for cleaner look

    # Add a single colorbar for both maps
    fig.colorbar(im1, ax=axes.ravel().tolist(), shrink=0.8, label='Temperature (°C)')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Spatial prediction map saved to {output_path}")
    logging.info(f"Spatial prediction map saved to {output_path}")

def denormalize(data: np.ndarray, mean: float, std: float) -> np.ndarray:
    """
    Denormalizes the data using the given mean and standard deviation.

    Args:
        data: The normalized data array.
        mean: The mean used for normalization.
        std: The standard deviation used for normalization.

    Returns:
        The denormalized data array.
    """
    return data * std + mean

def main():
    """Main function to run the evaluation pipeline."""
    start_time = time.time()
    logging.info("🚀 Starting evaluation process...")

    # --- Configuration and Setup ---
    cfg = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    exp_dir = Path(cfg['experiment'].get('path', 'experiments/latest'))
    model_path = exp_dir / 'best_model.pth'
    log_path = exp_dir / 'training_log.txt'

    # Create results directory
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)

    # --- Load Data ---
    logging.info("📂 Loading test data and normalization stats...")
    X_test = np.load(cfg['paths'].get('test_X_path', 'data/processed/tensors/test_X.npy'))
    y_test_norm = np.load(cfg['paths'].get('test_y_path', 'data/processed/tensors/test_y.npy'))

    norm_stats_path = cfg['paths'].get('norm_stats_path', 'data/processed/mean_std.json')
    with open(norm_stats_path, 'r') as f:
        norm_stats = json.load(f)
    mean = norm_stats.get('mean', {}).get('t2m', 0.0)
    std = norm_stats.get('std', {}).get('t2m', 1.0)

    # --- Load Model ---
    logging.info(f"🧠 Loading model from {model_path} onto {device}...")
    model = ConvLSTMModel(
        input_channels=1,
        hidden_dim=cfg['model']['hidden_dim'],
        num_layers=cfg['model']['num_layers'],
        output_size=y_test_norm.shape[-2] * y_test_norm.shape[-1]
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # --- Run Predictions ---
    logging.info("⚡ Running predictions on the test set...")
    predictions_norm = []
    batch_size = cfg['training'].get('batch_size', 8)
    test_tensor = torch.from_numpy(X_test).float()

    with torch.no_grad():
        for i in range(0, len(test_tensor), batch_size):
            batch_X = test_tensor[i:i+batch_size].to(device)
            pred = model(batch_X)
            predictions_norm.append(pred.cpu().numpy())
    
    y_pred_norm = np.concatenate(predictions_norm, axis=0)

    # --- Denormalize Data ---
    logging.info("🔄 Denormalizing predictions and ground truth...")
    y_pred = denormalize(y_pred_norm, mean, std) - 273.15 # Convert to Celsius
    y_test = denormalize(y_test_norm, mean, std) - 273.15 # Convert to Celsius

    # --- Calculate Metrics ---
    logging.info("📊 Calculating performance metrics...")
    rmse = np.sqrt(mean_squared_error(y_test.flatten(), y_pred.flatten()))
    mae = mean_absolute_error(y_test.flatten(), y_pred.flatten())
    r2 = r2_score(y_test.flatten(), y_pred.flatten())

    metrics = {
        'RMSE': rmse,
        'MAE': mae,
        'R2 Score': r2
    }

    print("\n--- Evaluation Results ---")
    print(f"{'Metric':<10} {'Value':<10}")
    print("-" * 25)
    for name, value in metrics.items():
        print(f"{name:<10} {value:<10.4f}")
    print("-" * 25)

    # Save metrics to CSV
    metrics_df = pd.DataFrame([metrics])
    metrics_path = results_dir / 'metrics.csv'
    metrics_df.to_csv(metrics_path, index=False)
    logging.info(f"✅ Metrics saved to {metrics_path}")

    # --- Save Predictions ---
    # For simplicity, saving predictions for the first sample date
    logging.info("💾 Saving sample predictions to CSV...")
    print("💾 Saving sample predictions to CSV...")
    first_sample_true = y_test[0].flatten()
    first_sample_pred = y_pred[0].flatten()
    
    # Create lat/lon grids
    lat_dim, lon_dim = y_test.shape[2], y_test.shape[3]
    lats = np.linspace(cfg['region']['lat_max'], cfg['region']['lat_min'], lat_dim)
    lons = np.linspace(cfg['region']['lon_min'], cfg['region']['lon_max'], lon_dim)
    lon_grid, lat_grid = np.meshgrid(lons, lats)

    pred_df = pd.DataFrame({
        'Date': pd.to_datetime('today').strftime('%Y-%m-%d'), # Placeholder date
        'Latitude': lat_grid.flatten(),
        'Longitude': lon_grid.flatten(),
        'Actual_Temperature_C': first_sample_true,
        'Predicted_Temperature_C': first_sample_pred
    })
    predictions_path = results_dir / 'predictions.csv'
    pred_df.to_csv(predictions_path, index=False)
    logging.info(f"✅ Sample predictions saved to {predictions_path}")

    # --- Generate Visualizations ---
    logging.info("🎨 Generating visualizations...")
    plot_loss_curve(log_path, results_dir / 'loss_curve.png')
    plot_actual_vs_predicted(y_test, y_pred, results_dir / 'actual_vs_predicted.png')
    plot_spatial_maps(y_test, y_pred, results_dir / 'spatial_prediction.png')

    # --- Final Verification ---
    print("\n--- Final Verification ---")
    print(f"✔ {'Graphs generated':<25}: {all((results_dir / f).exists() for f in ['loss_curve.png', 'actual_vs_predicted.png', 'spatial_prediction.png'])}")
    print(f"✔ {'CSV files generated':<25}: {all((results_dir / f).exists() for f in ['metrics.csv', 'predictions.csv'])}")
    print(f"✔ {'Results folder created':<25}: {results_dir.exists()}")
    
    end_time = time.time()
    logging.info(f"✨ Evaluation complete in {end_time - start_time:.2f} seconds.")

if __name__ == '__main__':
    main()