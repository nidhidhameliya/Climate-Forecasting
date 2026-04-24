import torch
import numpy as np
from models.convlstm import ConvLSTMModel
import json
import yaml

print("="*70)
print("🔍 DEBUGGING PREDICTIONS - Why all splits are identical?")
print("="*70)

# Load config
with open("config.yaml") as f:
    config = yaml.safe_load(f)

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ConvLSTMModel(config)
model.to(device)
model.load_state_dict(torch.load("experiments/latest/model.pth", map_location=device))
model.eval()

print(f"✓ Model loaded on {device}")

# Load a sample
X_test = np.load("data/processed/tensors/test_X.npy")
y_test = np.load("data/processed/tensors/test_y.npy")

print(f"\n📊 DATA SHAPES:")
print(f"  X_test: {X_test.shape}")
print(f"  y_test: {y_test.shape}")

# Check data values
print(f"\n🔬 DATA STATISTICS:")
print(f"  X_test range: [{np.min(X_test):.6f}, {np.max(X_test):.6f}]")
print(f"  X_test mean:  {np.mean(X_test):.6f}")
print(f"  X_test std:   {np.std(X_test):.6f}")
print(f"  y_test range: [{np.min(y_test):.6f}, {np.max(y_test):.6f}]")
print(f"  y_test mean:  {np.mean(y_test):.6f}")
print(f"  y_test std:   {np.std(y_test):.6f}")

# Check how many unique values
print(f"\n  Unique values in X_test: {len(np.unique(X_test))}")
print(f"  Unique values in y_test: {len(np.unique(y_test))}")

# Test prediction
X_sample = torch.tensor(X_test[:1], dtype=torch.float32).to(device)
print(f"\n🔮 SAMPLE PREDICTION:")
print(f"  Input shape to model: {X_sample.shape}")

with torch.no_grad():
    pred = model(X_sample)

print(f"  Output shape: {pred.shape}")
print(f"  Output range: [{torch.min(pred):.6f}, {torch.max(pred):.6f}]")
print(f"  Output mean: {torch.mean(pred):.6f}")
print(f"  Output std: {torch.std(pred):.6f}")

# Compare with target
target = torch.tensor(y_test[:1], dtype=torch.float32).to(device)
print(f"\n  Target range: [{torch.min(target):.6f}, {torch.max(target):.6f}]")
print(f"  Target mean: {torch.mean(target):.6f}")

# Check if predictions are constant
pred_np = pred.cpu().numpy()
print(f"\n📌 KEY FINDINGS:")
print(f"  All prediction values same? {len(np.unique(pred_np)) == 1}")
print(f"  All target values same? {len(np.unique(target.cpu().numpy())) == 1}")

# Sample actual values
print(f"\n  First 5 pred values: {pred_np.flatten()[:5]}")
print(f"  First 5 target values: {target.cpu().numpy().flatten()[:5]}")

# Load normalization stats
try:
    with open("data/processed/mean_std.json") as f:
        stats = json.load(f)
    mean = float(stats.get("mean", 288.15))
    std = float(stats.get("std", 10.0))
    print(f"\n📐 NORMALIZATION STATS:")
    print(f"  Mean: {mean}")
    print(f"  Std: {std}")
    
    # What do these values denormalize to?
    denorm_pred = pred_np * std + mean
    denorm_target = target.cpu().numpy() * std + mean
    
    print(f"\n🌡️  DENORMALIZED VALUES:")
    print(f"  Pred (K): {denorm_pred.flatten()[0]:.2f} K")
    print(f"  Target (K): {denorm_target.flatten()[0]:.2f} K")
    print(f"  Pred (°C): {denorm_pred.flatten()[0] - 273.15:.2f}°C")
    print(f"  Target (°C): {denorm_target.flatten()[0] - 273.15:.2f}°C")
except Exception as e:
    print(f"  Could not load normalization stats: {e}")

print("\n" + "="*70)
print("INTERPRETATION:")
print("="*70)
print("""
If all values are ~0.0 in normalized space:
  → Model predicting mean temperature (0 after normalization)
  → Targets also at mean (all ~0)
  → RMSE ≈ 0 because no variance to capture

Possible issues:
1. Data was normalized to zero mean (check mean_std.json)
2. Model learns to output constant mean (overfitting to mean)
3. Data has no temporal variation (all same temperature)
4. Preprocessing removed all variance
""")
print("="*70)
