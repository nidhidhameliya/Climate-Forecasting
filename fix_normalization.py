"""
Fix corrupted normalization - regenerate tensors with proper statistics
"""
import xarray as xr
import numpy as np
import json
import os
from pathlib import Path

print("="*70)
print("FIXING DATA NORMALIZATION")
print("="*70)

# Step 1: Load and check raw data
print("\nSTEP 1: Loading source data...")
train_ds = xr.open_dataset("data/interim/train_daily.nc")
val_ds   = xr.open_dataset("data/interim/val_daily.nc")  
test_ds  = xr.open_dataset("data/interim/test_daily.nc")

# Get temperature variable (should be t2m in Kelvin)
var_name = "t2m"  # 2-meter air temperature

train = train_ds[var_name].values  # Shape: (time, lat, lon)
val   = val_ds[var_name].values
test  = test_ds[var_name].values

print("   Train shape: {}".format(train.shape))
print("   Val shape:   {}".format(val.shape))
print("   Test shape:  {}".format(test.shape))

# Step 2: Compute proper statistics ONLY from training data
print("\nSTEP 2: Computing normalization statistics from TRAINING data...")

# Remove NaNs for statistics computation
train_valid = train[~np.isnan(train)]

if len(train_valid) == 0:
    print("   ERROR: All training data is NaN!")
    print("   This should not happen. Aborting.")
    exit(1)

mean = float(np.mean(train_valid))
std = float(np.std(train_valid))

print("   Mean: {:.4f} K".format(mean))
print("   Std:  {:.4f} K".format(std))
print("   Valid samples: {}/{}".format(len(train_valid), train.size))

# Sanity check
if np.isnan(mean) or np.isnan(std) or std == 0:
    print("   ERROR: Invalid statistics computed!")
    exit(1)

# Save statistics
print("\nSTEP 3: Saving normalization statistics...")
os.makedirs("data/processed", exist_ok=True)
with open("data/processed/mean_std.json", "w") as f:
    json.dump({"mean": mean, "std": std}, f, indent=2)
print(f"   ✓ Saved to data/processed/mean_std.json")

# Step 3: Normalize all data
print("\nSTEP 4: Normalizing data...")

# Convert to Celsius (subtract 273.15 from Kelvin)
# Then normalize: (x - mean) / std
train_norm = (train - 273.15 - (mean - 273.15)) / std
val_norm   = (val - 273.15 - (mean - 273.15)) / std
test_norm  = (test - 273.15 - (mean - 273.15)) / std

# Handle NaNs: replace with 0
train_norm = np.nan_to_num(train_norm, nan=0.0)
val_norm   = np.nan_to_num(val_norm, nan=0.0)
test_norm  = np.nan_to_num(test_norm, nan=0.0)

print(f"   Train norm range: [{np.min(train_norm):.4f}, {np.max(train_norm):.4f}]")
print(f"   Train norm mean:  {np.mean(train_norm):.6f}")
print(f"   Train norm std:   {np.std(train_norm):.4f}")

# Step 4: Verify data has variance
print("\n5️⃣  Checking data variance...")
if np.std(train_norm) < 0.01:
    print("   WARNING: Normalized data has very low variance!")
    print("   This might indicate data issues.")
else:
    print("   OK: Data has good variance")

# Step 5: Save normalized data
print("\nSTEP 6: Saving normalized data...")
np.save("data/processed/train.npy", train_norm)
np.save("data/processed/val.npy", val_norm)
np.save("data/processed/test.npy", test_norm)
print("   OK: Saved normalized arrays")

# Step 6: Create sequences (7-day input => 1-day output)
print("\nSTEP 7: Creating sequences...")

def create_sequences(data, seq_len=7):
    """Create sequences of length seq_len"""
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len])
    return np.array(X), np.array(y)

seq_len = 7

# Create train sequences
X_train, y_train = create_sequences(train_norm, seq_len)
print("   Train sequences: X={}, y={}".format(X_train.shape, y_train.shape))

# Create val sequences
X_val, y_val = create_sequences(val_norm, seq_len)
print("   Val sequences:   X={}, y={}".format(X_val.shape, y_val.shape))

# Create test sequences
X_test, y_test = create_sequences(test_norm, seq_len)
print("   Test sequences:  X={}, y={}".format(X_test.shape, y_test.shape))

# Convert to required shape: (batch, time, channels, height, width)
# Current: (batch, time, height, width)
# Need: (batch, time, 1, height, width) - add channel dimension

X_train = X_train[:, :, np.newaxis, :, :]  # (B, T, 1, H, W)
X_val   = X_val[:, :, np.newaxis, :, :]
X_test  = X_test[:, :, np.newaxis, :, :]

y_train = y_train[:, np.newaxis, :, :]  # (B, 1, H, W) 
y_val   = y_val[:, np.newaxis, :, :]
y_test  = y_test[:, np.newaxis, :, :]

print("\n   After reshaping:")
print("   X_train: {}, y_train: {}".format(X_train.shape, y_train.shape))
print("   X_val:   {}, y_val:   {}".format(X_val.shape, y_val.shape))
print("   X_test:  {}, y_test:  {}".format(X_test.shape, y_test.shape))

# Step 7: Save sequences
print("\nSTEP 8: Saving sequence tensors...")
os.makedirs("data/processed/tensors", exist_ok=True)

np.save("data/processed/tensors/train_X.npy", X_train)
np.save("data/processed/tensors/train_y.npy", y_train)
np.save("data/processed/tensors/val_X.npy", X_val)
np.save("data/processed/tensors/val_y.npy", y_val)
np.save("data/processed/tensors/test_X.npy", X_test)
np.save("data/processed/tensors/test_y.npy", y_test)

print("   OK: All tensor files saved")

# Step 8: Verify saved data
print("\nSTEP 9: Verifying saved data...")
X_test_check = np.load("data/processed/tensors/test_X.npy")
y_test_check = np.load("data/processed/tensors/test_y.npy")

print("   test_X range: [{:.4f}, {:.4f}]".format(np.min(X_test_check), np.max(X_test_check)))
print("   test_y range: [{:.4f}, {:.4f}]".format(np.min(y_test_check), np.max(y_test_check)))

if np.min(X_test_check) == 0 and np.max(X_test_check) == 0:
    print("   ERROR: Data is still all zeros!")
    exit(1)
else:
    print("   OK: Data has proper range!")

print("\n" + "="*70)
print("NORMALIZATION FIX COMPLETE!")
print("="*70)
print("\nNext steps:")
print("  1. Run: python main.py --train")
print("  2. Run: python evaluate_model.py")
print("  3. Run: streamlit run dashboard.py")
print("="*70)
