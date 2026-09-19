import json
import os

import numpy as np
import xarray as xr

print("Loading daily datasets...")

with xr.open_dataset("data/interim/train_daily.nc") as train_ds, xr.open_dataset("data/interim/val_daily.nc") as val_ds, xr.open_dataset("data/interim/test_daily.nc") as test_ds:
    var = "t2m"
    train = train_ds[var].values - 273.15
    val = val_ds[var].values - 273.15
    test = test_ds[var].values - 273.15

print("Computing statistics from TRAIN only...")
mean = np.nanmean(train)
std = np.nanstd(train)
std = std if std != 0 else 1.0

if not np.isfinite(mean) or not np.isfinite(std):
    raise ValueError(f"Non-finite training statistics: mean={mean}, std={std}")

train_norm = (train - mean) / std
val_norm = (val - mean) / std
test_norm = (test - mean) / std

train_norm = np.nan_to_num(train_norm, nan=0.0, posinf=0.0, neginf=0.0)
val_norm = np.nan_to_num(val_norm, nan=0.0, posinf=0.0, neginf=0.0)
test_norm = np.nan_to_num(test_norm, nan=0.0, posinf=0.0, neginf=0.0)

for arr in [train_norm, val_norm, test_norm]:
    if not np.isfinite(arr).all():
        raise ValueError("Normalization produced non-finite values")

os.makedirs("data/processed", exist_ok=True)
np.save("data/processed/train.npy", train_norm)
np.save("data/processed/val.npy", val_norm)
np.save("data/processed/test.npy", test_norm)

with open("data/processed/mean_std.json", "w") as f:
    json.dump({"mean": float(mean), "std": float(std)}, f)

print(f"Normalization complete. mean={mean:.6f}, std={std:.6f}")
