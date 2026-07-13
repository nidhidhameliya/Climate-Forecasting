import xarray as xr
import numpy as np
import json
import os

print("Loading daily datasets...")

train_ds = xr.open_dataset("data/interim/train_daily.nc")
val_ds   = xr.open_dataset("data/interim/val_daily.nc")
test_ds  = xr.open_dataset("data/interim/test_daily.nc")

var = "t2m"

# Convert Kelvin → Celsius
train = train_ds[var].values - 273.15
val   = val_ds[var].values - 273.15
test  = test_ds[var].values - 273.15

print("Computing statistics from TRAIN only...")

mean = np.nanmean(train)
std  = np.nanstd(train)
std = std if std != 0 else 1.0

train_norm = (train - mean) / std
val_norm   = (val - mean) / std
test_norm  = (test - mean) / std

train_norm = np.nan_to_num(train_norm)
val_norm   = np.nan_to_num(val_norm)
test_norm  = np.nan_to_num(test_norm)

os.makedirs("data/processed", exist_ok=True)

np.save("data/processed/train.npy", train_norm)
np.save("data/processed/val.npy", val_norm)
np.save("data/processed/test.npy", test_norm)

with open("data/processed/mean_std.json", "w") as f:
    json.dump({"mean": float(mean), "std": float(std)}, f)

print("Normalization complete.")