import xarray as xr
import numpy as np
import json
import os

print("Computing extreme thresholds from TRAIN set...")

train_ds = xr.open_dataset("data/interim/train_daily.nc")
var = "t2m"

# Kelvin → Celsius
train_data = train_ds[var].values - 273.15

threshold_95 = np.nanquantile(train_data, 0.95)
threshold_05 = np.nanquantile(train_data, 0.05)

os.makedirs("data/processed", exist_ok=True)

with open("data/processed/extreme_thresholds.json", "w") as f:
    json.dump({
        "top_5_percent_celsius": float(threshold_95),
        "bottom_5_percent_celsius": float(threshold_05)
    }, f)

print("Extreme thresholds saved.")