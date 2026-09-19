import os

import numpy as np
import xarray as xr

for name in ["train.nc", "val.nc", "test.nc"]:
    path = os.path.join("data/interim", name)
    if os.path.exists(path):
        os.remove(path)

print("Loading region subset...")
with xr.open_dataset("data/interim/region_subset_raw_verified.nc") as ds:
    train = ds.sel(time=slice("2019-01-01", "2023-12-31"))
    val = ds.sel(time=slice("2024-01-01", "2024-12-31"))
    test = ds.sel(time=slice("2025-01-01", "2025-12-31"))

    for split_name, split_obj in {"train": train, "val": val, "test": test}.items():
        arr = split_obj["t2m"].values
        if not np.isfinite(arr).all():
            raise ValueError(f"{split_name} split contains non-finite values: NaNs={np.isnan(arr).sum()}, infs={np.isinf(arr).sum()}")
        temp_path = f"data/interim/{split_name}.nc.tmp"
        if os.path.exists(temp_path):
            os.remove(temp_path)
        split_obj.to_netcdf(temp_path, engine="netcdf4")
        os.replace(temp_path, f"data/interim/{split_name}.nc")

print("Time-based split complete.")