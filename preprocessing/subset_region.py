import glob
import os

import numpy as np
import xarray as xr

OUTPUT = "data/interim/region_subset_raw_verified.nc"

print("Loading raw ERA5 files and selecting the study region...")

for stale in [
    OUTPUT,
    "data/interim/train.nc",
    "data/interim/val.nc",
    "data/interim/test.nc",
    "data/interim/train_daily.nc",
    "data/interim/val_daily.nc",
    "data/interim/test_daily.nc",
]:
    if os.path.exists(stale):
        os.remove(stale)
        print(f"Removed stale file: {stale}")

datasets = []
for path in sorted(glob.glob("data/raw/era5/*.nc")):
    with xr.open_dataset(path, decode_times=True) as dataset:
        if "valid_time" in dataset.dims:
            dataset = dataset.rename({"valid_time": "time"})
        subset = dataset[["t2m"]].sel(
            latitude=slice(35, 5),
            longitude=slice(65, 100),
        )
        datasets.append(subset)

ds_region = xr.concat(
    datasets,
    dim="time",
    data_vars="minimal",
    coords="minimal",
    compat="equals",
).sortby("time").load()
array = ds_region["t2m"].values
if not np.isfinite(array).all():
    raise ValueError(f"Regional subset contains non-finite values: NaNs={np.isnan(array).sum()}, infs={np.isinf(array).sum()}")

temp_path = OUTPUT + ".tmp"
if os.path.exists(temp_path):
    os.remove(temp_path)
ds_region.to_netcdf(temp_path, engine="netcdf4")
os.replace(temp_path, OUTPUT)
print(f"Region subset saved to {OUTPUT} with shape {array.shape} and finite values.")