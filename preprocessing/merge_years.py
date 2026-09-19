import glob
import xarray as xr
import os

RAW_PATH = "data/raw/era5/*.nc"
OUTPUT_PATH = "data/interim/merged.nc"

files = sorted(glob.glob(RAW_PATH))

if len(files) == 0:
    raise RuntimeError("❌ No files found in data/raw/era5/")

print(f"Found {len(files)} files")

# ERA5 downloads use ``valid_time``. Normalize that coordinate before
# concatenating so xarray cannot align files against an unrelated time index.
datasets = []
for path in files:
    dataset = xr.open_dataset(path, chunks={"valid_time": 100})
    if "valid_time" in dataset.dims:
        dataset = dataset.rename({"valid_time": "time"})
    datasets.append(dataset[["t2m"]])

ds = xr.concat(datasets, dim="time", data_vars="minimal", coords="minimal", compat="equals")
ds = ds.sortby("time")

os.makedirs("data/interim", exist_ok=True)

print("Saving merged dataset...")
ds.to_netcdf(OUTPUT_PATH)

print("✅ Merging completed successfully.")