import glob
import xarray as xr
import os

RAW_PATH = "data/raw/era5/*.nc"
OUTPUT_PATH = "data/interim/merged.nc"

files = sorted(glob.glob(RAW_PATH))

if len(files) == 0:
    raise RuntimeError("❌ No files found in data/raw/era5/")

print(f"Found {len(files)} files")

# 🔥 Memory-efficient multi-file loading
ds = xr.open_mfdataset(
    files,
    combine="by_coords",
    chunks={"time": 100},   # enables dask chunking
    parallel=True
)

# Fix time dimension if needed
if "valid_time" in ds.dims:
    ds = ds.rename({"valid_time": "time"})

os.makedirs("data/interim", exist_ok=True)

print("Saving merged dataset...")
ds.to_netcdf(OUTPUT_PATH)

print("✅ Merging completed successfully.")