import xarray as xr

print("Loading region subset...")
ds = xr.open_dataset("data/interim/region_subset.nc")

train = ds.sel(time=slice("2019-01-01", "2023-12-31"))
val   = ds.sel(time=slice("2024-01-01", "2024-12-31"))
test  = ds.sel(time=slice("2025-01-01", "2025-12-31"))

train.to_netcdf("data/interim/train.nc")
val.to_netcdf("data/interim/val.nc")
test.to_netcdf("data/interim/test.nc")

print("Time-based split complete.")