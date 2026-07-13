import xarray as xr

print("Loading merged dataset...")
ds = xr.open_dataset("data/interim/merged.nc")

# Select South Asia region
ds_region = ds.sel(
    latitude=slice(35, 5),
    longitude=slice(65, 100)
)

ds_region.to_netcdf("data/interim/region_subset.nc")

print("Region subset saved.")