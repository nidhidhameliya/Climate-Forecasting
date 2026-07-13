import xarray as xr

for split in ["train", "val", "test"]:
    print(f"Resampling {split} to daily max...")

    ds = xr.open_dataset(f"data/interim/{split}.nc")

    daily = ds.resample(time="1D").max()

    daily.to_netcdf(f"data/interim/{split}_daily.nc")

print("Daily max resampling done.")