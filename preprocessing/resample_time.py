import os

import numpy as np
import xarray as xr

for split in ["train", "val", "test"]:
    out_path = f"data/interim/{split}_daily.nc"
    if os.path.exists(out_path):
        os.remove(out_path)
    print(f"Resampling {split} to daily max...")
    with xr.open_dataset(f"data/interim/{split}.nc") as ds:
        daily = ds.resample(time="1D").max().load()
        arr = daily["t2m"].values
        if not np.isfinite(arr).all():
            raise ValueError(f"Daily {split} data contains non-finite values: NaNs={np.isnan(arr).sum()}, infs={np.isinf(arr).sum()}")
        temp_path = out_path + ".tmp"
        if os.path.exists(temp_path):
            os.remove(temp_path)
        daily.to_netcdf(temp_path, engine="netcdf4")
        os.replace(temp_path, out_path)

print("Daily max resampling done.")