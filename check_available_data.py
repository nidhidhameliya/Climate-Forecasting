import xarray as xr
import numpy as np

print("Checking available data files...")

files = {
    "data/interim/train.nc": "train",
    "data/interim/val.nc": "val",
    "data/interim/test.nc": "test",
}

for path, name in files.items():
    try:
        ds = xr.open_dataset(path)
        print("\n{} ({})".format(name.upper(), path))
        print("  Variables: {}".format(list(ds.data_vars.keys())))
        
        # Try to find temperature variable
        for var_name in ["t2m", "temperature", "temp", "t"]:
            if var_name in ds:
                data = ds[var_name].values
                print("  {} shape: {}".format(var_name, data.shape))
                print("  {} range: [{:.2f}, {:.2f}]".format(var_name, np.nanmin(data), np.nanmax(data)))
                print("  {} NaN: {:.1f}%".format(var_name, 100*np.sum(np.isnan(data))/data.size))
                break
    except Exception as e:
        print("\nERROR loading {}: {}".format(path, e))
