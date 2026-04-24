import xarray as xr
import numpy as np

print("Checking merged data file...")

try:
    ds = xr.open_dataset("data/interim/merged.nc")
    print("\nVariables: {}".format(list(ds.data_vars.keys())))
    print("Dimensions: {}".format(dict(ds.dims)))
    
    # Try each variable
    for var in ds.data_vars:
        data = ds[var].values
        valid = np.sum(~np.isnan(data))
        print("\n{}: shape={}, dtype={}, NaN%={:.1f}".format(
            var, data.shape, data.dtype, 100*(1-valid/data.size)))
        
        if valid > 0:
            print("  Range: [{:.2f}, {:.2f}]".format(np.nanmin(data), np.nanmax(data)))
            
except Exception as e:
    print("ERROR: {}".format(e))
