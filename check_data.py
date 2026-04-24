import xarray as xr
import numpy as np

print("="*70)
print("🔍 CHECKING INTERIM DATA FILES")
print("="*70)

for split in ["train", "val", "test"]:
    path = f"data/interim/{split}_daily.nc"
    print(f"\n📁 {split.upper()}: {path}")
    
    try:
        ds = xr.open_dataset(path)
        print(f"  ✓ File loaded successfully")
        print(f"  Dimensions: {dict(ds.dims)}")
        print(f"  Variables: {list(ds.data_vars.keys())}")
        
        if "t2m" in ds:
            t2m = ds["t2m"].values
            print(f"  t2m shape: {t2m.shape}")
            print(f"  t2m dtype: {t2m.dtype}")
            print(f"  t2m range: [{np.nanmin(t2m) if not np.all(np.isnan(t2m)) else 'NaN'}, {np.nanmax(t2m) if not np.all(np.isnan(t2m)) else 'NaN'}]")
            print(f"  t2m NaNs: {np.sum(np.isnan(t2m))}/{t2m.size} ({100*np.sum(np.isnan(t2m))/t2m.size:.1f}%)")
            print(f"  t2m mean: {np.nanmean(t2m)}")
            print(f"  t2m std: {np.nanstd(t2m)}")
        else:
            print(f"  ❌ 't2m' variable not found!")
            
    except Exception as e:
        print(f"  ❌ ERROR: {e}")

print("\n" + "="*70)
