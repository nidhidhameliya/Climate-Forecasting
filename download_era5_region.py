import os
import cdsapi

print("="*70)
print("DOWNLOADING ERA5 DATA (Optimized for India Region)")
print("="*70)

YEARS = ["2019", "2020", "2021", "2022", "2023"]

OUTPUT_DIR = "data/raw/era5"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def download_era5_region(year):
    """Download ERA5 data for specific region (5-35N, 65-100E) - much faster!"""
    print("\nDownloading ERA5 for {}...".format(year))

    c = cdsapi.Client()

    request = {
        "product_type": "reanalysis",
        "variable": [
            "2m_temperature",
            "total_precipitation",
        ],
        "year": year,
        "month": [
            "01","02","03","04","05","06",
            "07","08","09","10","11","12"
        ],
        "day": ["{:02d}".format(d) for d in range(1, 32)],
        "time": ["00:00"],  # Just one time per day (faster)
        "area": [35, 65, 5, 100],  # North, West, South, East (India region)
        "format": "netcdf"
    }
    
    output_file = os.path.join(OUTPUT_DIR, "era5_{}.nc".format(year))
    
    print("  Submitting request...")
    print("  Region: 5-35N, 65-100E")
    print("  This may take 5-15 minutes depending on Copernicus queue")
    
    try:
        c.retrieve(
            "reanalysis-era5-single-levels",
            request,
            output_file
        )
        print("  OK: Saved to {}".format(output_file))
        return True
    except Exception as e:
        print("  ERROR: {}".format(str(e)[:100]))
        return False

if __name__ == "__main__":
    print("\nThis is 5x smaller than full global ERA5")
    print("- Full ERA5 (global): ~500GB, takes hours")
    print("- Regional ERA5 (India): ~100GB, takes 30 min - 2 hours")
    print("\nStarting download (can be stopped with Ctrl+C)...\n")
    
    success_count = 0
    for year in YEARS:
        if download_era5_region(year):
            success_count += 1
    
    print("\n" + "="*70)
    print("Download complete: {}/{} years successful".format(success_count, len(YEARS)))
    print("="*70)
    
    if success_count > 0:
        print("\nNext: Process the data with:")
        print("  python -m preprocessing.merge_years")
        print("  python main.py --preprocess")
        print("  python main.py --train")
