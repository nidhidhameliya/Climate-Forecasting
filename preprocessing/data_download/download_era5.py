import os
import cdsapi

# ==============================
# Configuration
# ==============================

YEARS = ["2018", "2019", "2020", "2021", "2022", "2023", "2024", "2025"]

OUTPUT_DIR = "../data/raw/era5"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==============================
# Download Function
# ==============================

def download_era5(year):
    print(f"Downloading ERA5 data for {year}...")

    c = cdsapi.Client()

    c.retrieve(
        "reanalysis-era5-single-levels",
        {
            "product_type": "reanalysis",
            "variable": [
                "2m_temperature",
                "total_precipitation",
                "surface_pressure",
                "10m_u_component_of_wind",
                "10m_v_component_of_wind"
            ],
            "year": year,
            "month": [
                "01","02","03","04","05","06",
                "07","08","09","10","11","12"
            ],
            "day": [f"{d:02d}" for d in range(1, 32)],
            "time": [
                "00:00","06:00","12:00","18:00"
            ],
            "format": "netcdf"
        },
        os.path.join(OUTPUT_DIR, f"era5_{year}.nc")
    )

    print(f"Saved: era5_{year}.nc")


# ==============================
# Main
# ==============================

if __name__ == "__main__":
    for year in YEARS:
        download_era5(year)

    print("All downloads completed.")