"""
Generate synthetic climate data with realistic temperature patterns
Uses ERA5-like characteristics: seasonal variation, diurnal cycle, spatial gradients
"""
import numpy as np
import xarray as xr
from datetime import datetime, timedelta
import os

print("="*70)
print("GENERATING SYNTHETIC CLIMATE DATA")
print("="*70)

# Parameters matching your project
LAT_START, LAT_END, LAT_STEP = 5, 35, 0.25       # 5-35N, India region
LON_START, LON_END, LON_STEP = 65, 100, 0.25      # 65-100E
LAT_DIM, LON_DIM = 121, 141                       # Your grid size

# Create coordinate arrays
lats = np.arange(LAT_START, LAT_END + LAT_STEP, LAT_STEP)
lons = np.arange(LON_START, LON_END + LON_STEP, LON_STEP)
print("\nGrid: {}x{} ({} to {}N, {} to {}E)".format(
    LAT_DIM, LON_DIM, LAT_START, LAT_END, LON_START, LON_END))

# Generate dates
def generate_dates(start_year, end_year):
    """Generate daily dates for multiple years"""
    dates = []
    for year in range(start_year, end_year + 1):
        current = datetime(year, 1, 1)
        while current.year == year:
            dates.append(current)
            current += timedelta(days=1)
    return np.array(dates)

# Train: 2019-2021 (3 years = ~1100 days)
# Val:   2022 (1 year = ~365 days)
# Test:  2023 (1 year = ~365 days)

train_dates = generate_dates(2019, 2021)
val_dates   = generate_dates(2022, 2022)
test_dates  = generate_dates(2023, 2023)

print("\nTrain dates: {} to {} ({} days)".format(
    train_dates[0].date(), train_dates[-1].date(), len(train_dates)))
print("Val dates:   {} to {} ({} days)".format(
    val_dates[0].date(), val_dates[-1].date(), len(val_dates)))
print("Test dates:  {} to {} ({} days)".format(
    test_dates[0].date(), test_dates[-1].date(), len(test_dates)))

def generate_temperature_field(lats, lons, dates, base_year=2019):
    """
    Generate realistic temperature field with:
    - Latitude gradient (warm at equator, cool at poles)
    - Seasonal variation (warm in summer, cool in winter)
    - Diurnal cycle (small daily variation)
    - Spatial autocorrelation
    """
    times = len(dates)
    data = np.zeros((times, len(lats), len(lons)))
    
    for t, date in enumerate(dates):
        # Base temperature: latitude gradient
        # Equator is ~25-30C, poles are ~0C
        lat_gradient = 30 - (lats / 90) * 30
        
        # Seasonal component: 10K variation peak-to-peak
        doy = date.timetuple().tm_yday
        seasonal = 10 * np.sin(2 * np.pi * (doy / 365))
        
        # Diurnal component: small variation
        diurnal = 2 * np.sin(2 * np.pi * (t / 10))
        
        # Spatial variation: add some smoothness
        spatial_noise = np.random.randn(1, len(lons)) * 0.5
        spatial_noise = np.broadcast_to(spatial_noise, (len(lats), len(lons)))
        
        # Combine components
        temp_c = np.outer(lat_gradient + seasonal, np.ones(len(lons))) + spatial_noise + diurnal
        
        # Convert to Kelvin
        data[t] = temp_c + 273.15
    
    return data

# Generate data for each split
print("\nGenerating data...")

print("  Train...") 
train_data = generate_temperature_field(lats, lons, train_dates)
print("    Shape: {}, Range: [{:.2f}, {:.2f}] K".format(
    train_data.shape, np.nanmin(train_data), np.nanmax(train_data)))

print("  Val...")
val_data = generate_temperature_field(lats, lons, val_dates, base_year=2022)
print("    Shape: {}, Range: [{:.2f}, {:.2f}] K".format(
    val_data.shape, np.nanmin(val_data), np.nanmax(val_data)))

print("  Test...")
test_data = generate_temperature_field(lats, lons, test_dates, base_year=2023)
print("    Shape: {}, Range: [{:.2f}, {:.2f}] K".format(
    test_data.shape, np.nanmin(test_data), np.nanmax(test_data)))

# Save raw data
print("\nSaving raw data...")
os.makedirs("data/interim", exist_ok=True)

# Save to NetCDF for consistency
for name, data, dates in [("train", train_data, train_dates), 
                          ("val", val_data, val_dates),
                          ("test", test_data, test_dates)]:
    # Create daily averaged version
    daily_data = data[::1]  # Already daily
    
    ds = xr.Dataset(
        {
            "t2m": (["time", "latitude", "longitude"], daily_data)
        },
        coords={
            "time": dates,
            "latitude": lats,
            "longitude": lons
        }
    )
    
    # Save both versions
    daily_path = "data/interim/{}_daily.nc".format(name)
    ds.to_netcdf(daily_path)
    print("  Saved: {}".format(daily_path))

# Now create normalized tensors
print("\nComputing statistics...")

train_valid = train_data[~np.isnan(train_data)]
mean = float(np.mean(train_valid))
std = float(np.std(train_valid))

print("  Mean: {:.4f} K".format(mean))
print("  Std:  {:.4f} K".format(std))

# Normalize
print("\nNormalizing data...")
train_norm = (train_data - mean) / std
val_norm   = (val_data - mean) / std
test_norm  = (test_data - mean) / std

train_norm = np.nan_to_num(train_norm)
val_norm = np.nan_to_num(val_norm)
test_norm = np.nan_to_num(test_norm)

print("  Train norm range: [{:.4f}, {:.4f}], std: {:.4f}".format(
    np.min(train_norm), np.max(train_norm), np.std(train_norm)))

# Save normalized arrays
os.makedirs("data/processed", exist_ok=True)
np.save("data/processed/train.npy", train_norm)
np.save("data/processed/val.npy", val_norm)
np.save("data/processed/test.npy", test_norm)
print("  Saved normalized arrays")

# Save statistics
import json
with open("data/processed/mean_std.json", "w") as f:
    json.dump({"mean": mean, "std": std}, f, indent=2)
print("  Saved statistics")

# Create sequences
print("\nCreating sequences...")

def create_sequences(data, seq_len=7):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len])
    return np.array(X), np.array(y)

X_train, y_train = create_sequences(train_norm)
X_val, y_val     = create_sequences(val_norm)
X_test, y_test   = create_sequences(test_norm)

# Add channel dimension
X_train = X_train[:, :, np.newaxis, :, :]
X_val   = X_val[:, :, np.newaxis, :, :]
X_test  = X_test[:, :, np.newaxis, :, :]

y_train = y_train[:, np.newaxis, :, :]
y_val   = y_val[:, np.newaxis, :, :]
y_test  = y_test[:, np.newaxis, :, :]

print("  X_train: {}, y_train: {}".format(X_train.shape, y_train.shape))
print("  X_val:   {}, y_val:   {}".format(X_val.shape, y_val.shape))
print("  X_test:  {}, y_test:  {}".format(X_test.shape, y_test.shape))

# Save tensors
print("\nSaving tensors...")
os.makedirs("data/processed/tensors", exist_ok=True)

np.save("data/processed/tensors/train_X.npy", X_train)
np.save("data/processed/tensors/train_y.npy", y_train)
np.save("data/processed/tensors/val_X.npy", X_val)
np.save("data/processed/tensors/val_y.npy", y_val)
np.save("data/processed/tensors/test_X.npy", X_test)
np.save("data/processed/tensors/test_y.npy", y_test)

print("  All tensors saved!")

print("\n" + "="*70)
print("SUCCESS: Synthetic data generated and ready for training!")
print("="*70)
print("\nNext steps:")
print("  1. python main.py --train  (retrain model with proper data)")
print("  2. python evaluate_model.py")
print("  3. streamlit run dashboard.py")
print("="*70)
