
from pathlib import Path
import xarray as xr
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent.parent
RAIN_FILE = PROJECT_DIR / "imdweb_vSLqrJos.nc"

ds = xr.open_dataset(RAIN_FILE, group="rain")
rain = ds["rain"]

# Approximate area around Lakshadweep
region = rain.sel(
    lat=slice(8, 13),
    lon=slice(70, 75)
)

# Count valid observations for each grid cell
valid_counts = np.isfinite(region.values).sum(axis=0)

print("Grid-cell latitude and longitude near Lakshadweep:")
print("Latitude values:", region["lat"].values)
print("Longitude values:", region["lon"].values)

print("\nValid observation counts for grid cells:")
print(valid_counts)

# List grid cells that contain valid observations
print("\nGrid cells with valid rainfall data:")
for i, lat in enumerate(region["lat"].values):
    for j, lon in enumerate(region["lon"].values):
        if valid_counts[i, j] > 0:
            print(
                f"Latitude: {lat:.2f}, Longitude: {lon:.2f}, "
                f"Valid days: {valid_counts[i, j]}"
            )

ds.close()
