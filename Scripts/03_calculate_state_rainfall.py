
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
import regionmask

# --------------------------------------------------
# 1. Define project paths
# --------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parent.parent

RAIN_FILE = PROJECT_DIR / "imdweb_vSLqrJos.nc"
BOUNDARY_FILE = PROJECT_DIR / "geoBoundaries-IND-ADM1.geojson"

OUTPUT_DIR = PROJECT_DIR / "Processed_Data"
OUTPUT_DIR.mkdir(exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "state_monthly_rainfall.csv"

# --------------------------------------------------
# 2. Load rainfall and state boundaries
# --------------------------------------------------
print("Loading rainfall data...")
ds = xr.open_dataset(RAIN_FILE, group="rain")

print("Loading state boundaries...")
states = gpd.read_file(BOUNDARY_FILE)

print("Boundary columns:", states.columns.tolist())

# Identify the state-name column automatically.
possible_name_columns = [
    "shapeName", "shapeName_en", "NAME_1",
    "state", "State", "name"
]

name_column = next(
    (col for col in possible_name_columns if col in states.columns),
    None
)

if name_column is None:
    raise ValueError(
        "Could not identify the state-name column. "
        "Check the printed boundary columns."
    )

print("State name column:", name_column)

# --------------------------------------------------
# 3. Prepare state geometries
# --------------------------------------------------
states = states.to_crs("EPSG:4326").reset_index(drop=True)

# Create one mask value for each state polygon.
regions = regionmask.from_geopandas(
    states,
    names=name_column,
    name="Indian states and union territories"
)

# Rainfall coordinates and array
rain = ds["rain"]

# Create a mask matching the rainfall grid.
print("Creating state mask...")
mask = regions.mask(rain)
print("\nChecking grid cells inside each state:")

for state_index, state_row in states.iterrows():
    state_name = state_row[name_column]
    cell_count = int((mask == state_index).sum().item())

    print(f"{state_name}: {cell_count} grid cells")
# --------------------------------------------------
# 4. Calculate monthly rainfall totals
# --------------------------------------------------
print("Calculating monthly rainfall...")

# Sum daily rainfall for each calendar month.
monthly_rain = rain.resample(time="MS").sum(skipna=True, min_count=1)

records = []

for state_index, state_row in states.iterrows():
    state_name = str(state_row[name_column])

    # Grid cells whose centres fall inside this state.
    state_cells = mask == state_index

    # Select rainfall grid cells for this state.
    state_rain = monthly_rain.where(state_cells)

    # Average accumulated rainfall over the selected grid cells.
    # This gives a spatial mean monthly rainfall in mm.
    state_monthly = state_rain.mean(
        dim=("lat", "lon"),
        skipna=True
    )

    for date, value in zip(
        pd.to_datetime(state_monthly["time"].values),
        state_monthly.values
    ):
        records.append({
            "State": state_name,
            "Year": date.year,
            "Month": date.month,
            "Monthly_Rainfall_mm": (
                float(value) if np.isfinite(value) else np.nan
            )
        })

# --------------------------------------------------
# 5. Save the result
# --------------------------------------------------
result = pd.DataFrame(records)

result.to_csv(OUTPUT_FILE, index=False)

print("\nRainfall processing completed.")
print("Output file:", OUTPUT_FILE)
print("Rows:", len(result))
print("\nSample results:")
print(result.head(10).to_string(index=False))

print("\nMissing monthly rainfall values:",
      result["Monthly_Rainfall_mm"].isna().sum())

ds.close()
