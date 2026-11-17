
from pathlib import Path
import pandas as pd
import numpy as np
from netCDF4 import Dataset, num2date

# 1. Find the project folder
PROJECT = Path(__file__).resolve().parent.parent

print("PROJECT FOLDER:", PROJECT)

# 2. Inspect the dengue CSV
dengue_files = [
    f for f in PROJECT.rglob("RS_249_AU2634_Annexure-I.csv")
    if ".venv" not in f.parts
]

if dengue_files:
    dengue = pd.read_csv(dengue_files[0])
    print("\n--- DENGUE DATASET ---")
    print("File:", dengue_files[0].name)
    print("Shape:", dengue.shape)
    print("Columns:", dengue.columns.tolist())
    print(dengue.head())
else:
    print("\nDengue CSV not found.")

# 3. Inspect rainfall NetCDF group
rain_files = [
    f for f in PROJECT.rglob("imdweb_vSLqrJos.nc")
    if ".venv" not in f.parts
]

if rain_files:
    print("\n--- RAINFALL DATASET ---")

    with Dataset(rain_files[0]) as ds:
        print("Top-level groups:", list(ds.groups))

        g = ds.groups["rain"]
        rain = g.variables["rain"]
        time = g.variables["time"]
        lat = g.variables["lat"]
        lon = g.variables["lon"]

        print("Rainfall shape:", rain.shape)
        print("Latitude range:",
              float(np.min(lat[:])), "to", float(np.max(lat[:])))
        print("Longitude range:",
              float(np.min(lon[:])), "to", float(np.max(lon[:])))
        print("Rainfall units:",
              getattr(rain, "units", "not specified"))
        print("Time units:", getattr(time, "units", "not specified"))

        dates = num2date(
            time[:],
            units=time.units,
            calendar=getattr(time, "calendar", "standard")
        )

        print("First date:", dates[0])
        print("Last date:", dates[-1])
        print("Number of time steps:", len(dates))

        sample = np.ma.asarray(rain[0, :, :])
        print("Valid cells on first day:", int(sample.count()))

        if sample.count() > 0:
            print("Minimum rainfall:", float(sample.min()))
            print("Maximum rainfall:", float(sample.max()))
else:
    print("\nRainfall NetCDF file not found.")

