
import geopandas as gpd
from pathlib import Path

# Project folder is the parent of the Scripts folder
PROJECT_DIR = Path(__file__).resolve().parent.parent
boundary_file = PROJECT_DIR / "geoBoundaries-IND-ADM1.geojson"

print("Checking boundary file...")

gdf = gpd.read_file(boundary_file)

print("\nNumber of boundary features:", len(gdf))
print("Coordinate reference system:", gdf.crs)
print("\nState/UT names:")

for name in sorted(gdf["shapeName"].dropna().unique()):
    print("-", name)

print("\nInvalid geometries:", (~gdf.geometry.is_valid).sum())
print("\nBoundary file check completed.")
