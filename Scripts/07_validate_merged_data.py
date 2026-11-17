
from pathlib import Path
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
FILE = PROJECT_DIR / "Processed_Data" / "merged_dengue_rainfall.csv"

df = pd.read_csv(FILE)

print("Dataset shape:", df.shape)
print("\nColumns:", df.columns.tolist())
print("\nMissing values:")
print(df.isna().sum())

print("\nDuplicate state-year-month records:",
      df.duplicated(["State", "Year", "Month"]).sum())

print("\nRainfall coverage:")
print(df.groupby("State")["Monthly_Rainfall_mm"].apply(
    lambda x: x.notna().sum()
).to_string())

print("\nDengue case summary:")
print(df["Dengue_Cases"].describe())
