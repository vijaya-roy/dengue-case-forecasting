
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "Processed_Data"

INPUT_FILE = DATA_DIR / "merged_dengue_rainfall.csv"
OUTPUT_FILE = DATA_DIR / "model_data.csv"

# Load merged data
df = pd.read_csv(INPUT_FILE)

print("Original dataset shape:", df.shape)

# Check required columns
required = [
    "State", "Year", "Month",
    "Dengue_Cases", "Monthly_Rainfall_mm"
]

missing_columns = [c for c in required if c not in df.columns]

if missing_columns:
    raise ValueError(f"Missing required columns: {missing_columns}")

# Keep only records with observed rainfall and dengue cases
model_df = df.dropna(
    subset=["Dengue_Cases", "Monthly_Rainfall_mm"]
).copy()

# Create a date column for time-based analysis
model_df["Date"] = pd.to_datetime(
    dict(
        year=model_df["Year"],
        month=model_df["Month"],
        day=1
    )
)

# Log-transform dengue counts.
# log1p handles zero counts safely: log(1 + 0) = 0.
model_df["Log_Dengue_Cases"] = np.log1p(
    model_df["Dengue_Cases"]
)

# Save the modelling dataset
model_df = model_df.sort_values(["State", "Date"])
model_df.to_csv(OUTPUT_FILE, index=False)

print("\nModelling dataset saved to:", OUTPUT_FILE)
print("Modelling dataset shape:", model_df.shape)
print("States included:", model_df["State"].nunique())
print("Missing values:\n", model_df[required].isna().sum())

print("\nRecords per state:")
print(model_df.groupby("State").size().to_string())

print("\nSample:")
print(model_df.head(10).to_string(index=False))
