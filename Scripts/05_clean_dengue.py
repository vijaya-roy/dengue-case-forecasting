
from pathlib import Path
import pandas as pd

# --------------------------------------------------
# 1. Define project paths
# --------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = PROJECT_DIR / "RS_249_AU2634_Annexure-I.csv"
OUTPUT_DIR = PROJECT_DIR / "Processed_Data"
OUTPUT_DIR.mkdir(exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "clean_monthly_dengue.csv"

# --------------------------------------------------
# 2. Load the original dengue dataset
# --------------------------------------------------
print("Loading dengue dataset...")

df = pd.read_csv(INPUT_FILE)

print("Original shape:", df.shape)
print("Original columns:", df.columns.tolist())

# Remove extra spaces from column names and state names.
df.columns = df.columns.str.strip()
df["State/UT"] = df["State/UT"].astype(str).str.strip()

# Remove the national total row because it is not a state.
df = df[df["State/UT"].str.lower() != "total"].copy()

# Standardize known state-name variations.
state_name_mapping = {
    "Andhara Pradesh": "Andhra Pradesh",
    "Andaman & Nicobar Islands": "Andaman and Nicobar Islands",
    "Dadra & Nagar Haveli and Daman & Diu":
        "Dadra and Nagar Haveli and Daman and Diu",
}

df["State/UT"] = df["State/UT"].replace(state_name_mapping)

# --------------------------------------------------
# 3. Convert monthly columns into long format
# --------------------------------------------------
monthly_columns = [
    col for col in df.columns
    if col.startswith("2018 - ") or col.startswith("2019 - ")
]

# Exclude annual total columns.
monthly_columns = [
    col for col in monthly_columns
    if "Total" not in col
]

print("\nNumber of monthly columns:", len(monthly_columns))

long_df = df.melt(
    id_vars=["State/UT"],
    value_vars=monthly_columns,
    var_name="Year_Month",
    value_name="Dengue_Cases"
)

# Extract year and month name from column labels.
long_df["Year"] = long_df["Year_Month"].str.extract(r"(2018|2019)").astype(int)

long_df["Month_Name"] = (
    long_df["Year_Month"]
    .str.replace(r"^\d{4} - ", "", regex=True)
    .str.strip()
)

month_mapping = {
    "January": 1,
    "February": 2,
    "March": 3,
    "April": 4,
    "May": 5,
    "June": 6,
    "July": 7,
    "August": 8,
    "September": 9,
    "October": 10,
    "November": 11,
    "December": 12,
}

long_df["Month"] = long_df["Month_Name"].map(month_mapping)

# Convert case counts to numeric values.
long_df["Dengue_Cases"] = pd.to_numeric(
    long_df["Dengue_Cases"],
    errors="coerce"
)

# Keep only valid rows with a recognized month.
long_df = long_df.dropna(
    subset=["State/UT", "Year", "Month", "Dengue_Cases"]
).copy()

long_df["Year"] = long_df["Year"].astype(int)
long_df["Month"] = long_df["Month"].astype(int)
long_df["Dengue_Cases"] = long_df["Dengue_Cases"].astype(int)

# --------------------------------------------------
# 4. Sort and save the cleaned dataset
# --------------------------------------------------
long_df = long_df.rename(columns={"State/UT": "State"})

long_df = long_df[
    ["State", "Year", "Month", "Dengue_Cases"]
].sort_values(["State", "Year", "Month"])

long_df.to_csv(OUTPUT_FILE, index=False)

# --------------------------------------------------
# 5. Report data quality checks
# --------------------------------------------------
print("\nDengue cleaning completed.")
print("Output file:", OUTPUT_FILE)
print("Cleaned shape:", long_df.shape)
print("Number of states/UTs:", long_df["State"].nunique())

print("\nYear and month coverage:")
print(long_df.groupby("Year")["Month"].agg(["min", "max", "count"]))

print("\nSample cleaned records:")
print(long_df.head(10).to_string(index=False))

print("\nMissing values:")
print(long_df.isna().sum())

print("\nDuplicate state-year-month records:",
      long_df.duplicated(["State", "Year", "Month"]).sum())
