
from pathlib import Path
import pandas as pd
import unicodedata

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "Processed_Data"

DENGUE_FILE = DATA_DIR / "clean_monthly_dengue.csv"
RAINFALL_FILE = DATA_DIR / "state_monthly_rainfall.csv"

OUTPUT_FILE = DATA_DIR / "merged_dengue_rainfall.csv"

# --------------------------------------------------
# 1. Load cleaned datasets
# --------------------------------------------------
dengue = pd.read_csv(DENGUE_FILE)
rainfall = pd.read_csv(RAINFALL_FILE)

print("Dengue rows:", len(dengue))
print("Rainfall rows:", len(rainfall))

# --------------------------------------------------
# 2. Standardize state names
# --------------------------------------------------

def normalize_state(name):
    import unicodedata

    name = unicodedata.normalize("NFKD", str(name))
    name = "".join(
        ch for ch in name
        if not unicodedata.combining(ch)
    )
    name = name.lower().strip()
    name = " ".join(name.replace("&", "and").split())

    mapping = {
        "a and n island": "andaman and nicobar islands",
        "andaman and nicobar islands": "andaman and nicobar islands",
        "andhara pradesh": "andhra pradesh",
        "telengana": "telangana",
        "uttrakhand": "uttarakhand",
        "j and k": "jammu and kashmir",
        "d and n haveli": "dadra and nagar haveli and daman and diu",
        "daman and diu": "dadra and nagar haveli and daman and diu",
        "dadra and nagar haveli and daman and diu":
            "dadra and nagar haveli and daman and diu",
        "dadn haveli": "dadra and nagar haveli and daman and diu",
        "aandn island": "andaman and nicobar islands",
    }

    return mapping.get(name, name)


dengue["State_Key"] = dengue["State"].apply(normalize_state)
rainfall["State_Key"] = rainfall["State"].apply(normalize_state)

# --------------------------------------------------
# 3. Check state-name matching
# --------------------------------------------------
dengue_states = set(dengue["State_Key"].unique())
rainfall_states = set(rainfall["State_Key"].unique())

print("\nDengue states without rainfall match:")
print(sorted(dengue_states - rainfall_states))

print("\nRainfall states without dengue match:")
print(sorted(rainfall_states - dengue_states))

# --------------------------------------------------
# 4. Merge by state, year and month
# --------------------------------------------------
merged = dengue.merge(
    rainfall[[
        "State_Key", "Year", "Month", "Monthly_Rainfall_mm"
    ]],
    on=["State_Key", "Year", "Month"],
    how="left",
    validate="many_to_one"
)

# Remove the temporary matching key.
merged = merged.drop(columns=["State_Key"])

# Save the merged dataset.
merged.to_csv(OUTPUT_FILE, index=False)

# --------------------------------------------------
# 5. Report merge quality
# --------------------------------------------------
print("\nMerged dataset saved to:")
print(OUTPUT_FILE)

print("\nMerged shape:", merged.shape)
print("\nMissing rainfall values after merge:")
print(merged["Monthly_Rainfall_mm"].isna().sum())

print("\nSample merged records:")
print(merged.head(10).to_string(index=False))

print("\nRows with missing rainfall by state:")
print(
    merged.groupby("State")["Monthly_Rainfall_mm"]
    .apply(lambda values: values.isna().sum())
    .loc[lambda values: values > 0]
    .to_string()
)
