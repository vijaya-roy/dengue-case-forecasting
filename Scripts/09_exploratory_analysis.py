
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "Processed_Data"
RESULTS_DIR = PROJECT_DIR / "Results"
FIGURES_DIR = RESULTS_DIR / "Figures"
TABLES_DIR = RESULTS_DIR / "Tables"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)

# Load prepared data
df = pd.read_csv(DATA_DIR / "model_data.csv")
df["Date"] = pd.to_datetime(df["Date"])

print("Dataset shape:", df.shape)
print("Number of states:", df["State"].nunique())
print("Date range:", df["Date"].min(), "to", df["Date"].max())

# 1. Summary statistics
summary = df[
    ["Dengue_Cases", "Monthly_Rainfall_mm"]
].describe()

summary.to_csv(TABLES_DIR / "descriptive_statistics.csv")
print("\nDescriptive statistics:")
print(summary)

# 2. Monthly dengue cases across all included states
monthly = (
    df.groupby("Date", as_index=False)
    .agg(
        Total_Dengue_Cases=("Dengue_Cases", "sum"),
        Mean_Rainfall_mm=("Monthly_Rainfall_mm", "mean")
    )
)

monthly.to_csv(TABLES_DIR / "monthly_national_summary.csv", index=False)

plt.figure(figsize=(11, 5))
plt.plot(
    monthly["Date"],
    monthly["Total_Dengue_Cases"],
    marker="o"
)
plt.title("Reported Dengue Cases Across Included States")
plt.xlabel("Month")
plt.ylabel("Total Reported Dengue Cases")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "monthly_dengue_cases.png", dpi=300)
plt.close()

# 3. Rainfall and dengue relationship
plt.figure(figsize=(8, 5))
plt.scatter(
    df["Monthly_Rainfall_mm"],
    df["Dengue_Cases"],
    alpha=0.6
)
plt.title("Monthly Rainfall vs Dengue Cases")
plt.xlabel("Monthly Rainfall (mm)")
plt.ylabel("Reported Dengue Cases")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "rainfall_vs_dengue.png", dpi=300)
plt.close()

# 4. Relationship using log-transformed dengue counts
plt.figure(figsize=(8, 5))
plt.scatter(
    df["Monthly_Rainfall_mm"],
    np.log1p(df["Dengue_Cases"]),
    alpha=0.6
)
plt.title("Rainfall vs Log-Transformed Dengue Cases")
plt.xlabel("Monthly Rainfall (mm)")
plt.ylabel("log(1 + Dengue Cases)")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "rainfall_vs_log_dengue.png", dpi=300)
plt.close()

# 5. Correlation
correlation = df[
    ["Dengue_Cases", "Monthly_Rainfall_mm"]
].corr()

correlation.to_csv(TABLES_DIR / "correlation_matrix.csv")

print("\nCorrelation matrix:")
print(correlation)

print("\nFiles saved:")
print(FIGURES_DIR / "monthly_dengue_cases.png")
print(FIGURES_DIR / "rainfall_vs_dengue.png")
print(FIGURES_DIR / "rainfall_vs_log_dengue.png")
print(TABLES_DIR / "descriptive_statistics.csv")
print(TABLES_DIR / "monthly_national_summary.csv")
print(TABLES_DIR / "correlation_matrix.csv")

print("\nExploratory analysis completed.")
