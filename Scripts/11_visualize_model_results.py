
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_DIR / "Results"
TABLES_DIR = RESULTS_DIR / "Tables"
FIGURES_DIR = RESULTS_DIR / "Figures"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Load model metrics
metrics = pd.read_csv(TABLES_DIR / "model_evaluation_metrics.csv")
metrics = metrics.sort_values("RMSE")

print("Model evaluation metrics:")
print(metrics.to_string(index=False))

# Plot 1: MAE comparison
plt.figure(figsize=(9, 5))
plt.bar(metrics["Model"], metrics["MAE"])
plt.title("Comparison of Models Using MAE")
plt.xlabel("Model")
plt.ylabel("Mean Absolute Error")
plt.xticks(rotation=20, ha="right")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "model_comparison_mae.png", dpi=300)
plt.close()

# Plot 2: RMSE comparison
plt.figure(figsize=(9, 5))
plt.bar(metrics["Model"], metrics["RMSE"])
plt.title("Comparison of Models Using RMSE")
plt.xlabel("Model")
plt.ylabel("Root Mean Squared Error")
plt.xticks(rotation=20, ha="right")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "model_comparison_rmse.png", dpi=300)
plt.close()

# Plot 3: R-squared comparison
plt.figure(figsize=(9, 5))
plt.bar(metrics["Model"], metrics["R2"])
plt.axhline(0, linewidth=1)
plt.title("Comparison of Models Using R-squared")
plt.xlabel("Model")
plt.ylabel("R-squared")
plt.xticks(rotation=20, ha="right")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "model_comparison_r2.png", dpi=300)
plt.close()

# Plot 4: Actual vs predicted cases for the best model by RMSE
predictions = pd.read_csv(RESULTS_DIR / "test_set_predictions.csv")
best_model = metrics.iloc[0]["Model"]
prediction_column = best_model.replace(" ", "_")

plt.figure(figsize=(7, 7))
plt.scatter(
    predictions["Dengue_Cases"],
    predictions[prediction_column],
    alpha=0.6
)

maximum = max(
    predictions["Dengue_Cases"].max(),
    predictions[prediction_column].max()
)

plt.plot([0, maximum], [0, maximum], linestyle="--")
plt.title(f"Actual vs Predicted Cases: {best_model}")
plt.xlabel("Actual Dengue Cases")
plt.ylabel("Predicted Dengue Cases")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "best_model_actual_vs_predicted.png", dpi=300)
plt.close()

print("\nFigures saved in:", FIGURES_DIR)
print("Best model by RMSE:", best_model)
print("Visualization completed.")
