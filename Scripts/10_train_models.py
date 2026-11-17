
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Project paths
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "Processed_Data"
RESULTS_DIR = PROJECT_DIR / "Results"
TABLES_DIR = RESULTS_DIR / "Tables"

TABLES_DIR.mkdir(parents=True, exist_ok=True)

# Load modelling dataset
df = pd.read_csv(DATA_DIR / "model_data.csv")
df["Date"] = pd.to_datetime(df["Date"])

# Use only records with valid values
df = df.dropna(
    subset=[
        "State",
        "Year",
        "Month",
        "Dengue_Cases",
        "Monthly_Rainfall_mm",
    ]
).copy()

# Add a nonlinear rainfall feature for the polynomial-style model
df["Rainfall_Squared"] = df["Monthly_Rainfall_mm"] ** 2

# Chronological split: do not randomly mix future and past records
train = df[df["Date"] <= "2018-12-31"].copy()
test = df[df["Date"] >= "2019-01-01"].copy()

print("Training rows:", len(train))
print("Testing rows:", len(test))
print("Training period:", train["Date"].min(), "to", train["Date"].max())
print("Testing period:", test["Date"].min(), "to", test["Date"].max())

if train.empty or test.empty:
    raise ValueError("Training or testing data is empty. Check the dates.")

# Predict log(1 + cases) to reduce the influence of very large counts
y_train = np.log1p(train["Dengue_Cases"])
y_test_actual = test["Dengue_Cases"].to_numpy()

# Features used by the models
numeric_features = ["Year", "Month", "Monthly_Rainfall_mm"]
polynomial_numeric_features = [
    "Year", "Month", "Monthly_Rainfall_mm", "Rainfall_Squared"
]
categorical_features = ["State"]

def make_preprocessor(numeric_columns):
    return ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), numeric_columns),
            (
                "state",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_features,
            ),
        ]
    )

models = {
    "Linear Regression": Pipeline([
        ("preprocessor", make_preprocessor(numeric_features)),
        ("model", LinearRegression()),
    ]),
    "Ridge Regression": Pipeline([
        ("preprocessor", make_preprocessor(numeric_features)),
        ("model", Ridge(alpha=1.0)),
    ]),
    "KNN Regression": Pipeline([
        ("preprocessor", make_preprocessor(numeric_features)),
        ("model", KNeighborsRegressor(n_neighbors=5)),
    ]),
    "Polynomial-style Regression": Pipeline([
        ("preprocessor", make_preprocessor(polynomial_numeric_features)),
        ("model", Ridge(alpha=1.0)),
    ]),
}

# Save predictions and calculate metrics
results = []
predictions = test[["State", "Year", "Month", "Date", "Dengue_Cases"]].copy()

for name, model in models.items():
    feature_columns = (
        polynomial_numeric_features
        if name == "Polynomial-style Regression"
        else numeric_features
    )
    features = feature_columns + categorical_features

    model.fit(train[features], y_train)

    predicted_log = model.predict(test[features])
    predicted_cases = np.maximum(0, np.expm1(predicted_log))

    mae = mean_absolute_error(y_test_actual, predicted_cases)
    rmse = np.sqrt(mean_squared_error(y_test_actual, predicted_cases))
    r2 = r2_score(y_test_actual, predicted_cases)

    results.append({
        "Model": name,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
    })

    predictions[name.replace(" ", "_")] = predicted_cases

    print(
        f"{name}: MAE={mae:.3f}, "
        f"RMSE={rmse:.3f}, R2={r2:.3f}"
    )

# Save evaluation results
results_df = pd.DataFrame(results).sort_values("RMSE")
results_path = TABLES_DIR / "model_evaluation_metrics.csv"
results_df.to_csv(results_path, index=False)

predictions_path = RESULTS_DIR / "test_set_predictions.csv"
predictions.to_csv(predictions_path, index=False)

print("\nModel comparison:")
print(results_df.to_string(index=False))

print("\nSaved evaluation metrics to:", results_path)
print("Saved test predictions to:", predictions_path)
print("\nModel training and evaluation completed.")
print(
    "\nCAUTION: Only 18 months of data are available per state. "
    "These results are exploratory and do not establish reliable "
    "long-term forecasting performance."
)
