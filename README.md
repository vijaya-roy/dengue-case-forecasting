# State-wise Dengue Case Forecasting and Its Link with Rainfall

## 1. Project Overview

This project investigates the relationship between reported dengue cases and rainfall across Indian states and union territories. It uses historical monthly dengue data and rainfall data to prepare a combined dataset, explore patterns, and compare machine learning regression models.

The project is related to:
- **SDG 3:** Good Health and Well-being
- **SDG 13:** Climate Action

The task is a regression problem because the objective is to estimate dengue case counts.

## 2. Objectives

- Clean and organize state-wise monthly dengue case data.
- Calculate monthly rainfall summaries from gridded rainfall data.
- Combine dengue and rainfall data using state, year, and month.
- Explore dengue case patterns and their association with rainfall.
- Compare Linear Regression, Ridge Regression, K-Nearest Neighbours (KNN) Regression, and polynomial-style regression.
- Evaluate models using Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and R-squared (R²).

## 3. Data Sources

### Dengue data

The main dengue dataset contains monthly state/UT case counts for 2018 and January–June 2019.

Source: National Centre for Vector Borne Diseases Control (NCVBDC), Ministry of Health and Family Welfare, Government of India.

Official source: https://ncvbdc.mohfw.gov.in/index4.php?lang=1&level=0&linkid=431&lid=3715

Monthly data source: https://www.data.gov.in/resource/stateut-and-month-wise-dengue-cases-during-2018-and-2019-ministry-health-and-family

### Rainfall data

The rainfall dataset is based on India Meteorological Department (IMD) gridded rainfall data. The downloaded NetCDF file covers 1 January 2018 to 30 June 2019, with rainfall values expressed in millimetres per day.

Source: https://imdpune.gov.in/

## 4. Project Structure

- `Scripts/` — Python scripts for data inspection, cleaning, merging, analysis, and model evaluation.
- `Processed_Data/` — cleaned dengue data, state-wise rainfall summaries, merged data, and modelling data.
- `Results/Figures/` — plots generated during exploratory analysis and model evaluation.
- `Results/Tables/` — descriptive statistics, correlation results, and model evaluation metrics.
- `Results/test_set_predictions.csv` — observed and predicted dengue case counts for the test set.
- `README.md` — project documentation.

The exact script filenames may differ slightly as the project develops.

## 5. Environment and Requirements

The project was developed in Python using a virtual environment.

Main libraries used include:
- pandas
- NumPy
- Matplotlib
- scikit-learn
- xarray
- netCDF4
- GeoPandas
- regionmask

Activate the virtual environment in the VS Code PowerShell terminal before running scripts:

```powershell
.\.venv\Scripts\Activate.ps1
```

If the environment has not been created, create it with:

```powershell
python -m venv .venv
```

Install the required packages in the active environment if they are not already installed.

## 6. Workflow

1. Inspect the source files and rainfall data.
2. Check state boundary data and calculate state-wise rainfall summaries.
3. Clean the monthly dengue case data.
4. Merge dengue and rainfall data using state, year, and month.
5. Validate the merged dataset for missing values and duplicate records.
6. Prepare the modelling dataset and log-transformed dengue case feature.
7. Perform exploratory data analysis and generate figures and summary tables.
8. Train and compare regression models.
9. Visualize model evaluation results.

Run each script using the filename that exists in the `Scripts` folder. For example:

```powershell
python Scripts/09_exploratory_analysis.py
python Scripts/10_train_models.py
python Scripts/11_visualize_model_results.py
```

Run prerequisite scripts first because later scripts depend on files created by earlier steps.

## 7. Preliminary Model Results

The current experiment produced the following test-set metrics:

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| KNN Regression | 28.62 | 70.87 | 0.402 |
| Polynomial-style Regression | 29.01 | 72.26 | 0.378 |
| Ridge Regression | 28.98 | 72.41 | 0.375 |
| Linear Regression | 30.67 | 73.93 | 0.349 |

KNN Regression achieved the lowest MAE and RMSE and the highest R² among these four models in the current experiment.

These metrics are preliminary and should be interpreted cautiously. The dataset covers only 18 months per included state, and the evaluation uses a limited historical period. The results do not establish reliable long-term forecasting performance.

## 8. Limitations

- The dengue data covers only January 2018 to June 2019.
- Only states with matched dengue and rainfall records are included in the current modelling dataset.
- State-name differences and administrative boundary differences can affect data matching.
- Rainfall is only one possible factor associated with dengue incidence.
- The observed association does not establish that rainfall caused changes in dengue cases.
- More years of data and further validation would be needed to assess forecasting performance reliably.

## 9. Reproducibility

The Python scripts and generated CSV files should be retained with the project. Record any changes made to the source data, state-name mappings, excluded records, model settings, and evaluation procedure. All reported metrics and conclusions should be based on the outputs actually generated by the scripts.

## 10. Author and Course

**Student:** Vijaya Roy  
**Roll Number:** 24075570061  
**Course:** Machine Learning  
**Programme:** B.Sc. (Hons.) Computer Science, Semester VI  
**Institution:** University of Delhi
