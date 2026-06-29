# Energy Demand Forecasting with 3 Machine Learning Algorithms

This is a beginner-friendly machine learning project using a **Kaggle time-series dataset**.

## Project goal

Predict hourly electricity demand using past demand and date/time features.

This project includes **time series forecasting** because the data is ordered by time and the model uses previous-hour, previous-day, and previous-week values.

## Dataset

**Kaggle dataset:** Hourly Energy Consumption  
**Kaggle handle:** `robikscube/hourly-energy-consumption`  
**File used:** `PJME_hourly.csv`

Target column:

- `PJME_MW` = hourly energy demand in megawatts

## ML algorithms used

| Algorithm | Why used |
|---|---|
| Ridge Regression | Simple baseline model |
| Random Forest Regressor | Strong non-linear model |
| Gradient Boosting Regressor | Boosting model for better forecasting |

## Time-series features created

The project creates these features:

- hour
- day of week
- month
- year
- weekend flag
- lag 1 hour
- lag 24 hours
- lag 168 hours / previous week
- 24-hour rolling average
- 168-hour rolling average
- 24-hour rolling standard deviation

Important: lag and rolling features are shifted so the model does **not** look into the future.

## Results

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Random Forest | 303.00 | 422.17 | 0.9958 |
| Gradient Boosting | 547.69 | 714.22 | 0.9879 |
| Ridge Regression | 970.26 | 1239.75 | 0.9635 |

## Best model

**Random Forest** performed best.

It had the lowest error:

- MAE: 303.00
- RMSE: 422.17
- R²: 0.9958

## Folder structure

```text
ML_TimeSeries_3_Algos/
├── data/
│   ├── raw/
│   │   └── PJME_hourly.csv
│   └── processed/
│       └── pjme_features.csv
├── results/
│   ├── model_results.csv
│   ├── summary.txt
│   ├── top_features.csv
│   └── figures/
│       ├── actual_vs_predicted.png
│       └── top_features.png
├── src/
│   └── train_models.py
├── requirements.txt
└── README.md
```

## How to run

From the project folder:

```bash
source .venv/bin/activate
python src/train_models.py
```

If you create a new environment later:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/train_models.py
```

## Simple explanation for presentation

This project predicts hourly electricity demand using machine learning. I used a Kaggle time-series dataset and created lag features, rolling averages, and date-time features. Then I compared three algorithms: Ridge Regression, Random Forest, and Gradient Boosting. Random Forest gave the best performance because it captured non-linear patterns in electricity demand better than the simpler regression model.
