# Project Report: Energy Demand Forecasting

## 1. Introduction

This project uses machine learning to predict hourly electricity demand. The dataset is a time-series dataset from Kaggle, so each row is connected to a specific date and hour.

The goal is to compare three machine learning algorithms and see which one predicts future energy demand best.

## 2. Dataset

The dataset comes from Kaggle:

- Dataset name: Hourly Energy Consumption
- Kaggle handle: `robikscube/hourly-energy-consumption`
- File used: `PJME_hourly.csv`

The target variable is:

- `PJME_MW`: electricity demand in megawatts

After feature engineering, the dataset had 145,194 usable rows.

## 3. Time-Series Feature Engineering

To make the data useful for forecasting, I created time-based features:

- hour
- day of week
- month
- year
- weekend flag

I also created lag and rolling features:

- previous hour demand
- previous day demand
- previous week demand
- 24-hour rolling average
- 168-hour rolling average
- 24-hour rolling standard deviation

These features help the model learn daily and weekly electricity demand patterns.

## 4. Machine Learning Algorithms

I used three machine learning algorithms:

1. Ridge Regression
2. Random Forest Regressor
3. Gradient Boosting Regressor

Ridge Regression was used as a simple baseline. Random Forest and Gradient Boosting were used because they can learn more complex patterns.

## 5. Model Evaluation

The data was split by time, not randomly. The first 80% was used for training and the last 20% was used for testing. This is important for time-series projects because future data should not be used to train the model.

Metrics used:

- MAE: Mean Absolute Error
- RMSE: Root Mean Squared Error
- R²: How well the model explains the target values

## 6. Results

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Random Forest | 303.00 | 422.17 | 0.9958 |
| Gradient Boosting | 547.69 | 714.22 | 0.9879 |
| Ridge Regression | 970.26 | 1239.75 | 0.9635 |

## 7. Conclusion

Random Forest was the best model because it had the lowest MAE and RMSE. This means its predictions were closest to the real energy demand values.

The project shows how time-series forecasting can be done using normal machine learning models when good lag and rolling features are created.

## 8. Future Improvements

In the future, this project can be improved by:

- adding weather data
- adding holiday features
- trying XGBoost or LightGBM
- using deep learning models like LSTM
- building a Streamlit dashboard
