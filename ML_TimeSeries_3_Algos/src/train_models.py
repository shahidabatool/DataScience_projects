import shutil
from pathlib import Path

import kagglehub
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
RESULTS_DIR = PROJECT_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

DATASET_HANDLE = "robikscube/hourly-energy-consumption"
SOURCE_FILE = "PJME_hourly.csv"
TARGET_COLUMN = "PJME_MW"
DATE_COLUMN = "Datetime"


def make_dirs():
    for folder in [RAW_DIR, PROCESSED_DIR, RESULTS_DIR, FIGURES_DIR]:
        folder.mkdir(parents=True, exist_ok=True)


def download_kaggle_data() -> Path:
    """Download Kaggle data and copy one CSV into this project."""
    kaggle_path = Path(kagglehub.dataset_download(DATASET_HANDLE))
    src = kaggle_path / SOURCE_FILE
    dst = RAW_DIR / SOURCE_FILE
    if not dst.exists():
        shutil.copy2(src, dst)
    return dst


def load_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN])
    df = df.sort_values(DATE_COLUMN).drop_duplicates(DATE_COLUMN)
    df = df.set_index(DATE_COLUMN)
    return df


def create_time_series_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create calendar, lag, and rolling features for time-series forecasting."""
    data = df.copy()
    data["hour"] = data.index.hour
    data["dayofweek"] = data.index.dayofweek
    data["quarter"] = data.index.quarter
    data["month"] = data.index.month
    data["year"] = data.index.year
    data["dayofyear"] = data.index.dayofyear
    data["weekofyear"] = data.index.isocalendar().week.astype(int)
    data["is_weekend"] = (data.index.dayofweek >= 5).astype(int)

    # Lag features: previous hour, previous day, previous week
    data["lag_1"] = data[TARGET_COLUMN].shift(1)
    data["lag_24"] = data[TARGET_COLUMN].shift(24)
    data["lag_168"] = data[TARGET_COLUMN].shift(168)

    # Rolling demand features, shifted to avoid data leakage
    shifted = data[TARGET_COLUMN].shift(1)
    data["rolling_24_mean"] = shifted.rolling(window=24).mean()
    data["rolling_168_mean"] = shifted.rolling(window=168).mean()
    data["rolling_24_std"] = shifted.rolling(window=24).std()

    data = data.dropna()
    return data


def train_test_split_time(data: pd.DataFrame):
    feature_cols = [c for c in data.columns if c != TARGET_COLUMN]
    X = data[feature_cols]
    y = data[TARGET_COLUMN]

    split_index = int(len(data) * 0.8)
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]
    return X_train, X_test, y_train, y_test, feature_cols


def get_models():
    return {
        "Ridge Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=1.0)),
        ]),
        "Random Forest": RandomForestRegressor(
            n_estimators=120,
            max_depth=16,
            min_samples_leaf=3,
            random_state=42,
            n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=180,
            learning_rate=0.05,
            max_depth=3,
            random_state=42,
        ),
    }


def evaluate_model(name, model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    # Small time-series cross validation on training data
    tscv = TimeSeriesSplit(n_splits=3)
    cv_scores = cross_val_score(
        model,
        X_train,
        y_train,
        cv=tscv,
        scoring="neg_mean_absolute_error",
        n_jobs=None,
    )

    return {
        "model": name,
        "MAE": round(mae, 2),
        "RMSE": round(rmse, 2),
        "R2": round(r2, 4),
        "CV_MAE_mean": round(float(-cv_scores.mean()), 2),
    }, pd.Series(preds, index=y_test.index, name=name)


def save_prediction_plot(y_test, predictions):
    sample_days = 14 * 24
    plt.figure(figsize=(14, 6))
    plt.plot(y_test.iloc[:sample_days].index, y_test.iloc[:sample_days], label="Actual", linewidth=2)
    for name, pred in predictions.items():
        plt.plot(pred.iloc[:sample_days].index, pred.iloc[:sample_days], label=name, alpha=0.85)
    plt.title("Hourly Energy Demand Forecast: Actual vs Predicted (First 14 Test Days)")
    plt.xlabel("Date")
    plt.ylabel("Energy Demand (MW)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "actual_vs_predicted.png", dpi=160)
    plt.close()


def save_feature_importance(best_model_name, best_model, feature_cols):
    if hasattr(best_model, "feature_importances_"):
        importances = pd.Series(best_model.feature_importances_, index=feature_cols)
    elif isinstance(best_model, Pipeline):
        ridge = best_model.named_steps["model"]
        importances = pd.Series(np.abs(ridge.coef_), index=feature_cols)
    else:
        return

    top = importances.sort_values(ascending=False).head(12)
    top.to_csv(RESULTS_DIR / "top_features.csv", header=["importance"])

    plt.figure(figsize=(10, 5))
    top.sort_values().plot(kind="barh")
    plt.title(f"Top Features: {best_model_name}")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "top_features.png", dpi=160)
    plt.close()


def main():
    make_dirs()
    csv_path = download_kaggle_data()
    raw_df = load_data(csv_path)
    data = create_time_series_features(raw_df)
    data.to_csv(PROCESSED_DIR / "pjme_features.csv")

    X_train, X_test, y_train, y_test, feature_cols = train_test_split_time(data)
    models = get_models()

    rows = []
    predictions = {}
    fitted_models = {}
    for name, model in models.items():
        row, pred = evaluate_model(name, model, X_train, X_test, y_train, y_test)
        rows.append(row)
        predictions[name] = pred
        fitted_models[name] = model
        print(f"{name}: MAE={row['MAE']}, RMSE={row['RMSE']}, R2={row['R2']}")

    results = pd.DataFrame(rows).sort_values("MAE")
    results.to_csv(RESULTS_DIR / "model_results.csv", index=False)

    best_name = results.iloc[0]["model"]
    save_prediction_plot(y_test, predictions)
    save_feature_importance(best_name, fitted_models[best_name], feature_cols)

    summary = (
        f"Dataset: Kaggle - {DATASET_HANDLE}\n"
        f"Rows after feature engineering: {len(data):,}\n"
        f"Train rows: {len(X_train):,}\n"
        f"Test rows: {len(X_test):,}\n"
        f"Best model by MAE: {best_name}\n"
        f"Best MAE: {results.iloc[0]['MAE']}\n"
        f"Best RMSE: {results.iloc[0]['RMSE']}\n"
        f"Best R2: {results.iloc[0]['R2']}\n"
    )
    (RESULTS_DIR / "summary.txt").write_text(summary, encoding="utf-8")
    print("\n" + summary)
    print(f"Saved results to: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
