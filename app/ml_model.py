import os, json, joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

DATA_PATH    = "data/uploaded_data.csv"
MODEL_PATH   = "models/model.pkl"
SCALER_PATH  = "models/scaler.pkl"
METRICS_PATH = "models/metrics.json"

MODELS = {
    "linear_regression":  LinearRegression,
    "random_forest":      RandomForestRegressor,
    "gradient_boosting":  GradientBoostingRegressor,
}

FEATURE_COLS = ["year", "month", "quarter", "day_of_year", "day_of_week", "week_of_year"]


def _add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"]         = pd.to_datetime(df["date"])
    df["year"]         = df["date"].dt.year
    df["month"]        = df["date"].dt.month
    df["quarter"]      = df["date"].dt.quarter
    df["day_of_year"]  = df["date"].dt.dayofyear
    df["day_of_week"]  = df["date"].dt.dayofweek
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
    return df


def train_model(model_type: str = "random_forest", target: str = "fee_collected") -> dict:
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError("No data found. Upload data first via /data/upload.")

    df = pd.read_csv(DATA_PATH)
    df = _add_features(df)

    X = df[FEATURE_COLS]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    cls = MODELS.get(model_type, RandomForestRegressor)
    model = cls()
    model.fit(X_train_s, y_train)
    preds = model.predict(X_test_s)

    metrics = {
        "model_type":     model_type,
        "target":         target,
        "mae":            round(float(mean_absolute_error(y_test, preds)), 4),
        "rmse":           round(float(np.sqrt(mean_squared_error(y_test, preds))), 4),
        "r2_score":       round(float(r2_score(y_test, preds)), 4),
        "train_samples":  int(len(X_train)),
        "test_samples":   int(len(X_test)),
    }

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f)

    return metrics


def forecast_range(start_date: str, end_date: str, target: str = "fee_collected") -> list:
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Model not trained yet. Call /forecast/train first.")

    model  = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    dates = pd.date_range(start=start_date, end=end_date, freq="D")
    df = pd.DataFrame({"date": dates})
    df = _add_features(df)

    X = scaler.transform(df[FEATURE_COLS])
    preds = model.predict(X)

    return [
        {"date": str(d.date()), target: round(float(p), 2)}
        for d, p in zip(dates, preds)
    ]
