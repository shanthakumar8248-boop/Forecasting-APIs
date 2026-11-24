import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# Paths are anchored to the project root so the API works from any working directory.
BASE_DIR  = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "uploaded_data.csv"
MODEL_DIR = BASE_DIR / "models"

TARGETS   = ("fee_collected", "pending_amount")
MIN_ROWS  = 30          # minimum rows needed to train a meaningful model
TEST_FRAC = 0.2         # share of the most recent rows held out for evaluation

MODELS = {
    "linear_regression": LinearRegression,
    "random_forest":     lambda: RandomForestRegressor(random_state=42),
    "gradient_boosting": lambda: GradientBoostingRegressor(random_state=42),
}

FEATURE_COLS = ["year", "month", "quarter", "day_of_year", "day_of_week", "week_of_year"]


def artifact_paths(target: str) -> dict:
    """Each target gets its own model/scaler/metrics so training one never overwrites another."""
    return {
        "model":   MODEL_DIR / f"{target}_model.pkl",
        "scaler":  MODEL_DIR / f"{target}_scaler.pkl",
        "metrics": MODEL_DIR / f"{target}_metrics.json",
    }


def _check_target(target: str) -> None:
    if target not in TARGETS:
        raise ValueError(f"Invalid target '{target}'. Choose from: {', '.join(TARGETS)}.")


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
    _check_target(target)
    if model_type not in MODELS:
        raise ValueError(f"Invalid model_type '{model_type}'. Choose from: {', '.join(MODELS)}.")
    if not DATA_PATH.exists():
        raise FileNotFoundError("No data found. Upload data first via /data/upload.")

    df = pd.read_csv(DATA_PATH)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    if len(df) < MIN_ROWS:
        raise ValueError(f"Need at least {MIN_ROWS} rows to train, but the data has {len(df)}.")
    df = _add_features(df)

    # Chronological split: train on the past, evaluate on the most recent rows.
    # (A random split leaks future information on time series and inflates the scores.)
    split = int(len(df) * (1 - TEST_FRAC))
    train_df, test_df = df.iloc[:split], df.iloc[split:]
    X_train, y_train = train_df[FEATURE_COLS], train_df[target]
    X_test,  y_test  = test_df[FEATURE_COLS],  test_df[target]

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    model = MODELS[model_type]()
    model.fit(X_train_s, y_train)
    preds = model.predict(X_test_s)

    metrics = {
        "model_type":    model_type,
        "target":        target,
        "mae":           round(float(mean_absolute_error(y_test, preds)), 4),
        "rmse":          round(float(np.sqrt(mean_squared_error(y_test, preds))), 4),
        "r2_score":      round(float(r2_score(y_test, preds)), 4),
        "train_samples": int(len(X_train)),
        "test_samples":  int(len(X_test)),
        "data_start":    str(df["date"].min().date()),
        "data_end":      str(df["date"].max().date()),
        "trained_at":    datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }

    paths = artifact_paths(target)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model,  paths["model"])
    joblib.dump(scaler, paths["scaler"])
    paths["metrics"].write_text(json.dumps(metrics))

    return metrics


def load_metrics(target: str | None = None) -> dict:
    """Metrics for a target, or for the most recently trained model when no target is given."""
    if target is not None:
        _check_target(target)
        path = artifact_paths(target)["metrics"]
        if not path.exists():
            raise FileNotFoundError(f"No metrics for '{target}'. Train the model first.")
        return json.loads(path.read_text())

    files = list(MODEL_DIR.glob("*_metrics.json")) if MODEL_DIR.exists() else []
    if not files:
        raise FileNotFoundError("No metrics yet. Train the model first.")
    return json.loads(max(files, key=lambda p: p.stat().st_mtime).read_text())


def forecast_range(start_date, end_date, target: str = "fee_collected") -> list:
    _check_target(target)
    paths = artifact_paths(target)
    if not paths["model"].exists() or not paths["scaler"].exists():
        raise FileNotFoundError(f"No trained model for '{target}'. Call /forecast/train first.")

    model  = joblib.load(paths["model"])
    scaler = joblib.load(paths["scaler"])

    dates = pd.date_range(start=start_date, end=end_date, freq="D")
    df = _add_features(pd.DataFrame({"date": dates}))

    preds = model.predict(scaler.transform(df[FEATURE_COLS]))
    preds = np.clip(preds, 0, None)  # fee amounts can't be negative

    return [
        {"date": str(d.date()), target: round(float(p), 2)}
        for d, p in zip(dates, preds)
    ]
