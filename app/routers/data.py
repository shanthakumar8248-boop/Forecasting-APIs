from io import StringIO
from typing import Literal

import pandas as pd
from fastapi import APIRouter, File, HTTPException, Query, UploadFile

from app.ml_model import DATA_PATH, MIN_ROWS

router = APIRouter()

REQUIRED       = ["date", "fee_collected", "pending_amount"]
NUMERIC        = ["fee_collected", "pending_amount"]
MAX_UPLOAD_MB  = 10


def _load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise HTTPException(404, "No data uploaded yet.")
    return pd.read_csv(DATA_PATH)


@router.post("/upload", summary="Upload historical fee collection CSV")
async def upload(file: UploadFile = File(...)):
    if not (file.filename or "").lower().endswith(".csv"):
        raise HTTPException(400, "Only CSV files are accepted.")

    content = await file.read()
    if len(content) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, f"File too large. Maximum size is {MAX_UPLOAD_MB} MB.")

    try:
        df = pd.read_csv(StringIO(content.decode("utf-8-sig")))  # utf-8-sig also handles Excel's BOM
    except Exception:
        raise HTTPException(422, "Could not parse CSV. Check file format.")

    df.columns = df.columns.str.strip().str.lower()
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise HTTPException(422, f"Missing required columns: {', '.join(missing)}")
    df = df[REQUIRED]

    # Clean and validate so bad data fails here with a clear message, not later during training.
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for col in NUMERIC:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    bad_rows = int(df.isna().any(axis=1).sum())
    df = df.dropna()
    if df.empty:
        raise HTTPException(422, "No valid rows found. Check the date format (YYYY-MM-DD) and numeric columns.")
    if len(df) < MIN_ROWS:
        raise HTTPException(422, f"Need at least {MIN_ROWS} valid rows, but found {len(df)}.")

    df = df.sort_values("date").drop_duplicates(subset="date", keep="last")
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")

    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(DATA_PATH, index=False)

    return {
        "message":      "Uploaded successfully.",
        "rows":         len(df),
        "skipped_rows": bad_rows,
        "date_range":   [df["date"].iloc[0], df["date"].iloc[-1]],
        "columns":      list(df.columns),
        "preview":      df.head(5).to_dict(orient="records"),
    }


@router.get("/preview", summary="Preview the uploaded dataset")
def preview():
    df = _load_data()
    return {
        "rows":       len(df),
        "date_range": [str(df["date"].iloc[0]), str(df["date"].iloc[-1])],
        "columns":    list(df.columns),
        "preview":    df.head(10).to_dict(orient="records"),
    }


@router.get("/history", summary="Recent historical values (for charting against forecasts)")
def history(
    target: Literal["fee_collected", "pending_amount"] = "fee_collected",
    days:   int = Query(180, ge=1, le=3650, description="Number of most recent rows to return"),
):
    df = _load_data()
    return {"target": target, "history": df[["date", target]].tail(days).to_dict(orient="records")}
