import os
import pandas as pd
from io import StringIO
from fastapi import APIRouter, UploadFile, File, HTTPException

router = APIRouter()
DATA_PATH = "data/uploaded_data.csv"
REQUIRED   = {"date", "fee_collected", "pending_amount"}


@router.post("/upload", summary="Upload historical fee collection CSV")
async def upload(file: UploadFile = File(...)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(400, "Only CSV files are accepted.")

    content = await file.read()
    try:
        df = pd.read_csv(StringIO(content.decode("utf-8")))
    except Exception:
        raise HTTPException(422, "Could not parse CSV. Check file format.")

    df.columns = df.columns.str.strip().str.lower()
    missing = REQUIRED - set(df.columns)
    if missing:
        raise HTTPException(422, f"Missing required columns: {missing}")

    os.makedirs("data", exist_ok=True)
    df.to_csv(DATA_PATH, index=False)

    return {
        "message": "Uploaded successfully.",
        "rows": len(df),
        "columns": list(df.columns),
        "preview": df.head(5).to_dict(orient="records"),
    }


@router.get("/preview", summary="Preview the uploaded dataset")
def preview():
    if not os.path.exists(DATA_PATH):
        raise HTTPException(404, "No data uploaded yet.")
    df = pd.read_csv(DATA_PATH)
    return {
        "rows": len(df),
        "columns": list(df.columns),
        "preview": df.head(10).to_dict(orient="records"),
    }
