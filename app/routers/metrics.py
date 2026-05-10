import os, json
from fastapi import APIRouter, HTTPException

router   = APIRouter()
METRICS_PATH = "models/metrics.json"


@router.get("/summary", summary="Get model performance metrics")
def summary():
    if not os.path.exists(METRICS_PATH):
        raise HTTPException(404, "No metrics yet. Train the model first.")
    with open(METRICS_PATH) as f:
        return json.load(f)
