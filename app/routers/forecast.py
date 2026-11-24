import logging
from datetime import date
from typing import Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.ml_model import forecast_range, train_model

router = APIRouter()
log = logging.getLogger(__name__)

MAX_FORECAST_DAYS = 1826  # ~5 years


class TrainRequest(BaseModel):
    model_type: Literal["random_forest", "gradient_boosting", "linear_regression"] = "random_forest"
    target:     Literal["fee_collected", "pending_amount"] = "fee_collected"


class ForecastRequest(BaseModel):
    start_date: date   # YYYY-MM-DD
    end_date:   date   # YYYY-MM-DD
    target:     Literal["fee_collected", "pending_amount"] = "fee_collected"


@router.post("/train", summary="Train ML model on uploaded data")
def train(req: TrainRequest):
    try:
        metrics = train_model(model_type=req.model_type, target=req.target)
        return {"message": "Model trained successfully.", "metrics": metrics}
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    except ValueError as e:
        raise HTTPException(422, str(e))
    except Exception:
        log.exception("Training failed")
        raise HTTPException(500, "Training failed due to an internal error.")


@router.post("/predict", summary="Forecast fee collection over a date range")
def predict(req: ForecastRequest):
    if req.end_date < req.start_date:
        raise HTTPException(422, "end_date must be on or after start_date.")
    days = (req.end_date - req.start_date).days + 1
    if days > MAX_FORECAST_DAYS:
        raise HTTPException(422, f"Date range too long ({days} days). Maximum is {MAX_FORECAST_DAYS}.")

    try:
        results = forecast_range(req.start_date, req.end_date, req.target)
        return {
            "target":           req.target,
            "start_date":       str(req.start_date),
            "end_date":         str(req.end_date),
            "total_days":       len(results),
            "total_forecasted": round(sum(r[req.target] for r in results), 2),
            "forecast":         results,
        }
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    except ValueError as e:
        raise HTTPException(422, str(e))
    except Exception:
        log.exception("Forecast failed")
        raise HTTPException(500, "Forecast failed due to an internal error.")
