from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
from app.ml_model import train_model, forecast_range

router = APIRouter()


class TrainRequest(BaseModel):
    model_type: Optional[str] = "random_forest"  # random_forest | gradient_boosting | linear_regression
    target:     Optional[str] = "fee_collected"   # fee_collected | pending_amount


class ForecastRequest(BaseModel):
    start_date: str            # YYYY-MM-DD
    end_date:   str            # YYYY-MM-DD
    target:     Optional[str] = "fee_collected"


@router.post("/train", summary="Train ML model on uploaded data")
def train(req: TrainRequest):
    try:
        metrics = train_model(model_type=req.model_type, target=req.target)
        return {"message": "Model trained successfully.", "metrics": metrics}
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    except Exception as e:
        raise HTTPException(500, str(e))


@router.post("/predict", summary="Forecast fee collection over a date range")
def predict(req: ForecastRequest):
    try:
        results = forecast_range(req.start_date, req.end_date, req.target)
        total   = round(sum(r[req.target] for r in results), 2)
        return {
            "target":     req.target,
            "start_date": req.start_date,
            "end_date":   req.end_date,
            "total_days": len(results),
            "total_forecasted": total,
            "forecast":   results,
        }
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    except Exception as e:
        raise HTTPException(500, str(e))
