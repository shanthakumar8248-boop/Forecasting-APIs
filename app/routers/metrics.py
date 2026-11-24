from typing import Literal, Optional

from fastapi import APIRouter, HTTPException, Query

from app.ml_model import load_metrics

router = APIRouter()


@router.get("/summary", summary="Get model performance metrics")
def summary(
    target: Optional[Literal["fee_collected", "pending_amount"]] = Query(
        None, description="Target to get metrics for. Defaults to the most recently trained model."
    ),
):
    try:
        return load_metrics(target)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
