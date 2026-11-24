from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import data, forecast, metrics

app = FastAPI(
    title="Fee Collection Forecast API",
    description="AI/ML API to forecast fee collection trends and pending amount patterns in education.",
    version="2.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(data.router,     prefix="/data",     tags=["Data Upload"])
app.include_router(forecast.router, prefix="/forecast", tags=["Forecast"])
app.include_router(metrics.router,  prefix="/metrics",  tags=["Model Metrics"])


@app.get("/", tags=["Health"])
def root():
    return {"status": "ok", "message": "Fee Collection Forecast API v2.1 is running 🎓"}


@app.get("/health", tags=["Health"], summary="Data and model readiness")
def health():
    from app.ml_model import DATA_PATH, TARGETS, artifact_paths

    return {
        "status":         "ok",
        "data_uploaded":  DATA_PATH.exists(),
        "trained_models": [t for t in TARGETS if artifact_paths(t)["model"].exists()],
    }
