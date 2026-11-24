# 🎓 Fee Collection Forecast API

AI/ML API to forecast fee collection trends and pending amount patterns for educational institutions.

**Stack:** FastAPI · scikit-learn · Pandas · Streamlit · Plotly

---

## 🗂 Project Structure

```
fee-forecast-api/
├── app/
│   ├── main.py           → FastAPI entry point
│   ├── ml_model.py       → ML training & forecasting
│   └── routers/
│       ├── data.py       → Upload & preview endpoints
│       ├── forecast.py   → Train + predict endpoints
│       └── metrics.py    → Metrics endpoint
├── streamlit_app/
│   └── dashboard.py      → Interactive UI
├── data/
│   └── generate_sample.py
├── models/               → Saved model files, one set per target (auto-created, git-ignored)
├── requirements.txt
└── README.md
```

---

## 🚀 Setup & Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Start the API
```bash
python -m uvicorn app.main:app --reload
```
- API: http://localhost:8000
- Swagger Docs: http://localhost:8000/docs

### 3. Start the Dashboard (new terminal)
```bash
streamlit run streamlit_app/dashboard.py
```
- Dashboard: http://localhost:8501

---

## 📄 CSV Format

```csv
date,fee_collected,pending_amount
2024-01-01,15000,5000
2024-01-02,22000,3200
```

Generate sample data:
```bash
python data/generate_sample.py
```

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Data / trained-model readiness |
| POST | `/data/upload` | Upload & validate CSV (min 30 valid rows, max 10 MB) |
| GET | `/data/preview` | Preview data |
| GET | `/data/history?target=&days=` | Recent history (used to chart against forecasts) |
| POST | `/forecast/train` | Train ML model for one target |
| POST | `/forecast/predict` | Date-range forecast (max ~5 years) |
| GET | `/metrics/summary?target=` | Model metrics (latest model if `target` omitted) |

> Each target (`fee_collected`, `pending_amount`) has its own saved model, so training one never overwrites the other.
> Metrics are computed on the most recent 20% of the data (chronological split), which is a realistic test for forecasting.

The dashboard reads the API address from the `FORECAST_API_URL` environment variable (default `http://localhost:8000`).

---

## 🤖 Supported Models

- `random_forest` (default)
- `gradient_boosting`
- `linear_regression`
