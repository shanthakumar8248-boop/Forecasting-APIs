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
├── models/               → Saved model files (auto-created)
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
| POST | `/data/upload` | Upload CSV |
| GET | `/data/preview` | Preview data |
| POST | `/forecast/train` | Train ML model |
| POST | `/forecast/predict` | Date-range forecast |
| GET | `/metrics/summary` | Model metrics |

---

## 🤖 Supported Models

- `random_forest` (default)
- `gradient_boosting`
- `linear_regression`
