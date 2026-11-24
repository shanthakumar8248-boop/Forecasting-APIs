import os
from datetime import date, timedelta

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

API = os.getenv("FORECAST_API_URL", "http://localhost:8000")
TIMEOUT = 60  # seconds; training can take a while on large files

ACCENT_1, ACCENT_2, ACCENT_3 = "#e94560", "#f7971e", "#43cea2"

st.set_page_config(page_title="Fee Forecast", page_icon="🎓", layout="wide")

# Styling is theme-neutral (works in light and dark mode) and avoids overriding
# the global font on every element, which breaks Streamlit's icon glyphs.
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
.stApp { font-family: 'Inter', sans-serif; }
.block-container { padding-top: 2rem; max-width: 1200px; }
div[data-testid="stMetric"] {
    background: rgba(128, 128, 128, 0.08);
    border-radius: 10px; padding: 14px 18px;
    border-left: 4px solid #e94560;
}
.stTabs [data-baseweb="tab"] { font-weight: 600; font-size: 0.95rem; }
.stButton > button, .stDownloadButton > button {
    border-radius: 8px; font-weight: 600;
}
.stButton > button[kind="primary"] {
    background: linear-gradient(90deg, #e94560, #f7971e);
    color: white; border: none; padding: 10px 28px;
}
.stButton > button[kind="primary"]:hover { filter: brightness(1.08); color: white; }
</style>
""", unsafe_allow_html=True)


# ── API helpers ───────────────────────────────────────────────────────────────
def _error_text(res: requests.Response, fallback: str) -> str:
    """FastAPI returns either a string or a list of validation errors as `detail`."""
    try:
        detail = res.json().get("detail", fallback)
    except ValueError:
        return fallback
    if isinstance(detail, list):
        return "; ".join(str(d.get("msg", d)) for d in detail)
    return str(detail)


def call_api(method: str, path: str, **kwargs):
    """Returns (data, error). Never raises, so the dashboard survives an offline API."""
    try:
        res = requests.request(method, f"{API}{path}", timeout=TIMEOUT, **kwargs)
    except requests.RequestException:
        return None, f"Cannot reach the API at {API}. Is it running?"
    if res.ok:
        return res.json(), None
    return None, _error_text(res, f"Request failed ({res.status_code}).")


@st.cache_data(ttl=10, show_spinner=False)
def api_status():
    try:
        r = requests.get(f"{API}/health", timeout=2)
        return r.json() if r.ok else None
    except (requests.RequestException, ValueError):
        return None


def style_fig(fig: go.Figure, title: str) -> go.Figure:
    fig.update_layout(
        title=title, margin=dict(l=10, r=10, t=50, b=10),
        legend=dict(orientation="h", y=1.1, x=0),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        hovermode="x unified",
    )
    return fig


def pretty(name: str) -> str:
    return name.replace("_", " ").title()


# ── Header ────────────────────────────────────────────────────────────────────
status = api_status()
col_h1, col_h2 = st.columns([3, 1])
with col_h1:
    st.markdown("## 🎓 Fee Collection Forecast Dashboard")
    st.caption("AI-powered forecasting for educational institutions")
with col_h2:
    if status:
        st.success("API Connected ✅")
    else:
        st.error("API Offline ❌")
        st.caption(f"Expected at `{API}`")

st.divider()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Model Settings")
    model_type = st.selectbox("Algorithm", [
        "random_forest", "gradient_boosting", "linear_regression"
    ], format_func=pretty)
    target = st.selectbox("Forecast Target", ["fee_collected", "pending_amount"], format_func=pretty)
    st.divider()

    st.subheader("Status")
    if status:
        st.write("📁 Data uploaded" if status.get("data_uploaded") else "📁 No data yet")
        trained = status.get("trained_models", [])
        for t in ("fee_collected", "pending_amount"):
            st.write(f"{'✅' if t in trained else '⬜'} {pretty(t)} model")
    else:
        st.caption("Start the API to see status.")
    st.divider()
    st.info("**Workflow**\n1. Upload CSV\n2. Train Model\n3. Run Forecast\n4. View Metrics")

# ── Tabs ──────────────────────────────────────────────────────────────────────
t1, t2, t3, t4 = st.tabs(["📤 Upload Data", "🤖 Train Model", "📈 Forecast", "📊 Metrics"])


def metric_row(m: dict):
    c1, c2, c3 = st.columns(3)
    c1.metric("MAE", f"{m['mae']:,.2f}", help="Mean absolute error — average miss, in currency units.")
    c2.metric("RMSE", f"{m['rmse']:,.2f}", help="Root mean squared error — penalises large misses more.")
    c3.metric("R² Score", f"{m['r2_score']:.3f}", help="1.0 is perfect; 0 is no better than predicting the average.")


# ── TAB 1 ─────────────────────────────────────────────────────────────────────
with t1:
    st.subheader("Upload Historical Fee Data")

    c1, c2 = st.columns([2, 1])
    with c1:
        st.markdown("Your CSV must have these **3 columns** (at least 30 rows):")
        st.code("date, fee_collected, pending_amount", language="text")
    with c2:
        sample = pd.DataFrame({
            "date":           ["2024-01-01", "2024-01-02", "2024-01-03"],
            "fee_collected":  [15000, 22000, 18500],
            "pending_amount": [5000, 3200, 7100],
        })
        st.download_button("⬇️ Sample CSV", sample.to_csv(index=False),
                           "sample_fee_data.csv", "text/csv", width="stretch")

    uploaded = st.file_uploader("Choose CSV file", type=["csv"])
    if uploaded and st.button("Upload to API", type="primary", width="stretch"):
        with st.spinner("Uploading..."):
            d, err = call_api("POST", "/data/upload",
                              files={"file": (uploaded.name, uploaded.getvalue(), "text/csv")})
        if err:
            st.error(err)
        else:
            api_status.clear()
            st.success(f"✅ {d['rows']:,} rows uploaded ({d['date_range'][0]} → {d['date_range'][1]})")
            if d.get("skipped_rows"):
                st.warning(f"{d['skipped_rows']} row(s) with invalid dates or numbers were skipped.")
            st.dataframe(pd.DataFrame(d["preview"]), width="stretch", hide_index=True)

    st.divider()
    st.subheader("Current Data Preview")
    if st.button("🔄 Load Preview"):
        d, err = call_api("GET", "/data/preview")
        if err:
            st.warning(err)
        else:
            st.caption(f"{d['rows']:,} rows · {d['date_range'][0]} → {d['date_range'][1]} · "
                       f"Columns: {', '.join(d['columns'])}")
            st.dataframe(pd.DataFrame(d["preview"]), width="stretch", hide_index=True)


# ── TAB 2 ─────────────────────────────────────────────────────────────────────
with t2:
    st.subheader("Train Forecasting Model")
    st.markdown(f"**Algorithm:** `{model_type}` &nbsp;|&nbsp; **Target:** `{target}`")
    st.caption("The most recent 20% of your data is held out to evaluate the model.")

    if st.button("🚀 Train Model", type="primary", width="stretch"):
        with st.spinner("Training model — please wait..."):
            d, err = call_api("POST", "/forecast/train",
                              json={"model_type": model_type, "target": target})
        if err:
            st.error(err)
        else:
            api_status.clear()
            st.success("✅ Model trained successfully!")
            metric_row(d["metrics"])
            st.caption("See the **Metrics** tab for full details.")


# ── TAB 3 ─────────────────────────────────────────────────────────────────────
with t3:
    st.subheader("Date-Range Forecast")
    c1, c2 = st.columns(2)
    start = c1.date_input("Start Date", value=date.today() + timedelta(days=1))
    end   = c2.date_input("End Date",   value=date.today() + timedelta(days=30))

    if st.button("📈 Run Forecast", type="primary", width="stretch"):
        if end < start:
            st.error("End date must be on or after the start date.")
        else:
            with st.spinner("Generating forecast..."):
                d, err = call_api("POST", "/forecast/predict", json={
                    "start_date": str(start), "end_date": str(end), "target": target,
                })
                hist, _ = call_api("GET", "/data/history", params={"target": target, "days": 180})
            if err:
                st.error(err)
                st.session_state.pop("forecast", None)
            else:
                # Stored in session state so results survive reruns (e.g. clicking Download).
                st.session_state["forecast"] = {"data": d, "history": hist, "target": target}

    fc = st.session_state.get("forecast")
    if fc:
        d, hist, ftarget = fc["data"], fc["history"], fc["target"]
        df = pd.DataFrame(d["forecast"])
        df["date"] = pd.to_datetime(df["date"])

        st.success(f"{pretty(ftarget)} forecast for {d['total_days']} days "
                   f"({d['start_date']} → {d['end_date']})")

        c1, c2, c3 = st.columns(3)
        c1.metric("Total Forecasted", f"{d['total_forecasted']:,.0f}")
        c2.metric("Daily Average",    f"{df[ftarget].mean():,.0f}")
        c3.metric("Peak Day Value",   f"{df[ftarget].max():,.0f}",
                  help=f"Peak on {df.loc[df[ftarget].idxmax(), 'date']:%Y-%m-%d}")

        fig1 = go.Figure()
        if hist and hist.get("history"):
            h = pd.DataFrame(hist["history"])
            h["date"] = pd.to_datetime(h["date"])
            fig1.add_trace(go.Scatter(x=h["date"], y=h[ftarget], name="Historical",
                                      line=dict(color="#8a8d93", width=1.5)))
        fig1.add_trace(go.Scatter(x=df["date"], y=df[ftarget], name="Forecast",
                                  line=dict(color=ACCENT_1, width=2),
                                  fill="tozeroy", fillcolor="rgba(233,69,96,0.15)"))
        st.plotly_chart(style_fig(fig1, f"{pretty(ftarget)} — Daily Forecast"), width="stretch")

        monthly = (df.assign(month=df["date"].dt.to_period("M").astype(str))
                     .groupby("month")[ftarget].sum().reset_index())
        fig2 = go.Figure(go.Bar(x=monthly["month"], y=monthly[ftarget], marker_color=ACCENT_2,
                                name="Monthly total"))
        st.plotly_chart(style_fig(fig2, "Monthly Aggregated Forecast"), width="stretch")

        with st.expander("Forecast table"):
            st.dataframe(df, width="stretch", hide_index=True,
                         column_config={"date": st.column_config.DateColumn("Date", format="YYYY-MM-DD"),
                                        ftarget: st.column_config.NumberColumn(pretty(ftarget), format="%.2f")})
        st.download_button("⬇️ Download Forecast CSV", df.to_csv(index=False),
                           "forecast.csv", "text/csv")


# ── TAB 4 ─────────────────────────────────────────────────────────────────────
with t4:
    st.subheader("Model Performance Metrics")
    st.caption(f"Showing the model trained for **{pretty(target)}** (change in the sidebar).")

    m, err = call_api("GET", "/metrics/summary", params={"target": target})
    if err:
        st.warning(err)
    else:
        metric_row(m)
        st.divider()
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**Model Type:** `{m['model_type']}`")
            st.markdown(f"**Target:** `{m['target']}`")
            if m.get("trained_at"):
                st.markdown(f"**Trained At:** `{m['trained_at'].replace('T', ' ')[:16]} UTC`")
        with col2:
            st.markdown(f"**Train Samples:** `{m['train_samples']:,}`")
            st.markdown(f"**Test Samples:** `{m['test_samples']:,}`")
            if m.get("data_start"):
                st.markdown(f"**Data Range:** `{m['data_start']} → {m['data_end']}`")
