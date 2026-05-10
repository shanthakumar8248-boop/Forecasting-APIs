import streamlit as st
import pandas as pd
import requests
import plotly.express as px
import plotly.graph_objects as go

API = "http://localhost:8000"

st.set_page_config(page_title="Fee Forecast", page_icon="🎓", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
* { font-family: 'Inter', sans-serif; }
.block-container { padding-top: 2rem; }
.metric-box {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    border-radius: 14px; padding: 18px 24px; color: white;
    border-left: 5px solid #f7971e;
}
div[data-testid="stMetric"] {
    background: #1a1a2e; border-radius: 10px;
    padding: 16px; border-left: 4px solid #e94560;
}
div[data-testid="stMetric"] label { color: #aaa !important; }
div[data-testid="stMetric"] div { color: white !important; font-size: 1.4rem !important; }
.stTabs [data-baseweb="tab"] { font-weight: 600; font-size: 0.95rem; }
.stButton > button {
    background: linear-gradient(90deg, #e94560, #f7971e);
    color: white; border: none; border-radius: 8px;
    padding: 10px 28px; font-weight: 600; font-size: 0.95rem;
}
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
col_h1, col_h2 = st.columns([3, 1])
with col_h1:
    st.markdown("## 🎓 Fee Collection Forecast Dashboard")
    st.caption("AI-powered forecasting for educational institutions")
with col_h2:
    try:
       r = requests.get(f"{API}/", timeout=2)
       if r.status_code == 200:
          st.success("API Connected ✅")
       else:
          st.error("API Offline ❌")
    except Exception:
        st.error("API Offline ❌")

st.divider()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Model Settings")
    model_type = st.selectbox("Algorithm", [
        "random_forest", "gradient_boosting", "linear_regression"
    ])
    target = st.selectbox("Forecast Target", ["fee_collected", "pending_amount"])
    st.divider()
    st.info("**Workflow**\n1. Upload CSV\n2. Train Model\n3. Run Forecast\n4. View Metrics")

# ── Tabs ──────────────────────────────────────────────────────────────────────
t1, t2, t3, t4 = st.tabs(["📤 Upload Data", "🤖 Train Model", "📈 Forecast", "📊 Metrics"])


# ── TAB 1 ─────────────────────────────────────────────────────────────────────
with t1:
    st.subheader("Upload Historical Fee Data")

    c1, c2 = st.columns([2, 1])
    with c1:
        st.markdown("Your CSV must have these **3 columns**:")
        st.code("date, fee_collected, pending_amount", language="text")
    with c2:
        sample = pd.DataFrame({
            "date":            ["2024-01-01", "2024-01-02", "2024-01-03"],
            "fee_collected":   [15000, 22000, 18500],
            "pending_amount":  [5000, 3200, 7100],
        })
        st.download_button("⬇️ Sample CSV", sample.to_csv(index=False),
                           "sample_fee_data.csv", "text/csv", use_container_width=True)

    uploaded = st.file_uploader("Choose CSV file", type=["csv"])
    if uploaded:
        if st.button("Upload to API", use_container_width=True):
            with st.spinner("Uploading..."):
                res = requests.post(f"{API}/data/upload",
                                    files={"file": (uploaded.name, uploaded.getvalue(), "text/csv")})
            if res.status_code == 200:
                d = res.json()
                st.success(f"✅ {d['rows']} rows uploaded successfully!")
                st.dataframe(pd.DataFrame(d["preview"]), use_container_width=True)
            else:
                st.error(res.json().get("detail", "Upload failed."))

    st.divider()
    st.subheader("Current Data Preview")
    if st.button("🔄 Load Preview"):
        res = requests.get(f"{API}/data/preview")
        if res.status_code == 200:
            d = res.json()
            st.caption(f"{d['rows']} rows | Columns: {', '.join(d['columns'])}")
            st.dataframe(pd.DataFrame(d["preview"]), use_container_width=True)
        else:
            st.warning("No data uploaded yet.")


# ── TAB 2 ─────────────────────────────────────────────────────────────────────
with t2:
    st.subheader("Train Forecasting Model")
    st.markdown(f"**Algorithm:** `{model_type}` &nbsp;|&nbsp; **Target:** `{target}`")

    if st.button("🚀 Train Model", use_container_width=True):
        with st.spinner("Training model — please wait..."):
            res = requests.post(f"{API}/forecast/train",
                                json={"model_type": model_type, "target": target})
        if res.status_code == 200:
            m = res.json()["metrics"]
            st.success("✅ Model trained successfully!")
            c1, c2, c3 = st.columns(3)
            c1.metric("MAE",      m["mae"])
            c2.metric("RMSE",     m["rmse"])
            c3.metric("R² Score", m["r2_score"])

            fig = go.Figure(go.Bar(
                x=["MAE", "RMSE", "R² Score"],
                y=[m["mae"], m["rmse"], m["r2_score"]],
                marker_color=["#e94560", "#f7971e", "#43cea2"],
            ))
            fig.update_layout(title="Training Metrics", template="plotly_dark",
                              plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.error(res.json().get("detail", "Training failed."))


# ── TAB 3 ─────────────────────────────────────────────────────────────────────
with t3:
    st.subheader("Date-Range Forecast")
    c1, c2 = st.columns(2)
    start = c1.date_input("Start Date")
    end   = c2.date_input("End Date")

    if st.button("📈 Run Forecast", use_container_width=True):
        with st.spinner("Generating forecast..."):
            res = requests.post(f"{API}/forecast/predict", json={
                "start_date": str(start),
                "end_date":   str(end),
                "target":     target,
            })
        if res.status_code == 200:
            d  = res.json()
            df = pd.DataFrame(d["forecast"])
            df["date"] = pd.to_datetime(df["date"])

            st.success(f"Forecast generated for {d['total_days']} days")

            c1, c2, c3 = st.columns(3)
            c1.metric("Total Forecasted",  f"{d['total_forecasted']:,.0f}")
            c2.metric("Daily Average",     f"{df[target].mean():,.0f}")
            c3.metric("Peak Day Value",    f"{df[target].max():,.0f}")

            # Area chart
            fig1 = px.area(df, x="date", y=target,
                           title=f"{target.replace('_',' ').title()} — Daily Forecast",
                           color_discrete_sequence=["#e94560"], template="plotly_dark")
            fig1.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig1, use_container_width=True)

            # Monthly bar chart
            df["month"] = df["date"].dt.to_period("M").astype(str)
            monthly = df.groupby("month")[target].sum().reset_index()
            fig2 = px.bar(monthly, x="month", y=target,
                          title="Monthly Aggregated Forecast",
                          color_discrete_sequence=["#f7971e"], template="plotly_dark")
            fig2.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig2, use_container_width=True)

            st.dataframe(df, use_container_width=True)
            st.download_button("⬇️ Download Forecast CSV",
                               df.to_csv(index=False), "forecast.csv", "text/csv")
        else:
            st.error(res.json().get("detail", "Forecast failed."))


# ── TAB 4 ─────────────────────────────────────────────────────────────────────
with t4:
    st.subheader("Model Performance Metrics")
    if st.button("🔄 Refresh Metrics", use_container_width=True):
        res = requests.get(f"{API}/metrics/summary")
        if res.status_code == 200:
            m = res.json()
            c1, c2, c3 = st.columns(3)
            c1.metric("MAE",      m["mae"])
            c2.metric("RMSE",     m["rmse"])
            c3.metric("R² Score", m["r2_score"])
            st.divider()
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**Model Type:** `{m['model_type']}`")
                st.markdown(f"**Target:** `{m['target']}`")
            with col2:
                st.markdown(f"**Train Samples:** `{m['train_samples']}`")
                st.markdown(f"**Test Samples:** `{m['test_samples']}`")
        else:
            st.warning("No metrics found. Train the model first.")
