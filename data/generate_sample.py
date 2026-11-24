"""
Generate realistic sample fee collection data.
Run: python data/generate_sample.py
"""
from pathlib import Path

import pandas as pd
import numpy as np

OUT_PATH = Path(__file__).resolve().parent / "sample_fee_data.csv"  # works from any directory

np.random.seed(42)
dates = pd.date_range("2022-01-01", "2024-12-31", freq="D")

trend      = np.linspace(10000, 35000, len(dates))
seasonal   = 6000 * np.sin(2 * np.pi * dates.dayofyear / 365)
noise      = np.random.normal(0, 1500, len(dates))
collected  = np.clip(trend + seasonal + noise, 500, None).round(2)
pending    = (collected * np.random.uniform(0.1, 0.35, len(dates))).round(2)

df = pd.DataFrame({
    "date":           dates.strftime("%Y-%m-%d"),
    "fee_collected":  collected,
    "pending_amount": pending,
})
df.to_csv(OUT_PATH, index=False)
print(f"✅ Generated {len(df)} rows → {OUT_PATH}")
print(df.head())
