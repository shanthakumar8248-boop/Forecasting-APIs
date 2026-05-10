"""
Generate realistic sample fee collection data.
Run: python data/generate_sample.py
"""
import pandas as pd
import numpy as np

np.random.seed(42)
dates = pd.date_range("2022-01-01", "2024-12-31", freq="D")

trend      = np.linspace(10000, 35000, len(dates))
seasonal   = 6000 * np.sin(2 * np.pi * dates.dayofyear / 365)
noise      = np.random.normal(0, 1500, len(dates))
collected  = (trend + seasonal + noise).clip(min=500).round(2)
pending    = (collected * np.random.uniform(0.1, 0.35, len(dates))).round(2)

df = pd.DataFrame({
    "date":           dates.strftime("%Y-%m-%d"),
    "fee_collected":  collected,
    "pending_amount": pending,
})
df.to_csv("data/sample_fee_data.csv", index=False)
print(f"✅ Generated {len(df)} rows → data/sample_fee_data.csv")
print(df.head())
