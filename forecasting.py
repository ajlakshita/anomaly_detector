import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import joblib
import os

# ── Configuration ──────────────────────────────────────
DOMAINS = {
    "Medical":           "data/medical",
    "IT Infrastructure": "data/it_infrastructure",
    "Industrial":        "data/industrial"
}

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

FORECAST_STEPS = 10        # how many future steps to predict
LOOKBACK       = 20        # how many past rows to learn from


# ══════════════════════════════════════════════════════
#  FORECAST ONE COLUMN
# ══════════════════════════════════════════════════════

def forecast_column(series, steps=FORECAST_STEPS, lookback=LOOKBACK):
    """
    Given a series of values, predict the next N steps.
    Uses the last `lookback` rows to fit a Linear Regression.
    
    Think of it like drawing a trend line through the last
    20 points and extending it forward 10 steps.
    """
    series = series.dropna().values

    if len(series) < lookback:
        lookback = len(series)

    # Use last `lookback` points
    recent = series[-lookback:]

    # X = position (0,1,2,...), y = actual values
    X = np.arange(len(recent)).reshape(-1, 1)
    y = recent

    # Fit trend line
    model = LinearRegression()
    model.fit(X, y)

    # Predict next N steps
    future_X = np.arange(len(recent), len(recent) + steps).reshape(-1, 1)
    forecast  = model.predict(future_X)

    return forecast, recent


# ══════════════════════════════════════════════════════
#  CHECK IF FORECAST IS AN ANOMALY
# Uses the same Z-score logic as statistical_tests.py
# but applied to the forecasted values
# ══════════════════════════════════════════════════════

def is_forecast_anomalous(series, forecast, threshold=2.5):
    """
    Compare forecasted values against the historical mean/std.
    If any forecasted value is more than `threshold` std devs
    away from the historical mean → flag it.
    """
    mean = series.mean()
    std  = series.std()
    if std == 0:
        return np.zeros(len(forecast), dtype=bool)
    z_scores = np.abs((forecast - mean) / std)
    return z_scores > threshold


# ══════════════════════════════════════════════════════
#  FORECAST ALL COLUMNS IN ONE DATAFRAME
# ══════════════════════════════════════════════════════

def forecast_dataset(df, dataset_name, domain_name):
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != "anomaly"]

    print(f"\n{'─'*50}")
    print(f"  {dataset_name}")
    print(f"{'─'*50}")

    forecasts     = {}
    anomaly_flags = {}

    for col in numeric_cols:
        forecast, recent = forecast_column(df[col])
        flagged = is_forecast_anomalous(df[col], forecast)
        forecasts[col]     = forecast
        anomaly_flags[col] = flagged

        status = "🚨 ANOMALY PREDICTED" if flagged.any() else "✅ Normal trend"
        print(f"  {col:30s} → {status}")
        if flagged.any():
            steps = np.where(flagged)[0] + 1
            print(f"     ⚠️  Anomaly at future step(s): {steps.tolist()}")

    # ── Plot forecasts ─────────────────────────────────
    fig, axes = plt.subplots(
        len(numeric_cols), 1,
        figsize=(14, 3 * len(numeric_cols))
    )
    if len(numeric_cols) == 1:
        axes = [axes]

    fig.suptitle(
        f"{domain_name} — {dataset_name} — Forecast (next {FORECAST_STEPS} steps)",
        fontsize=13
    )

    for ax, col in zip(axes, numeric_cols):
        # Plot historical data (last 50 rows for clarity)
        hist = df[col].values[-50:]
        hist_x = np.arange(len(hist))
        ax.plot(hist_x, hist, color="steelblue", linewidth=1, label="Historical")

        # Plot forecast
        fore_x = np.arange(len(hist), len(hist) + FORECAST_STEPS)
        ax.plot(fore_x, forecasts[col], color="green",
                linewidth=1.5, linestyle="--", label="Forecast")

        # Highlight anomalous forecast points
        if anomaly_flags[col].any():
            flag_x = fore_x[anomaly_flags[col]]
            flag_y = forecasts[col][anomaly_flags[col]]
            ax.scatter(flag_x, flag_y, color="red", s=60,
                      zorder=5, label="Predicted Anomaly")

        # Draw a vertical line separating history from forecast
        ax.axvline(x=len(hist)-1, color="gray", linestyle=":", linewidth=1)
        ax.set_ylabel(col, fontsize=8)
        ax.legend(fontsize=7, loc="upper left")

    plt.tight_layout()
    safe_name  = dataset_name.replace(" ", "_").replace("/", "_")
    chart_path = os.path.join(
        OUTPUT_DIR,
        f"{domain_name.replace(' ','_')}_{safe_name}_forecast.png"
    )
    plt.savefig(chart_path)
    plt.close()
    print(f"\n  ✅ Forecast chart saved → {chart_path}")

    return forecasts, anomaly_flags


# ══════════════════════════════════════════════════════
#  MAIN — Run forecasting on all domains
# ══════════════════════════════════════════════════════

for domain_name, folder_path in DOMAINS.items():
    print(f"\n{'='*50}")
    print(f"  DOMAIN: {domain_name}")
    print(f"{'='*50}")

    csv_files = sorted([
        f for f in os.listdir(folder_path) if f.endswith(".csv")
    ])

    for filename in csv_files:
        filepath     = os.path.join(folder_path, filename)
        df           = pd.read_csv(filepath)
        dataset_name = filename.replace(".csv", "")
        forecast_dataset(df, dataset_name, domain_name)

print(f"\n{'='*50}")
print("  ALL FORECASTS COMPLETE — check outputs/ folder!")
print(f"{'='*50}\n")