import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import os

# ── Configuration ──────────────────────────────────────
DOMAINS = {
    "Medical":           "data/medical",
    "IT Infrastructure": "data/it_infrastructure",
    "Industrial":        "data/industrial"
}

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Thresholds ─────────────────────────────────────────
ZSCORE_THRESHOLD    = 3.0   # flag if Z-score above this
IQR_MULTIPLIER      = 1.5   # standard IQR fence
ROLLING_WINDOW      = 10    # rows to look back for variance
VARIANCE_MULTIPLIER = 2.0   # flag if variance spikes this much


# ══════════════════════════════════════════════════════
#  THE 3 TESTS
# ══════════════════════════════════════════════════════

def zscore_test(df, numeric_cols):
    """Flag rows where any column has Z-score above threshold."""
    flagged = pd.Series(False, index=df.index)
    details = {}
    for col in numeric_cols:
        z = np.abs(stats.zscore(df[col].dropna()))
        z = pd.Series(z, index=df[col].dropna().index)
        col_flagged = z > ZSCORE_THRESHOLD
        details[col] = z
        flagged = flagged | col_flagged.reindex(df.index, fill_value=False)
    return flagged, details


def iqr_test(df, numeric_cols):
    """Flag rows that fall outside the IQR fence."""
    flagged = pd.Series(False, index=df.index)
    details = {}
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - IQR_MULTIPLIER * IQR
        upper = Q3 + IQR_MULTIPLIER * IQR
        col_flagged = (df[col] < lower) | (df[col] > upper)
        details[col] = {"lower": round(lower, 2), "upper": round(upper, 2), "flagged": col_flagged}
        flagged = flagged | col_flagged
    return flagged, details


def rolling_variance_test(df, numeric_cols):
    """Flag rows where rolling variance suddenly spikes."""
    flagged = pd.Series(False, index=df.index)
    details = {}
    for col in numeric_cols:
        rolling_std = df[col].rolling(window=ROLLING_WINDOW).std()
        mean_std    = rolling_std.mean()
        col_flagged = rolling_std > (VARIANCE_MULTIPLIER * mean_std)
        details[col] = rolling_std
        flagged = flagged | col_flagged.fillna(False)
    return flagged, details


# ══════════════════════════════════════════════════════
#  RUN ALL TESTS ON A SINGLE DATAFRAME
# ══════════════════════════════════════════════════════

def run_all_tests(df, dataset_name, domain_name):
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != "anomaly"]

    print(f"\n{'─'*50}")
    print(f"  {dataset_name}")
    print(f"{'─'*50}")

    # Run the 3 tests
    z_flagged,   z_details   = zscore_test(df, numeric_cols)
    iqr_flagged, iqr_details = iqr_test(df, numeric_cols)
    var_flagged, var_details = rolling_variance_test(df, numeric_cols)

    # Combined flag — anomaly if ANY test flags it
    combined = z_flagged | iqr_flagged | var_flagged

    # ── Print Results ──────────────────────────────────
    print(f"\n  Z-Score Test    → {z_flagged.sum():3d} rows flagged")
    print(f"  IQR Test        → {iqr_flagged.sum():3d} rows flagged")
    print(f"  Variance Test   → {var_flagged.sum():3d} rows flagged")
    print(f"  Combined Total  → {combined.sum():3d} rows flagged")
    print(f"  Actual Anomalies (ground truth) → {df['anomaly'].sum()}")

    # ── Show exactly which rows are flagged ───────────
    flagged_rows = df[combined].copy()
    flagged_rows["z_flagged"]   = z_flagged[combined].values
    flagged_rows["iqr_flagged"] = iqr_flagged[combined].values
    flagged_rows["var_flagged"] = var_flagged[combined].values

    if len(flagged_rows) > 0:
        print(f"\n  Sample of flagged rows (first 5):")
        print(flagged_rows[["z_flagged","iqr_flagged","var_flagged","anomaly"]].head())

    # ── Save Chart ─────────────────────────────────────
    fig, axes = plt.subplots(len(numeric_cols), 1, figsize=(14, 3 * len(numeric_cols)))
    if len(numeric_cols) == 1:
        axes = [axes]

    fig.suptitle(f"{domain_name} — {dataset_name} — Anomaly Locations", fontsize=13)

    for ax, col in zip(axes, numeric_cols):
        ax.plot(df.index, df[col], color="steelblue", linewidth=0.8, label=col)

        # Mark flagged points
        ax.scatter(df.index[z_flagged],   df[col][z_flagged],
                   color="orange", s=40, label="Z-Score", zorder=5)
        ax.scatter(df.index[iqr_flagged], df[col][iqr_flagged],
                   color="red",    s=40, label="IQR",     zorder=5, marker="x")
        ax.scatter(df.index[var_flagged], df[col][var_flagged],
                   color="purple", s=40, label="Variance", zorder=5, marker="^")

        ax.set_ylabel(col, fontsize=8)
        ax.legend(fontsize=7, loc="upper right")

    plt.tight_layout()
    safe_name = dataset_name.replace(" ", "_").replace("/", "_")
    chart_path = os.path.join(OUTPUT_DIR, f"{domain_name.replace(' ','_')}_{safe_name}_tests.png")
    plt.savefig(chart_path)
    plt.close()
    print(f"\n  ✅ Chart saved → {chart_path}")

    return combined


# ══════════════════════════════════════════════════════
#  MAIN — Loop through all domains and datasets
# ══════════════════════════════════════════════════════

for domain_name, folder_path in DOMAINS.items():
    print(f"\n{'='*50}")
    print(f"  DOMAIN: {domain_name}")
    print(f"{'='*50}")

    csv_files = sorted([f for f in os.listdir(folder_path) if f.endswith(".csv")])

    for filename in csv_files:
        filepath = os.path.join(folder_path, filename)
        df = pd.read_csv(filepath)
        dataset_name = filename.replace(".csv", "")
        run_all_tests(df, dataset_name, domain_name)

print(f"\n{'='*50}")
print("  ALL TESTS COMPLETE — check outputs/ folder!")
print(f"{'='*50}\n")