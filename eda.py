import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ── Where your data lives ──────────────────────────────
DOMAINS = {
    "Medical":        "data/medical",
    "IT Infrastructure": "data/it_infrastructure",
    "Industrial":     "data/industrial"
}

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Loop through every domain and every CSV ────────────
for domain_name, folder_path in DOMAINS.items():

    print(f"\n{'='*50}")
    print(f"  DOMAIN: {domain_name}")
    print(f"{'='*50}")

    csv_files = sorted([
        f for f in os.listdir(folder_path) if f.endswith(".csv")
    ])

    all_data = []

    for filename in csv_files:
        filepath = os.path.join(folder_path, filename)
        df = pd.read_csv(filepath)
        all_data.append(df)

        print(f"\n--- {filename} ---")
        print(f"Rows: {len(df)}  |  Columns: {len(df.columns)}")
        print(f"Anomalies: {df['anomaly'].sum()}  |  Normal: {(df['anomaly']==0).sum()}")
        print("\nBasic Stats:")
        print(df.describe().round(2))

    # ── Combine all 3 datasets for this domain ─────────
    combined = pd.concat(all_data, ignore_index=True)
    numeric_cols = combined.select_dtypes(include="number").columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != "anomaly"]

    # ── Plot 1: Histograms for each numeric column ──────
    fig, axes = plt.subplots(
        nrows=2, ncols=len(numeric_cols)//2 + len(numeric_cols)%2,
        figsize=(18, 8)
    )
    axes = axes.flatten()
    fig.suptitle(f"{domain_name} — Value Distributions", fontsize=14)

    for i, col in enumerate(numeric_cols):
        axes[i].hist(
            combined[combined["anomaly"]==0][col], bins=30,
            color="steelblue", alpha=0.7, label="Normal"
        )
        axes[i].hist(
            combined[combined["anomaly"]==1][col], bins=30,
            color="red", alpha=0.6, label="Anomaly"
        )
        axes[i].set_title(col)
        axes[i].legend(fontsize=7)

    # hide unused subplots
    for j in range(i+1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    chart_path = os.path.join(OUTPUT_DIR, f"{domain_name.replace(' ','_')}_distributions.png")
    plt.savefig(chart_path)
    plt.close()
    print(f"\n✅ Chart saved → {chart_path}")

    # ── Plot 2: Anomaly count per dataset ───────────────
    labels = [f"Dataset {i+1}" for i in range(len(all_data))]
    anomaly_counts = [df["anomaly"].sum() for df in all_data]
    normal_counts  = [(df["anomaly"]==0).sum() for df in all_data]

    x = range(len(labels))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x, normal_counts,  label="Normal",  color="steelblue")
    ax.bar(x, anomaly_counts, bottom=normal_counts, label="Anomaly", color="red")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_title(f"{domain_name} — Anomaly vs Normal Count")
    ax.legend()
    plt.tight_layout()
    count_path = os.path.join(OUTPUT_DIR, f"{domain_name.replace(' ','_')}_anomaly_counts.png")
    plt.savefig(count_path)
    plt.close()
    print(f"✅ Chart saved → {count_path}")

print(f"\n{'='*50}")
print("  EDA COMPLETE — check your outputs/ folder!")
print(f"{'='*50}\n")