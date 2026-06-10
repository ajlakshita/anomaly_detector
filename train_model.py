import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from pyod.models.mcd import MCD
from pyod.models.lof import LOF
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import os

# ── Configuration ──────────────────────────────────────
DOMAINS = {
    "Medical":           "data/medical",
    "IT Infrastructure": "data/it_infrastructure",
    "Industrial":        "data/industrial"
}

MODEL_NAMES = {
    "Medical":           "models/medical_model.pkl",
    "IT Infrastructure": "models/it_model.pkl",
    "Industrial":        "models/industrial_model.pkl"
}

SCALER_NAMES = {
    "Medical":           "models/medical_scaler.pkl",
    "IT Infrastructure": "models/it_scaler.pkl",
    "Industrial":        "models/industrial_scaler.pkl"
}

os.makedirs("models", exist_ok=True)


# ══════════════════════════════════════════════════════
#  HELPER — Load all CSVs for a domain
# ══════════════════════════════════════════════════════

def load_domain_data(folder_path):
    """Load and combine all CSVs in a domain folder."""
    csv_files = sorted([f for f in os.listdir(folder_path) if f.endswith(".csv")])
    all_dfs = []
    for f in csv_files:
        df = pd.read_csv(os.path.join(folder_path, f))
        all_dfs.append(df)
    combined = pd.concat(all_dfs, ignore_index=True)
    return combined


# ══════════════════════════════════════════════════════
#  HELPER — Get numeric feature columns
# ══════════════════════════════════════════════════════

def get_features(df):
    """Return only numeric columns, excluding anomaly label."""
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    return [c for c in numeric_cols if c != "anomaly"]


# ══════════════════════════════════════════════════════
#  TRAIN ONE MODEL FOR ONE DOMAIN
# ══════════════════════════════════════════════════════

def train_domain_model(domain_name, folder_path, model_path, scaler_path):
    print(f"\n{'='*50}")
    print(f"  Training: {domain_name}")
    print(f"{'='*50}")

    # Load data
    df = load_domain_data(folder_path)
    feature_cols = get_features(df)

    print(f"  Total rows    : {len(df)}")
    print(f"  Features used : {feature_cols}")
    print(f"  Anomalies     : {df['anomaly'].sum()} / {len(df)}")

    # Separate features
    X = df[feature_cols].values
    y = df["anomaly"].values

    # Scale the data — brings all columns to same range
    # so one column doesn't dominate others
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train Isolation Forest
    # contamination = roughly what % of data we expect to be anomalies
    anomaly_ratio = round(df["anomaly"].mean(), 3)
    print(f"  Contamination : {anomaly_ratio}")

    if domain_name == "Medical":
        print(f"  Using: Local Outlier Factor (best for Medical)")
        model = LOF(contamination=anomaly_ratio)
    else:
        print(f"  Using: Minimum Covariance Determinant (best for IT/Industrial)")
        model = MCD(contamination=anomaly_ratio, random_state=42)
    model.fit(X_scaled)

    # ── Evaluate the model ─────────────────────────────
    # Isolation Forest returns -1 for anomaly, 1 for normal
    # We convert to 0/1 to match our labels
    y_pred = model.labels_

    print(f"\n  📊 Model Performance:")
    print(classification_report(y, y_pred, target_names=["Normal", "Anomaly"]))

    cm = confusion_matrix(y, y_pred)
    print(f"  Confusion Matrix:")
    print(f"  True Normal  flagged as Normal  : {cm[0][0]}")
    print(f"  True Normal  flagged as Anomaly : {cm[0][1]}")
    print(f"  True Anomaly flagged as Normal  : {cm[1][0]}")
    print(f"  True Anomaly flagged as Anomaly : {cm[1][1]}")

    # ── Save model and scaler ──────────────────────────
    joblib.dump(model,  model_path)
    joblib.dump(scaler, scaler_path)
    print(f"\n  ✅ Model  saved → {model_path}")
    print(f"  ✅ Scaler saved → {scaler_path}")

    return model, scaler, feature_cols


# ══════════════════════════════════════════════════════
#  MAIN — Train all 3 domain models
# ══════════════════════════════════════════════════════

trained_models = {}

for domain_name, folder_path in DOMAINS.items():
    model, scaler, features = train_domain_model(
        domain_name,
        folder_path,
        MODEL_NAMES[domain_name],
        SCALER_NAMES[domain_name]
    )
    trained_models[domain_name] = {
        "model":    model,
        "scaler":   scaler,
        "features": features
    }

print(f"\n{'='*50}")
print("  ALL 3 MODELS TRAINED AND SAVED!")
print(f"{'='*50}")
print("\n  Models saved in models/ folder:")
for domain, paths in MODEL_NAMES.items():
    print(f"  🤖 {domain:20s} → {paths}")
print()