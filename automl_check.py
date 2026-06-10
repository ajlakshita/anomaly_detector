import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, precision_score, recall_score
import warnings
warnings.filterwarnings("ignore")

# PyOD models
from pyod.models.iforest import IForest        # Isolation Forest
from pyod.models.lof import LOF                # Local Outlier Factor
from pyod.models.ocsvm import OCSVM            # One Class SVM
from pyod.models.knn import KNN                # K Nearest Neighbors
from pyod.models.hbos import HBOS              # Histogram Based
from pyod.models.pca import PCA as PyOD_PCA   # PCA Based
from pyod.models.mcd import MCD                # Minimum Covariance

# ── Configuration ──────────────────────────────────────
DOMAINS = {
    "Medical":           "data/medical",
    "IT Infrastructure": "data/it_infrastructure",
    "Industrial":        "data/industrial"
}

# ── All models to compare ──────────────────────────────
def get_models(contamination):
    return {
        "Isolation Forest": IForest(contamination=contamination, random_state=42),
        "Local Outlier Factor": LOF(contamination=contamination),
        "One Class SVM": OCSVM(contamination=contamination),
        "KNN": KNN(contamination=contamination),
        "HBOS": HBOS(contamination=contamination),
        "PCA Based": PyOD_PCA(contamination=contamination),
        "Min Cov Det": MCD(contamination=contamination, random_state=42),
    }


# ── Load all CSVs for a domain ─────────────────────────
def load_domain(folder):
    import os
    files = sorted([f for f in os.listdir(folder) if f.endswith(".csv")])
    dfs   = [pd.read_csv(f"{folder}/{f}") for f in files]
    return pd.concat(dfs, ignore_index=True)


# ══════════════════════════════════════════════════════
#  MAIN — Compare all models on all domains
# ══════════════════════════════════════════════════════

overall_scores = {}

for domain, folder in DOMAINS.items():
    print(f"\n{'='*60}")
    print(f"  DOMAIN: {domain}")
    print(f"{'='*60}")

    df           = load_domain(folder)
    numeric_cols = [c for c in df.select_dtypes(include="number").columns
                    if c != "anomaly"]
    X            = df[numeric_cols].values
    y_true       = df["anomaly"].values

    # Scale the data
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    contamination = round(y_true.mean(), 3)
    print(f"  Rows: {len(df)} | Anomaly rate: {contamination:.1%}")
    print(f"\n  {'Model':<25} {'F1':>8} {'Precision':>10} {'Recall':>8} {'Verdict'}")
    print(f"  {'-'*65}")

    models       = get_models(contamination)
    domain_scores = {}

    for name, model in models.items():
        try:
            model.fit(X_scaled)
            y_pred = model.labels_   # 0 = normal, 1 = anomaly

            f1   = f1_score(y_true, y_pred, zero_division=0)
            prec = precision_score(y_true, y_pred, zero_division=0)
            rec  = recall_score(y_true, y_pred, zero_division=0)

            domain_scores[name] = f1
            verdict = "BEST" if f1 == max(f1, 0) else ""
            print(f"  {name:<25} {f1:>8.3f} {prec:>10.3f} {rec:>8.3f}")

        except Exception as e:
            print(f"  {name:<25} {'ERROR':>8} — {str(e)[:40]}")
            domain_scores[name] = 0

    # ── Winner for this domain ─────────────────────────
    best_model = max(domain_scores, key=domain_scores.get)
    best_score = domain_scores[best_model]
    print(f"\n  WINNER: {best_model} (F1 = {best_score:.3f})")

    # Track overall
    for name, score in domain_scores.items():
        overall_scores[name] = overall_scores.get(name, 0) + score


# ══════════════════════════════════════════════════════
#  OVERALL RECOMMENDATION
# ══════════════════════════════════════════════════════

print(f"\n{'='*60}")
print("  OVERALL MODEL RANKING (sum of F1 across all domains)")
print(f"{'='*60}")

ranked = sorted(overall_scores.items(), key=lambda x: x[1], reverse=True)

for i, (name, score) in enumerate(ranked):
    medal  = ["1st", "2nd", "3rd"][i] if i < 3 else f"{i+1}th"
    marker = " <-- RECOMMENDED" if i == 0 else ""
    print(f"  {medal}  {name:<25} total F1 = {score:.3f}{marker}")

print(f"\n  CONCLUSION:")
winner = ranked[0][0]
if winner == "Isolation Forest":
    print(f"  Isolation Forest is confirmed as the best model.")
    print(f"  Your original decision was correct.")
else:
    print(f"  {winner} outperformed Isolation Forest.")
    print(f"  Consider switching to {winner} for better accuracy.")

print(f"\n{'='*60}\n")