import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib
import os

# ── Configuration ──────────────────────────────────────
DOMAINS = {
    "Medical":           "data/medical",
    "IT Infrastructure": "data/it_infrastructure",
    "Industrial":        "data/industrial"
}

os.makedirs("models", exist_ok=True)


# ══════════════════════════════════════════════════════
#  BUILD FEATURES FROM COLUMN NAMES + BASIC STATS
# ══════════════════════════════════════════════════════

def extract_features(df):
    """
    Turn a dataframe into a feature vector the classifier can use.
    We use:
      1. Which numeric columns exist (as a fixed vocabulary)
      2. Basic stats of each column (mean, std, min, max)
    """
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != "anomaly"]

    stats = []
    for col in numeric_cols:
        stats.append(df[col].mean())
        stats.append(df[col].std())
        stats.append(df[col].min())
        stats.append(df[col].max())
        stats.append(df[col].skew())

    return stats, numeric_cols


# ══════════════════════════════════════════════════════
#  BUILD COLUMN VOCABULARY
# We need a fixed-size feature vector for the classifier
# so we build a master list of all possible columns
# ══════════════════════════════════════════════════════

print("Building column vocabulary...")
all_columns = set()
domain_dataframes = {}

for domain_name, folder_path in DOMAINS.items():
    csv_files = sorted([f for f in os.listdir(folder_path) if f.endswith(".csv")])
    domain_dataframes[domain_name] = []
    for f in csv_files:
        df = pd.read_csv(os.path.join(folder_path, f))
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        numeric_cols = [c for c in numeric_cols if c != "anomaly"]
        all_columns.update(numeric_cols)
        domain_dataframes[domain_name].append(df)

all_columns = sorted(list(all_columns))
print(f"Total unique columns across all domains: {len(all_columns)}")
print(f"Columns: {all_columns}")


# ══════════════════════════════════════════════════════
#  BUILD FEATURE VECTOR USING FIXED VOCABULARY
# ══════════════════════════════════════════════════════

def build_feature_vector(df, vocabulary):
    """
    Build a fixed-length feature vector.
    For columns that exist in this df: use their stats.
    For columns that don't exist: fill with 0.
    """
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != "anomaly"]

    feature_vector = []
    for col in vocabulary:
        if col in numeric_cols:
            feature_vector.append(df[col].mean())
            feature_vector.append(df[col].std())
            feature_vector.append(df[col].min())
            feature_vector.append(df[col].max())
            feature_vector.append(df[col].skew())
        else:
            # Column doesn't exist in this domain → fill with 0
            feature_vector.extend([0, 0, 0, 0, 0])

    return feature_vector


# ══════════════════════════════════════════════════════
#  BUILD TRAINING DATA
# ══════════════════════════════════════════════════════

print("\nBuilding training data...")
X = []   # feature vectors
y = []   # domain labels

label_map = {
    "Medical":           0,
    "IT Infrastructure": 1,
    "Industrial":        2
}

label_names = {v: k for k, v in label_map.items()}

for domain_name, dataframes in domain_dataframes.items():
    for df in dataframes:
        feature_vector = build_feature_vector(df, all_columns)
        X.append(feature_vector)
        y.append(label_map[domain_name])

X = np.array(X)
y = np.array(y)

print(f"Training samples : {len(X)}")
print(f"Feature size     : {X.shape[1]}")
print(f"Labels           : {[label_names[i] for i in y]}")


# ══════════════════════════════════════════════════════
#  TRAIN THE CLASSIFIER
# ══════════════════════════════════════════════════════

print("\nTraining domain classifier...")

classifier = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)
classifier.fit(X, y)

# Evaluate on training data
y_pred = classifier.predict(X)
print("\n📊 Classifier Performance:")
print(classification_report(y, y_pred, target_names=list(label_map.keys())))


# ══════════════════════════════════════════════════════
#  SAVE EVERYTHING
# ══════════════════════════════════════════════════════

joblib.dump(classifier,  "models/domain_classifier.pkl")
joblib.dump(all_columns, "models/column_vocabulary.pkl")
joblib.dump(label_map,   "models/label_map.pkl")

print("✅ Classifier     saved → models/domain_classifier.pkl")
print("✅ Vocabulary     saved → models/column_vocabulary.pkl")
print("✅ Label map      saved → models/label_map.pkl")

print(f"\n{'='*50}")
print("  DOMAIN CLASSIFIER READY!")
print(f"{'='*50}\n")