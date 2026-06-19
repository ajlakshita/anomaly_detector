import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go
import joblib
import os
import re
import warnings
from groq import Groq
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Anomaly Detector",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;700&family=Inter:wght@300;400;500;600&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [data-testid="stAppViewContainer"] {
    background-color: #0a0a08;
    color: #d4cfc6;
    font-family: 'JetBrains Mono', monospace;
}

[data-testid="stAppViewContainer"] { background-color: #0a0a08; }
[data-testid="stHeader"] { background: transparent; display: none; }
.main .block-container { padding: 0; max-width: 100%; }
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stDecoration"] { display: none; }

/* ── Page wrapper ── */
.page-wrap {
    max-width: 1280px;
    margin: 0 auto;
    padding: 0 3rem;
}

/* ── Nav bar ── */
.nav-bar {
    border-bottom: 1px solid #1c1c18;
    padding: 1.2rem 3rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    position: sticky;
    top: 0;
    background: #0a0a08;
    z-index: 100;
}

.nav-logo {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.8rem;
    font-weight: 700;
    color: #f5a623;
    letter-spacing: 0.15em;
    text-transform: uppercase;
}

.nav-tag {
    font-size: 0.65rem;
    color: #3a3a34;
    letter-spacing: 0.12em;
    text-transform: uppercase;
}

/* ── Hero section ── */
.hero {
    padding: 5rem 3rem 4rem 3rem;
    max-width: 1280px;
    margin: 0 auto;
    border-bottom: 1px solid #1c1c18;
}

.hero-eyebrow {
    font-size: 0.68rem;
    color: #f5a623;
    letter-spacing: 0.25em;
    text-transform: uppercase;
    margin-bottom: 1.5rem;
    display: flex;
    align-items: center;
    gap: 0.6rem;
}

.hero-eyebrow::before {
    content: '';
    display: inline-block;
    width: 20px;
    height: 1px;
    background: #f5a623;
}

.hero-title {
    font-family: 'Inter', sans-serif;
    font-size: clamp(2.4rem, 5vw, 4.2rem);
    font-weight: 600;
    color: #f0ebe0;
    letter-spacing: -0.03em;
    line-height: 1.1;
    margin-bottom: 1.2rem;
}

.hero-title span {
    color: #f5a623;
}

.hero-sub {
    font-size: 0.78rem;
    color: #4a4a42;
    letter-spacing: 0.05em;
    line-height: 1.8;
    max-width: 520px;
}

/* ── Upload zone ── */
.upload-section {
    padding: 3rem 3rem;
    max-width: 1280px;
    margin: 0 auto;
    border-bottom: 1px solid #1c1c18;
}

.upload-label {
    font-size: 0.65rem;
    color: #3a3a34;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    margin-bottom: 0.8rem;
}

[data-testid="stFileUploader"] {
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
}

[data-testid="stFileUploadDropzone"] {
    background: #0e0e0b !important;
    border: 1px solid #2a2a22 !important;
    border-radius: 4px !important;
    padding: 2.5rem !important;
    transition: border-color 0.2s !important;
}

[data-testid="stFileUploadDropzone"]:hover {
    border-color: #f5a623 !important;
}

[data-testid="stFileUploadDropzone"] p {
    color: #3a3a34 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.75rem !important;
}

/* ── Domain pill ── */
.domain-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    background: #141410;
    border: 1px solid #f5a623;
    color: #f5a623;
    font-size: 0.65rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    padding: 0.35rem 0.9rem;
    border-radius: 2px;
    margin-bottom: 2rem;
}

.domain-pill::before {
    content: '';
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: #f5a623;
    animation: pulse 2s infinite;
}

@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.3; }
}

/* ── Stat grid ── */
.kpi-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    border: 1px solid #1c1c18;
    margin: 2rem 0;
}

.kpi-card {
    padding: 1.8rem 2rem;
    border-right: 1px solid #1c1c18;
}

.kpi-card:last-child { border-right: none; }

.kpi-label {
    font-size: 0.6rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #3a3a34;
    margin-bottom: 0.6rem;
}

.kpi-value {
    font-family: 'Inter', sans-serif;
    font-size: 2.4rem;
    font-weight: 600;
    color: #f0ebe0;
    line-height: 1;
    letter-spacing: -0.03em;
}

.kpi-value.warn { color: #f5a623; }
.kpi-value.danger { color: #e05c3a; }

/* ── Section headers ── */
.sec-head {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 2.5rem 0 1rem 0;
    border-bottom: 1px solid #1c1c18;
    margin-bottom: 1.5rem;
}

.sec-title {
    font-family: 'Inter', sans-serif;
    font-size: 0.85rem;
    font-weight: 600;
    color: #f0ebe0;
    letter-spacing: 0.01em;
    text-transform: uppercase;
}

.sec-meta {
    font-size: 0.6rem;
    color: #3a3a34;
    letter-spacing: 0.15em;
    text-transform: uppercase;
}

.sec-desc {
    font-size: 0.7rem;
    color: #3a3a34;
    line-height: 1.8;
    margin-bottom: 1.2rem;
    letter-spacing: 0.02em;
}

/* ── Metrics row ── */
[data-testid="stMetric"] {
    background: #0e0e0b !important;
    border: 1px solid #1c1c18 !important;
    border-radius: 2px !important;
    padding: 1.2rem 1.4rem !important;
}

[data-testid="stMetricLabel"] {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.6rem !important;
    letter-spacing: 0.15em !important;
    text-transform: uppercase !important;
    color: #3a3a34 !important;
}

[data-testid="stMetricValue"] {
    font-family: 'Inter', sans-serif !important;
    font-size: 1.6rem !important;
    font-weight: 600 !important;
    color: #f0ebe0 !important;
    letter-spacing: -0.02em !important;
}

/* ── Dataframe ── */
[data-testid="stDataFrame"] {
    border: 1px solid #1c1c18 !important;
    border-radius: 2px !important;
}

/* ── Selectbox ── */
[data-testid="stSelectbox"] > div > div {
    background: #0e0e0b !important;
    border: 1px solid #2a2a22 !important;
    border-radius: 2px !important;
    color: #d4cfc6 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.75rem !important;
}

/* ── Expander ── */
[data-testid="stExpander"] {
    border: 1px solid #1c1c18 !important;
    border-radius: 2px !important;
    background: #0a0a08 !important;
}

[data-testid="stExpander"] summary {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.7rem !important;
    letter-spacing: 0.08em !important;
    color: #3a3a34 !important;
}

/* ── Status bars ── */
.status-ok {
    background: #0c140a;
    border: 1px solid #2a4020;
    border-left: 3px solid #5a9040;
    color: #7ab860;
    font-size: 0.7rem;
    letter-spacing: 0.06em;
    padding: 0.8rem 1.2rem;
    border-radius: 2px;
    margin-top: 0.8rem;
}

.status-warn {
    background: #140e08;
    border: 1px solid #3a2810;
    border-left: 3px solid #f5a623;
    color: #f5a623;
    font-size: 0.7rem;
    letter-spacing: 0.06em;
    padding: 0.8rem 1.2rem;
    border-radius: 2px;
    margin-top: 0.8rem;
}

.model-tag {
    font-size: 0.62rem;
    color: #3a3a34;
    letter-spacing: 0.1em;
    margin-top: 0.6rem;
    font-family: 'JetBrains Mono', monospace;
}

/* ── Divider ── */
.divider { border: none; border-top: 1px solid #1c1c18; margin: 0; }

/* ── Chat ── */
.chat-wrap {
    padding: 3rem 3rem;
    max-width: 1280px;
    margin: 0 auto;
}

.chat-eyebrow {
    font-size: 0.65rem;
    color: #f5a623;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    margin-bottom: 0.5rem;
}

.chat-title {
    font-family: 'Inter', sans-serif;
    font-size: 1.1rem;
    font-weight: 600;
    color: #f0ebe0;
    margin-bottom: 0.4rem;
    letter-spacing: -0.01em;
}

.chat-hints {
    font-size: 0.65rem;
    color: #2e2e28;
    letter-spacing: 0.06em;
    line-height: 2;
    margin-bottom: 1.5rem;
}

[data-testid="stChatMessage"] {
    background: #0e0e0b !important;
    border: 1px solid #1c1c18 !important;
    border-radius: 2px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.78rem !important;
}

[data-testid="stChatInput"] > div {
    background: #0e0e0b !important;
    border: 1px solid #2a2a22 !important;
    border-radius: 2px !important;
}

[data-testid="stChatInput"] textarea {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.78rem !important;
    color: #d4cfc6 !important;
    background: transparent !important;
}

/* ── Content sections wrapper ── */
.content-wrap {
    max-width: 1280px;
    margin: 0 auto;
    padding: 0 3rem;
}

/* ── Footer ── */
.footer {
    border-top: 1px solid #1c1c18;
    padding: 1.5rem 3rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
    max-width: 1280px;
    margin: 3rem auto 0 auto;
}

.footer-left {
    font-size: 0.62rem;
    color: #2e2e28;
    letter-spacing: 0.1em;
}

.footer-right {
    font-size: 0.62rem;
    color: #2e2e28;
    letter-spacing: 0.1em;
}
</style>
""", unsafe_allow_html=True)

# ── Plotly theme ───────────────────────────────────────
PLOTLY_LAYOUT = dict(
    paper_bgcolor="#0a0a08",
    plot_bgcolor="#0a0a08",
    font=dict(family="JetBrains Mono, monospace", color="#3a3a34", size=10),
    xaxis=dict(gridcolor="#141410", linecolor="#1c1c18",
               tickfont=dict(color="#3a3a34", size=9),
               title_font=dict(color="#3a3a34", size=10)),
    yaxis=dict(gridcolor="#141410", linecolor="#1c1c18",
               tickfont=dict(color="#3a3a34", size=9),
               title_font=dict(color="#3a3a34", size=10)),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#3a3a34", size=9),
                orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    margin=dict(l=50, r=20, t=40, b=50),
    hoverlabel=dict(bgcolor="#141410", bordercolor="#2a2a22",
                    font=dict(family="JetBrains Mono, monospace", color="#d4cfc6", size=11)),
)


# ── Load Models ────────────────────────────────────────
@st.cache_resource
def load_models():
    m  = { "Medical": joblib.load("models/medical_model.pkl"),
           "IT Infrastructure": joblib.load("models/it_model.pkl"),
           "Industrial": joblib.load("models/industrial_model.pkl") }
    sc = { "Medical": joblib.load("models/medical_scaler.pkl"),
           "IT Infrastructure": joblib.load("models/it_scaler.pkl"),
           "Industrial": joblib.load("models/industrial_scaler.pkl") }
    cl = joblib.load("models/domain_classifier.pkl")
    vo = joblib.load("models/column_vocabulary.pkl")
    lm = joblib.load("models/label_map.pkl")
    return m, sc, cl, vo, lm

models, scalers, classifier, vocabulary, label_map = load_models()
label_names = {v: k for k, v in label_map.items()}


# ══════════════════════════════════════════════════════
#  CORE FUNCTIONS
# ══════════════════════════════════════════════════════

def detect_domain(df):
    nc = [c for c in df.select_dtypes(include="number").columns if c != "anomaly"]
    fv = []
    for col in vocabulary:
        if col in nc:
            fv.extend([df[col].mean(), df[col].std(), df[col].min(), df[col].max(), df[col].skew()])
        else:
            fv.extend([0, 0, 0, 0, 0])
    return label_names[classifier.predict([fv])[0]]


def run_tests(df, numeric_cols):
    z_f, iqr_f, zscores = pd.Series(False, index=df.index), pd.Series(False, index=df.index), {}
    for col in numeric_cols:
        z            = np.abs(stats.zscore(df[col].dropna()))
        zs           = pd.Series(z, index=df[col].dropna().index).reindex(df.index, fill_value=0)
        zscores[col] = zs
        z_f          = z_f | (zs > 3.0)
        Q1, Q3       = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr_f        = iqr_f | ((df[col] < Q1 - 2.0*(Q3-Q1)) | (df[col] > Q3 + 2.0*(Q3-Q1)))
    return z_f, iqr_f, zscores


def run_ml(df, numeric_cols, domain):
    Xs = scalers[domain].transform(df[numeric_cols].values)
    preds = models[domain].predict(Xs)
    try:    scores = models[domain].decision_function(Xs)
    except: scores = np.zeros(len(df))
    return preds, scores


def make_hover(df, nc, zscores, z_f, iqr_f, ml_f, ml_scores):
    hover = []
    for i in df.index:
        row   = df.iloc[i]
        flags = [t for t, f in [("Z-Score", z_f.iloc[i]), ("IQR", iqr_f.iloc[i]), ("ML", ml_f.iloc[i])] if f]
        lines = [f"<b>Row {i}</b>", f"<b>Flagged: {', '.join(flags)}</b>" if flags else "Normal"]
        lines.append("─────────────")
        for col in sorted(nc, key=lambda c: zscores[c].iloc[i], reverse=True)[:3]:
            v, z = row[col], zscores[col].iloc[i]
            lines.append(f"{col}: {v:.1f} {'↑' if v > df[col].mean() else '↓'} z={z:.2f}")
        lines.append(f"ML score: {ml_scores[i]:.3f}")
        hover.append("<br>".join(lines))
    return hover


def forecast_col(series, steps=10, lookback=50):
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.stattools import adfuller
    vals     = series.dropna().values
    lookback = min(lookback, len(vals))
    recent   = vals[-lookback:]
    try:    d = 0 if adfuller(recent)[1] < 0.05 else 1
    except: d = 1
    best_model, best_aic = None, np.inf
    for p in [1, 2, 3]:
        for q in [0, 1, 2]:
            try:
                m = ARIMA(recent, order=(p, d, q)).fit()
                if m.aic < best_aic:
                    best_aic, best_model = m.aic, m
            except: continue
    if best_model is None:
        X = np.arange(len(recent)).reshape(-1, 1)
        reg = LinearRegression().fit(X, recent)
        pred = reg.predict(np.arange(len(recent), len(recent)+steps).reshape(-1, 1))
        ci   = np.std(recent) * (1 + 0.1 * np.arange(1, steps+1))
        return pred, pred-ci, pred+ci, "Linear Regression (fallback)"
    fc   = best_model.get_forecast(steps=steps)
    pred = np.array(fc.predicted_mean)
    conf = fc.conf_int(alpha=0.10)
    if hasattr(conf, 'iloc'):
        lo, hi = np.array(conf.iloc[:, 0]), np.array(conf.iloc[:, 1])
    else:
        conf = np.array(conf)
        lo, hi = conf[:, 0], conf[:, 1]
    ar = best_model.model_orders.get('ar', '?')
    ma = best_model.model_orders.get('ma', '?')
    return pred, lo, hi, f"ARIMA({ar},{d},{ma}) · AIC {best_aic:.1f}"


# ── Chatbox helpers ────────────────────────────────────
COLUMN_SYNONYMS = {
    "blood pressure": ["bp"], "systolic": ["systolic"], "diastolic": ["diastolic"],
    "oxygen": ["spo2"], "saturation": ["spo2"], "sugar": ["glucose"],
    "fever": ["temperature"], "temp": ["temperature"], "processor": ["cpu"],
    "ram": ["memory"], "lag": ["latency"], "shake": ["vibration"],
    "shaking": ["vibration"], "traffic": ["requests"], "errors": ["error"], "speed": ["rpm"],
}
UNIT_TOKENS = {"pct","mmhg","bpm","mgdl","c","kw","mms","per","hr","units","ms","mbps","sec"}

def find_column(text, nc):
    text_l, best_col, best_score = text.lower(), None, 0
    for col in nc:
        tokens = col.lower().split("_")
        score  = sum(1 for t in tokens if t not in UNIT_TOKENS and len(t) > 2 and t in text_l)
        for syn, req in COLUMN_SYNONYMS.items():
            if syn in text_l and all(t in tokens for t in req):
                score += 2
        if score > best_score:
            best_score, best_col = score, col
    return best_col if best_score > 0 else None

def get_trend(series, lookback=30):
    vals = series.dropna().values
    recent = vals[-min(lookback, len(vals)):]
    X = np.arange(len(recent)).reshape(-1, 1)
    slope = LinearRegression().fit(X, recent).coef_[0]
    pct = (slope * len(recent)) / (abs(recent.mean()) + 1e-9) * 100
    return slope, pct

def extract_number(text):
    m = re.search(r"(-?\d+\.?\d*)", text)
    return float(m.group(1)) if m else None


# ══════════════════════════════════════════════════════
#  NAV BAR
# ══════════════════════════════════════════════════════

st.markdown("""
<div class="nav-bar">
    <div class="nav-logo">Anomaly Detector</div>
    <div class="nav-tag">ML · Statistical · Predictive</div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════
#  HERO
# ══════════════════════════════════════════════════════

st.markdown("""
<div class="hero">
    <div class="hero-eyebrow">Intelligent anomaly detection</div>
    <div class="hero-title">Detect what doesn't<br><span>belong.</span></div>
    <div class="hero-sub">Upload any structured dataset. Domain is detected automatically. Statistical tests, ML models and time series forecasting run in one pass.</div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════
#  UPLOAD
# ══════════════════════════════════════════════════════

st.markdown('<div class="upload-section"><div class="upload-label">Dataset input</div>', unsafe_allow_html=True)
uploaded_file = st.file_uploader("Upload CSV", type=["csv"], label_visibility="collapsed")
if uploaded_file is None:
    st.markdown("<div style='margin-top:0.6rem;font-size:0.62rem;color:#2e2e28;letter-spacing:0.12em;'>MEDICAL &nbsp;·&nbsp; IT INFRASTRUCTURE &nbsp;·&nbsp; INDUSTRIAL</div>", unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════
#  ANALYSIS
# ══════════════════════════════════════════════════════

if uploaded_file is not None:
    df           = pd.read_csv(uploaded_file)
    numeric_cols = [c for c in df.select_dtypes(include="number").columns if c != "anomaly"]
    domain       = detect_domain(df)
    z_f, iqr_f, zscores = run_tests(df, numeric_cols)
    ml_preds, ml_scores = run_ml(df, numeric_cols, domain)
    ml_f         = pd.Series(ml_preds == 1, index=df.index)
    flag_count   = z_f.astype(int) + iqr_f.astype(int) + ml_f.astype(int)
    any_anom     = flag_count >= 2
    total        = int(any_anom.sum())
    hover        = make_hover(df, numeric_cols, zscores, z_f, iqr_f, ml_f, ml_scores)

    st.markdown('<div class="content-wrap">', unsafe_allow_html=True)

    # ── Domain + KPIs ──────────────────────────────────
    anom_rate = 100*total/len(df)
    danger_class = "danger" if total > 0 else "ok"
    warn_class   = "warn"   if total > 0 else "ok"

    st.markdown(f'<div class="domain-pill">{domain}</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="kpi-grid">
        <div class="kpi-card">
            <div class="kpi-label">Total rows</div>
            <div class="kpi-value">{len(df)}</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Features</div>
            <div class="kpi-value">{len(numeric_cols)}</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Anomalies detected</div>
            <div class="kpi-value {danger_class}">{total}</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Anomaly rate</div>
            <div class="kpi-value {warn_class}">{anom_rate:.1f}%</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Statistical Analysis ───────────────────────────
    st.markdown("""
    <div class="sec-head">
        <div class="sec-title">Statistical Analysis</div>
        <div class="sec-meta">Z-Score · IQR · ML Model</div>
    </div>
    <div class="sec-desc">Z-Score flags values beyond 3σ. IQR flags values outside 2× the interquartile fence. ML model uses a domain-tuned outlier detector. A row is marked anomalous only when 2 or more tests agree — reducing false positives.</div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("Z-Score flags",       int(z_f.sum()))
    with c2: st.metric("IQR flags",           int(iqr_f.sum()))
    with c3: st.metric("ML model flags",      int(ml_f.sum()))
    with c4: st.metric("Consensus anomalies", total)

    # ── Flagged rows ───────────────────────────────────
    st.markdown("""<div class="sec-head" style="margin-top:2rem;">
        <div class="sec-title">Flagged rows</div>
    </div>""", unsafe_allow_html=True)

    results               = df.copy()
    results["z_score"]    = z_f.astype(int)
    results["iqr"]        = iqr_f.astype(int)
    results["ml_model"]   = ml_preds
    results["flagged"]    = any_anom.astype(int)
    results["confidence"] = (flag_count / 3 * 100).round(0).astype(int)

    anom_rows = results[results["flagged"] == 1]
    if len(anom_rows) > 0:
        st.markdown(f"<p style='font-size:0.62rem;color:#3a3a34;letter-spacing:0.12em;margin-bottom:0.6rem;'>{len(anom_rows)} ROWS FLAGGED — 2 OR MORE TESTS IN AGREEMENT</p>", unsafe_allow_html=True)
        st.dataframe(anom_rows, use_container_width=True, height=240)
    else:
        st.markdown("<p style='font-size:0.72rem;color:#3a3a34;'>No anomalies detected.</p>", unsafe_allow_html=True)

    with st.expander("View raw data (first 30 rows)"):
        st.dataframe(df.head(30), use_container_width=True)

    # ── Anomaly chart ──────────────────────────────────
    st.markdown("""<div class="sec-head" style="margin-top:2rem;">
        <div class="sec-title">Anomaly visualization</div>
        <div class="sec-meta">Hover any point for details</div>
    </div>
    <div class="sec-desc">Orange = Z-Score flag · Red = confirmed anomaly (2+ tests) · Amber triangle = ML model. Dashed lines show IQR fences.</div>
    """, unsafe_allow_html=True)

    sel_col   = st.selectbox("Feature", numeric_cols, label_visibility="collapsed", key="vis_col")
    mean_v    = df[sel_col].mean()
    Q1        = df[sel_col].quantile(0.25)
    Q3        = df[sel_col].quantile(0.75)
    IQR_v     = Q3 - Q1
    iqr_upper = Q3 + 2.0 * IQR_v
    iqr_lower = Q1 - 2.0 * IQR_v

    normal_idx = list(df.index[~any_anom])
    anom_idx   = list(df.index[any_anom])
    z_only     = list(df.index[z_f & ~any_anom])
    ml_only    = list(df.index[ml_f & ~any_anom])

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(df.index), y=list(df[sel_col]),
        mode="lines", line=dict(color="#1c1c18", width=1), name="Signal", hoverinfo="skip"))
    fig.add_hline(y=mean_v, line_dash="dot", line_color="#2a2a22",
                  annotation_text=f"mean {mean_v:.1f}", annotation_font_color="#3a3a34", annotation_font_size=9)
    fig.add_hline(y=iqr_upper, line_dash="dash", line_color="#2a2010", line_width=1,
                  annotation_text=f"upper {iqr_upper:.1f}", annotation_font_color="#4a3820", annotation_font_size=9)
    fig.add_hline(y=iqr_lower, line_dash="dash", line_color="#2a2010", line_width=1,
                  annotation_text=f"lower {iqr_lower:.1f}", annotation_font_color="#4a3820", annotation_font_size=9)
    if normal_idx:
        fig.add_trace(go.Scatter(x=normal_idx, y=[df[sel_col].iloc[i] for i in normal_idx],
            mode="markers", marker=dict(color="#1e1e1a", size=4), name="Normal",
            text=[hover[i] for i in normal_idx], hovertemplate="%{text}<extra></extra>"))
    if z_only:
        fig.add_trace(go.Scatter(x=z_only, y=[df[sel_col].iloc[i] for i in z_only],
            mode="markers", marker=dict(color="#f5a623", size=7, opacity=0.7), name="Z-Score flag",
            text=[hover[i] for i in z_only], hovertemplate="%{text}<extra></extra>"))
    if ml_only:
        fig.add_trace(go.Scatter(x=ml_only, y=[df[sel_col].iloc[i] for i in ml_only],
            mode="markers", marker=dict(color="#d4a030", size=7, symbol="triangle-up"), name="ML flag",
            text=[hover[i] for i in ml_only], hovertemplate="%{text}<extra></extra>"))
    if anom_idx:
        fig.add_trace(go.Scatter(x=anom_idx, y=[df[sel_col].iloc[i] for i in anom_idx],
            mode="markers", marker=dict(color="#e05c3a", size=10, line=dict(color="#c03020", width=1)),
            name="Confirmed anomaly",
            text=[hover[i] for i in anom_idx], hovertemplate="%{text}<extra></extra>"))
    layout = dict(**PLOTLY_LAYOUT)
    layout["height"] = 380
    layout["title"]  = dict(text=f"{sel_col}", font=dict(color="#3a3a34", size=11), x=0)
    layout["xaxis"]["title"] = "row index"
    layout["yaxis"]["title"] = sel_col
    fig.update_layout(**layout)
    st.plotly_chart(fig, use_container_width=True)

    # ── Heatmap ────────────────────────────────────────
    st.markdown("""<div class="sec-head" style="margin-top:2rem;">
        <div class="sec-title">Z-Score heatmap</div>
        <div class="sec-meta">All features</div>
    </div>
    <div class="sec-desc">Brighter amber means higher deviation from the column mean. Hover any cell for exact value and z-score.</div>
    """, unsafe_allow_html=True)

    z_matrix   = np.array([[zscores[col].iloc[i] for i in df.index] for col in numeric_cols])
    heat_hover = []
    for col in numeric_cols:
        rh = []
        for i in df.index:
            z, val = zscores[col].iloc[i], df[col].iloc[i]
            rh.append(f"<b>{col}</b><br>Row {i}<br>Value: {val:.2f}<br>Z-score: {z:.2f}<br>{'<b>ANOMALOUS</b>' if z > 3.0 else 'Normal'}")
        heat_hover.append(rh)

    fig_heat = go.Figure(data=go.Heatmap(
        z=z_matrix, x=list(df.index), y=numeric_cols,
        colorscale=[[0,"#0a0a08"],[0.3,"#141008"],[0.6,"#3a2008"],[0.8,"#8a5010"],[1.0,"#f5a623"]],
        zmin=0, zmax=5, text=heat_hover, hovertemplate="%{text}<extra></extra>",
        colorbar=dict(title=dict(text="z-score", font=dict(color="#3a3a34", size=9)),
                      tickfont=dict(color="#3a3a34", size=8), bgcolor="#0a0a08", bordercolor="#1c1c18")
    ))
    heat_layout = dict(**PLOTLY_LAYOUT)
    heat_layout["height"] = 280
    heat_layout["margin"] = dict(l=160, r=60, t=10, b=40)
    heat_layout["xaxis"]["title"] = "row index"
    fig_heat.update_layout(**heat_layout)
    st.plotly_chart(fig_heat, use_container_width=True)

    # ── Forecast ───────────────────────────────────────
    st.markdown("""<div class="sec-head" style="margin-top:2rem;">
        <div class="sec-title">Forecast</div>
        <div class="sec-meta">ARIMA · Next 10 steps</div>
    </div>
    <div class="sec-desc">Auto-selected ARIMA model using AIC scoring. Amber band = 90% confidence interval. Red points = predicted anomalies. Hover for exact values and CI bounds.</div>
    """, unsafe_allow_html=True)

    fore_col_sel            = st.selectbox("Column to forecast", numeric_cols, label_visibility="collapsed", key="fore_col")
    forecast, ci_lower, ci_upper, model_name = forecast_col(df[fore_col_sel])

    mean_v2   = df[fore_col_sel].mean()
    std_v2    = df[fore_col_sel].std()
    fore_z    = np.abs((forecast - mean_v2) / std_v2) if std_v2 > 0 else np.zeros(10)
    Q1_f      = df[fore_col_sel].quantile(0.25)
    Q3_f      = df[fore_col_sel].quantile(0.75)
    iqr_anom  = (forecast < Q1_f - 2.0*(Q3_f-Q1_f)) | (forecast > Q3_f + 2.0*(Q3_f-Q1_f))
    any_fa    = (fore_z > 3.0) | iqr_anom

    hist   = list(df[fore_col_sel].values[-60:])
    hist_x = list(range(len(hist)))
    fore_x = list(range(len(hist), len(hist)+10))

    fig2 = go.Figure()
    hist_hover = [f"<b>Row {len(df)-len(hist)+i}</b><br>Value: {v:.2f}<br>Z-score: {zscores[fore_col_sel].iloc[len(df)-len(hist)+i]:.2f}"
                  for i, v in enumerate(hist)]
    fig2.add_trace(go.Scatter(x=hist_x, y=hist, mode="lines+markers",
        line=dict(color="#1c1c18", width=1), marker=dict(color="#2a2a22", size=3),
        name="Historical", text=hist_hover, hovertemplate="%{text}<extra></extra>"))
    fig2.add_trace(go.Scatter(
        x=fore_x + fore_x[::-1],
        y=list(ci_upper) + list(ci_lower[::-1]),
        fill="toself", fillcolor="rgba(245,166,35,0.06)",
        line=dict(color="rgba(0,0,0,0)"), name="90% CI", hoverinfo="skip"))
    fig2.add_trace(go.Scatter(x=fore_x, y=list(ci_upper), mode="lines",
        line=dict(color="#3a2808", width=1, dash="dot"), name="Upper CI",
        text=[f"Upper CI: {v:.2f}" for v in ci_upper], hovertemplate="%{text}<extra></extra>"))
    fig2.add_trace(go.Scatter(x=fore_x, y=list(ci_lower), mode="lines",
        line=dict(color="#3a2808", width=1, dash="dot"), name="Lower CI",
        text=[f"Lower CI: {v:.2f}" for v in ci_lower], hovertemplate="%{text}<extra></extra>"))
    fore_hover = [f"<b>Step {i+1}</b><br>Predicted: <b>{v:.2f}</b><br>90% CI: {ci_lower[i]:.2f} — {ci_upper[i]:.2f}<br>Z-score: {fore_z[i]:.2f}<br><b>{'ANOMALY PREDICTED' if any_fa[i] else 'Normal'}</b>"
                  for i, v in enumerate(forecast)]
    fig2.add_trace(go.Scatter(x=fore_x, y=list(forecast), mode="lines+markers",
        line=dict(color="#f5a623", width=2, dash="dash"),
        marker=dict(color="#f5a623", size=5),
        name="Forecast", text=fore_hover, hovertemplate="%{text}<extra></extra>"))
    if any_fa.any():
        ax = [fore_x[i] for i in range(10) if any_fa[i]]
        ay = [forecast[i] for i in range(10) if any_fa[i]]
        ah = [fore_hover[i] for i in range(10) if any_fa[i]]
        fig2.add_trace(go.Scatter(x=ax, y=ay, mode="markers",
            marker=dict(color="#e05c3a", size=12, line=dict(color="#c03020", width=1)),
            name="Predicted anomaly", text=ah, hovertemplate="%{text}<extra></extra>"))
    fig2.add_vline(x=len(hist)-1, line_dash="dot", line_color="#1c1c18",
                   annotation_text="forecast →", annotation_font_color="#3a3a34", annotation_font_size=9)
    fore_layout = dict(**PLOTLY_LAYOUT)
    fore_layout["height"] = 380
    fore_layout["title"]  = dict(text=fore_col_sel, font=dict(color="#3a3a34", size=11), x=0)
    fore_layout["xaxis"]["title"] = "row index"
    fore_layout["yaxis"]["title"] = fore_col_sel
    fig2.update_layout(**fore_layout)
    st.plotly_chart(fig2, use_container_width=True)
    st.markdown(f"<div class='model-tag'>model: {model_name}</div>", unsafe_allow_html=True)

    if any_fa.any():
        steps = list(np.where(any_fa)[0] + 1)
        st.markdown(f'<div class="status-warn">Anomaly predicted at step(s) {steps} — values expected outside normal boundaries</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="status-ok">No anomalies predicted in the next 10 steps</div>', unsafe_allow_html=True)

    with st.expander("Forecast details — all 10 steps"):
        fore_table = pd.DataFrame({
            "Step":         list(range(1, 11)),
            "Predicted":    [round(v, 2) for v in forecast],
            "Lower CI":     [round(v, 2) for v in ci_lower],
            "Upper CI":     [round(v, 2) for v in ci_upper],
            "Z-score":      [round(v, 2) for v in fore_z],
            "IQR anomaly":  ["Yes" if v else "No" for v in iqr_anom],
            "Status":       ["ANOMALY" if v else "Normal" for v in any_fa],
        })
        st.dataframe(fore_table, use_container_width=True)

    st.session_state["df"]           = df
    st.session_state["domain"]       = domain
    st.session_state["results"]      = results
    st.session_state["numeric_cols"] = numeric_cols
    st.session_state["zscores"]      = zscores

    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('<hr class="divider">', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════
#  CHATBOX
# ══════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════
#  GENAI CHATBOX — powered by Groq
# ══════════════════════════════════════════════════════

@st.cache_resource
def get_groq_client():
    return Groq(api_key=st.secrets["GROQ_API_KEY"])


def build_data_context(df, results, domain, nc, zscores, model_name=None):
    """Build a compact text summary of the dataset for the AI to reason about."""
    total = int(results["flagged"].sum())

    context = f"DATASET DOMAIN: {domain}\n"
    context += f"TOTAL ROWS: {len(df)}\n"
    context += f"TOTAL ANOMALIES: {total} ({100*total/len(df):.1f}%)\n\n"

    context += "COLUMN STATISTICS:\n"
    for col in nc:
        context += (f"- {col}: mean={df[col].mean():.2f}, std={df[col].std():.2f}, "
                    f"min={df[col].min():.2f}, max={df[col].max():.2f}, "
                    f"max_z_score={zscores[col].max():.2f}\n")

    context += f"\nFLAGGED ANOMALY ROWS (showing up to 30 of {total}):\n"
    anom_rows = results[results["flagged"] == 1].head(30)
    for idx, row in anom_rows.iterrows():
        vals = ", ".join(f"{col}={row[col]:.2f}" for col in nc)
        context += (f"Row {idx}: confidence={int(row['confidence'])}%, "
                    f"z_score_test={'Yes' if row['z_score'] else 'No'}, "
                    f"iqr_test={'Yes' if row['iqr'] else 'No'}, "
                    f"ml_model={'Yes' if row['ml_model'] else 'No'} | {vals}\n")

    context += ("\nDETECTION METHOD: A row is flagged as anomalous only when 2 or more of these "
               "3 tests agree: Z-Score (threshold 3.0), IQR (2x interquartile fence), "
               "and an ML outlier detection model trained specifically for this domain.\n")

    return context


st.markdown("""
<div class="chat-wrap">
    <div class="chat-eyebrow">Data query — AI powered</div>
    <div class="chat-title">Ask anything about your dataset</div>
    <div class="chat-hints">
        Ask in plain language — "is row 45 concerning and why?", "what's driving the anomalies?",<br>
        "summarize the health of this dataset", "what should I investigate first?"
    </div>
</div>
""", unsafe_allow_html=True)

if "df" not in st.session_state:
    st.markdown("<div class='chat-wrap'><p style='font-size:0.7rem;color:#2e2e28;letter-spacing:0.08em;'>Upload a dataset above to enable data queries.</p></div>", unsafe_allow_html=True)
else:
    st.markdown('<div class="chat-wrap" style="padding-top:0;">', unsafe_allow_html=True)

    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ask anything about your data..."):
        st.session_state["messages"].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        df, results, domain = st.session_state["df"], st.session_state["results"], st.session_state["domain"]
        nc, zscores = st.session_state["numeric_cols"], st.session_state["zscores"]

        with st.chat_message("assistant"):
            with st.spinner("Analyzing..."):
                try:
                    client       = get_groq_client()
                    data_context = build_data_context(df, results, domain, nc, zscores)

                    system_prompt = (
                        "You are a data analysis assistant embedded in an anomaly detection tool. "
                        "You answer questions about the uploaded dataset using ONLY the statistics "
                        "and flagged rows provided below. Be precise, cite actual row numbers and "
                        "values when relevant, and keep answers concise and well-formatted with "
                        "markdown tables where useful. If asked about something not in the data, "
                        "say so clearly rather than guessing.\n\n"
                        f"{data_context}"
                    )

                    chat_completion = client.chat.completions.create(
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": prompt}
                        ],
                        model="llama-3.3-70b-versatile",
                        temperature=0.3,
                        max_tokens=800,
                    )

                    response = chat_completion.choices[0].message.content

                except Exception as e:
                    response = f"Error reaching the AI service: {str(e)}"

                st.markdown(response)

        st.session_state["messages"].append({"role": "assistant", "content": response})

    st.markdown('</div>', unsafe_allow_html=True)

# ── Footer ─────────────────────────────────────────────
st.markdown("""
<div class="footer">
    <div class="footer-left">Anomaly Detector · ML + Statistical + Predictive</div>
    <div class="footer-right">LOF · MCD · ARIMA · Z-Score · IQR</div>
</div>
""", unsafe_allow_html=True)