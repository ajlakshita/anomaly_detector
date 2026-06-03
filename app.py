import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go
import joblib
import os
import re

# ── Page Config ────────────────────────────────────────
st.set_page_config(
    page_title="Anomaly Detector",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── Global Styles ──────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@300;400;500&family=Syne:wght@400;600;700;800&display=swap');
*, *::before, *::after { box-sizing: border-box; }
html, body, [data-testid="stAppViewContainer"] { background-color: #0d0d0d; color: #e8e4dc; font-family: 'DM Mono', monospace; }
[data-testid="stAppViewContainer"] { background-color: #0d0d0d; }
[data-testid="stHeader"] { background: transparent; }
.main .block-container { padding: 3rem 3rem 6rem 3rem; max-width: 1400px; }
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stDecoration"] { display: none; }
h1, h2, h3 { font-family: 'Syne', sans-serif; letter-spacing: -0.02em; }
.ad-header { border-bottom: 1px solid #2a2a2a; padding-bottom: 2rem; margin-bottom: 3rem; }
.ad-header-label { font-size: 0.7rem; letter-spacing: 0.2em; text-transform: uppercase; color: #555; margin-bottom: 0.5rem; }
.ad-header-title { font-family: 'Syne', sans-serif; font-size: 2.8rem; font-weight: 800; color: #e8e4dc; letter-spacing: -0.04em; line-height: 1; margin: 0; }
.ad-header-sub { font-size: 0.78rem; color: #555; margin-top: 0.6rem; letter-spacing: 0.05em; }
.domain-badge { display: inline-block; background: #0d0d0d; border: 1px solid #c8f060; color: #c8f060; font-size: 0.7rem; letter-spacing: 0.15em; text-transform: uppercase; padding: 0.3rem 0.8rem; border-radius: 2px; margin-bottom: 1rem; }
.stat-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 1px; background: #1a1a1a; border: 1px solid #1a1a1a; margin: 1.5rem 0; }
.stat-card { background: #0d0d0d; padding: 1.4rem 1.6rem; }
.stat-card-label { font-size: 0.65rem; letter-spacing: 0.15em; text-transform: uppercase; color: #555; margin-bottom: 0.4rem; }
.stat-card-value { font-family: 'Syne', sans-serif; font-size: 2rem; font-weight: 700; color: #e8e4dc; line-height: 1; }
.stat-card-value.alert { color: #ff6b6b; }
.stat-card-value.ok { color: #c8f060; }
.section-header { display: flex; align-items: baseline; gap: 1rem; margin: 2.5rem 0 1rem 0; padding-bottom: 0.75rem; border-bottom: 1px solid #1a1a1a; }
.section-title { font-family: 'Syne', sans-serif; font-size: 1rem; font-weight: 700; color: #e8e4dc; letter-spacing: -0.01em; text-transform: uppercase; margin: 0; }
.section-count { font-size: 0.65rem; color: #555; letter-spacing: 0.1em; }
.section-desc { font-size: 0.68rem; color: #444; margin-bottom: 1rem; letter-spacing: 0.03em; line-height: 1.6; }
hr { border: none; border-top: 1px solid #1a1a1a; margin: 2rem 0; }
[data-testid="stMetric"] { background: #111; border: 1px solid #1a1a1a; padding: 1rem 1.2rem; border-radius: 2px; }
[data-testid="stMetricLabel"] { font-family: 'DM Mono', monospace !important; font-size: 0.65rem !important; letter-spacing: 0.1em !important; text-transform: uppercase !important; color: #555 !important; }
[data-testid="stMetricValue"] { font-family: 'Syne', sans-serif !important; font-size: 1.8rem !important; font-weight: 700 !important; color: #e8e4dc !important; }
.forecast-ok { background: #0f1a05; border: 1px solid #c8f060; color: #c8f060; font-family: 'DM Mono', monospace; font-size: 0.72rem; padding: 0.6rem 1rem; border-radius: 2px; margin-top: 0.5rem; }
.forecast-alert { background: #1a0505; border: 1px solid #ff6b6b; color: #ff6b6b; font-family: 'DM Mono', monospace; font-size: 0.72rem; padding: 0.6rem 1rem; border-radius: 2px; margin-top: 0.5rem; }
.chat-header { margin: 2.5rem 0 1rem 0; padding-bottom: 0.75rem; border-bottom: 1px solid #1a1a1a; }
.chat-hint { font-size: 0.68rem; color: #444; letter-spacing: 0.05em; margin-top: 0.3rem; }
[data-testid="stChatMessage"] { background: #111 !important; border: 1px solid #1a1a1a !important; border-radius: 2px !important; }
[data-testid="stChatInput"] > div { background: #111 !important; border: 1px solid #2a2a2a !important; border-radius: 2px !important; }
[data-testid="stChatInput"] textarea { color: #e8e4dc !important; background: transparent !important; }
[data-testid="stFileUploadDropzone"] { background: transparent !important; border: 1px dashed #2a2a2a !important; border-radius: 2px !important; padding: 2rem !important; }
[data-testid="stExpander"] { border: 1px solid #1a1a1a !important; border-radius: 2px !important; background: #0d0d0d !important; }
</style>
""", unsafe_allow_html=True)

PLOTLY_LAYOUT = dict(
    paper_bgcolor="#0d0d0d",
    plot_bgcolor="#0d0d0d",
    font=dict(family="monospace", color="#888", size=11),
    xaxis=dict(gridcolor="#1a1a1a", linecolor="#2a2a2a", tickfont=dict(color="#555", size=10)),
    yaxis=dict(gridcolor="#1a1a1a", linecolor="#2a2a2a", tickfont=dict(color="#555", size=10)),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#666", size=10),
                orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    margin=dict(l=50, r=20, t=50, b=50),
    hoverlabel=dict(bgcolor="#111", bordercolor="#333",
                    font=dict(family="monospace", color="#e8e4dc", size=12)),
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
    z_f   = pd.Series(False, index=df.index)
    iqr_f = pd.Series(False, index=df.index)
    zscores = {}

    for col in numeric_cols:
        z        = np.abs(stats.zscore(df[col].dropna()))
        zs       = pd.Series(z, index=df[col].dropna().index).reindex(df.index, fill_value=0)
        zscores[col] = zs
        z_f      = z_f | (zs > 3.0)
        Q1, Q3   = df[col].quantile(0.25), df[col].quantile(0.75)
        IQR      = Q3 - Q1
        iqr_f    = iqr_f | ((df[col] < Q1 - 2.0*IQR) | (df[col] > Q3 + 2.0*IQR))

    return z_f, iqr_f, zscores


def run_ml(df, numeric_cols, domain):
    X      = df[numeric_cols].values
    Xs     = scalers[domain].transform(X)
    preds  = models[domain].predict(Xs)
    scores = models[domain].score_samples(Xs)
    return np.where(preds == -1, 1, 0), scores


def make_hover(df, numeric_cols, zscores, z_f, iqr_f, ml_f, ml_scores):
    hover = []
    for i in df.index:
        row   = df.iloc[i]
        flags = []
        if z_f.iloc[i]:   flags.append("Z-Score")
        if iqr_f.iloc[i]: flags.append("IQR")
        if ml_f.iloc[i]:  flags.append("ML Model")
        lines = [f"<b>Row {i}</b>"]
        if flags:
            lines.append(f"<b>Flagged: {', '.join(flags)}</b>")
        else:
            lines.append("Status: Normal")
        lines.append("──────────")
        top = sorted(numeric_cols, key=lambda c: zscores[c].iloc[i], reverse=True)[:3]
        for col in top:
            val  = row[col]
            z    = zscores[col].iloc[i]
            diff = val - df[col].mean()
            arr  = "↑" if diff > 0 else "↓"
            lines.append(f"{col}: {val:.1f} {arr} z={z:.2f}")
        lines.append(f"ML score: {ml_scores[i]:.3f}")
        hover.append("<br>".join(lines))
    return hover


def forecast_col(series, steps=10, lookback=20):
    vals     = series.dropna().values
    lookback = min(lookback, len(vals))
    recent   = vals[-lookback:]
    X        = np.arange(len(recent)).reshape(-1, 1)
    reg      = LinearRegression().fit(X, recent)
    fX       = np.arange(len(recent), len(recent)+steps).reshape(-1, 1)
    pred     = reg.predict(fX)
    resid    = recent - reg.predict(X).flatten()
    ci       = np.std(resid) * (1 + 0.1 * np.arange(1, steps+1))
    return pred, ci


# ══════════════════════════════════════════════════════
#  HEADER
# ══════════════════════════════════════════════════════

st.markdown("""
<div class="ad-header">
    <div class="ad-header-label">Machine Learning System</div>
    <div class="ad-header-title">Anomaly Detector</div>
    <div class="ad-header-sub">Upload a structured dataset — domain is detected automatically</div>
</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload CSV", type=["csv"], label_visibility="collapsed")

if uploaded_file is None:
    st.markdown("<div style='margin-top:1rem;font-size:0.7rem;color:#333;letter-spacing:0.1em;'>SUPPORTED DOMAINS / MEDICAL &middot; IT INFRASTRUCTURE &middot; INDUSTRIAL</div>", unsafe_allow_html=True)


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

    # ── Overview ───────────────────────────────────────
    st.markdown(f'<div class="domain-badge">{domain}</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="stat-grid">
        <div class="stat-card"><div class="stat-card-label">Total Rows</div><div class="stat-card-value">{len(df)}</div></div>
        <div class="stat-card"><div class="stat-card-label">Features</div><div class="stat-card-value">{len(numeric_cols)}</div></div>
        <div class="stat-card"><div class="stat-card-label">Anomalies Detected</div><div class="stat-card-value {'alert' if total>0 else 'ok'}">{total}</div></div>
        <div class="stat-card"><div class="stat-card-label">Anomaly Rate</div><div class="stat-card-value {'alert' if total>0 else 'ok'}">{100*total/len(df):.1f}%</div></div>
    </div>""", unsafe_allow_html=True)

    # ── Statistical metrics ────────────────────────────
    st.markdown("""<div class="section-header"><span class="section-title">Statistical Analysis</span><span class="section-count">Z-SCORE / IQR / ML MODEL</span></div>
    <div class="section-desc">Z-Score flags values beyond 3 standard deviations. IQR flags values outside 2x the interquartile fence. ML Model uses Isolation Forest. A row is anomalous only when 2 or more tests agree.</div>""", unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("Z-Score Flags",      int(z_f.sum()))
    with c2: st.metric("IQR Flags",          int(iqr_f.sum()))
    with c3: st.metric("ML Model Flags",     int(ml_f.sum()))
    with c4: st.metric("Consensus Anomalies", total)

    # ── Flagged rows table ─────────────────────────────
    st.markdown("""<div class="section-header"><span class="section-title">Flagged Rows</span></div>""", unsafe_allow_html=True)

    results              = df.copy()
    results["z_score"]   = z_f.astype(int)
    results["iqr"]       = iqr_f.astype(int)
    results["ml_model"]  = ml_preds
    results["flagged"]   = any_anom.astype(int)
    results["confidence"]= (flag_count / 3 * 100).round(0).astype(int)

    anom_rows = results[results["flagged"] == 1]
    if len(anom_rows) > 0:
        st.markdown(f"<p style='font-size:0.7rem;color:#555;'>{len(anom_rows)} ROWS FLAGGED</p>", unsafe_allow_html=True)
        st.dataframe(anom_rows, use_container_width=True, height=260)
    else:
        st.markdown("<p style='font-size:0.75rem;color:#555;'>No anomalies detected.</p>", unsafe_allow_html=True)

    with st.expander("View raw data"):
        st.dataframe(df.head(30), use_container_width=True)


    # ══════════════════════════════════════════════════
    #  ANOMALY CHART
    # ══════════════════════════════════════════════════

    st.markdown("""<div class="section-header"><span class="section-title">Anomaly Visualization</span><span class="section-count">HOVER FOR DETAILS</span></div>
    <div class="section-desc">Hover over any point to see the row number, exact value, z-score, and reason for flagging. Orange = Z-Score, Red = confirmed anomaly (2+ tests), Green triangle = ML model.</div>""", unsafe_allow_html=True)

    sel_col = st.selectbox("Select feature", numeric_cols, label_visibility="collapsed", key="vis_col")

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

    fig.add_trace(go.Scatter(
        x=list(df.index), y=list(df[sel_col]),
        mode="lines", line=dict(color="#222", width=1),
        name="Signal", hoverinfo="skip"
    ))

    fig.add_hline(y=mean_v, line_dash="dot", line_color="#333",
                  annotation_text=f"mean {mean_v:.1f}",
                  annotation_font_color="#444", annotation_font_size=10)
    fig.add_hline(y=iqr_upper, line_dash="dash", line_color="#1a3300", line_width=1,
                  annotation_text=f"upper fence {iqr_upper:.1f}",
                  annotation_font_color="#2a5200", annotation_font_size=9)
    fig.add_hline(y=iqr_lower, line_dash="dash", line_color="#1a3300", line_width=1,
                  annotation_text=f"lower fence {iqr_lower:.1f}",
                  annotation_font_color="#2a5200", annotation_font_size=9)

    if normal_idx:
        fig.add_trace(go.Scatter(
            x=normal_idx, y=[df[sel_col].iloc[i] for i in normal_idx],
            mode="markers", marker=dict(color="#2a2a2a", size=5),
            name="Normal",
            text=[hover[i] for i in normal_idx],
            hovertemplate="%{text}<extra></extra>"
        ))

    if z_only:
        fig.add_trace(go.Scatter(
            x=z_only, y=[df[sel_col].iloc[i] for i in z_only],
            mode="markers", marker=dict(color="#ff9f43", size=8),
            name="Z-Score flag",
            text=[hover[i] for i in z_only],
            hovertemplate="%{text}<extra></extra>"
        ))

    if ml_only:
        fig.add_trace(go.Scatter(
            x=ml_only, y=[df[sel_col].iloc[i] for i in ml_only],
            mode="markers", marker=dict(color="#c8f060", size=8, symbol="triangle-up"),
            name="ML flag",
            text=[hover[i] for i in ml_only],
            hovertemplate="%{text}<extra></extra>"
        ))

    if anom_idx:
        fig.add_trace(go.Scatter(
            x=anom_idx, y=[df[sel_col].iloc[i] for i in anom_idx],
            mode="markers", marker=dict(color="#ff6b6b", size=12,
                            line=dict(color="#ff0000", width=1)),
            name="Confirmed anomaly",
            text=[hover[i] for i in anom_idx],
            hovertemplate="%{text}<extra></extra>"
        ))

    layout = dict(**PLOTLY_LAYOUT)
    layout["height"] = 420
    layout["title"]  = dict(text=f"{sel_col} — anomaly map",
                             font=dict(color="#555", size=12), x=0)
    layout["xaxis"]["title"] = "Row Index"
    layout["yaxis"]["title"] = sel_col
    fig.update_layout(**layout)
    st.plotly_chart(fig, use_container_width=True)


    # ══════════════════════════════════════════════════
    #  Z-SCORE HEATMAP
    # ══════════════════════════════════════════════════

    st.markdown("""<div class="section-header"><span class="section-title">Z-Score Heatmap</span><span class="section-count">ALL FEATURES</span></div>
    <div class="section-desc">Brighter red means the value is further from its column mean. Hover to see exact z-score and value for any cell.</div>""", unsafe_allow_html=True)

    z_matrix   = np.array([[zscores[col].iloc[i] for i in df.index] for col in numeric_cols])
    heat_hover = []
    for col in numeric_cols:
        row_hover = []
        for i in df.index:
            z   = zscores[col].iloc[i]
            val = df[col].iloc[i]
            row_hover.append(f"<b>{col}</b><br>Row {i}<br>Value: {val:.2f}<br>Z-score: {z:.2f}<br>{'<b>ANOMALOUS</b>' if z > 3.0 else 'Normal'}")
        heat_hover.append(row_hover)

    fig_heat = go.Figure(data=go.Heatmap(
        z=z_matrix,
        x=list(df.index),
        y=numeric_cols,
        colorscale=[[0,"#0d0d0d"],[0.3,"#1a0a0a"],[0.6,"#5a1010"],[0.8,"#aa2020"],[1.0,"#ff6b6b"]],
        zmin=0, zmax=5,
        text=heat_hover,
        hovertemplate="%{text}<extra></extra>",
        colorbar=dict(
            title=dict(text="Z-score", font=dict(color="#555", size=10)),
            tickfont=dict(color="#555", size=9),
            bgcolor="#0d0d0d", bordercolor="#2a2a2a"
        )
    ))

    heat_layout = dict(**PLOTLY_LAYOUT)
    heat_layout["height"] = 300
    heat_layout["margin"] = dict(l=160, r=60, t=20, b=50)
    heat_layout["xaxis"]["title"] = "Row Index"
    fig_heat.update_layout(**heat_layout)
    st.plotly_chart(fig_heat, use_container_width=True)


    # ══════════════════════════════════════════════════
    #  FORECAST CHART
    # ══════════════════════════════════════════════════

    st.markdown("""<div class="section-header"><span class="section-title">Forecast</span><span class="section-count">NEXT 10 STEPS</span></div>
    <div class="section-desc">Linear trend forecast using the last 20 data points. Shaded band shows the confidence interval widening over time. Hover over forecast points to see predicted values.</div>""", unsafe_allow_html=True)

    fore_col        = st.selectbox("Column to forecast", numeric_cols, label_visibility="collapsed", key="fore_col")
    forecast, ci    = forecast_col(df[fore_col])
    mean_v2         = df[fore_col].mean()
    std_v2          = df[fore_col].std()
    fore_z          = np.abs((forecast - mean_v2) / std_v2) if std_v2 > 0 else np.zeros(10)
    fore_anom       = fore_z > 3.0

    hist   = list(df[fore_col].values[-60:])
    hist_x = list(range(len(hist)))
    fore_x = list(range(len(hist), len(hist) + 10))

    fig2 = go.Figure()

    hist_hover = [f"<b>Row {len(df)-len(hist)+i}</b><br>Value: {v:.2f}<br>Z-score: {zscores[fore_col].iloc[len(df)-len(hist)+i]:.2f}"
                  for i, v in enumerate(hist)]

    fig2.add_trace(go.Scatter(
        x=hist_x, y=hist,
        mode="lines+markers",
        line=dict(color="#2a2a2a", width=1),
        marker=dict(color="#3a3a3a", size=4),
        name="Historical",
        text=hist_hover,
        hovertemplate="%{text}<extra></extra>"
    ))

    fig2.add_trace(go.Scatter(
        x=fore_x + fore_x[::-1],
        y=list(forecast + ci) + list((forecast - ci)[::-1]),
        fill="toself", fillcolor="rgba(200,240,96,0.05)",
        line=dict(color="rgba(0,0,0,0)"),
        name="Confidence band", hoverinfo="skip"
    ))

    fore_hover = [f"<b>Future step {i+1}</b><br>Predicted: {v:.2f}<br>Z-score vs history: {fore_z[i]:.2f}<br>{'<b>ANOMALY PREDICTED</b>' if fore_anom[i] else 'Normal'}"
                  for i, v in enumerate(forecast)]

    fig2.add_trace(go.Scatter(
        x=fore_x, y=list(forecast),
        mode="lines+markers",
        line=dict(color="#c8f060", width=2, dash="dash"),
        marker=dict(color="#c8f060", size=6),
        name="Forecast",
        text=fore_hover,
        hovertemplate="%{text}<extra></extra>"
    ))

    if fore_anom.any():
        ax = [fore_x[i] for i in range(10) if fore_anom[i]]
        ay = [forecast[i] for i in range(10) if fore_anom[i]]
        ah = [fore_hover[i] for i in range(10) if fore_anom[i]]
        fig2.add_trace(go.Scatter(
            x=ax, y=ay, mode="markers",
            marker=dict(color="#ff6b6b", size=14, line=dict(color="#ff0000", width=1)),
            name="Predicted anomaly",
            text=ah,
            hovertemplate="%{text}<extra></extra>"
        ))

    fig2.add_vline(x=len(hist)-1, line_dash="dot", line_color="#2a2a2a",
                   annotation_text="forecast starts",
                   annotation_font_color="#444", annotation_font_size=9)

    fore_layout = dict(**PLOTLY_LAYOUT)
    fore_layout["height"] = 400
    fore_layout["title"]  = dict(text=f"{fore_col} — forecast",
                                  font=dict(color="#555", size=12), x=0)
    fore_layout["xaxis"]["title"] = "Row Index"
    fore_layout["yaxis"]["title"] = fore_col
    fig2.update_layout(**fore_layout)
    st.plotly_chart(fig2, use_container_width=True)

    if fore_anom.any():
        steps = list(np.where(fore_anom)[0] + 1)
        st.markdown(f'<div class="forecast-alert">Anomaly predicted at future step(s): {steps}</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="forecast-ok">No anomalies predicted in the next 10 steps</div>', unsafe_allow_html=True)

    # Store for chatbox
    st.session_state["df"]          = df
    st.session_state["domain"]      = domain
    st.session_state["results"]     = results
    st.session_state["numeric_cols"]= numeric_cols
    st.session_state["zscores"]     = zscores

    st.markdown("<hr>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════
#  CHATBOX
# ══════════════════════════════════════════════════════

st.markdown("""
<div class="chat-header">
    <div class="section-title">Data Query</div>
    <div class="chat-hint">TRY / "is row 45 an anomaly?" &middot; "how many anomalies?" &middot; "which are the worst rows?" &middot; "summarize the anomalies"</div>
</div>
""", unsafe_allow_html=True)

if "df" not in st.session_state:
    st.markdown("<p style='font-size:0.72rem;color:#333;letter-spacing:0.05em;'>Upload a dataset above to enable data queries.</p>", unsafe_allow_html=True)
else:
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Query your data..."):
        st.session_state["messages"].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        df       = st.session_state["df"]
        results  = st.session_state["results"]
        domain   = st.session_state["domain"]
        nc       = st.session_state["numeric_cols"]
        zscores  = st.session_state["zscores"]
        response = ""

        row_match = re.search(r"row\s+(\d+)", prompt.lower())

        if row_match:
            rn = int(row_match.group(1))
            if rn < len(results):
                row  = results.iloc[rn]
                is_a = row["flagged"] == 1
                conf = int(row["confidence"])
                response  = f"**Row {rn} — {'Anomaly' if is_a else 'Normal'}**"
                response += f" ({conf}% confidence)\n\n" if is_a else "\n\n"
                response += "| Test | Result |\n|---|---|\n"
                response += f"| Z-Score  | {'Flagged' if row['z_score']  else 'Pass'} |\n"
                response += f"| IQR      | {'Flagged' if row['iqr']      else 'Pass'} |\n"
                response += f"| ML Model | {'Flagged' if row['ml_model'] else 'Pass'} |\n\n"
                response += "**Column breakdown (sorted by z-score):**\n\n"
                sorted_nc = sorted(nc, key=lambda c: zscores[c].iloc[rn], reverse=True)
                for col in sorted_nc:
                    val  = row[col]
                    mean = df[col].mean()
                    z    = zscores[col].iloc[rn]
                    diff = val - mean
                    pct  = abs(diff/mean*100) if mean != 0 else 0
                    arr  = "↑" if diff > 0 else "↓"
                    flag = " — **anomalous**" if z > 3.0 else ""
                    response += f"- `{col}`: `{val:.2f}` {arr} {pct:.0f}% from mean (z=`{z:.2f}`){flag}\n"
            else:
                response = f"Row {rn} does not exist. Dataset has {len(df)} rows (0 to {len(df)-1})."

        elif any(w in prompt.lower() for w in ["how many", "total", "count"]):
            t = int(results["flagged"].sum())
            response = (f"**{t} anomalies** out of {len(df)} rows ({100*t/len(df):.1f}% rate)\n\n"
                        f"| Test | Flags |\n|---|---|\n"
                        f"| Z-Score | {int(results['z_score'].sum())} |\n"
                        f"| IQR | {int(results['iqr'].sum())} |\n"
                        f"| ML Model | {int(results['ml_model'].sum())} |\n"
                        f"| **Consensus (2+)** | **{t}** |")

        elif any(w in prompt.lower() for w in ["domain", "type", "kind"]):
            response = f"Domain classified as **{domain}**."

        elif any(w in prompt.lower() for w in ["worst", "highest", "most", "top"]):
            top5 = results[results["flagged"]==1].nlargest(5, "confidence")
            response = "**Top anomalous rows:**\n\n| Row | Confidence | Most anomalous column |\n|---|---|---|\n"
            for idx, row in top5.iterrows():
                tc  = max(nc, key=lambda c: zscores[c].iloc[idx])
                tz  = zscores[tc].iloc[idx]
                response += f"| {idx} | {int(row['confidence'])}% | `{tc}` (z={tz:.2f}) |\n"

        elif any(w in prompt.lower() for w in ["summarize","summary","overview","explain"]):
            t   = int(results["flagged"].sum())
            tc  = max(nc, key=lambda c: zscores[c].max())
            tz  = zscores[tc].max()
            response = (f"**Summary — {domain}**\n\n"
                        f"- {len(df)} rows analyzed\n"
                        f"- {t} anomalies detected ({100*t/len(df):.1f}% rate)\n"
                        f"- Most anomalous feature: `{tc}` (max z-score: {tz:.2f})\n"
                        f"- Method: Z-Score + IQR + Isolation Forest, consensus 2/3 required")

        elif any(w in prompt.lower() for w in ["column","feature","field"]):
            response = f"**Features — {domain}:**\n\n"
            for col in nc:
                mz = zscores[col].max()
                response += f"- `{col}` — mean `{df[col].mean():.2f}`, std `{df[col].std():.2f}`, max z `{mz:.2f}`\n"

        else:
            t = int(results["flagged"].sum())
            response = (f"**{domain}** — {len(df)} rows, {t} anomalies.\n\n"
                        f"Try: `is row 45 an anomaly?` / `how many anomalies?` / `summarize` / `worst rows`")

        st.session_state["messages"].append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.markdown(response)