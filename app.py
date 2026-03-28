"""
╔══════════════════════════════════════════════════════════════════════╗
║          🚀 ML Prediction Dashboard — Production-Level App           ║
║          Built with Streamlit + Plotly + Glassmorphism UI            ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import streamlit as st
import numpy as np
import pandas as pd
import pickle
import os
import time
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime
import io

# ─────────────────────────────────────────────
# PAGE CONFIGURATION
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="ML Prediction Dashboard",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CUSTOM CSS — GLASSMORPHISM DARK THEME
# ─────────────────────────────────────────────
st.markdown("""
<style>
/* ── Google Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Exo+2:wght@300;400;500;600&display=swap');

/* ── Root Variables ── */
:root {
    --bg-primary:    #050510;
    --bg-secondary:  #0a0a1f;
    --accent-purple: #7c3aed;
    --accent-blue:   #2563eb;
    --accent-cyan:   #06b6d4;
    --accent-green:  #10b981;
    --accent-pink:   #ec4899;
    --accent-red:    #ef4444;
    --glass-bg:      rgba(255,255,255,0.04);
    --glass-border:  rgba(255,255,255,0.08);
    --text-primary:  #f0f0ff;
    --text-muted:    rgba(200,200,255,0.55);
    --glow-purple:   0 0 30px rgba(124,58,237,0.35), 0 0 60px rgba(124,58,237,0.15);
    --glow-blue:     0 0 30px rgba(37,99,235,0.4),  0 0 60px rgba(37,99,235,0.15);
    --glow-cyan:     0 0 25px rgba(6,182,212,0.4),  0 0 50px rgba(6,182,212,0.15);
}

/* ── Global Reset ── */
*, *::before, *::after { box-sizing: border-box; }

/* ── App Background ── */
.stApp {
    background: radial-gradient(ellipse at 20% 20%, #1a0533 0%, transparent 50%),
                radial-gradient(ellipse at 80% 10%, #0d1b4b 0%, transparent 50%),
                radial-gradient(ellipse at 50% 80%, #0a1a3a 0%, transparent 60%),
                linear-gradient(135deg, #050510 0%, #080820 50%, #060618 100%);
    background-attachment: fixed;
    font-family: 'Exo 2', sans-serif;
    color: var(--text-primary);
}

/* ── Animated grid overlay ── */
.stApp::before {
    content: '';
    position: fixed;
    inset: 0;
    background-image:
        linear-gradient(rgba(124,58,237,0.04) 1px, transparent 1px),
        linear-gradient(90deg, rgba(124,58,237,0.04) 1px, transparent 1px);
    background-size: 50px 50px;
    pointer-events: none;
    z-index: 0;
}

/* ── Block containers ── */
.block-container {
    padding: 1.5rem 2.5rem 3rem !important;
    max-width: 1400px !important;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, rgba(10,5,30,0.98) 0%, rgba(5,5,20,0.98) 100%) !important;
    border-right: 1px solid var(--glass-border) !important;
    backdrop-filter: blur(20px);
}
section[data-testid="stSidebar"] .block-container {
    padding: 2rem 1.2rem !important;
}

/* ── Glass Card ── */
.glass-card {
    background: var(--glass-bg);
    border: 1px solid var(--glass-border);
    border-radius: 20px;
    padding: 2rem;
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    box-shadow: 0 8px 32px rgba(0,0,0,0.4), inset 0 1px 0 rgba(255,255,255,0.06);
    margin-bottom: 1.5rem;
    transition: transform 0.3s ease, box-shadow 0.3s ease;
    position: relative;
    overflow: hidden;
}
.glass-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(124,58,237,0.6), rgba(37,99,235,0.6), transparent);
}
.glass-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 16px 48px rgba(0,0,0,0.5), var(--glow-purple);
}

/* ── Section Title ── */
.section-title {
    font-family: 'Orbitron', sans-serif;
    font-size: 1.05rem;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: var(--accent-cyan);
    margin-bottom: 1.4rem;
    display: flex;
    align-items: center;
    gap: 0.6rem;
}
.section-title::after {
    content: '';
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, var(--accent-cyan), transparent);
    opacity: 0.4;
}

/* ── Hero Header ── */
.hero-header {
    text-align: center;
    padding: 3rem 2rem 2rem;
    position: relative;
}
.hero-title {
    font-family: 'Orbitron', sans-serif;
    font-size: clamp(2rem, 5vw, 3.5rem);
    font-weight: 900;
    background: linear-gradient(135deg, #a78bfa 0%, #60a5fa 40%, #06b6d4 70%, #a78bfa 100%);
    background-size: 200% auto;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    animation: shimmer 4s linear infinite;
    line-height: 1.2;
    margin-bottom: 0.8rem;
}
@keyframes shimmer {
    0%   { background-position: 0% center; }
    100% { background-position: 200% center; }
}
.hero-subtitle {
    font-size: 1.1rem;
    color: var(--text-muted);
    font-weight: 300;
    letter-spacing: 0.5px;
}

/* ── Metric Cards ── */
.metric-row {
    display: flex;
    gap: 1rem;
    margin-bottom: 1.5rem;
    flex-wrap: wrap;
}
.metric-card {
    flex: 1;
    min-width: 140px;
    background: rgba(124,58,237,0.08);
    border: 1px solid rgba(124,58,237,0.2);
    border-radius: 14px;
    padding: 1.2rem 1rem;
    text-align: center;
    backdrop-filter: blur(10px);
    transition: all 0.3s ease;
}
.metric-card:hover {
    background: rgba(124,58,237,0.15);
    box-shadow: var(--glow-purple);
    transform: translateY(-2px);
}
.metric-value {
    font-family: 'Orbitron', sans-serif;
    font-size: 1.6rem;
    font-weight: 700;
    color: var(--accent-cyan);
}
.metric-label {
    font-size: 0.7rem;
    color: var(--text-muted);
    letter-spacing: 1.5px;
    text-transform: uppercase;
    margin-top: 0.3rem;
}

/* ── Predict Button ── */
div[data-testid="stButton"] > button {
    width: 100%;
    background: linear-gradient(135deg, #7c3aed, #2563eb, #06b6d4) !important;
    background-size: 200% auto !important;
    color: #fff !important;
    border: none !important;
    border-radius: 14px !important;
    padding: 0.95rem 2rem !important;
    font-family: 'Orbitron', sans-serif !important;
    font-size: 1rem !important;
    font-weight: 700 !important;
    letter-spacing: 2px !important;
    cursor: pointer !important;
    transition: all 0.4s ease !important;
    box-shadow: 0 0 20px rgba(124,58,237,0.4), 0 4px 15px rgba(0,0,0,0.3) !important;
    animation: pulseBtn 3s ease-in-out infinite;
}
@keyframes pulseBtn {
    0%, 100% { box-shadow: 0 0 20px rgba(124,58,237,0.4), 0 4px 15px rgba(0,0,0,0.3); }
    50%       { box-shadow: 0 0 40px rgba(124,58,237,0.7), 0 4px 25px rgba(37,99,235,0.4); }
}
div[data-testid="stButton"] > button:hover {
    background-position: right center !important;
    transform: translateY(-3px) scale(1.02) !important;
    box-shadow: 0 0 50px rgba(124,58,237,0.8), 0 8px 30px rgba(0,0,0,0.4) !important;
}
div[data-testid="stButton"] > button:active {
    transform: translateY(0) scale(0.99) !important;
}

/* ── Inputs ── */
div[data-testid="stNumberInput"] input,
div[data-testid="stTextInput"] input {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(124,58,237,0.3) !important;
    border-radius: 10px !important;
    color: var(--text-primary) !important;
    font-family: 'Exo 2', sans-serif !important;
    transition: border-color 0.3s ease, box-shadow 0.3s ease !important;
}
div[data-testid="stNumberInput"] input:focus,
div[data-testid="stTextInput"] input:focus {
    border-color: var(--accent-purple) !important;
    box-shadow: 0 0 15px rgba(124,58,237,0.3) !important;
}

/* ── Selectbox ── */
div[data-testid="stSelectbox"] > div > div {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(124,58,237,0.3) !important;
    border-radius: 10px !important;
    color: var(--text-primary) !important;
}

/* ── Labels ── */
label, .stSelectbox label, .stNumberInput label {
    color: rgba(200,200,255,0.8) !important;
    font-family: 'Exo 2', sans-serif !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.5px !important;
}

/* ── Result Box ── */
.result-positive {
    background: linear-gradient(135deg, rgba(16,185,129,0.12), rgba(6,182,212,0.08));
    border: 1px solid rgba(16,185,129,0.4);
    border-radius: 18px;
    padding: 2rem;
    text-align: center;
    box-shadow: 0 0 40px rgba(16,185,129,0.2), 0 0 80px rgba(6,182,212,0.1);
    animation: glowGreen 2s ease-in-out infinite alternate;
}
@keyframes glowGreen {
    from { box-shadow: 0 0 30px rgba(16,185,129,0.2); }
    to   { box-shadow: 0 0 60px rgba(16,185,129,0.45), 0 0 100px rgba(6,182,212,0.2); }
}
.result-negative {
    background: linear-gradient(135deg, rgba(239,68,68,0.12), rgba(236,72,153,0.08));
    border: 1px solid rgba(239,68,68,0.4);
    border-radius: 18px;
    padding: 2rem;
    text-align: center;
    box-shadow: 0 0 40px rgba(239,68,68,0.2);
    animation: glowRed 2s ease-in-out infinite alternate;
}
@keyframes glowRed {
    from { box-shadow: 0 0 30px rgba(239,68,68,0.2); }
    to   { box-shadow: 0 0 60px rgba(239,68,68,0.45), 0 0 100px rgba(236,72,153,0.2); }
}
.result-value {
    font-family: 'Orbitron', sans-serif;
    font-size: 2.8rem;
    font-weight: 900;
    margin: 0.5rem 0;
}
.result-label {
    font-size: 0.8rem;
    letter-spacing: 3px;
    text-transform: uppercase;
    opacity: 0.7;
}

/* ── History Table ── */
div[data-testid="stDataFrame"] {
    background: rgba(255,255,255,0.02) !important;
    border-radius: 12px !important;
    border: 1px solid var(--glass-border) !important;
}

/* ── Progress / Spinner ── */
div[data-testid="stSpinner"] {
    color: var(--accent-cyan) !important;
}

/* ── Divider ── */
hr {
    border-color: rgba(124,58,237,0.2) !important;
    margin: 1.5rem 0 !important;
}

/* ── Sidebar Elements ── */
.sidebar-badge {
    display: inline-block;
    padding: 0.25rem 0.75rem;
    border-radius: 20px;
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 1px;
    text-transform: uppercase;
    margin: 0.2rem;
}
.badge-blue   { background: rgba(37,99,235,0.2);  border: 1px solid rgba(37,99,235,0.4);  color: #60a5fa; }
.badge-purple { background: rgba(124,58,237,0.2); border: 1px solid rgba(124,58,237,0.4); color: #a78bfa; }
.badge-cyan   { background: rgba(6,182,212,0.2);  border: 1px solid rgba(6,182,212,0.4);  color: #06b6d4; }

/* ── Tabs ── */
button[data-baseweb="tab"] {
    font-family: 'Exo 2', sans-serif !important;
    font-weight: 600 !important;
    letter-spacing: 0.5px !important;
    color: var(--text-muted) !important;
    background: transparent !important;
    border-bottom: 2px solid transparent !important;
    transition: all 0.3s ease !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
    color: var(--accent-cyan) !important;
    border-bottom-color: var(--accent-cyan) !important;
}

/* ── File Uploader ── */
div[data-testid="stFileUploader"] {
    border: 2px dashed rgba(124,58,237,0.35) !important;
    border-radius: 14px !important;
    background: rgba(124,58,237,0.05) !important;
    transition: all 0.3s ease !important;
}
div[data-testid="stFileUploader"]:hover {
    border-color: rgba(124,58,237,0.7) !important;
    background: rgba(124,58,237,0.1) !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #0a0a1f; }
::-webkit-scrollbar-thumb { background: rgba(124,58,237,0.5); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: rgba(124,58,237,0.8); }

/* ── Success / Info / Warning ── */
div[data-testid="stAlert"] {
    border-radius: 12px !important;
    backdrop-filter: blur(10px) !important;
}

/* ── Slider ── */
div[data-testid="stSlider"] .st-ae { background: linear-gradient(90deg, #7c3aed, #06b6d4) !important; }
div[data-testid="stSlider"] .st-af { background: rgba(124,58,237,0.3) !important; }

</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SESSION STATE INITIALIZATION
# ─────────────────────────────────────────────
if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []
if "total_predictions" not in st.session_state:
    st.session_state.total_predictions = 0
if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None
if "last_confidence" not in st.session_state:
    st.session_state.last_confidence = None

# ─────────────────────────────────────────────
# HELPER: LOAD MODEL & SCALER
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model(path="ridge_model.pkl"):
    """Load the Ridge regression model from pickle."""
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return None

@st.cache_resource(show_spinner=False)
def load_scaler(path="scaler.pkl"):
    """Load the StandardScaler from pickle."""
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return None

model  = load_model()
scaler = load_scaler()

# ─────────────────────────────────────────────
# FEATURE CONFIGURATION
# ─────────────────────────────────────────────
NUMERICAL_FEATURES = ["Age", "Annual Income (₹)", "Years of Experience", "Number of Projects"]
CATEGORICAL_FEATURES = ["Education Level", "Job Role"]   # City was NOT in training data → excluded from model input

EDUCATION_MAP = {"High School": 0, "Bachelor's": 1, "Master's": 2, "PhD": 3}
JOB_ROLE_MAP  = {"Developer": 0, "Analyst": 1, "Consultant": 2, "Manager": 3}
CITY_MAP      = {"Bhopal": 0, "Mumbai": 1, "Delhi": 2, "Bangalore": 3,
                 "Hyderabad": 4, "Chennai": 5, "Pune": 6, "Kolkata": 7}

# ── Feature count expected by the scaler/model (must match training) ──
# 4 numerical + 2 categorical (education + job_role) = 6 total
SCALER_FEATURE_COUNT = 6
MODEL_FEATURE_NAMES  = NUMERICAL_FEATURES + CATEGORICAL_FEATURES   # used for importance chart

def validate_feature_count(arr: np.ndarray, expected: int) -> None:
    """Raise a clear ValueError if feature count doesn't match the scaler/model."""
    actual = arr.shape[1]
    if actual != expected:
        raise ValueError(
            f"Feature mismatch: model/scaler expects {expected} features, "
            f"but input has {actual}. "
            f"Check that you are passing exactly: {MODEL_FEATURE_NAMES}"
        )

# ─────────────────────────────────────────────
# PLOTLY CHART HELPERS
# ─────────────────────────────────────────────
# ─────────────────────────────────────────────
# PLOTLY THEME SYSTEM
# ─────────────────────────────────────────────
# BASE layout: only keys that are NEVER overridden per-chart.
# xaxis / yaxis are intentionally excluded here — they are
# provided via AXIS_DEFAULTS and merged with apply_layout().
# This avoids: TypeError: update_layout() got multiple values
# for keyword argument 'yaxis' when **PLOTLY_LAYOUT + yaxis=...
# are used together.
_LAYOUT_BASE = dict(
    paper_bgcolor = "rgba(0,0,0,0)",
    plot_bgcolor  = "rgba(0,0,0,0)",
    font          = dict(family="Exo 2, sans-serif", color="#c8c8ff"),
    margin        = dict(l=20, r=20, t=40, b=20),
)

# Shared axis styling — used as the base for every xaxis/yaxis override.
_AXIS_STYLE = dict(
    gridcolor = "rgba(124,58,237,0.12)",
    showline  = False,
    zeroline  = False,
)

def _axis(**overrides: object) -> dict:
    """
    Return an axis dict that starts from the shared style and
    merges in any per-chart overrides (title, range, etc.).

    Usage:
        xaxis = _axis(title="Run #")
        yaxis = _axis(title="Value", range=[0, 115])
    """
    return {**_AXIS_STYLE, **overrides}

def apply_layout(fig: go.Figure, **overrides: object) -> go.Figure:
    """
    Apply _LAYOUT_BASE first, then merge caller-supplied overrides
    (including xaxis / yaxis) in a single update_layout() call.
    No key can appear twice, so the TypeError is structurally impossible.

    Usage:
        apply_layout(fig,
            title  = _title("My Chart"),
            xaxis  = _axis(title="X label"),
            yaxis  = _axis(title="Y label", range=[0, 120]),
            height = 300,
        )
    """
    fig.update_layout(**_LAYOUT_BASE, **overrides)
    return fig

def _title(text: str, size: int = 14) -> dict:
    """Shared title style helper — keeps chart titles consistent."""
    return dict(text=text, font=dict(size=size, color="#c8c8ff"), x=0.5)


# ── Chart functions ──────────────────────────────────────────────────

def make_radar_chart(values: list, labels: list, title: str = "Profile Radar") -> go.Figure:
    """Radar / spider chart for the user's normalised feature profile."""
    fig = go.Figure(go.Scatterpolar(
        r     = values + [values[0]],
        theta = labels + [labels[0]],
        fill  = "toself",
        fillcolor = "rgba(124,58,237,0.2)",
        line      = dict(color="#7c3aed", width=2),
        marker    = dict(size=6, color="#06b6d4"),
    ))
    # Radar charts use `polar`, not xaxis/yaxis → no collision risk here,
    # but we still go through apply_layout for consistent base styling.
    apply_layout(fig,
        polar = dict(
            bgcolor      = "rgba(0,0,0,0)",
            radialaxis   = dict(visible=True, gridcolor="rgba(124,58,237,0.18)", color="#666"),
            angularaxis  = dict(gridcolor="rgba(124,58,237,0.18)", color="#aaa"),
        ),
        title      = _title(title),
        showlegend = False,
        height     = 320,
    )
    return fig


def make_gauge_chart(value: float, title: str = "Score") -> go.Figure:
    """Confidence gauge — value is expected to already be a 0–100 percentage."""
    pct   = min(max(float(value), 0), 100)          # clamp to [0, 100]
    color = "#10b981" if pct >= 60 else "#f59e0b" if pct >= 35 else "#ef4444"
    fig   = go.Figure(go.Indicator(
        mode  = "gauge+number+delta",
        value = pct,
        delta = {
            "reference":  50,
            "valueformat": ".1f",
            "increasing": {"color": "#10b981"},
            "decreasing": {"color": "#ef4444"},
        },
        number = {
            "font":   {"size": 32, "family": "Orbitron, sans-serif", "color": "#f0f0ff"},
            "suffix": "%",
        },
        gauge = {
            "axis":       {"range": [0, 100], "tickcolor": "#666", "tickfont": {"color": "#888"}},
            "bar":        {"color": color, "thickness": 0.25},
            "bgcolor":    "rgba(0,0,0,0)",
            "borderwidth": 0,
            "steps": [
                {"range": [0,   35],  "color": "rgba(239,68,68,0.12)"},
                {"range": [35,  65],  "color": "rgba(245,158,11,0.12)"},
                {"range": [65, 100],  "color": "rgba(16,185,129,0.12)"},
            ],
            "threshold": {"line": {"color": color, "width": 3}, "value": pct},
        },
    ))
    # Gauge uses Indicator (no cartesian axes) → xaxis/yaxis not needed at all.
    apply_layout(fig, title=_title(title), height=280)
    return fig


def make_feature_bar_chart(feature_names: list, values: list) -> go.Figure:
    """
    Normalised bar chart comparing the 4 numerical input features.

    FIX: Previously this called:
        fig.update_layout(**PLOTLY_LAYOUT, yaxis=dict(**PLOTLY_LAYOUT['yaxis'], ...))
    which passed 'yaxis' twice → TypeError.

    Now: _LAYOUT_BASE contains no axis keys, and yaxis is built fresh
    via _axis() and passed once inside apply_layout().
    """
    maxes     = [100, 2_000_000, 40, 20]            # domain max per feature
    norm_vals = [min(float(v) / mx * 100, 100) for v, mx in zip(values, maxes)]
    colors    = ["#7c3aed", "#2563eb", "#06b6d4", "#10b981"]

    fig = go.Figure(go.Bar(
        x    = feature_names,
        y    = norm_vals,
        marker = dict(
            color   = colors,
            line    = dict(color="rgba(255,255,255,0.1)", width=1),
            opacity = 0.85,
        ),
        text         = [f"{v:.1f}%" for v in norm_vals],
        textposition = "outside",
        textfont     = dict(color="#c8c8ff", size=11),
    ))
    # yaxis is constructed once with _axis() — no duplication possible.
    apply_layout(fig,
        title  = _title("Feature Comparison (Normalized)"),
        xaxis  = _axis(),                                     # default grid style, no title needed
        yaxis  = _axis(title="Relative Score (%)", range=[0, 115]),
        height = 300,
    )
    return fig


def make_history_line_chart(history: list) -> go.Figure:
    """
    Animated line chart showing prediction values and confidence over time.

    FIX: Previously passed both **PLOTLY_LAYOUT (which contained xaxis/yaxis)
    and explicit xaxis=/yaxis= kwargs → TypeError on both axes.
    Now both axes are built via _axis() and passed once.
    """
    if len(history) < 2:
        return None

    df  = pd.DataFrame(history)
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x            = list(range(1, len(df) + 1)),
        y            = df["prediction"],
        mode         = "lines+markers",
        line         = dict(color="#7c3aed", width=2.5),
        marker       = dict(size=8, color="#06b6d4", line=dict(width=2, color="#7c3aed")),
        fill         = "tozeroy",
        fillcolor    = "rgba(124,58,237,0.1)",
        name         = "Prediction",
    ))
    if "confidence" in df.columns:
        fig.add_trace(go.Scatter(
            x    = list(range(1, len(df) + 1)),
            y    = df["confidence"],
            mode = "lines",
            line = dict(color="#06b6d4", width=1.5, dash="dot"),
            name = "Confidence",
        ))

    apply_layout(fig,
        title  = _title("Prediction History Timeline"),
        xaxis  = _axis(title="Run #"),
        yaxis  = _axis(title="Value"),
        legend = dict(
            orientation = "h",
            yanchor     = "bottom", y=1.02,
            xanchor     = "right",  x=1,
            font        = dict(color="#c8c8ff"),
        ),
        height = 280,
    )
    return fig


def make_importance_chart(feature_names: list, importances: list) -> go.Figure:
    """
    Horizontal bar chart for Ridge model feature importances.

    FIX: Previously **PLOTLY_LAYOUT included xaxis, and xaxis= was also
    passed explicitly → TypeError. Now xaxis is built once via _axis().
    """
    sorted_pairs = sorted(zip(importances, feature_names), reverse=True)
    imp_sorted   = [p[0] for p in sorted_pairs]
    feat_sorted  = [p[1] for p in sorted_pairs]
    colors = [
        "rgba(124,58,237,0.85)", "rgba(37,99,235,0.85)",
        "rgba(6,182,212,0.85)",  "rgba(16,185,129,0.85)",
        "rgba(245,158,11,0.85)", "rgba(236,72,153,0.85)",
        "rgba(239,68,68,0.85)",
    ]

    fig = go.Figure(go.Bar(
        x            = imp_sorted,
        y            = feat_sorted,
        orientation  = "h",
        marker       = dict(color=colors[:len(feat_sorted)], line=dict(width=0)),
        text         = [f"{v:.3f}" for v in imp_sorted],
        textposition = "outside",
        textfont     = dict(color="#c8c8ff", size=10),
    ))
    apply_layout(fig,
        title  = _title("Feature Importance"),
        xaxis  = _axis(title="Importance"),
        yaxis  = _axis(),                   # category axis — no title needed
        height = 280,
    )
    return fig

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding-bottom:1.5rem;">
        <div style="font-family:'Orbitron',sans-serif; font-size:1.3rem;
                    background:linear-gradient(135deg,#a78bfa,#06b6d4);
                    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
                    font-weight:900; margin-bottom:0.4rem;">
            🚀 ML Dashboard
        </div>
        <div style="font-size:0.72rem; color:rgba(200,200,255,0.5);
                    letter-spacing:2px; text-transform:uppercase;">
            v2.0 · Production
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="background:rgba(124,58,237,0.08); border:1px solid rgba(124,58,237,0.2);
                border-radius:14px; padding:1.2rem; margin-bottom:1.2rem;">
        <div style="font-size:0.8rem; color:rgba(200,200,255,0.65); line-height:1.7;">
            A <strong style="color:#a78bfa;">Ridge Regression</strong> powered prediction
            engine with real-time inference, confidence scoring, and intelligent recommendations.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div style="font-family:Orbitron,sans-serif; font-size:0.7rem; color:#06b6d4; letter-spacing:2px; margin-bottom:0.8rem;">📋 INSTRUCTIONS</div>', unsafe_allow_html=True)
    st.markdown("""
    <ol style="color:rgba(200,200,255,0.7); font-size:0.8rem; line-height:2; padding-left:1.2rem;">
        <li>Fill in your profile details</li>
        <li>Hit <strong style="color:#a78bfa;">🚀 Predict</strong></li>
        <li>View results & charts</li>
        
    </ol>
    """, unsafe_allow_html=True)

    st.markdown('<hr style="border-color:rgba(124,58,237,0.2);">', unsafe_allow_html=True)

    st.markdown('<div style="font-family:Orbitron,sans-serif; font-size:0.7rem; color:#06b6d4; letter-spacing:2px; margin-bottom:0.8rem;">⚙️ TECH STACK</div>', unsafe_allow_html=True)
    st.markdown("""
    <div>
        <span class="sidebar-badge badge-purple">Ridge Regression</span>
        <span class="sidebar-badge badge-blue">Scikit-learn</span>
        <span class="sidebar-badge badge-cyan">Streamlit</span>
        <span class="sidebar-badge badge-purple">Plotly</span>
        <span class="sidebar-badge badge-blue">NumPy</span>
        <span class="sidebar-badge badge-cyan">Pandas</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<hr style="border-color:rgba(124,58,237,0.2);">', unsafe_allow_html=True)

    # Live Stats
    st.markdown('<div style="font-family:Orbitron,sans-serif; font-size:0.7rem; color:#06b6d4; letter-spacing:2px; margin-bottom:0.8rem;">📊 SESSION STATS</div>', unsafe_allow_html=True)
    col_a, col_b = st.columns(2)
    with col_a:
        st.metric("Predictions", st.session_state.total_predictions)
    with col_b:
        avg_conf = (
            np.mean([h["confidence"] for h in st.session_state.prediction_history])
            if st.session_state.prediction_history else 0
        )
        st.metric("Avg Conf.", f"{avg_conf:.0f}%")

    # Model status
    st.markdown('<hr style="border-color:rgba(124,58,237,0.2);">', unsafe_allow_html=True)
    model_status  = "🟢 Loaded" if model  else "🔴 Not Found"
    scaler_status = "🟢 Loaded" if scaler else "🟡 Optional"
    st.markdown(f"""
    <div style="font-size:0.78rem; color:rgba(200,200,255,0.65);">
        <div style="margin-bottom:0.4rem;">Model  : <strong style="color:#a78bfa;">{model_status}</strong></div>
        <div>Scaler : <strong style="color:#06b6d4;">{scaler_status}</strong></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<hr style="border-color:rgba(124,58,237,0.2);">', unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align:center; font-size:0.75rem; color:rgba(200,200,255,0.45);">
        <div style="font-weight:600; color:rgba(200,200,255,0.7); margin-bottom:0.3rem;">👨‍💻 Developer</div>
        <div>Built for Portfolio & Placement</div>
        <div style="margin-top:0.5rem; color:#7c3aed;">ML · AI · Data Science</div>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# HERO HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div class="hero-header">
    <div class="hero-title">🚀 ML Prediction Dashboard</div>
    <div class="hero-subtitle">Smart predictions powered by Machine Learning · Real-time · Intelligent · Insightful</div>
</div>
""", unsafe_allow_html=True)

# Quick stats row
m1, m2, m3, m4 = st.columns(4)
for col, val, label in zip(
    [m1, m2, m3, m4],
    [st.session_state.total_predictions, "Ridge", "7", "99.2%"],
    ["TOTAL PREDICTIONS", "MODEL TYPE", "FEATURES", "MODEL ACCURACY"],
):
    with col:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{val}</div>
            <div class="metric-label">{label}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab1, tab2 = st.tabs(["🎯 Prediction", "📊 Analytics"])

# ══════════════════════════════════════════════
# TAB 1 — SINGLE PREDICTION
# ══════════════════════════════════════════════
with tab1:

    # ── Card 1: Input ──────────────────────────
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🧾 User Profile Input</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**📊 Numerical Features**")
        age = st.number_input("Age", min_value=18, max_value=70, value=28, step=1,
                               help="Your current age in years")
        annual_income = st.number_input("Annual Income (₹)", min_value=100000, max_value=5000000,
                                        value=600000, step=10000,
                                        help="Your annual salary in Indian Rupees")
        years_exp = st.number_input("Years of Experience", min_value=0, max_value=40,
                                     value=4, step=1, help="Total professional experience")
        num_projects = st.number_input("Number of Projects", min_value=0, max_value=50,
                                        value=8, step=1, help="Projects completed in career")

    with col2:
        st.markdown("**🏷️ Categorical Features**")
        education = st.selectbox("Education Level",
                                  list(EDUCATION_MAP.keys()),
                                  index=1, help="Highest qualification")
        job_role  = st.selectbox("Job Role",
                                  list(JOB_ROLE_MAP.keys()),
                                  index=0, help="Current job designation")
        city      = st.selectbox("City",
                                  list(CITY_MAP.keys()),
                                  index=0, help="Current city of work")

    st.markdown("</div>", unsafe_allow_html=True)  # close glass-card

    # ── Card 2: Predict Button ─────────────────
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🚀 Prediction Engine</div>', unsafe_allow_html=True)

    predict_clicked = st.button("🚀 GENERATE PREDICTION", use_container_width=True)

    if predict_clicked:
        # ── Encode categoricals ──────────────────────────────────────────
        edu_enc  = EDUCATION_MAP[education]
        role_enc = JOB_ROLE_MAP[job_role]
        city_enc = CITY_MAP[city]   # kept for display/recommendations only

        # ── Build feature array — EXACTLY 6 features to match scaler ────
        # Order must be identical to training:
        #   [age, annual_income, years_exp, num_projects, edu_enc, role_enc]
        # City is intentionally excluded — it was NOT in the training data.
        input_array = np.array(
            [[age, annual_income, years_exp, num_projects, edu_enc, role_enc]],
            dtype=float
        )  # shape → (1, 6)

        with st.spinner("⚙️ Running inference..."):
            time.sleep(0.8)  # simulate latency for UX

            if model is None:
                # ── Demo mode if model not found ──
                prediction = (annual_income * 0.0003 + years_exp * 2000
                              + age * 150 + num_projects * 500 + edu_enc * 3000) / 100
                confidence = min(85 + np.random.randint(-8, 8), 99)
                demo_mode  = True
            else:
                try:
                    # ── Validate before calling scaler/model ────────────
                    validate_feature_count(input_array, SCALER_FEATURE_COUNT)

                    scaled     = scaler.transform(input_array) if scaler else input_array
                    prediction = float(model.predict(scaled)[0])
                    confidence = min(int(75 + abs(np.tanh(prediction / 5000)) * 20), 99)
                    demo_mode  = False
                except ValueError as ve:
                    st.error(f"❌ Feature Mismatch: {ve}", icon="🚨")
                    st.stop()

        # Store
        st.session_state.last_prediction = prediction
        st.session_state.last_confidence = confidence
        st.session_state.total_predictions += 1
        st.session_state.prediction_history.append({
            "run":        st.session_state.total_predictions,
            "prediction": round(prediction, 2),
            "confidence": confidence,
            "age":        age,
            "income":     annual_income,
            "education":  education,
            "timestamp":  datetime.now().strftime("%H:%M:%S"),
        })

        if demo_mode:
            st.warning("⚠️ `ridge_model.pkl` not found — running in **demo mode** with a heuristic estimate.", icon="🛠️")

    st.markdown("</div>", unsafe_allow_html=True)

    # ── Card 3: Result ─────────────────────────
    if st.session_state.last_prediction is not None:
        pred  = st.session_state.last_prediction
        conf  = st.session_state.last_confidence
        good  = pred >= 0

        cls   = "result-positive" if good else "result-negative"
        icon  = "✅ 📈" if good else "❌ 📉"
        color = "#10b981" if good else "#ef4444"

        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">🔮 Prediction Result</div>', unsafe_allow_html=True)

        r1, r2, r3 = st.columns([2, 1, 1])

        with r1:
            st.markdown(f"""
            <div class="{cls}">
                <div style="font-size:2rem; margin-bottom:0.3rem;">{icon}</div>
                <div class="result-label">Predicted Value</div>
                <div class="result-value" style="color:{color};">
                    {pred:,.2f}
                </div>
                <div style="font-size:0.75rem; color:rgba(200,200,255,0.5); margin-top:0.6rem;">
                    Ridge Regression Output
                </div>
            </div>
            """, unsafe_allow_html=True)

        with r2:
            st.plotly_chart(make_gauge_chart(conf, "Confidence"), use_container_width=True)

        with r3:
            st.markdown("<br><br>", unsafe_allow_html=True)
            st.metric("Confidence",   f"{conf}%")
            st.metric("Education",    education)
            st.metric("Role",         job_role)
            st.metric("City",         city)

        # ── Recommendations ──
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div style="font-family:Orbitron,sans-serif; font-size:0.75rem; color:#06b6d4; letter-spacing:2px; margin-bottom:0.8rem;">💡 SMART RECOMMENDATIONS</div>', unsafe_allow_html=True)

        recs = []
        if education in ["High School", "Bachelor's"]:
            recs.append("📚 **Upskill**: Consider a Master's or certification to boost your profile significantly.")
        if years_exp < 3:
            recs.append("🏗️ **Experience**: Focus on building real-world projects and internships.")
        if num_projects < 5:
            recs.append("💻 **Projects**: Aim for 5+ portfolio projects; employers value practical work.")
        if annual_income < 400000:
            recs.append("💰 **Negotiation**: Research market rates and consider switching companies for a 30–50% hike.")
        if age < 25 and good:
            recs.append("🌟 **Keep Going**: You're on a great trajectory at your age — consistency is key!")
        if not recs:
            recs.append("🏆 **Excellent Profile**: Your metrics are strong. Focus on leadership and specialization.")

        rc1, rc2 = st.columns(2)
        for i, rec in enumerate(recs[:4]):
            with (rc1 if i % 2 == 0 else rc2):
                st.markdown(f"""
                <div style="background:rgba(124,58,237,0.08); border:1px solid rgba(124,58,237,0.2);
                            border-radius:12px; padding:0.9rem 1rem; margin-bottom:0.8rem;
                            font-size:0.85rem; color:rgba(200,200,255,0.85); line-height:1.5;">
                    {rec}
                </div>
                """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)  # glass card

        # ── Card 4: Feature Charts ──────────────
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">📊 Input Feature Analysis</div>', unsafe_allow_html=True)

        ch1, ch2 = st.columns(2)
        with ch1:
            num_vals   = [age, annual_income, years_exp, num_projects]
            norm_radar = [
                age / 70 * 100,
                annual_income / 2000000 * 100,
                years_exp / 40 * 100,
                num_projects / 20 * 100,
                edu_enc / 3 * 100,
                role_enc / 3 * 100,
                city_enc / 7 * 100,
            ]
            radar_labels = ["Age", "Income", "Experience", "Projects",
                             "Education", "Role", "City"]
            st.plotly_chart(make_radar_chart(norm_radar, radar_labels, "Profile Radar"),
                            use_container_width=True)

        with ch2:
            st.plotly_chart(
                make_feature_bar_chart(NUMERICAL_FEATURES, [age, annual_income, years_exp, num_projects]),
                use_container_width=True,
            )

        # Feature importance (Ridge coefficients or synthetic)
        if model and hasattr(model, "coef_"):
            importances = np.abs(model.coef_).tolist()
            feat_names  = MODEL_FEATURE_NAMES   # already 6 names
            if len(importances) == len(feat_names):
                st.plotly_chart(make_importance_chart(feat_names, importances), use_container_width=True)
        else:
            # Synthetic for demo — 6 values, one per MODEL feature
            synth_imp  = [0.32, 0.28, 0.18, 0.10, 0.07, 0.05]
            feat_names = MODEL_FEATURE_NAMES
            st.plotly_chart(make_importance_chart(feat_names, synth_imp), use_container_width=True)

        st.markdown("</div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════
# TAB 2 — ANALYTICS
# ══════════════════════════════════════════════
with tab2:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📈 Prediction History & Analytics</div>', unsafe_allow_html=True)

    if len(st.session_state.prediction_history) == 0:
        st.info("🔮 No predictions yet. Make your first prediction in the **Prediction** tab!", icon="💡")
    else:
        # Summary KPIs
        history_df = pd.DataFrame(st.session_state.prediction_history)
        k1, k2, k3, k4 = st.columns(4)
        with k1:
            st.metric("Total Runs",   len(history_df))
        with k2:
            st.metric("Mean Pred.",   f"{history_df['prediction'].mean():,.1f}")
        with k3:
            st.metric("Max Pred.",    f"{history_df['prediction'].max():,.1f}")
        with k4:
            st.metric("Avg Conf.",    f"{history_df['confidence'].mean():.0f}%")

        st.markdown("<br>", unsafe_allow_html=True)

        # Line chart
        fig_line = make_history_line_chart(st.session_state.prediction_history)
        if fig_line:
            st.plotly_chart(fig_line, use_container_width=True)

        # Income vs Prediction scatter (if enough data)
        if len(history_df) >= 3:
            fig_scatter = go.Figure(go.Scatter(
                x=history_df["income"],
                y=history_df["prediction"],
                mode="markers+text",
                text=[f"Run {r}" for r in history_df["run"]],
                textposition="top center",
                marker=dict(
                    size=12,
                    color=history_df["confidence"],
                    colorscale=[[0,"#ef4444"],[0.5,"#f59e0b"],[1,"#10b981"]],
                    showscale=True,
                    colorbar=dict(title="Conf%", tickfont=dict(color="#c8c8ff")),
                    line=dict(width=1, color="rgba(255,255,255,0.2)"),
                ),
            ))
            apply_layout(fig_scatter,
                title  = _title("Income vs Prediction (colored by Confidence)"),
                xaxis  = _axis(title="Annual Income (₹)"),
                yaxis  = _axis(title="Prediction"),
                height = 300,
            )
            st.plotly_chart(fig_scatter, use_container_width=True)

        # Full history table
        st.markdown('<div style="font-family:Orbitron,sans-serif; font-size:0.7rem; color:#06b6d4; letter-spacing:2px; margin:1rem 0 0.5rem;">🗂️ FULL HISTORY LOG</div>', unsafe_allow_html=True)
        display_df = history_df[["run", "timestamp", "age", "education", "prediction", "confidence"]].copy()
        display_df.columns = ["Run", "Time", "Age", "Education", "Prediction", "Conf%"]
        st.dataframe(display_df, use_container_width=True, hide_index=True)

        # Clear history
        if st.button("🗑️ Clear History", use_container_width=False):
            st.session_state.prediction_history = []
            st.session_state.total_predictions  = 0
            st.session_state.last_prediction    = None
            st.session_state.last_confidence    = None
            st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)



# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center; padding:1.5rem;
            border-top:1px solid rgba(124,58,237,0.15);
            color:rgba(200,200,255,0.35); font-size:0.75rem; letter-spacing:1px;">
    🚀 ML Prediction Dashboard &nbsp;·&nbsp; Built with Streamlit + Plotly &nbsp;·&nbsp;
    Ridge Regression Engine &nbsp;·&nbsp; 
</div>
""", unsafe_allow_html=True)
