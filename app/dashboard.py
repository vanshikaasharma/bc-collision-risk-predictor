# Streamlit dashboard — BC collision risk explorer
import json
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# --- Theme ---
COLORS = {
    "bg": "#0f1419",
    "card": "#1a2332",
    "accent": "#3b82f6",
    "accent2": "#10b981",
    "warn": "#f59e0b",
    "danger": "#ef4444",
    "text": "#e2e8f0",
    "muted": "#94a3b8",
}
CHART_TEMPLATE = "plotly_dark"
CHART_COLORS = ["#3b82f6", "#10b981", "#8b5cf6", "#f59e0b", "#ec4899"]

st.set_page_config(
    page_title="BC Collision Risk Predictor",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    f"""
    <style>
    .stApp {{ background: linear-gradient(160deg, {COLORS['bg']} 0%, #1a1f2e 50%, {COLORS['bg']} 100%); }}
    [data-testid="stSidebar"] {{ background: {COLORS['card']}; border-right: 1px solid #2d3a4f; }}
    h1 {{ color: {COLORS['text']}; font-weight: 700; letter-spacing: -0.02em; }}
    h2, h3 {{ color: {COLORS['text']}; }}
    .hero {{ color: {COLORS['muted']}; font-size: 1.05rem; margin-bottom: 1.5rem; }}
    div[data-testid="stMetric"] {{
        background: {COLORS['card']};
        border: 1px solid #2d3a4f;
        border-radius: 12px;
        padding: 1rem 1.25rem;
        box-shadow: 0 4px 14px rgba(0,0,0,0.25);
    }}
    div[data-testid="stMetric"] label {{ color: {COLORS['muted']}; font-size: 0.85rem; }}
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {{
        color: {COLORS['accent']}; font-size: 1.75rem; font-weight: 700;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_clean():
    return pd.read_csv(ROOT / "data/processed/clean.csv")


@st.cache_data
def load_metrics():
    path = ROOT / "models/metrics.json"
    if path.exists():
        return json.loads(path.read_text())
    return None


def chart_layout(fig, height=380):
    fig.update_layout(
        template=CHART_TEMPLATE,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=COLORS["text"], size=12),
        margin=dict(l=20, r=20, t=50, b=20),
        height=height,
        colorway=CHART_COLORS,
    )
    return fig


def risk_gauge(prob):
    pct = prob * 100
    bar_color = COLORS["danger"] if pct >= 66 else COLORS["warn"] if pct >= 33 else COLORS["accent2"]
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=pct,
            number={"suffix": "%", "font": {"size": 42, "color": COLORS["text"]}},
            title={"text": "High-risk probability", "font": {"size": 16, "color": COLORS["muted"]}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": COLORS["muted"]},
                "bar": {"color": bar_color, "thickness": 0.75},
                "bgcolor": COLORS["card"],
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 33], "color": "rgba(16, 185, 129, 0.25)"},
                    {"range": [33, 66], "color": "rgba(245, 158, 11, 0.25)"},
                    {"range": [66, 100], "color": "rgba(239, 68, 68, 0.25)"},
                ],
            },
        )
    )
    fig.update_layout(
        template=CHART_TEMPLATE,
        paper_bgcolor="rgba(0,0,0,0)",
        height=280,
        margin=dict(l=30, r=30, t=40, b=10),
    )
    return fig


# --- Sidebar ---
with st.sidebar:
    st.markdown("### Navigation")
    page = st.radio(
        "Go to",
        ["Overview", "Model results", "Predict risk"],
        label_visibility="collapsed",
    )
    st.divider()
    st.markdown(
        f"<p style='color:{COLORS['muted']}; font-size:0.8rem;'>"
        "ICBC Lower Mainland · 2021–2025<br>"
        "Open Data Licence applies to official ICBC data."
        "</p>",
        unsafe_allow_html=True,
    )

st.title("BC Collision Risk Predictor")
st.markdown(
    '<p class="hero">Explore crash patterns and score high-risk driving contexts using ICBC reported crash data.</p>',
    unsafe_allow_html=True,
)

# --- Overview ---
if page == "Overview":
    df = load_clean()
    casualty = df["severity"].astype(str).str.upper().str.contains("CASUALTY").sum()

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total crashes", f"{len(df):,}")
    m2.metric("Municipalities", df["municipality"].nunique())
    m3.metric("Years", f"{df['year'].min()}–{df['year'].max()}")
    m4.metric("Casualty crashes", f"{casualty:,}")

    left, right = st.columns(2)

    with left:
        by_city = df["municipality"].value_counts().head(12).reset_index()
        by_city.columns = ["municipality", "crashes"]
        fig = px.bar(
            by_city,
            x="crashes",
            y="municipality",
            orientation="h",
            title="Top municipalities",
            color="crashes",
            color_continuous_scale=["#1e3a5f", "#3b82f6"],
        )
        fig.update_layout(showlegend=False, coloraxis_showscale=False)
        st.plotly_chart(chart_layout(fig), use_container_width=True)

    with right:
        by_hour = df["hour"].value_counts().sort_index().reset_index()
        by_hour.columns = ["hour", "crashes"]
        fig = px.area(
            by_hour,
            x="hour",
            y="crashes",
            title="Crashes by time of day",
            markers=True,
        )
        fig.update_traces(fillcolor="rgba(59, 130, 246, 0.35)", line_color=COLORS["accent"])
        st.plotly_chart(chart_layout(fig), use_container_width=True)

    st.markdown("#### Collision types")
    by_type = df["collision_type"].value_counts().head(8).reset_index()
    by_type.columns = ["collision_type", "count"]
    fig = px.pie(
        by_type,
        names="collision_type",
        values="count",
        hole=0.45,
        color_discrete_sequence=CHART_COLORS,
    )
    st.plotly_chart(chart_layout(fig, height=340), use_container_width=True)

# --- Model results ---
elif page == "Model results":
    metrics = load_metrics()
    if not metrics:
        st.warning("Run `python src/train_models.py` first to generate model metrics.")
    else:
        st.success(
            f"Best model on test set: **{metrics['best_model']}** · ROC-AUC **{metrics['best_roc_auc']}**"
        )
        comp = pd.DataFrame(metrics["comparison"])

        c1, c2 = st.columns([1, 1])
        with c1:
            st.dataframe(
                comp.style.background_gradient(subset=["roc_auc"], cmap="Blues"),
                hide_index=True,
                use_container_width=True,
            )
        with c2:
            comp_sorted = comp.sort_values("roc_auc", ascending=True)
            fig = px.bar(
                comp_sorted,
                x="roc_auc",
                y="model",
                orientation="h",
                title="Model comparison (ROC-AUC)",
                color="roc_auc",
                color_continuous_scale=["#334155", "#3b82f6", "#10b981"],
            )
            fig.update_layout(showlegend=False, coloraxis_showscale=False)
            st.plotly_chart(chart_layout(fig), use_container_width=True)

        st.info(
            "All scores are on the **held-out 20% test split**. "
            "The best model (excluding Dummy baseline) is saved to `models/risk_model.joblib`."
        )

# --- Predict ---
else:
    import joblib

    model_path = ROOT / "models/risk_model.joblib"
    if not model_path.exists():
        st.warning("Run `python src/train_models.py` first.")
        st.stop()

    bundle = joblib.load(model_path)
    model = bundle["model"]
    encoder = bundle["encoder"]
    feature_cols = bundle["feature_cols"]
    model_name = bundle.get("model_name", "Model")

    df = load_clean()

    left, right = st.columns([1, 1])

    with left:
        st.markdown("#### Scenario")
        c1, c2 = st.columns(2)
        municipality = c1.selectbox("Municipality", sorted(df["municipality"].unique()))
        collision_type = c2.selectbox("Collision type", sorted(df["collision_type"].unique()))
        region = st.selectbox("Region", sorted(df["region"].unique()))
        c3, c4, c5 = st.columns(3)
        year = c3.selectbox("Year", sorted(df["year"].unique()))
        month = c4.selectbox("Month", list(range(1, 13)), index=5)
        hour = c5.selectbox("Hour", sorted(df["hour"].unique()), index=len(df["hour"].unique()) // 2)
        intersection = st.selectbox("Intersection crash", ["Yes", "No"])

    with right:
        st.markdown(f"#### Prediction · {model_name}")
        row = pd.DataFrame([{
            "municipality": municipality,
            "year": year,
            "month": month,
            "hour": hour,
            "collision_type": collision_type,
            "intersection_crash": intersection,
            "region": region,
        }])
        X = encoder.transform(row[feature_cols])
        prob = float(model.predict_proba(X)[0, 1])
        st.plotly_chart(risk_gauge(prob), use_container_width=True)

        if prob >= 0.66:
            st.error("High-risk context — elevated crash severity pattern for this combination.")
        elif prob >= 0.33:
            st.warning("Moderate risk context.")
        else:
            st.success("Lower risk context relative to training data.")

st.divider()
st.caption(
    "Contains information licensed under ICBC's Open Data Licence when using official ICBC data. "
    "For research and portfolio use; not operational road safety advice."
)
