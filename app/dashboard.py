# BC Road Risk Intelligence Platform — Streamlit dashboard
import json
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Premium earthy palette
THEME = {
    "primary": "#52352D",
    "sage": "#B1D4D0",
    "bg": "#F5F1EA",
    "card": "#FFFFFF",
    "surface": "#EEE8DF",
    "text": "#52352D",
    "muted": "#6B5B54",
    "border": "#D9D0C4",
    "warn": "#9A7B4F",
    "danger": "#8B4A3A",
    "safe": "#6E9A94",
}
CHART_COLORS = ["#52352D", "#B1D4D0", "#8A6F63", "#6E8B87", "#A89088", "#C4A99A"]

PAGES = [
    "Executive Summary",
    "Collision Hotspot Intelligence Map",
    "Risk Prediction Center",
    "Municipality Analytics",
]

st.set_page_config(
    page_title="BC Road Risk Intelligence",
    page_icon="🛣️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    .stApp {{
        background: {THEME['bg']};
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }}
    [data-testid="stSidebar"] {{
        background: {THEME['surface']};
        border-right: 1px solid {THEME['border']};
    }}
    [data-testid="stSidebar"] .stRadio label {{
        color: {THEME['text']};
        font-weight: 500;
    }}
    h1, h2, h3, h4 {{
        color: {THEME['primary']} !important;
        font-weight: 600 !important;
        letter-spacing: -0.02em;
    }}
    p, .stCaption, [data-testid="stMarkdownContainer"] p {{
        color: {THEME['muted']};
    }}
    h1 {{
        color: {THEME['primary']} !important;
        font-size: 2rem !important;
        font-weight: 700 !important;
        letter-spacing: -0.03em !important;
        padding: 0 !important;
        margin-bottom: 0.15rem !important;
    }}
    .page-subtitle {{
        color: {THEME['muted']};
        font-size: 1rem;
        margin: 0 0 1.5rem 0;
        line-height: 1.45;
    }}
    .section-card {{
        background: {THEME['card']};
        border: 1px solid {THEME['border']};
        border-radius: 20px;
        padding: 1rem 1.25rem;
        box-shadow: 0 4px 24px rgba(82, 53, 45, 0.06);
        margin-bottom: 1rem;
    }}
    .kpi-card {{
        background: {THEME['card']};
        border: 1px solid {THEME['border']};
        border-radius: 20px;
        padding: 1.25rem 1.5rem;
        box-shadow: 0 4px 24px rgba(82, 53, 45, 0.06);
        min-height: 110px;
    }}
    .kpi-label {{
        color: {THEME['muted']};
        font-size: 0.78rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }}
    .kpi-value {{
        color: {THEME['primary']};
        font-size: 1.85rem;
        font-weight: 700;
        margin: 0.35rem 0 0.15rem;
    }}
    .kpi-hint {{
        color: {THEME['muted']};
        font-size: 0.8rem;
        line-height: 1.35;
    }}
    .chart-card {{
        background: {THEME['card']};
        border: 1px solid {THEME['border']};
        border-radius: 20px;
        padding: 0.75rem 0.5rem 0.25rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 24px rgba(82, 53, 45, 0.06);
    }}
    .filter-panel {{
        background: {THEME['card']};
        border: 1px solid {THEME['border']};
        border-radius: 20px;
        padding: 1.25rem 1rem;
        box-shadow: 0 4px 24px rgba(82, 53, 45, 0.06);
    }}
    .filter-panel h4 {{
        color: {THEME['primary']} !important;
        font-size: 0.95rem !important;
        margin: 1rem 0 0.5rem !important;
    }}
    .muni-card {{
        background: {THEME['card']};
        border: 1px solid {THEME['border']};
        border-radius: 20px;
        padding: 1rem 1.1rem;
        box-shadow: 0 4px 20px rgba(82, 53, 45, 0.06);
        margin-bottom: 1rem;
        min-height: 220px;
    }}
    .muni-name {{
        color: {THEME['primary']};
        font-weight: 700;
        font-size: 1.05rem;
        margin: 0 0 0.5rem;
    }}
    .muni-stat {{
        color: {THEME['muted']};
        font-size: 0.78rem;
        margin: 0.15rem 0;
    }}
    .muni-stat b {{
        color: {THEME['primary']};
    }}
    .map-legend {{
        color: {THEME['muted']};
        font-size: 0.8rem;
        margin-top: 0.5rem;
    }}
    div[data-testid="stMetric"] {{
        background: {THEME['card']};
        border: 1px solid {THEME['border']};
        border-radius: 20px;
        padding: 1rem 1.25rem;
        box-shadow: 0 4px 20px rgba(82, 53, 45, 0.05);
    }}
    div[data-testid="stMetric"] label {{ color: {THEME['muted']}; }}
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {{
        color: {THEME['primary']};
    }}
    .stTabs [data-baseweb="tab-list"] {{
        gap: 8px;
        background: transparent;
    }}
    .stTabs [data-baseweb="tab"] {{
        background: {THEME['surface']};
        border-radius: 12px;
        color: {THEME['text']};
        padding: 0.5rem 1rem;
    }}
    .stAlert {{
        border-radius: 16px;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_clean():
    return pd.read_csv(ROOT / "data/processed/clean.csv")


@st.cache_data
def load_training():
    path = ROOT / "data/processed/training_data.csv"
    if path.exists():
        return pd.read_csv(path)
    return None


@st.cache_data
def load_metrics():
    path = ROOT / "models/metrics.json"
    if path.exists():
        return json.loads(path.read_text())
    return None


@st.cache_data
def geo_crash_data():
    df = load_clean()
    if "latitude" not in df.columns or "longitude" not in df.columns:
        return None
    geo = df.dropna(subset=["latitude", "longitude"]).copy()
    geo = geo[
        (geo["latitude"].between(48.5, 51.0))
        & (geo["longitude"].between(-125.5, -120.5))
    ]
    return geo


def top_predicted_contexts(n=10):
    """Highest historical risk_score groups — shown as prediction table."""
    td = load_training()
    if td is None:
        return pd.DataFrame()
    top = td.sort_values("risk_score", ascending=False).head(n).copy()
    top["time_of_day"] = top["hour"].map(format_hour)
    top["location"] = top["municipality"] + " · " + top["street"].astype(str).str[:40]
    return top


def page_heading(title: str, subtitle: str = ""):
    """Native Streamlit h1 + optional subtitle on page background (not in a card)."""
    st.title(title)
    if subtitle:
        st.markdown(f'<p class="page-subtitle">{subtitle}</p>', unsafe_allow_html=True)


def format_hour(h) -> str:
    """24h ICBC hour → readable label (e.g. 16 → 4 PM)."""
    h = int(h)
    if h == 0:
        return "12 AM"
    if h < 12:
        return f"{h} AM"
    if h == 12:
        return "12 PM"
    return f"{h - 12} PM"


def add_hour_labels(frame, hour_col="hour"):
    out = frame.copy()
    out["time_label"] = out[hour_col].map(format_hour)
    return out


def kpi_row_html(items):
    """items: list of (label, value, hint)"""
    cols = st.columns(len(items))
    for col, (label, value, hint) in zip(cols, items):
        with col:
            st.markdown(
                f"""
                <div class="kpi-card">
                    <div class="kpi-label">{label}</div>
                    <div class="kpi-value">{value}</div>
                    <div class="kpi-hint">{hint}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def chart_layout(fig, height=380):
    """Force dark readable text on light background (Plotly defaults are too light)."""
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=THEME["primary"], family="Inter, sans-serif", size=12),
        margin=dict(l=24, r=24, t=48, b=24),
        height=height,
        colorway=CHART_COLORS,
        title=dict(font=dict(color=THEME["primary"], size=14)),
        legend=dict(
            font=dict(color=THEME["primary"], size=11),
            bgcolor=THEME["card"],
            bordercolor=THEME["border"],
        ),
    )
    axis_font = dict(color=THEME["primary"], size=11)
    title_font = dict(color=THEME["primary"], size=12)
    fig.update_xaxes(
        tickfont=axis_font,
        title_font=title_font,
        color=THEME["primary"],
        gridcolor="#E8E2D8",
        linecolor=THEME["border"],
    )
    fig.update_yaxes(
        tickfont=axis_font,
        title_font=title_font,
        color=THEME["primary"],
        gridcolor="#E8E2D8",
        linecolor=THEME["border"],
    )
    return fig


def style_pie(fig):
    fig.update_traces(
        textfont=dict(color=THEME["primary"], size=11),
        textposition="inside",
    )
    fig.update_layout(
        legend=dict(
            font=dict(color=THEME["primary"], size=11),
            bgcolor=THEME["card"],
        ),
    )
    return fig


def show_chart(fig, height=380, pie=False):
    fig = chart_layout(fig, height=height)
    if pie:
        fig = style_pie(fig)
    with st.container():
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)


def risk_gauge(prob):
    pct = prob * 100
    bar_color = THEME["danger"] if pct >= 66 else THEME["warn"] if pct >= 33 else THEME["safe"]
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=pct,
            number={"suffix": "%", "font": {"size": 40, "color": THEME["primary"]}},
            title={
                "text": "High-risk context probability",
                "font": {"size": 14, "color": THEME["muted"]},
            },
            gauge={
                "axis": {"range": [0, 100], "tickcolor": THEME["muted"]},
                "bar": {"color": bar_color, "thickness": 0.72},
                "bgcolor": THEME["surface"],
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 33], "color": "rgba(177, 212, 208, 0.45)"},
                    {"range": [33, 66], "color": "rgba(154, 123, 79, 0.35)"},
                    {"range": [66, 100], "color": "rgba(139, 74, 58, 0.35)"},
                ],
            },
        )
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        height=300,
        margin=dict(l=28, r=28, t=48, b=12),
        font=dict(color=THEME["text"]),
    )
    return fig


def map_layout(fig, height=560):
    fig.update_layout(
        map_style="carto-positron",
        margin=dict(l=0, r=0, t=24, b=0),
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color=THEME["text"]),
        legend=dict(bgcolor=THEME["card"], bordercolor=THEME["border"]),
    )
    return fig


# —— Sidebar ——
with st.sidebar:
    st.markdown(
        f"<p style='color:{THEME['primary']}; font-weight:700; font-size:1.1rem; "
        f"margin-bottom:0.25rem;'>BC Road Risk</p>"
        f"<p style='color:{THEME['muted']}; font-size:0.75rem; margin-top:0;'>Intelligence Platform</p>",
        unsafe_allow_html=True,
    )
    st.divider()
    page = st.radio("Navigation", PAGES, label_visibility="collapsed")
    st.divider()
    metrics = load_metrics()
    if metrics and metrics.get("best_roc_auc"):
        st.markdown(
            f"<p style='color:{THEME['muted']}; font-size:0.8rem;'>"
            f"<b style='color:{THEME['primary']}'>Model</b><br>"
            f"{metrics.get('best_model', 'XGBoost')} · ROC-AUC {metrics['best_roc_auc']}<br><br>"
            f"ICBC Lower Mainland · 2021–2025<br>"
            f"Open Data Licence applies."
            f"</p>",
            unsafe_allow_html=True,
        )
    else:
        st.caption("ICBC Lower Mainland · 2021–2025 · Open Data Licence applies.")


# —— Executive Summary ——
if page == "Executive Summary":
    page_heading(
        "British Columbia Road Risk Intelligence Platform",
        "Executive Summary · ICBC Lower Mainland crash analytics (2021–2025)",
    )
    df = load_clean()
    casualty = df["severity"].astype(str).str.upper().str.contains("CASUALTY").sum()
    cas_pct = 100 * casualty / len(df)
    m = load_metrics()
    auc_txt = f"{m['best_roc_auc']:.2f}" if m and m.get("best_roc_auc") else "—"

    kpi_row_html([
        ("Total reported crashes", f"{len(df):,}", "ICBC records in scope"),
        ("Municipalities", str(df["municipality"].nunique()), "Lower Mainland"),
        ("Casualty share", f"{cas_pct:.1f}%", f"{casualty:,} casualty-related"),
        ("Forecast model quality", auc_txt, "ROC-AUC · trained ≤2023, tested on 2024"),
    ])

    st.subheader("Key findings")
    top_city = df["municipality"].value_counts().index[0]
    top_n = int(df["municipality"].value_counts().iloc[0])
    peak_h = int(df["hour"].mode().iloc[0])
    by_year = df["year"].value_counts().sort_index()
    year_note = ""
    if len(by_year) >= 2:
        year_note = (
            f" Reported volume grew from **{int(by_year.iloc[0]):,}** ({int(by_year.index[0])}) "
            f"to **{int(by_year.iloc[-1]):,}** ({int(by_year.index[-1])})."
        )
    auc_note = ""
    if m and m.get("best_roc_auc"):
        auc_note = (
            f" The classifier scores **~{m['best_roc_auc']:.2f} ROC-AUC** on next-year holdout "
            "(train through 2023, test 2024)."
        )
    findings = [
        f"**{top_city}** leads crash volume (**{top_n:,}** records) — risk is concentrated in major corridors, not spread evenly.",
        f"Peak reporting hour is **{format_hour(peak_h)}**; evening and afternoon periods dominate the hourly profile.",
        f"**{cas_pct:.1f}%** of crashes are casualty-related; models weight injury crashes more than property-only (PDO).{year_note}",
        f"Models rank **street + time + crash-type** contexts — they do **not** predict whether a specific driver will crash.{auc_note}",
    ]
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    for line in findings:
        st.markdown(f"- {line}")
    st.markdown("</div>", unsafe_allow_html=True)

    left, right = st.columns(2)
    with left:
        by_city = df["municipality"].value_counts().head(12).reset_index()
        by_city.columns = ["municipality", "crashes"]
        fig = px.bar(
            by_city,
            x="crashes",
            y="municipality",
            orientation="h",
            title="Top municipalities by crash volume",
            color="crashes",
            color_continuous_scale=[THEME["sage"], THEME["primary"]],
        )
        fig.update_layout(showlegend=False, coloraxis_showscale=False)
        show_chart(fig)

    with right:
        by_hour = df["hour"].value_counts().sort_index().reset_index()
        by_hour.columns = ["hour", "crashes"]
        by_hour = add_hour_labels(by_hour)
        fig = px.area(
            by_hour,
            x="time_label",
            y="crashes",
            title="Crashes by time of day",
            markers=True,
        )
        fig.update_traces(
            fillcolor="rgba(177, 212, 208, 0.5)",
            line_color=THEME["primary"],
            line_width=2,
        )
        show_chart(fig)

    c1, c2 = st.columns([1, 1])
    with c1:
        by_year = df["year"].value_counts().sort_index().reset_index()
        by_year.columns = ["year", "crashes"]
        fig = px.line(
            by_year,
            x="year",
            y="crashes",
            title="Reported crashes by year",
            markers=True,
        )
        fig.update_traces(line_color=THEME["primary"], marker=dict(size=8))
        show_chart(fig, height=320)

    with c2:
        by_type = df["collision_type"].value_counts().head(8).reset_index()
        by_type.columns = ["collision_type", "count"]
        fig = px.pie(
            by_type,
            names="collision_type",
            values="count",
            hole=0.5,
            title="Collision type mix",
            color_discrete_sequence=CHART_COLORS,
        )
        show_chart(fig, height=320, pie=True)


# —— Hotspot Map ——
elif page == "Collision Hotspot Intelligence Map":
    page_heading(
        "Collision Hotspot Intelligence Map",
        "Geospatial view of ICBC-reported crashes with coordinates",
    )

    geo = geo_crash_data()
    if geo is None:
        st.warning(
            "No coordinates in `clean.csv`. Re-run: `python src/check_data.py`"
        )
        st.stop()

    with st.container():
        fc1, fc2, fc3, fc4 = st.columns(4)
        cities = ["All"] + sorted(geo["municipality"].unique().tolist())
        years = ["All"] + sorted(geo["year"].unique().astype(int).tolist())
        sev_opts = ["All", "Casualty only", "Property damage only"]

        sel_city = fc1.selectbox("Municipality", cities)
        sel_year = fc2.selectbox("Year", years)
        sel_sev = fc3.selectbox("Severity", sev_opts)

        street_opts = ["All"]
        if sel_city != "All" and "street" in geo.columns:
            city_streets = geo.loc[geo["municipality"] == sel_city, "street"].dropna().unique()
            street_opts += sorted(
                [s for s in city_streets if str(s).upper() not in ("UNKNOWN", "NAN")]
            )[:200]
        sel_street = fc4.selectbox("Street", street_opts)

    filtered = geo.copy()
    if sel_city != "All":
        filtered = filtered[filtered["municipality"] == sel_city]
    if sel_year != "All":
        filtered = filtered[filtered["year"] == sel_year]
    if sel_sev == "Casualty only":
        filtered = filtered[
            filtered["severity"].astype(str).str.upper().str.contains("CASUALTY")
        ]
    elif sel_sev == "Property damage only":
        filtered = filtered[
            filtered["severity"].astype(str).str.upper().str.contains("PROPERTY")
        ]
    if sel_street != "All":
        filtered = filtered[filtered["street"] == sel_street]

    if len(filtered) == 0:
        st.warning("No crashes match these filters. Try broader filters.")
        st.stop()

    kpi_row_html([
        ("Crashes on map", f"{len(filtered):,}", "Current filter"),
        ("Geocoded records", f"{len(geo):,}", f"of {len(load_clean()):,} total"),
        ("Map sample", f"{min(25_000, len(filtered)):,}", "Max points for performance"),
    ])

    tab1, tab2, tab3 = st.tabs(["Crash density", "Hotspot clusters", "Top corridors"])

    with tab1:
        sample = filtered.sample(n=min(25_000, len(filtered)), random_state=42)
        sample["is_casualty"] = (
            sample["severity"].astype(str).str.upper().str.contains("CASUALTY")
        )
        fig = px.scatter_map(
            sample,
            lat="latitude",
            lon="longitude",
            color="is_casualty",
            color_discrete_map={True: THEME["danger"], False: THEME["primary"]},
            zoom=8,
            opacity=0.4,
            labels={"is_casualty": "Casualty crash"},
        )
        st.plotly_chart(map_layout(fig), use_container_width=True)

    with tab2:
        grid = filtered.copy()
        grid["lat_cell"] = grid["latitude"].round(2)
        grid["lon_cell"] = grid["longitude"].round(2)
        hotspots = (
            grid.groupby(["lat_cell", "lon_cell"])
            .size()
            .reset_index(name="crash_count")
            .sort_values("crash_count", ascending=False)
            .head(300)
        )
        fig = px.scatter_map(
            hotspots,
            lat="lat_cell",
            lon="lon_cell",
            size="crash_count",
            color="crash_count",
            color_continuous_scale=[THEME["sage"], THEME["primary"], THEME["danger"]],
            size_max=28,
            zoom=8,
            labels={"crash_count": "Crashes in cell"},
        )
        st.plotly_chart(map_layout(fig), use_container_width=True)

    with tab3:
        if "street" in filtered.columns:
            streets = (
                filtered["street"]
                .astype(str)
                .replace("", pd.NA)
                .dropna()
            )
            streets = streets[~streets.str.upper().isin(["UNKNOWN", "NAN"])]
            top_streets = streets.value_counts().head(20).reset_index()
            top_streets.columns = ["street", "crashes"]
            st.dataframe(top_streets, hide_index=True, use_container_width=True)
        else:
            st.info("Re-run `python src/check_data.py` to include street names.")


# —— Risk Prediction Center ——
elif page == "Risk Prediction Center":
    page_heading("Risk Prediction Center")

    tab_insights, tab_eval, tab_score = st.tabs(
        ["High-risk contexts", "Model evaluation", "Street risk lookup"]
    )

    with tab_insights:
        left, right = st.columns([1, 1])
        pred_table = top_predicted_contexts(12)
        with left:
            st.markdown(
                f'<p style="color:{THEME["primary"]};font-weight:600;">'
                "Highest severity-weighted contexts in ICBC data</p>",
                unsafe_allow_html=True,
            )
            if len(pred_table) == 0:
                st.warning("Run `python src/Target.py` first.")
            else:
                display = pred_table[
                    ["location", "time_of_day", "collision_type", "risk_score"]
                ].rename(columns={
                    "location": "Location",
                    "time_of_day": "Time",
                    "collision_type": "Crash type",
                    "risk_score": "Activity score",
                })
                st.dataframe(display, hide_index=True, use_container_width=True)

        with right:
            td = load_training()
            if td is not None:
                corridor = (
                    td.groupby(["hour", "month"])["risk_score"]
                    .mean()
                    .reset_index()
                )
                corridor = add_hour_labels(corridor)
                fig = px.area(
                    corridor,
                    x="time_label",
                    y="risk_score",
                    color="month",
                    title="Avg activity score by time of day",
                    color_discrete_sequence=CHART_COLORS,
                )
                fig.update_traces(line_width=1)
                show_chart(fig)

    with tab_eval:
        metrics = load_metrics()
        if not metrics:
            st.warning("Run `python src/train_models.py` first.")
        else:
            primary_year = metrics.get("primary_test_year", 2024)
            with st.container():
                st.markdown('<div class="section-card">', unsafe_allow_html=True)
                if metrics.get("temporal_folds"):
                    st.markdown(
                        f'<p style="color:{THEME["primary"]};font-weight:600;">'
                        "Temporal holdout — classifier</p>",
                        unsafe_allow_html=True,
                    )
                    st.caption("Train on past years → test next year (year excluded from features).")
                    fold_rows = [
                        {
                            "Train years": f["train_years"],
                            "Test year": f["test_year"],
                            "Best model": f["best_model"],
                            "ROC-AUC": f["best_roc_auc"],
                        }
                        for f in metrics["temporal_folds"]
                    ]
                    st.dataframe(pd.DataFrame(fold_rows), hide_index=True, use_container_width=True)

                    primary_fold = next(
                        (f for f in metrics["temporal_folds"] if f["test_year"] == primary_year),
                        metrics["temporal_folds"][0],
                    )
                    comp = pd.DataFrame(primary_fold["comparison"])
                else:
                    comp = pd.DataFrame(metrics["comparison"])

                c1, c2 = st.columns(2)
                with c1:
                    st.dataframe(comp, hide_index=True, use_container_width=True)
                with c2:
                    comp_sorted = comp.sort_values("roc_auc", ascending=True)
                    fig = px.bar(
                        comp_sorted,
                        x="roc_auc",
                        y="model",
                        orientation="h",
                        title=f"Classifier ROC-AUC (test year {primary_year})",
                        color="roc_auc",
                        color_continuous_scale=[THEME["sage"], THEME["primary"]],
                    )
                    fig.update_layout(showlegend=False, coloraxis_showscale=False)
                    show_chart(fig)
                st.markdown("</div>", unsafe_allow_html=True)

            reg_path = ROOT / "models/regressor_metrics.json"
            if reg_path.exists():
                reg_m = json.loads(reg_path.read_text())
                st.markdown('<div class="section-card">', unsafe_allow_html=True)
                st.markdown(
                    f'<p style="color:{THEME["primary"]};font-weight:600;">'
                    "Temporal holdout — regressor</p>",
                    unsafe_allow_html=True,
                )
                if reg_m.get("temporal_folds"):
                    reg_rows = []
                    for f in reg_m["temporal_folds"]:
                        best_row = next(
                            (r for r in f["comparison"] if r["model"] == f["best_model"]),
                            f["comparison"][-1],
                        )
                        reg_rows.append({
                            "Train years": f["train_years"],
                            "Test year": f["test_year"],
                            "R²": f["best_r2"],
                            "MAE": best_row["mae"],
                        })
                    st.dataframe(pd.DataFrame(reg_rows), hide_index=True, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)

    with tab_score:
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
        top_streets = bundle.get("top_streets", [])

        if "street" not in feature_cols or "region" in feature_cols:
            st.error("Re-run pipeline: check_data → Target → train_models.")
            st.stop()

        df = load_clean()
        left, right = st.columns([1, 1])

        with left:
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.markdown(
                f'<p style="color:{THEME["primary"]};font-weight:600;">Scenario inputs</p>',
                unsafe_allow_html=True,
            )
            municipality = st.selectbox("Municipality", sorted(df["municipality"].unique()))
            city_streets = df.loc[df["municipality"] == municipality, "street"].dropna().unique()
            city_streets = sorted(
                [s for s in city_streets if str(s).upper() not in ("UNKNOWN", "NAN")]
            )
            street_options = (
                [s for s in city_streets if s in top_streets] if top_streets else city_streets[:150]
            )
            if not street_options:
                street_options = ["OTHER"]
            street = st.selectbox("Street", street_options)

            c1, c2 = st.columns(2)
            collision_type = c1.selectbox("Collision type", sorted(df["collision_type"].unique()))
            intersection = c2.selectbox("Intersection crash", ["Yes", "No"])

            c3, c4, c5 = st.columns(3)
            year = c3.selectbox("Year", sorted(df["year"].unique()))
            month = c4.selectbox("Month", list(range(1, 13)), index=5)
            hours_sorted = sorted(df["hour"].unique())
            hour = c5.selectbox("Time of day", hours_sorted, format_func=format_hour)
            st.markdown("</div>", unsafe_allow_html=True)

        with right:
            street_for_model = street if street in top_streets else "OTHER"
            row = pd.DataFrame([{
                "municipality": municipality,
                "street": street_for_model,
                "year": year,
                "month": month,
                "hour": hour,
                "collision_type": collision_type,
                "intersection_crash": intersection,
            }])
            X = encoder.transform(row[feature_cols])

            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.markdown(
                f'<p style="color:{THEME["primary"]};font-weight:600;">'
                f"ML output · {model_name}</p>",
                unsafe_allow_html=True,
            )
            prob = float(model.predict_proba(X)[0, 1])
            show_chart(risk_gauge(prob), height=300)

            reg_model_path = ROOT / "models/risk_regressor.joblib"
            if reg_model_path.exists():
                reg_bundle = joblib.load(reg_model_path)
                X_reg = reg_bundle["encoder"].transform(row[reg_bundle["feature_cols"]])
                pred_score = float(reg_bundle["model"].predict(X_reg)[0])
                st.metric("Predicted activity score", f"{pred_score:.1f}")

            st.markdown(
                f"**{municipality}** · {street} · **{format_hour(hour)}** · {collision_type}",
            )
            if prob >= 0.66:
                st.error("High activity context — similar to top-tier ICBC groups.")
            elif prob >= 0.33:
                st.warning("Moderate activity vs other street/time contexts.")
            else:
                st.success("Lower activity vs typical contexts in the data.")
            st.markdown("</div>", unsafe_allow_html=True)


# —— Municipality Analytics ——
elif page == "Municipality Analytics":
    page_heading(
        "Municipality Analytics",
        "Risk profiles and crash statistics by city",
    )

    df = load_clean()
    td = load_training()

    top_muni = df["municipality"].value_counts().head(8).index.tolist()
    sel = st.selectbox("Focus municipality", top_muni)

    sub = df[df["municipality"] == sel]
    cas = sub["severity"].astype(str).str.upper().str.contains("CASUALTY").mean() * 100

    peak_h = int(sub["hour"].mode().iloc[0])
    kpi_row_html([
        ("Crashes", f"{len(sub):,}", sel),
        ("Casualty rate", f"{cas:.1f}%", "Share of records"),
        ("Peak hour", format_hour(peak_h), "Most frequent crash time"),
        (
            "Intersection share",
            f"{(sub['intersection_crash'].astype(str).str.upper() == 'YES').mean() * 100:.0f}%",
            "Yes responses",
        ),
    ])

    c1, c2 = st.columns(2)
    with c1:
        hourly = sub["hour"].value_counts().sort_index().reset_index()
        hourly.columns = ["hour", "crashes"]
        hourly = add_hour_labels(hourly)
        fig = px.bar(
            hourly,
            x="time_label",
            y="crashes",
            title=f"{sel} — crashes by time of day",
            color_discrete_sequence=[THEME["primary"]],
        )
        show_chart(fig)

    with c2:
        if td is not None and "risk_score" in td.columns:
            muni_td = td[td["municipality"] == sel]
            rs = (
                muni_td.groupby("year")["risk_score"]
                .mean()
                .reset_index()
            )
            fig = px.line(
                rs,
                x="year",
                y="risk_score",
                title=f"{sel} — avg severity-weighted group score by year",
                markers=True,
            )
            fig.update_traces(line_color=THEME["sage"], line_width=3)
            show_chart(fig)
        else:
            yearly = sub["year"].value_counts().sort_index().reset_index()
            yearly.columns = ["year", "crashes"]
            fig = px.line(yearly, x="year", y="crashes", markers=True)
            show_chart(fig)

    st.subheader("Municipality comparison")
    compare = df["municipality"].value_counts().head(15).reset_index()
    compare.columns = ["municipality", "crashes"]
    fig = px.bar(
        compare,
        x="municipality",
        y="crashes",
        title="Crash volume by municipality",
        color="crashes",
        color_continuous_scale=[THEME["sage"], THEME["primary"]],
    )
    fig.update_layout(showlegend=False, coloraxis_showscale=False, xaxis_tickangle=-35)
    show_chart(fig, height=400)

st.divider()
st.caption(
    "Contains information licensed under ICBC's Open Data Licence when using official ICBC data. "
    "For research and portfolio use; not operational road safety advice."
)
