from __future__ import annotations
import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from safora.config import TRENDS_PATH, ANILIST_PATH, REGIONAL_PATH, MODEL_PATH, METRICS_PATH
from safora.features import validate_trends
from safora.modeling import predict_latest
from safora.scoring import opportunity_scores
from safora.explainability import top_global_features, explain_forecast

st.set_page_config(page_title="SAFORA | Market Intelligence", page_icon="🌏", layout="wide")

st.markdown("""
<style>
.main {background: #f7f8fb;}
.hero {padding: 1.2rem 1.5rem; border-radius: 18px;
background: linear-gradient(120deg,#102b46,#174c45); color:white; margin-bottom:1rem;}
.hero h1 {color:white; margin-bottom:.25rem;}
.hero p {color:#dbe8e8; margin:0;}
.small-note {color:#64748b;font-size:.85rem;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
<h1>SAFORA 2.0</h1>
<p>AI-powered market intelligence for Japanese media entering India</p>
</div>
""", unsafe_allow_html=True)

pages = ["Overview", "TrendRadar", "WhyScore", "Regional Intelligence", "Launch Simulator", "Data & Model"]
page = st.sidebar.radio("Navigate", pages)
st.sidebar.caption("Predictions are demand proxies, not guaranteed commercial outcomes.")

def load_csv(path):
    try:
        return pd.read_csv(path) if Path(path).exists() else pd.DataFrame()
    except Exception as exc:
        st.error(f"Could not read {path.name}: {exc}")
        return pd.DataFrame()

trends = load_csv(TRENDS_PATH)
metadata = load_csv(ANILIST_PATH)
regional = load_csv(REGIONAL_PATH)
metrics = {}
if METRICS_PATH.exists():
    try:
        metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    except Exception:
        metrics = {}

trained = MODEL_PATH.exists() and bool(metrics)
forecasts = pd.DataFrame()
if trained and not trends.empty:
    try:
        forecasts = predict_latest()
    except Exception as exc:
        st.sidebar.warning(f"Forecast unavailable: {exc}")

ranked = opportunity_scores(forecasts, metadata if not metadata.empty else None) if not forecasts.empty else pd.DataFrame()

if page == "Overview":
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Catalogue titles", f"{len(metadata):,}")
    c2.metric("Trend observations", f"{len(trends):,}")
    c3.metric("Forecastable titles", f"{len(forecasts):,}")
    c4.metric("Regional records", f"{len(regional):,}")
    st.subheader("Market opportunity ranking")
    if ranked.empty:
        st.info("No trained forecast available yet. Collect time-series data and run `python -m safora train`.")
    else:
        cols = [c for c in ["title", "latest_interest", "forecast_interest", "change_vs_latest", "direction", "opportunity_score"] if c in ranked.columns]
        st.dataframe(ranked[cols], use_container_width=True, hide_index=True)
        st.caption("Opportunity score is a transparent relative heuristic (70% forecast level, 30% momentum), not a probability of success.")
        fig = px.bar(ranked.head(15), x="title", y="forecast_interest", color="direction",
                     title="Forecast search-interest index (four-week horizon)")
        st.plotly_chart(fig, use_container_width=True)
    st.subheader("Data health")
    st.write({
        "AniList metadata available": not metadata.empty,
        "Trends available": not trends.empty,
        "Trained model available": trained,
        "Regional data available": not regional.empty,
    })
    st.warning("Catalogue popularity and search interest are different signals. Do not describe metadata popularity as Indian demand.")

elif page == "TrendRadar":
    st.subheader("TrendRadar — demand forecast")
    if trends.empty:
        st.info("No trends.csv found. Collect Trends data or import a CSV in Data & Model.")
    else:
        try:
            clean = validate_trends(trends)
            title = st.selectbox("Choose a title", sorted(clean["title"].unique()))
            series = clean[clean["title"] == title].sort_values("date")
            fig = px.line(series, x="date", y="search_interest", title=f"Observed Google Trends index — {title}")
            st.plotly_chart(fig, use_container_width=True)
            if not forecasts.empty and title in forecasts["title"].values:
                row = forecasts[forecasts["title"] == title].iloc[0]
                a, b, c = st.columns(3)
                a.metric("Latest observed index", f"{row['latest_interest']:.1f}")
                b.metric("Forecast index", f"{row['forecast_interest']:.1f}")
                c.metric("Change", f"{row['change_vs_latest']:+.1f}")
                st.write("Forecast direction:", row["direction"])
            else:
                st.info("Not enough history or no trained model for this title.")
        except Exception as exc:
            st.error(str(exc))

elif page == "WhyScore":
    st.subheader("WhyScore — explainability and evidence")
    if forecasts.empty or not metrics:
        st.info("Train a model first to see model-level explanations.")
    else:
        importance = top_global_features(metrics, 12)
        if not importance.empty:
            st.write("Global permutation importance (held-out test set)")
            st.plotly_chart(px.bar(importance.sort_values("importance"), x="importance", y="feature", orientation="h"), use_container_width=True)
        if not ranked.empty:
            title = st.selectbox("Explain title", ranked["title"].tolist())
            row = ranked[ranked["title"] == title].iloc[0]
            for statement in explain_forecast(row, metrics):
                st.write("• " + statement)
            st.caption("Global permutation importance describes predictive usefulness on this test set; it is not a causal explanation for an individual title.")
        st.json({
            "model": metrics.get("model"),
            "target": metrics.get("target"),
            "model_metrics": metrics.get("model_metrics"),
            "naive_baseline_metrics": metrics.get("naive_baseline_metrics"),
            "beats_baseline_on_mae": metrics.get("beats_baseline_on_mae")
        })

elif page == "Regional Intelligence":
    st.subheader("Regional demand intelligence")
    if regional.empty:
        st.info("No regional_trends.csv found. Import real regional observations to activate this section.")
        st.markdown("Required columns: `title`, `region`, `search_interest`; optional `date`, `geo`.")
        st.warning("This panel intentionally does not fabricate state-level demand.")
    else:
        required = {"title", "region", "search_interest"}
        if not required.issubset(regional.columns):
            st.error(f"Regional file requires columns: {sorted(required)}")
        else:
            regional["search_interest"] = pd.to_numeric(regional["search_interest"], errors="coerce")
            title = st.selectbox("Title", sorted(regional["title"].dropna().unique()))
            view = regional[regional["title"] == title].dropna(subset=["search_interest"])
            view = view.groupby("region", as_index=False)["search_interest"].mean()
            st.plotly_chart(px.bar(view.sort_values("search_interest", ascending=False),
                                   x="region", y="search_interest", title=f"Observed regional index — {title}"),
                            use_container_width=True)
            st.caption("Regional indices may be normalized independently by the source. Compare only when collection methodology supports it.")

elif page == "Launch Simulator":
    st.subheader("LaunchSimulator — transparent scenario planning")
    st.caption("This is an assumption-based planning tool, not a causal marketing model.")
    title = st.selectbox("Campaign title", ranked["title"].tolist() if not ranked.empty else ["Example title — no model output"])
    base = float(ranked.loc[ranked["title"] == title, "forecast_interest"].iloc[0]) if not ranked.empty and title in ranked["title"].values else 50.0
    current_lang = st.selectbox("Localization scenario", ["Subtitles first", "Hindi dubbing", "Regional-language dubbing", "Multi-language release"])
    spend = st.slider("Hypothetical campaign intensity", 0, 100, 30, help="An assumption index, not rupees or measured spend.")
    uplift = st.slider("Assumed uplift from campaign (%)", 0, 50, 10)
    localization = st.slider("Assumed localization fit adjustment (%)", -20, 30, 5)
    scenario = max(0.0, min(100.0, base * (1 + uplift / 100) * (1 + localization / 100)))
    a, b, c = st.columns(3)
    a.metric("Baseline forecast index", f"{base:.1f}")
    b.metric("Scenario index", f"{scenario:.1f}")
    c.metric("Difference", f"{scenario-base:+.1f}")
    st.write(f"Scenario: **{current_lang}**")
    st.write("Assumptions used:")
    st.write(f"- Campaign uplift: {uplift}% (user-entered assumption)")
    st.write(f"- Localization adjustment: {localization}% (user-entered assumption)")
    st.write(f"- Campaign intensity index: {spend}/100 (not a validated spend-response curve)")
    st.warning("Do not present this as a trained causal estimate. Replace assumptions with measured campaign outcomes when available.")

elif page == "Data & Model":
    st.subheader("Data ingestion and model status")
    st.write("### Current files")
    st.write({
        "metadata": str(ANILIST_PATH),
        "trends": str(TRENDS_PATH),
        "regional": str(REGIONAL_PATH),
        "model": str(MODEL_PATH),
    })
    st.write("### Import canonical trend CSV")
    uploaded = st.file_uploader("Upload CSV with date, title, search_interest columns", type=["csv"])
    if uploaded is not None:
        incoming = pd.read_csv(uploaded)
        try:
            validated = validate_trends(incoming)
            if st.button("Append validated observations to trends.csv"):
                if TRENDS_PATH.exists():
                    existing = pd.read_csv(TRENDS_PATH)
                    merged = pd.concat([existing, validated], ignore_index=True)
                else:
                    merged = validated
                merged = validate_trends(merged)
                merged.to_csv(TRENDS_PATH, index=False)
                st.success(f"Saved {len(merged)} normalized observations.")
                st.rerun()
            st.dataframe(validated.head(20), use_container_width=True)
        except Exception as exc:
            st.error(str(exc))
    st.write("### Training metrics")
    if metrics:
        st.json(metrics)
    else:
        st.info("No metrics yet. Run `python -m safora train --horizon 4` after collecting sufficient data.")
    st.write("### Metadata")
    if not metadata.empty:
        st.dataframe(metadata.head(30), use_container_width=True)
