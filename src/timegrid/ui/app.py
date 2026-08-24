"""Streamlit demo: reconciliation leaderboard, forecast explorer, what-if planner."""

from __future__ import annotations

import os

import httpx
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

API_URL = os.environ.get("TIMEGRID_API_URL", "http://localhost:8450")

st.set_page_config(page_title="timegrid", page_icon="🗓️", layout="wide")
st.title("🗓️ timegrid")
st.caption(
    "Hierarchical KPI forecasting: base forecasts disagree with themselves — reconciliation fixes it"
)


def _ok() -> bool:
    try:
        return httpx.get(f"{API_URL}/health", timeout=3).status_code == 200
    except httpx.HTTPError:
        return False


if not _ok():
    st.error(f"API not reachable at {API_URL}. Start it with `make api`.")
    st.stop()

tab_board, tab_forecast, tab_whatif = st.tabs(
    ["🏆 Leaderboard", "📈 Forecast explorer", "🎛️ What-if"]
)

with tab_board:
    m = httpx.get(f"{API_URL}/leaderboard", timeout=30).json()["metrics"]
    rows = []
    for method in ["base", "bottom_up", "top_down", "ols"]:
        rows.append(
            {
                "method": method,
                "MAPE total": m[f"mape_total_{method}"],
                "MAPE region": m[f"mape_region_{method}"],
                "MAPE bottom": m[f"mape_bottom_{method}"],
                "coherence error": m[f"coherence_{method}"],
            }
        )
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    st.caption(
        "Base forecasts violate the hierarchy (nonzero coherence error); every "
        "reconciliation method restores exact coherence — the leaderboard shows what "
        "each one costs or gains in accuracy per level."
    )

with tab_forecast:
    method = st.selectbox("Reconciliation method", ["ols", "bottom_up", "top_down", "base"])
    body = httpx.get(f"{API_URL}/forecast", params={"method": method}, timeout=60).json()
    h = httpx.get(f"{API_URL}/hierarchy", timeout=30).json()
    pick = st.selectbox("Series", ["total", *h["regions"], *h["bottom"]])
    fig = go.Figure()
    tail = body["history_tail"]
    fig.add_trace(go.Scatter(x=tail["week"], y=tail[pick], name="history", mode="lines"))
    fig.add_trace(
        go.Scatter(
            x=body["weeks"],
            y=body["forecasts"][pick],
            name="forecast",
            mode="lines+markers",
            line={"dash": "dash"},
        )
    )
    fig.update_layout(height=420, xaxis_title="Week", yaxis_title=pick)
    st.plotly_chart(fig, use_container_width=True)
    st.metric("Coherence error", f"{body['coherence_error']:.4f}")

with tab_whatif:
    h = httpx.get(f"{API_URL}/hierarchy", timeout=30).json()
    series = st.selectbox("Bottom series to boost", h["bottom"])
    uplift = st.slider("Uplift %", -30, 60, 10)
    weeks = st.slider("For how many weeks", 1, 12, 8)
    if st.button("Propagate", type="primary"):
        body = httpx.post(
            f"{API_URL}/whatif",
            json={"series": series, "uplift_pct": uplift, "weeks": weeks},
            timeout=60,
        ).json()
        impact = body["impact"]
        cols = st.columns(len(impact))
        for col, (name, value) in zip(cols, impact.items(), strict=True):
            col.metric(name, f"{value:+,.0f}")
        st.caption(f"Coherence after propagation: {body['coherence_error']:.6f} (exact)")
