from __future__ import annotations

from pathlib import Path

import pandas as pd
import pydeck as pdk
import streamlit as st


st.set_page_config(page_title="Philly CHP Intelligence", layout="wide")

st.markdown(
    """
    <style>
        .stApp {
            background: radial-gradient(circle at 20% 20%, #0e1b2c 0%, #0a111b 45%, #060b12 100%);
            color: #d9e4f2;
        }
        .kpi-card {
            background: linear-gradient(145deg, rgba(24,38,56,0.9), rgba(11,19,31,0.9));
            border: 1px solid rgba(78, 145, 227, 0.3);
            border-radius: 12px;
            padding: 14px 16px;
        }
        .kpi-label {
            font-size: 0.85rem;
            color: #95a9c5;
        }
        .kpi-value {
            font-size: 1.6rem;
            font-weight: 700;
            color: #eaf2ff;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def load_data(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)


processed_path = Path("data/processed/master_buildings_scored.parquet")
if not processed_path.exists():
    st.error("Run `python scripts/run_pipeline.py` first to generate scored data.")
    st.stop()

df = load_data(str(processed_path))

st.title("Philly CHP Intelligence")
st.caption("Commercial and industrial CHP opportunity ranking with deferred maintenance signals")

with st.sidebar:
    st.header("Filters")
    score_floor = st.slider("Minimum final score", min_value=0, max_value=100, value=40)

    type_options = sorted([t for t in df["building_type"].dropna().astype(str).unique().tolist() if t])
    selected_types = st.multiselect("Building Type", options=type_options, default=type_options[:15])

    max_sqft = int(df["square_footage"].fillna(0).max()) if len(df) else 100000
    sqft_range = st.slider("Square Footage", min_value=0, max_value=max(1000, max_sqft), value=(0, max(1000, max_sqft)))

    confidence_options = sorted(df["data_confidence"].dropna().astype(str).unique().tolist()) if "data_confidence" in df.columns else []
    selected_confidence = st.multiselect("Data Confidence", options=confidence_options, default=confidence_options)

filtered = df[df["final_opportunity_score"] >= score_floor].copy()
if selected_types:
    filtered = filtered[filtered["building_type"].astype(str).isin(selected_types)]
filtered = filtered[
    (filtered["square_footage"].fillna(0) >= sqft_range[0])
    & (filtered["square_footage"].fillna(0) <= sqft_range[1])
]
if selected_confidence:
    filtered = filtered[filtered["data_confidence"].astype(str).isin(selected_confidence)]

filtered = filtered.sort_values("final_opportunity_score", ascending=False).reset_index(drop=True)

k1, k2, k3, k4 = st.columns(4)
with k1:
    st.markdown(f"<div class='kpi-card'><div class='kpi-label'>Filtered Buildings</div><div class='kpi-value'>{len(filtered):,}</div></div>", unsafe_allow_html=True)
with k2:
    mean_score = filtered["final_opportunity_score"].mean() if len(filtered) else 0
    st.markdown(f"<div class='kpi-card'><div class='kpi-label'>Avg Opportunity Score</div><div class='kpi-value'>{mean_score:.1f}</div></div>", unsafe_allow_html=True)
with k3:
    high_count = int((filtered["final_opportunity_score"] >= 75).sum()) if len(filtered) else 0
    st.markdown(f"<div class='kpi-card'><div class='kpi-label'>High Priority (75+)</div><div class='kpi-value'>{high_count:,}</div></div>", unsafe_allow_html=True)
with k4:
    deferred_avg = filtered["deferred_maintenance_score"].mean() if len(filtered) else 0
    st.markdown(f"<div class='kpi-card'><div class='kpi-label'>Avg Deferred Need Score</div><div class='kpi-value'>{deferred_avg:.1f}</div></div>", unsafe_allow_html=True)

st.subheader("Ranked Targets")
cols_to_show = [
    "final_opportunity_score",
    "chp_fit_score",
    "deferred_maintenance_score",
    "project_size_score",
    "pursuit_score",
    "building_name",
    "address",
    "building_type",
    "square_footage",
    "benchmark_site_eui",
    "permit_count_5y",
    "unsafe_violation_count_5y",
    "complaint_count_5y",
    "data_confidence",
]
present_cols = [c for c in cols_to_show if c in filtered.columns]
st.dataframe(filtered[present_cols], use_container_width=True, height=360)

export_csv = filtered.to_csv(index=False).encode("utf-8")
st.download_button("Export Filtered CSV", data=export_csv, file_name="philly_chp_targets_filtered.csv", mime="text/csv")

map_df = filtered.dropna(subset=["latitude", "longitude"]).copy()
if len(map_df):
    st.subheader("Philly Opportunity Map")
    map_df["final_opportunity_score"] = map_df["final_opportunity_score"].fillna(0)
    map_df["color_r"] = (255 - (map_df["final_opportunity_score"] * 2.0).clip(0, 200)).astype(int)
    map_df["color_g"] = (80 + (map_df["final_opportunity_score"] * 1.2).clip(0, 175)).astype(int)
    map_df["color_b"] = 90

    layer = pdk.Layer(
        "ScatterplotLayer",
        data=map_df,
        get_position="[longitude, latitude]",
        get_radius=70,
        get_fill_color="[color_r, color_g, color_b, 185]",
        pickable=True,
    )

    view_state = pdk.ViewState(latitude=39.9526, longitude=-75.1652, zoom=11, pitch=0)
    tooltip = {
        "html": "<b>{building_name}</b><br/>{address}<br/>Score: {final_opportunity_score}",
        "style": {"backgroundColor": "#091222", "color": "#e2ecff"},
    }
    st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view_state, tooltip=tooltip))

st.subheader("Building Detail")
if len(filtered):
    filtered = filtered.copy()
    filtered["_label"] = filtered["building_name"].fillna("Unknown") + " | " + filtered["address"].fillna("Unknown")
    selected_label = st.selectbox("Select building", options=filtered["_label"].tolist(), index=0)
    selected = filtered.loc[filtered["_label"] == selected_label].iloc[0]

    detail_cols = [
        "building_name",
        "address",
        "parcel_id",
        "account_id",
        "building_type",
        "land_use_category",
        "square_footage",
        "benchmark_site_eui",
        "benchmark_source_eui",
        "benchmark_ghg_intensity",
        "permit_count_5y",
        "permit_types_5y",
        "unsafe_violation_count_5y",
        "unsafe_violation_examples",
        "complaint_count_5y",
        "complaint_types_5y",
        "chp_fit_score",
        "deferred_maintenance_score",
        "project_size_score",
        "pursuit_score",
        "final_opportunity_score",
        "top_reason_1",
        "top_reason_2",
        "top_reason_3",
        "data_confidence",
        "validation_next_step",
    ]
    present_detail_cols = [c for c in detail_cols if c in selected.index]
    st.json({col: selected[col] for col in present_detail_cols})
else:
    st.info("No records match the current filters.")
