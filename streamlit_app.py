from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pydeck as pdk
import streamlit as st


st.set_page_config(page_title="Philly CHP Intelligence", layout="wide")

st.markdown(
    """
    <style>
        .stApp {
            background: radial-gradient(circle at 12% 12%, #122640 0%, #0b1524 45%, #070b12 100%);
            color: #dce8f8;
        }
        .section-title {
            font-size: 1.08rem;
            font-weight: 650;
            color: #d6e4f6;
            margin-top: 0.6rem;
            margin-bottom: 0.25rem;
        }
        .kpi-card {
            background: linear-gradient(145deg, rgba(26,43,68,0.95), rgba(11,20,35,0.95));
            border: 1px solid rgba(90, 158, 236, 0.33);
            border-radius: 12px;
            padding: 13px 14px;
            min-height: 88px;
        }
        .kpi-label {
            font-size: 0.82rem;
            color: #9cb3d1;
            margin-bottom: 0.25rem;
        }
        .kpi-value {
            font-size: 1.48rem;
            font-weight: 700;
            color: #f2f7ff;
            line-height: 1.2;
        }
        .panel {
            background: rgba(8,16,27,0.65);
            border: 1px solid rgba(98, 154, 224, 0.22);
            border-radius: 12px;
            padding: 14px 16px;
        }
        .muted {
            color: #97abc8;
            font-size: 0.88rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def load_data(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)


def _business_category(value: object) -> str:
    text = str(value).upper() if value is not None else ""
    if any(k in text for k in ["HOSPITAL", "MEDICAL", "HEALTH"]):
        return "Hospital / Healthcare"
    if any(k in text for k in ["UNIVERSITY", "COLLEGE", "SCHOOL", "CAMPUS"]):
        return "University / College"
    if any(k in text for k in ["INDUSTRIAL", "MANUFACTUR", "PLANT", "FACTORY", "PROCESS"]):
        return "Industrial / Process"
    if any(k in text for k in ["APARTMENT", "MULTI", "RESIDENTIAL"]):
        return "Large Multifamily"
    if any(k in text for k in ["HOTEL", "INN"]):
        return "Hotel / Lodging"
    if any(k in text for k in ["DATA CENTER", "SERVER"]):
        return "Data Center / Critical"
    if any(k in text for k in ["MIXED", "LAB", "MUNICIPAL", "GOVERNMENT"]):
        return "Institutional / Mixed"
    if any(k in text for k in ["OFFICE", "CORPORATE"]):
        return "Office"
    if any(k in text for k in ["RETAIL", "STORE", "MALL", "SHOP"]):
        return "Retail"
    if any(k in text for k in ["PARKING", "GARAGE", "SELF STORAGE", "STORAGE"]):
        return "Parking / Storage"
    if any(k in text for k in ["WAREHOUSE", "DISTRIBUTION", "LOGISTICS"]):
        return "Warehouse"
    return "Other / Unknown"


def _render_kpi(label: str, value: str, help_text: str) -> None:
    st.markdown(
        f"<div class='kpi-card'><div class='kpi-label' title='{help_text}'>{label}</div><div class='kpi-value'>{value}</div></div>",
        unsafe_allow_html=True,
    )


def _yes_no(value: bool) -> str:
    return "Yes" if bool(value) else "No"


def _matplotlib_available() -> bool:
    return importlib.util.find_spec("matplotlib") is not None


processed_path = Path("data/processed/master_buildings_scored.parquet")
if not processed_path.exists():
    st.error("Run `python scripts/run_pipeline.py` first to generate scored data.")
    st.stop()

df = load_data(str(processed_path)).copy()

numeric_cols = [
    "final_opportunity_score",
    "chp_fit_score",
    "historical_deferred_maintenance_score",
    "deferred_maintenance_score",
    "project_size_score",
    "pursuit_score",
    "benchmark_site_eui",
    "permit_count_5y",
    "unsafe_violation_count_5y",
    "complaint_count_5y",
    "square_footage",
]
for col in numeric_cols:
    if col not in df.columns:
        df[col] = 0
    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

if "historical_deferred_maintenance_score" not in df.columns:
    df["historical_deferred_maintenance_score"] = df["deferred_maintenance_score"]

if "building_type_category" not in df.columns:
    df["building_type_category"] = df["building_type"].map(_business_category)
else:
    df["building_type_category"] = df["building_type_category"].fillna("other").astype(str).str.replace("_", " ").str.title()

if "has_chp_thermal_signal" not in df.columns:
    df["has_chp_thermal_signal"] = df["benchmark_site_eui"] >= 80
if "has_historical_maintenance_burden" not in df.columns:
    df["has_historical_maintenance_burden"] = df["historical_deferred_maintenance_score"] >= 60

df["why_ranked"] = (
    df["top_reason_1"].fillna("")
    + " | "
    + df["top_reason_2"].fillna("")
    + " | "
    + df["top_reason_3"].fillna("")
).str.strip(" |")

st.title("Philly CHP Intelligence")
st.caption("Decision-support screening for CHP opportunity, capital backlog, and pursuit readiness")

with st.sidebar:
    st.header("Screening Filters")
    score_floor = st.slider(
        "Opportunity Score Floor",
        min_value=0,
        max_value=100,
        value=45,
        help="Only show buildings at or above this opportunity score threshold.",
    )

    category_options = sorted(df["building_type_category"].dropna().astype(str).unique().tolist())
    default_categories = [c for c in category_options if c in {"Hospital / Healthcare", "University / College", "Industrial / Process", "Large Multifamily", "Hotel / Lodging", "Institutional / Mixed"}]
    selected_categories = st.multiselect(
        "Building Category",
        options=category_options,
        default=default_categories if default_categories else category_options,
        help="Categories are CHP-oriented groupings to reduce building-type clutter.",
    )

    max_sqft = int(df["square_footage"].max()) if len(df) else 100000
    sqft_range = st.slider(
        "Building Size (sq ft)",
        min_value=0,
        max_value=max(1000, max_sqft),
        value=(25000, max(25000, max_sqft)),
    )

    confidence_options = sorted(df["data_confidence"].dropna().astype(str).unique().tolist()) if "data_confidence" in df.columns else []
    selected_confidence = st.multiselect("Data Confidence", options=confidence_options, default=confidence_options)

    st.markdown("<div class='section-title'>Priority Toggles</div>", unsafe_allow_html=True)
    high_chp_only = st.toggle("High CHP Fit only", value=False, help="CHP Fit >= 70")
    high_hist_deferred_only = st.toggle("High Historical Deferred Need only", value=False, help="Historical Deferred Need >= 60")
    high_confidence_only = st.toggle("High Confidence only", value=False)

filtered = df[df["final_opportunity_score"] >= score_floor].copy()
if selected_categories:
    filtered = filtered[filtered["building_type_category"].isin(selected_categories)]
filtered = filtered[
    (filtered["square_footage"] >= sqft_range[0])
    & (filtered["square_footage"] <= sqft_range[1])
]
if selected_confidence:
    filtered = filtered[filtered["data_confidence"].astype(str).isin(selected_confidence)]
if high_chp_only:
    filtered = filtered[filtered["chp_fit_score"] >= 70]
if high_hist_deferred_only:
    filtered = filtered[filtered["historical_deferred_maintenance_score"] >= 60]
if high_confidence_only:
    filtered = filtered[filtered["data_confidence"] == "High"]

filtered = filtered.sort_values("final_opportunity_score", ascending=False).reset_index(drop=True)

k1, k2, k3, k4, k5, k6 = st.columns(6)
with k1:
    _render_kpi("Filtered Buildings", f"{len(filtered):,}", "Number of buildings currently matching all filters.")
with k2:
    _render_kpi("Avg Opportunity Score", f"{filtered['final_opportunity_score'].mean() if len(filtered) else 0:.1f}", "Average overall priority score for filtered buildings.")
with k3:
    _render_kpi("High Priority Count", f"{int((filtered['final_opportunity_score'] >= 75).sum()) if len(filtered) else 0:,}", "Buildings with Opportunity Score >= 75.")
with k4:
    _render_kpi("Avg Deferred Need Score", f"{filtered['historical_deferred_maintenance_score'].mean() if len(filtered) else 0:.1f}", "Historical deferred maintenance severity across filtered results.")
with k5:
    avg_site_eui = filtered.loc[filtered["benchmark_site_eui"] > 0, "benchmark_site_eui"].mean() if len(filtered) else 0
    _render_kpi("Avg Site EUI", f"{avg_site_eui if pd.notna(avg_site_eui) else 0:.1f}", "Average site EUI for records with benchmark energy data.")
with k6:
    thermal_count = int(filtered["has_chp_thermal_signal"].sum()) if len(filtered) else 0
    _render_kpi("CHP Thermal Signals", f"{thermal_count:,}", "Filtered buildings showing CHP-relevant thermal signal patterns.")

st.markdown("<div class='section-title'>Ranked Targets</div>", unsafe_allow_html=True)
st.markdown("<div class='muted'>Sorted by Opportunity Score descending. Color intensity highlights stronger values.</div>", unsafe_allow_html=True)

table_cols = [
    "final_opportunity_score",
    "chp_fit_score",
    "historical_deferred_maintenance_score",
    "project_size_score",
    "pursuit_score",
    "building_name",
    "address",
    "building_type_category",
    "square_footage",
    "benchmark_site_eui",
    "permit_count_5y",
    "unsafe_violation_count_5y",
    "why_ranked",
    "data_confidence",
]
present_cols = [c for c in table_cols if c in filtered.columns]
display = filtered[present_cols].copy()
display = display.rename(
    columns={
        "final_opportunity_score": "Opportunity Score",
        "chp_fit_score": "CHP Fit",
        "historical_deferred_maintenance_score": "Deferred Need",
        "project_size_score": "Project Scale",
        "pursuit_score": "Pursuit Ease",
        "building_name": "Building",
        "address": "Address",
        "building_type_category": "Category",
        "square_footage": "Square Footage",
        "benchmark_site_eui": "Site EUI",
        "permit_count_5y": "Recent Permits (5y)",
        "unsafe_violation_count_5y": "Unsafe Violations (5y)",
        "why_ranked": "Why Ranked",
        "data_confidence": "Data Confidence",
    }
)

if _matplotlib_available():
    styler = (
        display.style.format(
            {
                "Opportunity Score": "{:.1f}",
                "CHP Fit": "{:.1f}",
                "Deferred Need": "{:.1f}",
                "Project Scale": "{:.1f}",
                "Pursuit Ease": "{:.1f}",
                "Square Footage": "{:,.0f}",
                "Site EUI": "{:.1f}",
                "Recent Permits (5y)": "{:.0f}",
                "Unsafe Violations (5y)": "{:.0f}",
            }
        )
        .background_gradient(subset=["Opportunity Score"], cmap="YlGn")
        .background_gradient(subset=["CHP Fit"], cmap="Blues")
        .background_gradient(subset=["Deferred Need"], cmap="Oranges")
    )
    st.dataframe(styler, use_container_width=True, height=400)
else:
    st.dataframe(display, use_container_width=True, height=400)

filtered.to_csv("data/processed/latest_filtered_targets.csv", index=False)
st.download_button(
    "Export Filtered Results CSV",
    data=filtered.to_csv(index=False).encode("utf-8"),
    file_name="philly_chp_targets_filtered.csv",
    mime="text/csv",
)

validation_cols = [
    "building_name",
    "address",
    "building_type",
    "final_opportunity_score",
    "chp_fit_score",
    "historical_deferred_maintenance_score",
    "project_size_score",
    "pursuit_score",
    "top_reason_1",
    "top_reason_2",
    "top_reason_3",
    "recommended_next_step",
    "data_confidence",
]
validation_export = filtered[[c for c in validation_cols if c in filtered.columns]].rename(
    columns={"final_opportunity_score": "opportunity_score"}
)
st.download_button(
    "Download Validation / Debug Export",
    data=validation_export.to_csv(index=False).encode("utf-8"),
    file_name="ranking_validation_export.csv",
    mime="text/csv",
)

st.markdown("<div class='section-title'>Philly Opportunity Map</div>", unsafe_allow_html=True)
map_df = filtered.copy()
if "latitude" in map_df.columns and "longitude" in map_df.columns:
    map_df["latitude"] = pd.to_numeric(map_df["latitude"], errors="coerce")
    map_df["longitude"] = pd.to_numeric(map_df["longitude"], errors="coerce")
    map_df = map_df.dropna(subset=["latitude", "longitude"]).copy()

if len(map_df):
    max_points = st.slider("Map point limit", min_value=100, max_value=5000, value=1500, step=100, help="Limit points for map responsiveness.")
    map_df = map_df.sort_values("final_opportunity_score", ascending=False).head(max_points).copy()
    map_df["final_opportunity_score"] = map_df["final_opportunity_score"].fillna(0)
    map_df["color_r"] = (245 - (map_df["final_opportunity_score"] * 1.7).clip(0, 180)).astype(int)
    map_df["color_g"] = (86 + (map_df["final_opportunity_score"] * 1.4).clip(0, 160)).astype(int)
    map_df["color_b"] = 98
    map_df["radius"] = (80 + map_df["final_opportunity_score"] * 2.8).clip(80, 420)
    map_df["elevation"] = (300 + map_df["final_opportunity_score"] * 45).clip(300, 5000)

    scatter_layer = pdk.Layer(
        "ScatterplotLayer",
        data=map_df,
        get_position="[longitude, latitude]",
        get_radius="radius",
        radius_min_pixels=3,
        radius_max_pixels=24,
        get_fill_color="[color_r, color_g, color_b, 200]",
        stroked=True,
        get_line_color=[220, 235, 255, 140],
        line_width_min_pixels=1,
        pickable=True,
    )
    column_layer = pdk.Layer(
        "ColumnLayer",
        data=map_df,
        get_position="[longitude, latitude]",
        radius=55,
        get_elevation="elevation",
        elevation_scale=1,
        get_fill_color="[color_r, color_g, color_b, 115]",
        pickable=True,
    )
    heat_layer = pdk.Layer(
        "HeatmapLayer",
        data=map_df,
        get_position="[longitude, latitude]",
        get_weight="final_opportunity_score",
        opacity=0.25,
    )

    view_state = pdk.ViewState(latitude=39.9526, longitude=-75.1652, zoom=10.9, pitch=35, bearing=10)
    tooltip = {
        "html": "<b>{building_name}</b><br/>{address}<br/>Opportunity: {final_opportunity_score}<br/>CHP Fit: {chp_fit_score}<br/>Deferred Need: {historical_deferred_maintenance_score}",
        "style": {"backgroundColor": "#0b1322", "color": "#e4efff"},
    }
    st.pydeck_chart(
        pdk.Deck(
            map_style="https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json",
            layers=[heat_layer, column_layer, scatter_layer],
            initial_view_state=view_state,
            tooltip=tooltip,
        )
    )
else:
    st.warning("No mapped coordinates are available yet. Re-run the pipeline after the coordinate patch to populate map points.")

st.markdown("<div class='section-title'>Building Detail</div>", unsafe_allow_html=True)
if len(filtered):
    pick_df = filtered.copy()
    pick_df["_label"] = pick_df["building_name"].fillna("Unknown") + " | " + pick_df["address"].fillna("Unknown")
    selected_label = st.selectbox("Select target", options=pick_df["_label"].tolist(), index=0)
    selected = pick_df.loc[pick_df["_label"] == selected_label].iloc[0]

    st.markdown("<div class='panel'>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        st.markdown(f"### {selected.get('building_name', 'Unknown Building')}")
        st.markdown(f"**Address:** {selected.get('address', 'Unknown Address')}")
        st.markdown(f"**Category:** {selected.get('building_type_category', 'Other / Unknown')}")
        st.markdown(f"**Data Confidence:** {selected.get('data_confidence', 'Unknown')}")
    with c2:
        st.metric("Opportunity Score", f"{selected.get('final_opportunity_score', 0):.1f}")
        st.metric("CHP Fit", f"{selected.get('chp_fit_score', 0):.1f}")
    with c3:
        st.metric("Deferred Need", f"{selected.get('historical_deferred_maintenance_score', selected.get('deferred_maintenance_score', 0)):.1f}")
        st.metric("Project Scale", f"{selected.get('project_size_score', 0):.1f}")

    st.markdown("#### Overview")
    st.markdown(
        f"- **Square Footage:** {selected.get('square_footage', 0):,.0f}\n"
        f"- **Building Type (raw):** {selected.get('building_type', 'UNKNOWN')}\n"
        f"- **CHP Thermal Signal:** {_yes_no(bool(selected.get('has_chp_thermal_signal', False)))}\n"
        f"- **Historical Maintenance Burden:** {_yes_no(bool(selected.get('has_historical_maintenance_burden', False)))}"
    )

    st.markdown("#### CHP Indicators")
    st.markdown(
        f"- **Site EUI:** {selected.get('benchmark_site_eui', 0):.1f}\n"
        f"- **Source EUI:** {selected.get('benchmark_source_eui', 0):.1f}\n"
        f"- **Benchmark GHG Intensity:** {selected.get('benchmark_ghg_intensity', 0):.1f}"
    )

    st.markdown("#### Deferred Maintenance History")
    st.markdown(
        f"- **Historical Deferred Maintenance Score:** {selected.get('historical_deferred_maintenance_score', selected.get('deferred_maintenance_score', 0)):.1f}\n"
        f"- **Historical Violation Burden:** {selected.get('historical_violation_burden_score', 0):.1f}\n"
        f"- **Issue Persistence:** {selected.get('issue_persistence_score', 0):.1f}\n"
        f"- **Underinvestment:** {selected.get('underinvestment_score', 0):.1f}\n"
        f"- **Benchmark Underperformance:** {selected.get('benchmarking_underperformance_score', 0):.1f}\n"
        f"- **Current Critical Issues:** {selected.get('current_critical_issue_score', 0):.1f}"
    )

    st.markdown("#### Permits & Violations")
    st.markdown(
        f"- **Recent Permits (5y):** {selected.get('permit_count_5y', 0):.0f}\n"
        f"- **Permit Summary:** {selected.get('permit_types_5y', 'N/A')}\n"
        f"- **Unsafe Violations (5y):** {selected.get('unsafe_violation_count_5y', 0):.0f}\n"
        f"- **Unsafe Violation Summary:** {selected.get('unsafe_violation_examples', 'N/A')}\n"
        f"- **Complaints (5y):** {selected.get('complaint_count_5y', 0):.0f}\n"
        f"- **Complaint Summary:** {selected.get('complaint_types_5y', 'N/A')}"
    )

    st.markdown("#### Scoring Explanation")
    st.markdown(
        f"- {selected.get('top_reason_1', 'No reason generated')}\n"
        f"- {selected.get('top_reason_2', 'No reason generated')}\n"
        f"- {selected.get('top_reason_3', 'No reason generated')}"
    )

    st.markdown("#### Recommended Next Step")
    st.info(selected.get("recommended_next_step", selected.get("validation_next_step", "Collect more building and utility context.")))

    missing_flags: list[str] = []
    if selected.get("benchmark_site_eui", 0) <= 0:
        missing_flags.append("Benchmark energy intensity is missing.")
    if selected.get("permit_count_5y", 0) <= 0:
        missing_flags.append("Permit history is sparse or missing.")
    if str(selected.get("building_type", "UNKNOWN")).upper() == "UNKNOWN":
        missing_flags.append("Building type is uncertain; CHP fit confidence is reduced.")
    if str(selected.get("address", "")).upper() == "UNKNOWN ADDRESS":
        missing_flags.append("Address quality is weak; cross-dataset joining confidence is lower.")
    if missing_flags:
        st.warning("Missing-data warnings: " + " ".join(missing_flags))

    st.markdown("</div>", unsafe_allow_html=True)
else:
    st.info("No records match the current filter selections.")

with st.expander("Ranking Sanity Check Summary", expanded=False):
    top20 = filtered.head(20)[["building_name", "address", "building_type_category", "final_opportunity_score"]].copy()
    top20 = top20.rename(columns={"building_name": "Building", "address": "Address", "building_type_category": "Category", "final_opportunity_score": "Opportunity Score"})
    st.markdown("**Top 20 Buildings by Opportunity Score**")
    st.dataframe(top20, use_container_width=True, height=260)

    st.markdown("**Distribution by Building Category**")
    category_dist = filtered["building_type_category"].value_counts(normalize=True).mul(100).round(1).rename("Percent")
    st.dataframe(category_dist.to_frame(), use_container_width=True)

    strategic = {"Hospital / Healthcare", "University / College", "Large Multifamily", "Industrial / Process", "Data Center / Critical"}
    top50 = filtered.head(50)
    strategic_pct = (top50["building_type_category"].isin(strategic).mean() * 100.0) if len(top50) else 0.0
    st.metric("Top 50 in strategic CHP categories", f"{strategic_pct:.1f}%")
