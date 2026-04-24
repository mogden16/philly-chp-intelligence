from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .cleaning.common import bounded_score


CHP_CATEGORY_KEYWORDS: dict[str, list[str]] = {
    "hospital_healthcare": ["HOSPITAL", "MEDICAL", "HEALTH", "CLINIC"],
    "university_college": ["UNIVERSITY", "COLLEGE", "CAMPUS", "SCHOOL", "ACADEMY"],
    "industrial_process": ["INDUSTRIAL", "MANUFACTURING", "PROCESS", "PLANT", "FACTORY"],
    "multifamily_large": ["APARTMENT", "MULTI FAMILY", "MULTIFAMILY", "RESIDENTIAL TOWER"],
    "hotel_lodging": ["HOTEL", "INN", "LODGING"],
    "data_center_critical": ["DATA CENTER", "DATA CENTRE", "SERVER"],
    "institutional_mixed_use": ["MIXED", "LAB", "LABORATORY", "MUNICIPAL", "GOVERNMENT"],
    "office_large": ["OFFICE", "CORPORATE", "HEADQUARTERS"],
    "warehouse_low_thermal": ["WAREHOUSE", "DISTRIBUTION", "LOGISTICS"],
    "retail_low_thermal": ["RETAIL", "SHOPPING", "STORE", "MALL"],
    "parking_storage": ["PARKING", "GARAGE", "SELF STORAGE", "STORAGE"],
}

CHP_CATEGORY_BASE_SCORE: dict[str, float] = {
    "hospital_healthcare": 98,
    "university_college": 96,
    "industrial_process": 95,
    "multifamily_large": 88,
    "hotel_lodging": 88,
    "data_center_critical": 96,
    "institutional_mixed_use": 84,
    "office_large": 70,
    "warehouse_low_thermal": 48,
    "retail_low_thermal": 40,
    "parking_storage": 12,
    "other": 55,
}


def _string_contains_any(value: object, keywords: list[str]) -> bool:
    if value is None or pd.isna(value):
        return False
    text = str(value).upper()
    return any(k in text for k in keywords)


def _classify_building_type(value: object) -> str:
    if value is None or pd.isna(value):
        return "other"
    text = str(value).upper()
    for category, keywords in CHP_CATEGORY_KEYWORDS.items():
        if any(k in text for k in keywords):
            return category
    return "other"


def _yes_no(value: bool) -> str:
    return "Yes" if bool(value) else "No"


def _compute_thermal_signal(master: pd.DataFrame) -> pd.Series:
    high_eui = pd.to_numeric(master["benchmark_site_eui"], errors="coerce").fillna(0) >= 80
    thermal_keywords = ["HOSPITAL", "HOTEL", "LAB", "UNIVERSITY", "PROCESS", "PLANT", "LAUNDRY", "KITCHEN"]
    type_signal = master["building_type"].map(lambda v: _string_contains_any(v, thermal_keywords))
    return high_eui | type_signal


def _compute_chp_fit(master: pd.DataFrame, config: dict[str, Any]) -> pd.Series:
    thresholds = config.get("thresholds", {})

    size_component = bounded_score(
        master["square_footage"],
        thresholds.get("large_building_sqft", 50000),
        thresholds.get("very_large_building_sqft", 200000),
    )
    eui_component = bounded_score(master["benchmark_site_eui"], 25, thresholds.get("high_site_eui", 90))
    type_component = master["building_type_category"].map(lambda c: CHP_CATEGORY_BASE_SCORE.get(c, 55.0))
    thermal_bonus = master["has_chp_thermal_signal"].map(lambda x: 100.0 if x else 35.0)

    score = 0.35 * size_component + 0.30 * eui_component + 0.25 * type_component + 0.10 * thermal_bonus
    return score.clip(0, 100)


def _compute_benchmarking_underperformance(master: pd.DataFrame) -> pd.Series:
    eui = pd.to_numeric(master["benchmark_site_eui"], errors="coerce").fillna(0)
    medians = (
        pd.DataFrame({"cat": master["building_type_category"], "eui": eui})
        .groupby("cat", as_index=False)["eui"]
        .median()
        .set_index("cat")["eui"]
        .to_dict()
    )
    peer_median = master["building_type_category"].map(lambda c: medians.get(c, 0)).replace(0, np.nan)
    ratio = (eui / peer_median).replace([np.inf, -np.inf], np.nan).fillna(0)
    return bounded_score(ratio, 0.9, 1.8)


def _compute_historical_deferred_components(master: pd.DataFrame) -> pd.DataFrame:
    unsafe = pd.to_numeric(master["unsafe_violation_count_5y"], errors="coerce").fillna(0)
    complaints = pd.to_numeric(master["complaint_count_5y"], errors="coerce").fillna(0)
    permits = pd.to_numeric(master["permit_count_5y"], errors="coerce").fillna(0)
    permit_value = pd.to_numeric(master.get("permit_value_5y", 0), errors="coerce").fillna(0)
    sqft = pd.to_numeric(master["square_footage"], errors="coerce").fillna(0)
    year_built = pd.to_numeric(master.get("year_built", np.nan), errors="coerce")
    benchmark_underperf = _compute_benchmarking_underperformance(master)

    historical_violation_burden = (0.55 * bounded_score(unsafe, 0, 20) + 0.45 * bounded_score(complaints, 0, 40)).clip(0, 100)
    repeated_issue_pattern = (0.5 * bounded_score(unsafe + complaints, 2, 35) + 0.5 * bounded_score(unsafe * complaints, 0, 60)).clip(0, 100)

    permit_value_per_sf = (permit_value / (sqft + 1)).replace([np.inf, -np.inf], np.nan).fillna(0)
    permit_intensity = (permits / ((sqft / 100000.0) + 1)).replace([np.inf, -np.inf], np.nan).fillna(0)
    year_now = pd.Timestamp.utcnow().year
    age_years = (year_now - year_built).clip(lower=0).fillna(45)
    underinvestment = (
        0.45 * (100 - bounded_score(permit_value_per_sf, 2, 60))
        + 0.30 * (100 - bounded_score(permit_intensity, 1, 15))
        + 0.25 * bounded_score(age_years, 20, 95)
    ).clip(0, 100)

    level_text = master.get("unsafe_violation_levels_5y", pd.Series([""] * len(master))).fillna("").astype(str).str.upper()
    has_active_flag = level_text.str.contains("OPEN|ACTIVE|IMMINENT|DANG", regex=True)
    current_critical = (bounded_score(unsafe, 0, 8) * 0.8 + has_active_flag.map(lambda x: 20.0 if x else 0.0)).clip(0, 100)

    historical_deferred = (
        0.30 * historical_violation_burden
        + 0.25 * repeated_issue_pattern
        + 0.25 * underinvestment
        + 0.15 * benchmark_underperf
        + 0.05 * current_critical
    ).clip(0, 100)

    return pd.DataFrame(
        {
            "historical_violation_burden_score": historical_violation_burden,
            "issue_persistence_score": repeated_issue_pattern,
            "underinvestment_score": underinvestment,
            "benchmarking_underperformance_score": benchmark_underperf,
            "current_critical_issue_score": current_critical,
            "historical_deferred_maintenance_score": historical_deferred,
        },
        index=master.index,
    )


def _compute_project_size(master: pd.DataFrame) -> pd.Series:
    area_component = bounded_score(master["square_footage"], 15000, 300000)
    energy_component = bounded_score(master["benchmark_source_eui"], 20, 170)
    return (0.65 * area_component + 0.35 * energy_component).clip(0, 100)


def _compute_pursuit(master: pd.DataFrame) -> pd.Series:
    completeness = bounded_score(master["data_sources_count"], 2, 7)
    geo_ready = np.where(master["latitude"].notna() & master["longitude"].notna(), 100.0, 45.0)
    distress_penalty = 100 - bounded_score(master["unsafe_violation_count_5y"], 0, 35)
    scale_readiness = bounded_score(master["square_footage"], 20000, 400000)
    score = 0.40 * completeness + 0.25 * scale_readiness + 0.20 * geo_ready + 0.15 * distress_penalty
    return pd.Series(score, index=master.index).clip(0, 100)


def _compute_data_confidence(row: pd.Series) -> str:
    points = 0
    if row.get("benchmark_site_eui", 0) and row.get("benchmark_site_eui", 0) > 0:
        points += 1
    if row.get("square_footage", 0) and row.get("square_footage", 0) > 0:
        points += 1
    if str(row.get("building_type_category", "other")) != "other":
        points += 1
    if row.get("permit_count_5y", 0) > 0 or row.get("complaint_count_5y", 0) > 0 or row.get("unsafe_violation_count_5y", 0) > 0:
        points += 1
    if row.get("data_sources_count", 0) >= 5:
        points += 1
    if str(row.get("address", "")).strip().upper() == "UNKNOWN ADDRESS":
        points -= 1
    if str(row.get("property_key", "")).upper().startswith("UNKNOWN"):
        points -= 1

    if points >= 4:
        return "High"
    if points >= 2:
        return "Medium"
    return "Low"


def _build_reasons(row: pd.Series) -> tuple[str, str, str, str]:
    reasons: list[str] = []
    cat = str(row.get("building_type_category", "other")).replace("_", " ")

    if row.get("chp_fit_score", 0) >= 70:
        reasons.append(f"Large {cat} profile with CHP-compatible load characteristics")
    if row.get("historical_deferred_maintenance_score", 0) >= 60:
        reasons.append("Long-run maintenance burden suggests deferred capital backlog")
    if row.get("underinvestment_score", 0) >= 60:
        reasons.append("Modernization permit activity appears low relative to building scale")
    if row.get("benchmarking_underperformance_score", 0) >= 60:
        reasons.append("Energy intensity underperforms peers in the same building category")
    if row.get("has_chp_thermal_signal", False):
        reasons.append("Thermal demand signals support CHP screening priority")

    while len(reasons) < 3:
        reasons.append("Supplement with utility interval and central-plant validation")

    next_step = "Request 12–24 months of interval utility data and review central plant operating profile"
    if row.get("data_confidence", "Low") == "Low":
        next_step = "Resolve address/parcel matching and fill missing benchmark and permit history before pursuit"
    elif row.get("chp_fit_score", 0) >= 70 and row.get("historical_deferred_maintenance_score", 0) >= 60:
        next_step = "Prioritize for CHP + capital renewal feasibility scoping with facilities leadership"

    return reasons[0], reasons[1], reasons[2], next_step


def score_master_table(master: pd.DataFrame, scoring_config: dict[str, Any]) -> pd.DataFrame:
    """Compute transparent rules-based opportunity scores and explainability fields."""
    out = master.copy()

    numeric_cols = [
        "square_footage",
        "benchmark_site_eui",
        "benchmark_source_eui",
        "permit_count_5y",
        "permit_value_5y",
        "unsafe_violation_count_5y",
        "complaint_count_5y",
        "data_sources_count",
        "year_built",
    ]
    for col in numeric_cols:
        if col not in out.columns:
            out[col] = 0
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0)

    out["building_type_category"] = out["building_type"].map(_classify_building_type)
    out["has_chp_thermal_signal"] = _compute_thermal_signal(out)

    out["chp_fit_score"] = _compute_chp_fit(out, scoring_config)
    deferred_components = _compute_historical_deferred_components(out)
    out = pd.concat([out, deferred_components], axis=1)
    out["historical_deferred_maintenance_score"] = out["historical_deferred_maintenance_score"].clip(0, 100)
    out["deferred_maintenance_score"] = out["historical_deferred_maintenance_score"]
    out["project_size_score"] = _compute_project_size(out)
    out["pursuit_score"] = _compute_pursuit(out)

    weights = scoring_config.get("weights", {})
    out["final_opportunity_score"] = (
        out["chp_fit_score"] * float(weights.get("chp_fit_score", 0.42))
        + out["historical_deferred_maintenance_score"] * float(weights.get("deferred_maintenance_score", 0.28))
        + out["project_size_score"] * float(weights.get("project_size_score", 0.18))
        + out["pursuit_score"] * float(weights.get("pursuit_score", 0.12))
    ).clip(0, 100)

    out["data_confidence"] = out.apply(_compute_data_confidence, axis=1)
    reason_values = out.apply(_build_reasons, axis=1, result_type="expand")
    reason_values.columns = ["top_reason_1", "top_reason_2", "top_reason_3", "recommended_next_step"]
    out = pd.concat([out, reason_values], axis=1)
    out["validation_next_step"] = out["recommended_next_step"]
    out["has_historical_maintenance_burden"] = out["historical_deferred_maintenance_score"] >= 60

    out = out.sort_values("final_opportunity_score", ascending=False).reset_index(drop=True)
    return out
