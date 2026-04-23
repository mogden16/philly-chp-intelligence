from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .cleaning.common import bounded_score


def _string_contains_any(value: object, keywords: list[str]) -> bool:
    if value is None or pd.isna(value):
        return False
    text = str(value).upper()
    return any(k in text for k in keywords)


def _compute_chp_fit(master: pd.DataFrame, config: dict[str, Any]) -> pd.Series:
    thresholds = config.get("thresholds", {})
    bt = config.get("building_type_keywords", {})

    size_component = bounded_score(
        master["square_footage"],
        thresholds.get("large_building_sqft", 50000),
        thresholds.get("very_large_building_sqft", 150000),
    )
    eui_component = bounded_score(master["benchmark_site_eui"], 20, thresholds.get("high_site_eui", 80))

    preferred_types = bt.get("industrial", []) + bt.get("commercial", [])
    type_bonus = master["building_type"].map(lambda v: 100.0 if _string_contains_any(v, preferred_types) else 40.0)

    return (0.4 * size_component + 0.4 * eui_component + 0.2 * type_bonus).clip(0, 100)


def _compute_deferred_maintenance(master: pd.DataFrame) -> pd.Series:
    unsafe = bounded_score(master["unsafe_violation_count_5y"], 0, 12)
    complaints = bounded_score(master["complaint_count_5y"], 0, 25)
    permit_gap = 100 - bounded_score(master["permit_count_5y"], 0, 10)
    return (0.5 * unsafe + 0.3 * complaints + 0.2 * permit_gap).clip(0, 100)


def _compute_project_size(master: pd.DataFrame) -> pd.Series:
    area_component = bounded_score(master["square_footage"], 10000, 250000)
    energy_component = bounded_score(master["benchmark_source_eui"], 20, 140)
    return (0.7 * area_component + 0.3 * energy_component).clip(0, 100)


def _compute_pursuit(master: pd.DataFrame) -> pd.Series:
    completeness = bounded_score(master["data_sources_count"], 1, 6)
    score = 0.6 * completeness + 0.2 * (100 - bounded_score(master["unsafe_violation_count_5y"], 0, 15))
    with_geo = np.where(master["latitude"].notna() & master["longitude"].notna(), 100.0, 30.0)
    score += 0.2 * with_geo
    return pd.Series(score, index=master.index).clip(0, 100)


def _build_reasons(row: pd.Series) -> tuple[str, str, str, str, str]:
    reasons: list[str] = []

    if row.get("square_footage", 0) >= 100000:
        reasons.append("Large floor area supports CHP economics")
    if row.get("benchmark_site_eui", 0) >= 80:
        reasons.append("High site EUI suggests strong thermal/electric load")
    if row.get("unsafe_violation_count_5y", 0) >= 3:
        reasons.append("Multiple unsafe violations indicate capital need")
    if row.get("complaint_count_5y", 0) >= 8:
        reasons.append("High complaint volume indicates maintenance pressure")
    if row.get("permit_count_5y", 0) == 0:
        reasons.append("No recent permits may indicate deferred investment")

    while len(reasons) < 3:
        reasons.append("Additional validation needed")

    confidence = "High" if row.get("data_sources_count", 0) >= 5 else "Medium" if row.get("data_sources_count", 0) >= 3 else "Low"

    next_step = "Validate utility interval data and verify mechanical room constraints"
    if row.get("data_sources_count", 0) < 3:
        next_step = "Confirm parcel match and collect missing benchmark/permit records"

    return reasons[0], reasons[1], reasons[2], confidence, next_step


def score_master_table(master: pd.DataFrame, scoring_config: dict[str, Any]) -> pd.DataFrame:
    """Compute transparent rules-based opportunity scores and explainability fields."""
    out = master.copy()

    out["chp_fit_score"] = _compute_chp_fit(out, scoring_config)
    out["deferred_maintenance_score"] = _compute_deferred_maintenance(out)
    out["project_size_score"] = _compute_project_size(out)
    out["pursuit_score"] = _compute_pursuit(out)

    weights = scoring_config.get("weights", {})
    out["final_opportunity_score"] = (
        out["chp_fit_score"] * float(weights.get("chp_fit_score", 0.35))
        + out["deferred_maintenance_score"] * float(weights.get("deferred_maintenance_score", 0.25))
        + out["project_size_score"] * float(weights.get("project_size_score", 0.20))
        + out["pursuit_score"] * float(weights.get("pursuit_score", 0.20))
    ).clip(0, 100)

    reason_values = out.apply(_build_reasons, axis=1, result_type="expand")
    reason_values.columns = ["top_reason_1", "top_reason_2", "top_reason_3", "data_confidence", "validation_next_step"]
    out = pd.concat([out, reason_values], axis=1)

    out = out.sort_values("final_opportunity_score", ascending=False).reset_index(drop=True)
    return out
