from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from .common import build_property_key, normalize_address, normalize_columns, parse_datetime, pick_first_column


def clean_unsafe_violations(df: pd.DataFrame, years_window: int = 5) -> pd.DataFrame:
    """Aggregate unsafe/imminently dangerous violations by property."""
    raw = normalize_columns(df.copy())
    address = pick_first_column(raw, ["address", "street_address", "location"]).map(normalize_address)
    parcel_id = pick_first_column(raw, ["parcel_id", "mapreg", "opa_account_num"])
    account_id = pick_first_column(raw, ["account_num", "opa_account_num"])
    violation_date = parse_datetime(pick_first_column(raw, ["violation_date", "date", "case_open_date"]))
    violation_desc = pick_first_column(raw, ["violation_description", "description", "case_type"]).fillna("UNKNOWN")
    level = pick_first_column(raw, ["violation_level", "severity", "priority"]).fillna("UNKNOWN")

    out = pd.DataFrame(
        {
            "address": address,
            "parcel_id": parcel_id,
            "account_id": account_id,
            "violation_date": violation_date,
            "violation_desc": violation_desc,
            "violation_level": level,
        }
    )
    out["property_key"] = out.apply(
        lambda r: build_property_key(r["parcel_id"], r["account_id"], r["address"]), axis=1
    )

    cutoff = pd.Timestamp(datetime.now(timezone.utc)).tz_convert("UTC") - pd.DateOffset(years=years_window)
    recent = out[out["violation_date"] >= cutoff].copy()

    summary = (
        recent.groupby("property_key", as_index=False)
        .agg(
            unsafe_violation_count_5y=("violation_desc", "count"),
            unsafe_violation_levels_5y=("violation_level", lambda x: ", ".join(sorted(set(x.astype(str)))[:5])),
            unsafe_violation_examples=("violation_desc", lambda x: ", ".join(sorted(set(x.astype(str)))[:5])),
            address=("address", "first"),
            parcel_id=("parcel_id", "first"),
            account_id=("account_id", "first"),
        )
    )
    return summary
