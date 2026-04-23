from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from .common import build_property_key, normalize_address, normalize_columns, parse_datetime, pick_first_column


def clean_complaints(df: pd.DataFrame, years_window: int = 5) -> pd.DataFrame:
    """Aggregate code-enforcement and related complaint signals."""
    raw = normalize_columns(df.copy())
    address = pick_first_column(raw, ["address", "street_address", "location"]).map(normalize_address)
    parcel_id = pick_first_column(raw, ["parcel_id", "mapreg", "opa_account_num"])
    account_id = pick_first_column(raw, ["account_num", "opa_account_num"])
    opened_date = parse_datetime(pick_first_column(raw, ["requested_datetime", "created_date", "open_date"]))
    complaint_type = pick_first_column(raw, ["service_name", "complaint_type", "case_type"]).fillna("UNKNOWN")

    out = pd.DataFrame(
        {
            "address": address,
            "parcel_id": parcel_id,
            "account_id": account_id,
            "opened_date": opened_date,
            "complaint_type": complaint_type,
        }
    )
    out["property_key"] = out.apply(
        lambda r: build_property_key(r["parcel_id"], r["account_id"], r["address"]), axis=1
    )

    cutoff = pd.Timestamp(datetime.now(timezone.utc)).tz_convert("UTC") - pd.DateOffset(years=years_window)
    recent = out[out["opened_date"] >= cutoff].copy()

    summary = (
        recent.groupby("property_key", as_index=False)
        .agg(
            complaint_count_5y=("complaint_type", "count"),
            complaint_types_5y=("complaint_type", lambda x: ", ".join(sorted(set(x.astype(str)))[:5])),
            address=("address", "first"),
            parcel_id=("parcel_id", "first"),
            account_id=("account_id", "first"),
        )
    )
    return summary
