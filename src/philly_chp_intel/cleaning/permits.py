from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from .common import build_property_key, normalize_address, normalize_columns, parse_datetime, pick_first_column, to_numeric


def clean_permits(df: pd.DataFrame, years_window: int = 5) -> pd.DataFrame:
    """Aggregate permit activity at building/property level."""
    raw = normalize_columns(df.copy())

    address = pick_first_column(raw, ["address", "street_address", "location"]).map(normalize_address)
    parcel_id = pick_first_column(raw, ["parcel_id", "mapreg", "opa_account_num"])
    account_id = pick_first_column(raw, ["account_num", "opa_account_num"])
    issued_date = parse_datetime(pick_first_column(raw, ["issued_date", "permit_issued_date", "date_issued"]))
    permit_value = to_numeric(pick_first_column(raw, ["estimated_cost", "permit_value", "value"]))
    permit_type = pick_first_column(raw, ["permit_type", "permit_description", "work_type"]).fillna("UNKNOWN")

    out = pd.DataFrame(
        {
            "address": address,
            "parcel_id": parcel_id,
            "account_id": account_id,
            "issued_date": issued_date,
            "permit_value": permit_value,
            "permit_type": permit_type,
        }
    )
    out["property_key"] = out.apply(
        lambda r: build_property_key(r["parcel_id"], r["account_id"], r["address"]), axis=1
    )

    cutoff = pd.Timestamp(datetime.now(timezone.utc)).tz_convert("UTC") - pd.DateOffset(years=years_window)
    recent = out[out["issued_date"] >= cutoff].copy()

    summary = (
        recent.groupby("property_key", as_index=False)
        .agg(
            permit_count_5y=("permit_type", "count"),
            permit_value_5y=("permit_value", "sum"),
            permit_types_5y=("permit_type", lambda x: ", ".join(sorted(set(x.dropna().astype(str)))[:5])),
            address=("address", "first"),
            parcel_id=("parcel_id", "first"),
            account_id=("account_id", "first"),
        )
    )
    return summary
