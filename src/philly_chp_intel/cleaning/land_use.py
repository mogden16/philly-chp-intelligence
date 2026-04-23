from __future__ import annotations

import pandas as pd

from .common import build_property_key, normalize_address, normalize_columns, pick_first_column


def clean_land_use(df: pd.DataFrame) -> pd.DataFrame:
    """Clean land use categories by property."""
    raw = normalize_columns(df.copy())
    address = pick_first_column(raw, ["address", "street_address"]).map(normalize_address)
    parcel_id = pick_first_column(raw, ["mapreg", "parcel_id", "parcel_number"])
    account_id = pick_first_column(raw, ["opa_account_num", "account_num"])
    land_use = pick_first_column(raw, ["land_use", "zoning_description", "use_description"])

    out = pd.DataFrame(
        {
            "address": address,
            "parcel_id": parcel_id,
            "account_id": account_id,
            "land_use_category": land_use,
        }
    )
    out["property_key"] = out.apply(
        lambda r: build_property_key(r["parcel_id"], r["account_id"], r["address"]), axis=1
    )

    return out.drop_duplicates(subset=["property_key"])
