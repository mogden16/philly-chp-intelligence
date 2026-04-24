from __future__ import annotations

import pandas as pd

from .common import build_property_key, normalize_address, normalize_columns, pick_first_column, to_numeric


def clean_opa_properties(df: pd.DataFrame) -> pd.DataFrame:
    """Clean OPA property and assessment dataset."""
    raw = normalize_columns(df.copy())

    address = pick_first_column(raw, ["street_address", "address", "mailing_address", "location"]).map(normalize_address)
    parcel_id = pick_first_column(raw, ["parcel_number", "mapreg", "parcel_id"])
    account_id = pick_first_column(raw, ["opa_account_num", "account_num", "account_id"])
    building_name = pick_first_column(raw, ["owner_1", "owner_name", "building_name"])
    building_type = pick_first_column(
        raw,
        ["market_value_description", "use_code_description", "building_type", "category_code_description", "building_code_description"],
    )
    sqft = to_numeric(pick_first_column(raw, ["total_livable_area", "gross_area", "building_area"]))
    assessed = to_numeric(pick_first_column(raw, ["market_value", "assessed_value", "total_assessment"]))
    year_built = to_numeric(pick_first_column(raw, ["year_built", "construction_year"]))

    cleaned = pd.DataFrame(
        {
            "building_name": building_name,
            "address": address,
            "parcel_id": parcel_id,
            "account_id": account_id,
            "building_type": building_type,
            "square_footage": sqft,
            "assessed_value": assessed,
            "year_built": year_built,
        }
    )
    cleaned["property_key"] = cleaned.apply(
        lambda r: build_property_key(r["parcel_id"], r["account_id"], r["address"]), axis=1
    )

    return cleaned.drop_duplicates(subset=["property_key"])
