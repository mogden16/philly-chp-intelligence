from __future__ import annotations

import pandas as pd

from .common import build_property_key, normalize_address, normalize_columns, pick_first_column, to_numeric


def clean_benchmarking(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize Philadelphia building benchmarking records."""
    raw = normalize_columns(df.copy())

    building_name = pick_first_column(raw, ["building_name", "property_name", "facility_name"])
    address = pick_first_column(raw, ["address", "property_address", "street_address"]).map(normalize_address)
    parcel_id = pick_first_column(raw, ["parcel_id", "opa_account_num", "mapreg"])
    account_id = pick_first_column(raw, ["account_num", "opa_account_num", "account_id"])
    building_type = pick_first_column(raw, ["primary_property_type", "building_type", "property_type"])
    sqft = to_numeric(pick_first_column(raw, ["gross_floor_area", "gross_floor_area_sq_ft", "square_feet", "bldg_sqft"]))
    site_eui = to_numeric(pick_first_column(raw, ["site_eui_kbtu_ft2", "site_eui", "weather_normalized_site_eui"]))
    source_eui = to_numeric(pick_first_column(raw, ["source_eui_kbtu_ft2", "source_eui", "weather_normalized_source_eui"]))
    ghg = to_numeric(pick_first_column(raw, ["ghg_emissions_intensity", "ghg_intensity", "total_ghg_emissions_intensity"]))

    cleaned = pd.DataFrame(
        {
            "building_name": building_name,
            "address": address,
            "parcel_id": parcel_id,
            "account_id": account_id,
            "building_type": building_type,
            "square_footage": sqft,
            "benchmark_site_eui": site_eui,
            "benchmark_source_eui": source_eui,
            "benchmark_ghg_intensity": ghg,
            "benchmark_record_count": 1,
        }
    )
    cleaned["property_key"] = cleaned.apply(
        lambda r: build_property_key(r["parcel_id"], r["account_id"], r["address"]), axis=1
    )

    grouped = (
        cleaned.groupby("property_key", as_index=False)
        .agg(
            {
                "building_name": "first",
                "address": "first",
                "parcel_id": "first",
                "account_id": "first",
                "building_type": "first",
                "square_footage": "max",
                "benchmark_site_eui": "mean",
                "benchmark_source_eui": "mean",
                "benchmark_ghg_intensity": "mean",
                "benchmark_record_count": "sum",
            }
        )
    )
    return grouped
