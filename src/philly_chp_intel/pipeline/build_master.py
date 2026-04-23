from __future__ import annotations

import pandas as pd


def _prepare(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["property_key"]).set_index("property_key")
    if "property_key" not in df.columns:
        raise ValueError("Input dataframe missing required 'property_key' column")
    return df.drop_duplicates(subset=["property_key"]).set_index("property_key")


def build_master_table(
    opa_df: pd.DataFrame,
    benchmark_df: pd.DataFrame,
    land_use_df: pd.DataFrame,
    permits_df: pd.DataFrame,
    violations_df: pd.DataFrame,
    complaints_df: pd.DataFrame,
    geometry_df: pd.DataFrame,
) -> pd.DataFrame:
    """Combine cleaned sources into one record per property_key."""
    inputs = [opa_df, benchmark_df, land_use_df, permits_df, violations_df, complaints_df, geometry_df]
    master_idx = pd.DataFrame(index=pd.Index([], name="property_key"))

    for next_df in inputs:
        if next_df.empty:
            continue
        master_idx = master_idx.combine_first(_prepare(next_df))

    master = master_idx.reset_index()

    if "land_use_category" not in master.columns:
        master["land_use_category"] = None

    numeric_defaults = {
        "square_footage": 0.0,
        "benchmark_site_eui": 0.0,
        "benchmark_source_eui": 0.0,
        "benchmark_ghg_intensity": 0.0,
        "permit_count_5y": 0,
        "permit_value_5y": 0.0,
        "unsafe_violation_count_5y": 0,
        "complaint_count_5y": 0,
    }
    for col, default in numeric_defaults.items():
        if col not in master.columns:
            master[col] = default
        master[col] = pd.to_numeric(master[col], errors="coerce").fillna(default)

    signal_cols = [
        "square_footage",
        "benchmark_site_eui",
        "permit_count_5y",
        "unsafe_violation_count_5y",
        "complaint_count_5y",
        "geometry_wkt",
    ]
    present = pd.DataFrame({c: master[c].notna() for c in signal_cols if c in master.columns})
    master["data_sources_count"] = present.sum(axis=1)

    master["building_name"] = master["building_name"].fillna("Unknown Building")
    master["address"] = master["address"].fillna("Unknown Address")
    master["building_type"] = master["building_type"].fillna("UNKNOWN")

    keep = [
        "property_key",
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
        "assessed_value",
        "year_built",
        "permit_count_5y",
        "permit_value_5y",
        "permit_types_5y",
        "unsafe_violation_count_5y",
        "unsafe_violation_levels_5y",
        "unsafe_violation_examples",
        "complaint_count_5y",
        "complaint_types_5y",
        "longitude",
        "latitude",
        "geometry_wkt",
        "data_sources_count",
    ]

    for col in keep:
        if col not in master.columns:
            master[col] = None

    output = master[keep].drop_duplicates(subset=["property_key"])
    return output
