from __future__ import annotations

from hashlib import md5
from typing import Iterable

import pandas as pd


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Convert raw column names to snake_case-ish names."""
    renamed = {
        c: c.strip().lower().replace(" ", "_").replace("/", "_").replace("-", "_")
        for c in df.columns
    }
    return df.rename(columns=renamed)


def pick_first_column(df: pd.DataFrame, candidates: Iterable[str]) -> pd.Series:
    """Return first existing candidate column or empty values."""
    for col in candidates:
        if col in df.columns:
            return df[col]
    return pd.Series([None] * len(df), index=df.index)


def normalize_address(value: object) -> str | None:
    """Normalize address text for joins."""
    if value is None or pd.isna(value):
        return None
    text = str(value).strip().upper()
    return " ".join(text.split()) if text else None


def to_numeric(series: pd.Series) -> pd.Series:
    """Coerce series to numeric with NaNs on parse errors."""
    return pd.to_numeric(series, errors="coerce")


def parse_datetime(series: pd.Series) -> pd.Series:
    """Parse date/datetime values robustly."""
    return pd.to_datetime(series, errors="coerce", utc=True)


def build_property_key(parcel_id: object, account_id: object, address: object) -> str:
    """Build deterministic property key from parcel/account/address."""
    if parcel_id is not None and str(parcel_id).strip() and str(parcel_id).strip().lower() != "nan":
        return f"PARCEL::{str(parcel_id).strip().upper()}"
    if account_id is not None and str(account_id).strip() and str(account_id).strip().lower() != "nan":
        return f"ACCOUNT::{str(account_id).strip().upper()}"
    normalized = normalize_address(address)
    if normalized:
        digest = md5(normalized.encode("utf-8")).hexdigest()[:12]
        return f"ADDR::{digest}"
    return "UNKNOWN"


def bounded_score(series: pd.Series, min_value: float, max_value: float) -> pd.Series:
    """Scale values into 0..100 with clipping."""
    if max_value <= min_value:
        return pd.Series([0.0] * len(series), index=series.index)
    scaled = (series - min_value) / (max_value - min_value)
    return (scaled.clip(lower=0.0, upper=1.0) * 100.0).fillna(0.0)
