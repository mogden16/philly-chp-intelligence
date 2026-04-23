from __future__ import annotations

import pandas as pd
from shapely import wkt
from shapely.geometry import Point

from .common import build_property_key, normalize_address, normalize_columns, pick_first_column, to_numeric


def _safe_point(lon: float | None, lat: float | None):
    if lon is None or lat is None:
        return None
    if pd.isna(lon) or pd.isna(lat):
        return None
    return Point(float(lon), float(lat))


def clean_geometry(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize geometry and coordinate fields."""
    raw = normalize_columns(df.copy())
    address = pick_first_column(raw, ["address", "street_address"]).map(normalize_address)
    parcel_id = pick_first_column(raw, ["parcel_id", "mapreg", "parcel_number"])
    account_id = pick_first_column(raw, ["account_num", "opa_account_num"])
    lat = to_numeric(pick_first_column(raw, ["lat", "latitude", "y"]))
    lon = to_numeric(pick_first_column(raw, ["lng", "lon", "longitude", "x"]))
    wkt_col = pick_first_column(raw, ["the_geom", "geometry", "wkt"])

    geom = []
    for i in range(len(raw)):
        shape = None
        raw_wkt = wkt_col.iloc[i]
        if raw_wkt is not None and not pd.isna(raw_wkt):
            try:
                text = str(raw_wkt)
                if text.startswith("SRID="):
                    text = text.split(";", 1)[1]
                shape = wkt.loads(text)
            except Exception:
                shape = None
        if shape is None:
            shape = _safe_point(lon.iloc[i], lat.iloc[i])
        geom.append(shape)

    out = pd.DataFrame(
        {
            "address": address,
            "parcel_id": parcel_id,
            "account_id": account_id,
            "latitude": lat,
            "longitude": lon,
            "geometry_wkt": [g.wkt if g is not None else None for g in geom],
        }
    )
    out["property_key"] = out.apply(
        lambda r: build_property_key(r["parcel_id"], r["account_id"], r["address"]), axis=1
    )
    return out.drop_duplicates(subset=["property_key"])
