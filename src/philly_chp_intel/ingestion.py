from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .cleaning.common import normalize_columns


def _arcgis_features_to_dataframe(payload: dict[str, Any]) -> pd.DataFrame:
    """Flatten ArcGIS FeatureServer JSON into a DataFrame."""
    features = payload.get("features", [])
    rows: list[dict[str, Any]] = []

    for feature in features:
        attributes = dict(feature.get("attributes") or {})
        geometry = feature.get("geometry")
        if geometry:
            attributes["geometry_json"] = json.dumps(geometry, separators=(",", ":"))
            attributes["geometry_x"] = geometry.get("x")
            attributes["geometry_y"] = geometry.get("y")
        rows.append(attributes)

    return pd.DataFrame(rows)


def load_raw_dataset(raw_path: Path, dataset_format: str) -> pd.DataFrame:
    """Load and normalize a raw dataset file by configured format."""
    if not raw_path.exists():
        return pd.DataFrame()

    fmt = dataset_format.lower().strip()
    if fmt == "json":
        try:
            payload = json.loads(raw_path.read_text(encoding="utf-8"))
            df = _arcgis_features_to_dataframe(payload)
            return normalize_columns(df)
        except Exception:
            return pd.DataFrame()

    try:
        df = pd.read_csv(raw_path, low_memory=False)
        return normalize_columns(df)
    except Exception:
        return pd.DataFrame()
