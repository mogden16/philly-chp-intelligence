from __future__ import annotations

from pathlib import Path

import pandas as pd


def _parquet_safe(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize mixed object columns so pyarrow can serialize them safely."""
    out = df.copy()
    object_cols = out.select_dtypes(include=["object"]).columns
    for col in object_cols:
        out[col] = out[col].map(lambda v: None if pd.isna(v) else str(v))
    return out


def write_parquet(df: pd.DataFrame, path: Path) -> None:
    """Persist a DataFrame to parquet, creating parent dirs when needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    _parquet_safe(df).to_parquet(path, index=False)


def read_parquet(path: Path) -> pd.DataFrame:
    """Read parquet data as pandas DataFrame."""
    return pd.read_parquet(path)
