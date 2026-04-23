from __future__ import annotations

from pathlib import Path

import pandas as pd


def write_parquet(df: pd.DataFrame, path: Path) -> None:
    """Persist a DataFrame to parquet, creating parent dirs when needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def read_parquet(path: Path) -> pd.DataFrame:
    """Read parquet data as pandas DataFrame."""
    return pd.read_parquet(path)
