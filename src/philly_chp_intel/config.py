from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class AppConfig:
    """Application configuration loaded from YAML."""

    project_name: str
    raw_data_dir: Path
    processed_data_dir: Path
    datasets: dict[str, dict[str, Any]]
    scoring: dict[str, Any]


def load_config(config_path: str | Path) -> AppConfig:
    """Load YAML config from disk into typed AppConfig."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    base_dir = path.parent.parent if path.parent.name == "config" else Path(".")

    raw_dir = base_dir / payload["paths"]["raw_data_dir"]
    processed_dir = base_dir / payload["paths"]["processed_data_dir"]

    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    return AppConfig(
        project_name=payload.get("project_name", "Philly CHP Intelligence"),
        raw_data_dir=raw_dir,
        processed_data_dir=processed_dir,
        datasets=payload.get("datasets", {}),
        scoring=payload.get("scoring", {}),
    )
