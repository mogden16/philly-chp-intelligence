from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import requests

DatasetFormat = Literal["csv", "json"]


@dataclass(frozen=True)
class DownloadResult:
    """Metadata for a download outcome."""

    dataset_name: str
    output_path: Path
    success: bool
    message: str


def _extension_for_format(dataset_format: DatasetFormat) -> str:
    """Map a dataset format to a raw-file extension."""
    return "json" if dataset_format == "json" else "csv"


def download_dataset(
    url: str,
    output_path: Path,
    dataset_format: DatasetFormat = "csv",
    timeout_seconds: int = 90,
) -> DownloadResult:
    """Download a public dataset (CSV or JSON) to disk."""
    try:
        response = requests.get(url, timeout=timeout_seconds)
        response.raise_for_status()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(response.content)
        return DownloadResult(
            dataset_name=output_path.stem,
            output_path=output_path,
            success=True,
            message=f"Downloaded {output_path.name}",
        )
    except requests.RequestException as exc:
        return DownloadResult(
            dataset_name=output_path.stem,
            output_path=output_path,
            success=False,
            message=f"Failed: {exc}",
        )


def download_all(datasets: dict[str, dict], raw_dir: Path) -> list[DownloadResult]:
    """Download all configured datasets to the raw directory."""
    results: list[DownloadResult] = []
    for dataset_name, meta in datasets.items():
        url = str(meta.get("url", "")).strip()
        dataset_format = str(meta.get("format", "csv")).strip().lower()
        if dataset_format not in {"csv", "json"}:
            dataset_format = "csv"
        ext = _extension_for_format(dataset_format)
        output_path = raw_dir / f"{dataset_name}.{ext}"

        if not url:
            results.append(
                DownloadResult(
                    dataset_name=dataset_name,
                    output_path=output_path,
                    success=False,
                    message="No URL configured",
                )
            )
            continue
        result = download_dataset(url=url, output_path=output_path, dataset_format=dataset_format)
        results.append(
            DownloadResult(
                dataset_name=dataset_name,
                output_path=result.output_path,
                success=result.success,
                message=result.message,
            )
        )
    return results
