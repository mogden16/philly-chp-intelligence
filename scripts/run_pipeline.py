from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from philly_chp_intel.cleaning.benchmarking import clean_benchmarking
from philly_chp_intel.cleaning.complaints import clean_complaints
from philly_chp_intel.cleaning.geometry import clean_geometry
from philly_chp_intel.cleaning.land_use import clean_land_use
from philly_chp_intel.cleaning.opa import clean_opa_properties
from philly_chp_intel.cleaning.permits import clean_permits
from philly_chp_intel.cleaning.violations import clean_unsafe_violations
from philly_chp_intel.config import load_config
from philly_chp_intel.downloader import download_all
from philly_chp_intel.io_utils import write_parquet
from philly_chp_intel.pipeline.build_master import build_master_table
from philly_chp_intel.scoring import score_master_table


SOURCE_TO_CLEANER = {
    "benchmarking": clean_benchmarking,
    "opa_properties": clean_opa_properties,
    "land_use": clean_land_use,
    "li_permits": clean_permits,
    "li_unsafe_violations": clean_unsafe_violations,
    "complaints": clean_complaints,
    "parcels": clean_geometry,
}


def _read_csv_or_empty(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, low_memory=False)
    except Exception:
        return pd.DataFrame()


def run_pipeline(config_path: str) -> None:
    """Run full ingest-clean-score pipeline and persist parquet outputs."""
    config = load_config(config_path)

    download_results = download_all(config.datasets, config.raw_data_dir)
    for result in download_results:
        print(f"[{result.dataset_name}] {result.message}")

    cleaned: dict[str, pd.DataFrame] = {}
    years_window = int(config.scoring.get("thresholds", {}).get("recent_years_window", 5))

    for dataset_name, cleaner in SOURCE_TO_CLEANER.items():
        raw_path = config.raw_data_dir / f"{dataset_name}.csv"
        raw_df = _read_csv_or_empty(raw_path)

        if raw_df.empty:
            cleaned_df = pd.DataFrame(columns=["property_key"])
        elif dataset_name == "li_permits":
            cleaned_df = cleaner(raw_df, years_window=years_window)
        elif dataset_name == "li_unsafe_violations":
            cleaned_df = cleaner(raw_df, years_window=years_window)
        elif dataset_name == "complaints":
            cleaned_df = cleaner(raw_df, years_window=years_window)
        else:
            cleaned_df = cleaner(raw_df)

        cleaned[dataset_name] = cleaned_df
        write_parquet(cleaned_df, config.processed_data_dir / f"{dataset_name}_clean.parquet")

    master = build_master_table(
        opa_df=cleaned.get("opa_properties", pd.DataFrame(columns=["property_key"])),
        benchmark_df=cleaned.get("benchmarking", pd.DataFrame(columns=["property_key"])),
        land_use_df=cleaned.get("land_use", pd.DataFrame(columns=["property_key"])),
        permits_df=cleaned.get("li_permits", pd.DataFrame(columns=["property_key"])),
        violations_df=cleaned.get("li_unsafe_violations", pd.DataFrame(columns=["property_key"])),
        complaints_df=cleaned.get("complaints", pd.DataFrame(columns=["property_key"])),
        geometry_df=cleaned.get("parcels", pd.DataFrame(columns=["property_key"])),
    )
    write_parquet(master, config.processed_data_dir / "master_buildings.parquet")

    scored = score_master_table(master, config.scoring)
    write_parquet(scored, config.processed_data_dir / "master_buildings_scored.parquet")

    print(f"Pipeline complete. Records in scored master table: {len(scored):,}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Philly CHP Intelligence pipeline")
    parser.add_argument("--config", default="config/settings.sample.yml", help="Path to YAML config")
    args = parser.parse_args()
    run_pipeline(args.config)
