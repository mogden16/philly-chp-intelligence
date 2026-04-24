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
from philly_chp_intel.ingestion import load_raw_dataset
from philly_chp_intel.io_utils import write_parquet
from philly_chp_intel.pipeline.build_master import build_master_table
from philly_chp_intel.scoring import score_master_table


SOURCE_TO_CLEANER = {
    "benchmarking_reported_2023": clean_benchmarking,
    "benchmarking_not_reported_2023": clean_benchmarking,
    "opa_properties": clean_opa_properties,
    "land_use": clean_land_use,
    "li_permits": clean_permits,
    "li_unsafe_violations": clean_unsafe_violations,
    "complaints": clean_complaints,
    "parcels": clean_geometry,
}


BENCHMARK_REPORTED = "benchmarking_reported_2023"
BENCHMARK_NOT_REPORTED = "benchmarking_not_reported_2023"


def _parquet_safe(df: pd.DataFrame) -> pd.DataFrame:
    """Convert mixed-type object columns to strings for stable parquet writes."""
    out = df.copy()
    object_cols = out.select_dtypes(include=["object"]).columns
    for col in object_cols:
        out[col] = out[col].map(lambda v: None if pd.isna(v) else str(v))
    return out


def run_pipeline(config_path: str) -> None:
    """Run full ingest-clean-score pipeline and persist parquet outputs."""
    config = load_config(config_path)

    download_results = download_all(config.datasets, config.raw_data_dir)
    for result in download_results:
        print(f"[{result.dataset_name}] {result.message}")

    raw_normalized: dict[str, pd.DataFrame] = {}
    for dataset_name, meta in config.datasets.items():
        dataset_format = str(meta.get("format", "csv")).lower()
        ext = "json" if dataset_format == "json" else "csv"
        raw_path = config.raw_data_dir / f"{dataset_name}.{ext}"
        raw_df = load_raw_dataset(raw_path, dataset_format=dataset_format)
        raw_normalized[dataset_name] = raw_df

        if dataset_name in {BENCHMARK_REPORTED, BENCHMARK_NOT_REPORTED}:
            write_parquet(_parquet_safe(raw_df), config.processed_data_dir / f"{dataset_name}.parquet")

    benchmark_master_parts: list[pd.DataFrame] = []
    if BENCHMARK_REPORTED in raw_normalized and not raw_normalized[BENCHMARK_REPORTED].empty:
        reported = raw_normalized[BENCHMARK_REPORTED].copy()
        reported["benchmarking_source"] = "reported_2023"
        benchmark_master_parts.append(reported)
    if BENCHMARK_NOT_REPORTED in raw_normalized and not raw_normalized[BENCHMARK_NOT_REPORTED].empty:
        not_reported = raw_normalized[BENCHMARK_NOT_REPORTED].copy()
        not_reported["benchmarking_source"] = "not_reported_2023"
        benchmark_master_parts.append(not_reported)

    benchmark_master = (
        pd.concat(benchmark_master_parts, ignore_index=True, sort=False)
        if benchmark_master_parts
        else pd.DataFrame(columns=["benchmarking_source"])
    )
    write_parquet(_parquet_safe(benchmark_master), config.processed_data_dir / "benchmarking_master_2023.parquet")

    cleaned: dict[str, pd.DataFrame] = {}
    years_window = int(config.scoring.get("thresholds", {}).get("recent_years_window", 5))

    for dataset_name, cleaner in SOURCE_TO_CLEANER.items():
        raw_df = raw_normalized.get(dataset_name, pd.DataFrame())

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
        benchmark_df=pd.concat(
            [
                cleaned.get(BENCHMARK_REPORTED, pd.DataFrame(columns=["property_key"])),
                cleaned.get(BENCHMARK_NOT_REPORTED, pd.DataFrame(columns=["property_key"])),
            ],
            ignore_index=True,
            sort=False,
        ),
        land_use_df=cleaned.get("land_use", pd.DataFrame(columns=["property_key"])),
        permits_df=cleaned.get("li_permits", pd.DataFrame(columns=["property_key"])),
        violations_df=cleaned.get("li_unsafe_violations", pd.DataFrame(columns=["property_key"])),
        complaints_df=cleaned.get("complaints", pd.DataFrame(columns=["property_key"])),
        geometry_df=cleaned.get("parcels", pd.DataFrame(columns=["property_key"])),
    )
    write_parquet(master, config.processed_data_dir / "master_buildings.parquet")

    scored = score_master_table(master, config.scoring)
    write_parquet(scored, config.processed_data_dir / "master_buildings_scored.parquet")

    validation_cols = [
        "building_name",
        "address",
        "building_type",
        "final_opportunity_score",
        "chp_fit_score",
        "historical_deferred_maintenance_score",
        "project_size_score",
        "pursuit_score",
        "top_reason_1",
        "top_reason_2",
        "top_reason_3",
        "recommended_next_step",
        "data_confidence",
    ]
    validation_export = scored[[c for c in validation_cols if c in scored.columns]].rename(
        columns={"final_opportunity_score": "opportunity_score"}
    )
    validation_export.to_csv(config.processed_data_dir / "ranking_validation_export.csv", index=False)

    if "building_type_category" in scored.columns:
        top20 = scored.head(20)[["building_name", "address", "building_type_category", "final_opportunity_score"]]
        print("\nTop 20 buildings by opportunity score:")
        print(top20.to_string(index=False))

        dist = scored["building_type_category"].value_counts(normalize=True).mul(100).round(1)
        print("\nDistribution by building category (%):")
        print(dist.to_string())

        strategic = {
            "hospital_healthcare",
            "university_college",
            "multifamily_large",
            "industrial_process",
            "data_center_critical",
        }
        top50 = scored.head(50)
        strategic_pct = (top50["building_type_category"].isin(strategic).mean() * 100.0) if len(top50) else 0.0
        print(f"\nTop 50 strategic CHP categories: {strategic_pct:.1f}%")

    print(f"Pipeline complete. Records in scored master table: {len(scored):,}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Philly CHP Intelligence pipeline")
    parser.add_argument("--config", default="config/settings.sample.yml", help="Path to YAML config")
    args = parser.parse_args()
    run_pipeline(args.config)
