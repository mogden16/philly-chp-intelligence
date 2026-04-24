# Philly CHP Intelligence

Python-first MVP for identifying and ranking Philadelphia commercial and industrial buildings with high combined heat and power (CHP) potential and deferred maintenance signals.

## Features
- Public-data ingestion from Philadelphia open data endpoints
- Modular cleaning pipeline into consistent building-level schema
- Master table assembly with one row per property/building
- Transparent, editable rules-based scoring
- Historical deferred maintenance scoring built on multi-signal subcomponents (not just current open issues)
- Explainability outputs and confidence flags
- Streamlit dashboard with KPIs, filters, map, detail panel, and CSV export

## Quick Start
1. Create and activate a virtual environment.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Copy config and update URLs/paths if needed:
   ```bash
   cp config/settings.sample.yml config/settings.local.yml
   ```
4. Run the data pipeline:
   ```bash
   python scripts/run_pipeline.py --config config/settings.local.yml
   ```
5. Launch Streamlit:
   ```bash
   streamlit run streamlit_app.py
   ```

## Data Flow
- Raw downloads: `data/raw/*.csv`
- Cleaned source tables: `data/processed/*_clean.parquet`
- Master table: `data/processed/master_buildings.parquet`
- Scored output: `data/processed/master_buildings_scored.parquet`

## Notes
- Benchmarking now uses two official 2023 City ArcGIS FeatureServer datasets (`reported` and `not_reported`) instead of one generic source.
- ArcGIS benchmarking feeds are ingested from JSON (`features[].attributes`) and normalized before cleaning/scoring.
- The sample config contains candidate public endpoints. Validate and update URLs as needed.
- Scoring weights and thresholds are in the config file and can be edited without code changes.
- Validation export is written to `data/processed/ranking_validation_export.csv` for scoring QA.
