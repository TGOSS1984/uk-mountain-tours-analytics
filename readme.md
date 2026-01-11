# Power BI Winter Mountain Tours Analytics (Portfolio)

This repo will demonstrate:
- Synthetic dataset generation (2024–2025)
- API ingestion (UK bank holidays + mountain forecasts)
- Data cleaning + modelling (star schema for Power BI)
- Optional SQL warehouse (SQLite/DuckDB)
- ML forecasting layer (predict 2026 demand)
- Power BI dashboard as the end product

## Repo Structure
- data/raw: seed + API pulls (raw)
- data/interim: lightly transformed
- data/processed: model-ready tables for Power BI
- src: pipelines (extract/synth/transform/sql/ml)
- powerbi: PBIX/PBIT (optional in git)
