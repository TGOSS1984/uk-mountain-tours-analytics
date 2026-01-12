from __future__ import annotations

import argparse

from src.utils.io import step

# Dims
from src.synth.generate_routes import build_dim_route
from src.synth.generate_guides import build_dim_guide

# APIs + transforms
from src.extract.bank_holidays import pull_bank_holidays
from src.transform.build_dim_date import build_dim_date
from src.transform.clean_bank_holidays import (
    build_dim_bank_holiday,
    build_bridge_bank_holiday_date,
    build_dim_region_division,
)

# Weather
from src.extract.weather_openmeteo_ukmo import pull_all_routes
from src.transform.build_weather_daily import build_daily

# Facts + modelling tables
from src.synth.generate_bookings import generate_fact_bookings
from src.transform.build_model_tables import build_route_day
from src.transform.validate_schema import validate

# SQL
from src.sql.build_sqlite import build_sqlite

# ML
from src.ml.train_forecast import train as train_baseline
from src.ml.predict_2026 import predict as predict_2026_baseline
from src.ml.train_forecast_time_split_v2 import train_time_split
from src.ml.predict_2026_v2 import predict as predict_2026_v2

from src.config import SEEDS, PROCESSED_DIR, RAW_DIR


def run(all_steps: bool = True, skip_weather: bool = False, skip_sql: bool = False, skip_ml: bool = False):
    # 1) Dims
    with step("DIM: routes"):
        build_dim_route(SEEDS.routes_json, PROCESSED_DIR / "dim_route.csv")

    with step("DIM: guides"):
        build_dim_guide(SEEDS.guides_json, PROCESSED_DIR / "dim_guide.csv")

    # 2) Bank holidays API -> raw
    with step("API: bank holidays (raw pull)"):
        pull_bank_holidays(RAW_DIR / "bank_holidays.json")

    # 3) dim_date
    with step("DIM: date (2024-2026) with bank holiday flags"):
        build_dim_date(
            start_date="2024-01-01",
            end_date="2026-12-31",
            bank_holidays_json=RAW_DIR / "bank_holidays.json",
            out_csv=PROCESSED_DIR / "dim_date.csv",
        )

    # 4) bank holiday modelling dims/bridge
    with step("TRANSFORM: bank holiday dimensions"):
        dim_bh = build_dim_bank_holiday(RAW_DIR / "bank_holidays.json", PROCESSED_DIR / "dim_bank_holiday.csv")
        build_bridge_bank_holiday_date(dim_bh, PROCESSED_DIR / "bridge_bank_holiday_date.csv")
        build_dim_region_division(PROCESSED_DIR / "dim_region_division.csv")

    # 5) Weather (optional)
    if not skip_weather:
        with step("API: weather (UKMO) pull -> hourly"):
            pull_all_routes()  # writes interim hourly + raw json

        with step("TRANSFORM: weather daily features"):
            build_daily()

    # 6) Synthetic facts
    with step("SYNTH: generate fact_bookings (2024-2025)"):
        generate_fact_bookings()

    with step("MODEL: build fact_route_day (2024-2025)"):
        build_route_day()

    # 7) Validate
    with step("QUALITY: validate schema + ranges"):
        validate()

    # 8) SQL warehouse (optional)
    if not skip_sql:
        with step("SQL: build sqlite warehouse"):
            build_sqlite()

    # 9) ML (optional)
    if not skip_ml:
        with step("ML: train baseline + predict 2026"):
            train_baseline()
            predict_2026_baseline()

        with step("ML: train time-split v2 + predict 2026 v2"):
            train_time_split()
            predict_2026_v2()


def main():
    parser = argparse.ArgumentParser(description="Run the end-to-end pipeline for the Power BI analytics project.")
    parser.add_argument("--skip-weather", action="store_true", help="Skip weather API pull + processing.")
    parser.add_argument("--skip-sql", action="store_true", help="Skip building SQLite warehouse.")
    parser.add_argument("--skip-ml", action="store_true", help="Skip ML training + 2026 predictions.")
    args = parser.parse_args()

    run(skip_weather=args.skip_weather, skip_sql=args.skip_sql, skip_ml=args.skip_ml)


if __name__ == "__main__":
    main()
