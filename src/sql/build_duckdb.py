from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import duckdb

from src.config import PROCESSED_DIR


@dataclass(frozen=True)
class Outputs:
    duckdb_path: Path = PROCESSED_DIR / "winter_tours.duckdb"


# Map processed CSV stems -> DuckDB table names
TABLES = [
    ("dim_route", "dim_route"),
    ("dim_guide", "dim_guide"),
    ("dim_date", "dim_date"),
    ("dim_bank_holiday", "dim_bank_holiday"),
    ("bridge_bank_holiday_date", "bridge_bank_holiday_date"),
    ("dim_region_division", "dim_region_division"),
    ("fact_bookings_2024_2025", "fact_bookings"),
    ("fact_route_day_2024_2025", "fact_route_day"),
    ("fact_route_week_2024_2025", "fact_route_week"),
    ("fact_forecast_week_2026", "fact_forecast_week_2026"),
]


def build_duckdb(outputs: Outputs = Outputs()) -> Path:
    outputs.duckdb_path.parent.mkdir(parents=True, exist_ok=True)

    if outputs.duckdb_path.exists():
        outputs.duckdb_path.unlink()

    con = duckdb.connect(str(outputs.duckdb_path))

    try:
        # Load each CSV into a DuckDB table
        for file_stem, table in TABLES:
            csv_path = PROCESSED_DIR / f"{file_stem}.csv"
            if not csv_path.exists():
                raise FileNotFoundError(f"Missing processed table: {csv_path}")

            con.execute(f"""
                CREATE TABLE {table} AS
                SELECT * FROM read_csv_auto('{csv_path.as_posix()}', HEADER=TRUE);
            """)

        # Optional: simple views for convenience
        con.execute("""
            CREATE VIEW v_weekly_actual_vs_forecast AS
            SELECT
                a.iso_year AS actual_year,
                a.iso_week,
                a.route_id,
                a.region,
                a.bookings_count AS actual_bookings,
                f.predicted_bookings_count AS forecast_bookings
            FROM fact_route_week a
            LEFT JOIN fact_forecast_week_2026 f
              ON a.iso_week = f.iso_week AND a.route_id = f.route_id;
        """)

    finally:
        con.close()

    return outputs.duckdb_path


if __name__ == "__main__":
    out = build_duckdb()
    print(f"Saved {out}")
