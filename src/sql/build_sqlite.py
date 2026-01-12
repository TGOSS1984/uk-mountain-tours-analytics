from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
import pandas as pd

from src.config import PROCESSED_DIR


@dataclass(frozen=True)
class Inputs:
    dim_route: Path = PROCESSED_DIR / "dim_route.csv"
    dim_guide: Path = PROCESSED_DIR / "dim_guide.csv"
    dim_date: Path = PROCESSED_DIR / "dim_date.csv"
    dim_bank_holiday: Path = PROCESSED_DIR / "dim_bank_holiday.csv"
    bridge_bank_holiday_date: Path = PROCESSED_DIR / "bridge_bank_holiday_date.csv"
    dim_region_division: Path = PROCESSED_DIR / "dim_region_division.csv"
    fact_bookings: Path = PROCESSED_DIR / "fact_bookings_2024_2025.csv"
    fact_route_day: Path = PROCESSED_DIR / "fact_route_day_2024_2025.csv"


@dataclass(frozen=True)
class Outputs:
    sqlite_db: Path = PROCESSED_DIR / "winter_tours.sqlite"


TABLES = [
    ("dim_route", "dim_route"),
    ("dim_guide", "dim_guide"),
    ("dim_date", "dim_date"),
    ("dim_bank_holiday", "dim_bank_holiday"),
    ("bridge_bank_holiday_date", "bridge_bank_holiday_date"),
    ("dim_region_division", "dim_region_division"),
    ("fact_bookings_2024_2025", "fact_bookings"),
    ("fact_route_day_2024_2025", "fact_route_day"),

    # NEW: weekly
    ("fact_route_week_2024_2025", "fact_route_week"),

    # NEW: 2026 weekly forecast
    ("fact_forecast_week_2026", "fact_forecast_week_2026"),

]


def _to_sqlite(df: pd.DataFrame, con: sqlite3.Connection, table: str) -> None:
    df.to_sql(table, con, if_exists="replace", index=False)


def _create_indexes(con: sqlite3.Connection) -> None:
    cur = con.cursor()

    # Dim keys
    cur.execute("CREATE INDEX IF NOT EXISTS idx_dim_date_date_key ON dim_date(date_key);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_dim_route_route_id ON dim_route(route_id);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_dim_guide_guide_id ON dim_guide(guide_id);")

    # Facts
    cur.execute("CREATE INDEX IF NOT EXISTS idx_fact_bookings_date_key ON fact_bookings(date_key);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_fact_bookings_route_id ON fact_bookings(route_id);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_fact_bookings_guide_id ON fact_bookings(guide_id);")

    cur.execute("CREATE INDEX IF NOT EXISTS idx_fact_route_day_date_key ON fact_route_day(date_key);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_fact_route_day_route_id ON fact_route_day(route_id);")

    # Holidays
    cur.execute("CREATE INDEX IF NOT EXISTS idx_bridge_hol_date ON bridge_bank_holiday_date(date);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_bridge_hol_id ON bridge_bank_holiday_date(bank_holiday_id);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_dim_bank_holiday_id ON dim_bank_holiday(bank_holiday_id);")

    # Weekly actuals
    cur.execute("CREATE INDEX IF NOT EXISTS idx_fact_route_week_route_id ON fact_route_week(route_id);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_fact_route_week_iso ON fact_route_week(iso_year, iso_week);")

    # Weekly forecast
    cur.execute("CREATE INDEX IF NOT EXISTS idx_fcst_week_route_id ON fact_forecast_week_2026(route_id);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_fcst_week_iso ON fact_forecast_week_2026(iso_year, iso_week);")

    con.commit()


def build_sqlite(inputs: Inputs = Inputs(), outputs: Outputs = Outputs()) -> Path:
    outputs.sqlite_db.parent.mkdir(parents=True, exist_ok=True)

    if outputs.sqlite_db.exists():
        outputs.sqlite_db.unlink()

    con = sqlite3.connect(outputs.sqlite_db)

    try:
        # Load & write tables
        for file_stem, target_table in TABLES:
            csv_path = PROCESSED_DIR / f"{file_stem}.csv"
            if not csv_path.exists():
                raise FileNotFoundError(f"Missing processed table: {csv_path}")

            # Parse dates where useful
            if target_table in ("dim_date", "dim_bank_holiday", "bridge_bank_holiday_date"):
                df = pd.read_csv(csv_path, parse_dates=["date"])
            else:
                df = pd.read_csv(csv_path)

            _to_sqlite(df, con, target_table)

        _create_indexes(con)
    finally:
        con.close()

    return outputs.sqlite_db


if __name__ == "__main__":
    out = build_sqlite()
    print(f"Saved {out}")
