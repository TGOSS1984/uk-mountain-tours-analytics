from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.config import PROCESSED_DIR


@dataclass(frozen=True)
class Files:
    fact_bookings: Path = PROCESSED_DIR / "fact_bookings_2024_2025.csv"
    dim_date: Path = PROCESSED_DIR / "dim_date.csv"
    dim_route: Path = PROCESSED_DIR / "dim_route.csv"
    out_route_day: Path = PROCESSED_DIR / "fact_route_day_2024_2025.csv"


def build_route_day(files: Files = Files()) -> Path:
    fact = pd.read_csv(files.fact_bookings)
    dim_date = pd.read_csv(files.dim_date)
    dim_route = pd.read_csv(files.dim_route)

    # Aggregate booking-level to route-day grain
    grp = (
        fact.groupby(["date_key", "route_id", "region"], as_index=False)
        .agg(
            bookings_count=("booking_id", "count"),
            party_size_total=("party_size", "sum"),
            party_size_avg=("party_size", "mean"),
            sales_ex_vat=("sales_ex_vat", "sum"),
            vat_amount=("vat_amount", "sum"),
            sales_inc_vat=("sales_inc_vat", "sum"),
            staff_cost=("staff_cost", "sum"),
            margin_amount=("margin_amount", "sum"),
            discount_bookings=("discount_flag", "sum"),
        )
    )

    grp["discount_rate"] = grp["discount_bookings"] / grp["bookings_count"]
    grp["margin_pct_weighted"] = grp["margin_amount"] / grp["sales_ex_vat"].replace(0, pd.NA)

    # Join date attributes (for ML features + BI slicing)
    model = grp.merge(dim_date, on="date_key", how="left")

    # Join route attributes (difficulty, duration, distance, lat/lon)
    model = model.merge(
        dim_route[["route_id", "difficulty", "duration_hours", "distance_km", "route_lat", "route_lon"]],
        on="route_id",
        how="left",
    )

    # Order columns for readability
    preferred = [
        "date_key",
        "date",
        "year",
        "quarter",
        "month",
        "month_name",
        "iso_year",
        "iso_week",
        "day_name",
        "is_weekend",
        "season",
        "route_id",
        "region",
        "difficulty",
        "distance_km",
        "duration_hours",
        "route_lat",
        "route_lon",
        "bookings_count",
        "party_size_total",
        "party_size_avg",
        "discount_bookings",
        "discount_rate",
        "sales_ex_vat",
        "vat_amount",
        "sales_inc_vat",
        "staff_cost",
        "margin_amount",
        "margin_pct_weighted",
        "is_bank_holiday_england_wales",
        "is_bank_holiday_scotland",
        "is_bank_holiday_northern_ireland",
        "is_bank_holiday_any",
    ]
    cols = [c for c in preferred if c in model.columns] + [c for c in model.columns if c not in preferred]
    model = model[cols].sort_values(["date_key", "route_id"])

    files.out_route_day.parent.mkdir(parents=True, exist_ok=True)
    model.to_csv(files.out_route_day, index=False)
    return files.out_route_day


if __name__ == "__main__":
    out = build_route_day()
    print(f"Saved {out}")
