from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.config import PROCESSED_DIR


@dataclass(frozen=True)
class Files:
    fact_route_day: Path = PROCESSED_DIR / "fact_route_day_2024_2025.csv"
    dim_date: Path = PROCESSED_DIR / "dim_date.csv"
    dim_route: Path = PROCESSED_DIR / "dim_route.csv"
    out_route_week: Path = PROCESSED_DIR / "fact_route_week_2024_2025.csv"


def _ensure_dim_date_iso_columns(dd: pd.DataFrame) -> pd.DataFrame:
    if "date" not in dd.columns:
        raise KeyError(
            "dim_date must contain a 'date' column to derive ISO week/year. "
            f"Found columns: {list(dd.columns)}"
        )

    dd["date"] = pd.to_datetime(dd["date"], errors="coerce")
    dd = dd.dropna(subset=["date"])

    if "iso_year" not in dd.columns or "iso_week" not in dd.columns:
        iso = dd["date"].dt.isocalendar()
        dd["iso_year"] = iso.year.astype(int)
        dd["iso_week"] = iso.week.astype(int)

    return dd


def _resolve_prefer_y(df: pd.DataFrame, base: str) -> pd.DataFrame:
    """
    Normalise potentially suffixed columns back to base name.

    Prefer *_y (dim_date) over *_x (fact) when both exist.
    """
    y = f"{base}_y"
    x = f"{base}_x"

    if base in df.columns:
        return df

    if y in df.columns:
        df[base] = df[y]
        return df

    if x in df.columns:
        df[base] = df[x]
        return df

    return df


def build_route_week(files: Files = Files()) -> Path:
    day = pd.read_csv(files.fact_route_day)
    dim_date = pd.read_csv(files.dim_date)
    dim_route = pd.read_csv(files.dim_route)

    # ---- Ensure key types ----
    if "date_key" not in day.columns:
        raise KeyError(f"fact_route_day missing date_key. Found: {list(day.columns)}")
    if "route_id" not in day.columns:
        raise KeyError(f"fact_route_day missing route_id. Found: {list(day.columns)}")

    day["date_key"] = pd.to_numeric(day["date_key"], errors="coerce").astype("Int64")
    day["route_id"] = pd.to_numeric(day["route_id"], errors="coerce").astype("Int64")
    day = day.dropna(subset=["date_key", "route_id"]).copy()
    day["date_key"] = day["date_key"].astype(int)
    day["route_id"] = day["route_id"].astype(int)

    dim_date["date_key"] = pd.to_numeric(dim_date["date_key"], errors="coerce").astype("Int64")
    dim_date = dim_date.dropna(subset=["date_key"]).copy()
    dim_date["date_key"] = dim_date["date_key"].astype(int)

    dim_route["route_id"] = pd.to_numeric(dim_route["route_id"], errors="coerce").astype("Int64")
    dim_route = dim_route.dropna(subset=["route_id"]).copy()
    dim_route["route_id"] = dim_route["route_id"].astype(int)

    # ---- Ensure iso columns exist on dim_date ----
    dim_date = _ensure_dim_date_iso_columns(dim_date)

    # ---- Select columns from dim_date (only those that exist) ----
    dd_cols = [
        "date_key",
        "date",
        "iso_year",
        "iso_week",
        "year",
        "month",
        "quarter",
        "season",
        "is_bank_holiday_any",
        "is_weekend",
        "day_name",
    ]
    dd_cols = [c for c in dd_cols if c in dim_date.columns]
    dd = dim_date[dd_cols].copy()

    # Merge (suffixes to prevent overwrites)
    merged = day.merge(dd, on="date_key", how="left", suffixes=("_x", "_y"))

    # Normalise ISO + calendar columns back to base names
    for col in ["iso_year", "iso_week", "is_bank_holiday_any", "is_weekend"]:
        merged = _resolve_prefer_y(merged, col)

    # If region isn't in fact_route_day, bring it from dim_route
    if "region" not in merged.columns:
        if "region" not in dim_route.columns:
            raise KeyError(
                "No 'region' column found in fact_route_day or dim_route. "
                "Weekly aggregation expects region."
            )
        merged = merged.merge(dim_route[["route_id", "region"]], on="route_id", how="left")

    # Ensure ISO columns exist
    missing_iso = [c for c in ["iso_year", "iso_week"] if c not in merged.columns]
    if missing_iso:
        raise KeyError(
            f"Missing {missing_iso} after merging dim_date. "
            f"dim_date columns: {list(dim_date.columns)}; merged columns: {list(merged.columns)}"
        )

    # Coerce ISO numeric
    merged["iso_year"] = pd.to_numeric(merged["iso_year"], errors="coerce").astype("Int64")
    merged["iso_week"] = pd.to_numeric(merged["iso_week"], errors="coerce").astype("Int64")
    merged = merged.dropna(subset=["iso_year", "iso_week"]).copy()
    merged["iso_year"] = merged["iso_year"].astype(int)
    merged["iso_week"] = merged["iso_week"].astype(int)

    # If calendar flags still missing (different naming upstream), default to 0 so pipeline runs
    if "is_bank_holiday_any" not in merged.columns:
        merged["is_bank_holiday_any"] = 0
    if "is_weekend" not in merged.columns:
        merged["is_weekend"] = 0

    merged["is_bank_holiday_any"] = pd.to_numeric(merged["is_bank_holiday_any"], errors="coerce").fillna(0).astype(int)
    merged["is_weekend"] = pd.to_numeric(merged["is_weekend"], errors="coerce").fillna(0).astype(int)

    # ---- Weekly aggregation at route-week grain ----
    weekly = (
        merged.groupby(["iso_year", "iso_week", "route_id", "region"], as_index=False)
        .agg(
            # Demand
            bookings_count=("bookings_count", "sum"),
            party_size_total=("party_size_total", "sum"),

            # Commercials
            sales_ex_vat=("sales_ex_vat", "sum"),
            vat_amount=("vat_amount", "sum"),
            sales_inc_vat=("sales_inc_vat", "sum"),
            staff_cost=("staff_cost", "sum"),
            margin_amount=("margin_amount", "sum"),
            discount_bookings=("discount_bookings", "sum"),

            # Calendar signals (counts inside the week)
            bank_holiday_days_any=("is_bank_holiday_any", "sum"),
            weekend_days=("is_weekend", "sum"),
        )
    )

    weekly["discount_rate"] = weekly["discount_bookings"] / weekly["bookings_count"].replace(0, pd.NA)
    weekly["margin_pct_weighted"] = weekly["margin_amount"] / weekly["sales_ex_vat"].replace(0, pd.NA)

    # ---- Join route attributes (for ML features) ----
    route_attr_cols = ["route_id", "difficulty", "distance_km", "duration_hours"]
    route_attr_cols = [c for c in route_attr_cols if c in dim_route.columns]
    weekly = weekly.merge(dim_route[route_attr_cols], on="route_id", how="left")

    # ---- Add stable week_start (Monday) ----
    weekly["week_start"] = pd.to_datetime(
        weekly["iso_year"].astype(str)
        + "-W"
        + weekly["iso_week"].astype(str).str.zfill(2)
        + "-1",
        format="%G-W%V-%u",
        errors="coerce",
    )

    # ---- Filter to just the years we care about (2024–2025) ----
    weekly = weekly[(weekly["iso_year"] >= 2024) & (weekly["iso_year"] <= 2025)].copy()

    weekly = weekly.sort_values(["iso_year", "iso_week", "route_id"])

    files.out_route_week.parent.mkdir(parents=True, exist_ok=True)
    weekly.to_csv(files.out_route_week, index=False)
    return files.out_route_week


if __name__ == "__main__":
    out = build_route_week()
    print(f"Saved {out}")
