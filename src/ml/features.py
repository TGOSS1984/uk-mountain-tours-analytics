from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

from src.config import PROCESSED_DIR


@dataclass(frozen=True)
class Paths:
    fact_route_day: Path = PROCESSED_DIR / "fact_route_day_2024_2025.csv"
    dim_date: Path = PROCESSED_DIR / "dim_date.csv"
    dim_route: Path = PROCESSED_DIR / "dim_route.csv"
    dim_region_division: Path = PROCESSED_DIR / "dim_region_division.csv"
    dim_bank_holiday: Path = PROCESSED_DIR / "dim_bank_holiday.csv"


CLOSED_HOLIDAY_KEYWORDS = (
    "Christmas Day",
    "Boxing Day",
    "New Year",
    "New Year's Day",
    "New Year’s Day",
    "Good Friday",
    "Easter Monday",
)


def _build_closed_dates_by_division(bh: pd.DataFrame) -> dict[str, set]:
    bh = bh.copy()
    bh["title"] = bh["title"].astype(str)

    mask = False
    for kw in CLOSED_HOLIDAY_KEYWORDS:
        mask = mask | bh["title"].str.contains(kw, case=False, na=False)

    closed = bh[mask]
    out: dict[str, set] = {}
    for division, grp in closed.groupby("division"):
        out[division] = set(grp["date"].dt.date)
    return out


def _encode_categoricals(df: pd.DataFrame) -> Tuple[pd.DataFrame, list[str]]:
    """
    One-hot encode a small set of categoricals.
    Returns (df_encoded, feature_columns).
    """
    cat_cols = ["region", "difficulty", "season", "day_name"]
    existing = [c for c in cat_cols if c in df.columns]
    df2 = pd.get_dummies(df, columns=existing, drop_first=False)
    feature_cols = [c for c in df2.columns if c not in ("bookings_count",)]
    return df2, feature_cols


def build_training_frame(paths: Paths = Paths()) -> Tuple[pd.DataFrame, list[str]]:
    """
    Build ML training dataset from route-day facts (2024–2025).
    Target: bookings_count
    Features: calendar + route attributes + holiday flags + region mapping
    """
    df = pd.read_csv(paths.fact_route_day)

    # Keep only the columns we want, but be flexible
    keep = [
        "date_key",
        "date",
        "year",
        "quarter",
        "month",
        "iso_week",
        "day_name",
        "is_weekend",
        "season",
        "route_id",
        "region",
        "difficulty",
        "distance_km",
        "duration_hours",
        "is_bank_holiday_england_wales",
        "is_bank_holiday_scotland",
        "is_bank_holiday_northern_ireland",
        "is_bank_holiday_any",
        "bookings_count",
    ]
    df = df[[c for c in keep if c in df.columns]].copy()

    # Ensure types
    df["route_id"] = df["route_id"].astype(int)
    df["date_key"] = df["date_key"].astype(int)
    df["bookings_count"] = df["bookings_count"].astype(float)

    # Add "holiday_division" per region (helps pick the right holiday flag later)
    region_div = pd.read_csv(paths.dim_region_division)
    df = df.merge(region_div, on="region", how="left")
    df["division"] = df["division"].fillna("england-and-wales")

    # Division-aware bank holiday flag
    def div_bh(row) -> int:
        if row["division"] == "scotland":
            return int(row.get("is_bank_holiday_scotland", 0))
        if row["division"] == "northern-ireland":
            return int(row.get("is_bank_holiday_northern_ireland", 0))
        return int(row.get("is_bank_holiday_england_wales", 0))

    df["is_bank_holiday_division"] = df.apply(div_bh, axis=1)

    # Closed days flag (for later; training set already excludes some closures via generator)
    bh = pd.read_csv(paths.dim_bank_holiday, parse_dates=["date"])
    closed_by_div = _build_closed_dates_by_division(bh)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["is_closed_day"] = df.apply(
        lambda r: int(r["date"].date() in closed_by_div.get(r["division"], set())),
        axis=1,
    )

    # Encode categoricals
    encoded, feature_cols = _encode_categoricals(df)

    # Remove raw helper cols you don’t want as features
    drop_non_features = ["date", "division"]
    encoded = encoded.drop(columns=[c for c in drop_non_features if c in encoded.columns], errors="ignore")

    # Ensure features exclude target
    feature_cols = [c for c in encoded.columns if c != "bookings_count"]
    return encoded, feature_cols


def build_scoring_frame_2026(paths: Paths = Paths()) -> Tuple[pd.DataFrame, list[str]]:
    """
    Create the 2026 route-day scaffold (all routes x all dates in 2026),
    build the same features as training frame (no target).
    """
    dim_date = pd.read_csv(paths.dim_date, parse_dates=["date"])
    dim_route = pd.read_csv(paths.dim_route)
    region_div = pd.read_csv(paths.dim_region_division)
    bh = pd.read_csv(paths.dim_bank_holiday, parse_dates=["date"])

    # 2026 only
    d26 = dim_date[(dim_date["date"] >= "2026-01-01") & (dim_date["date"] <= "2026-12-31")].copy()
    d26["date_key"] = d26["date_key"].astype(int)

    # Route scaffold
    routes = dim_route[["route_id", "region", "difficulty", "distance_km", "duration_hours"]].copy()
    routes["route_id"] = routes["route_id"].astype(int)

    # Cartesian product: dates x routes
    d26["key"] = 1
    routes["key"] = 1
    base = d26.merge(routes, on="key", how="outer").drop(columns=["key"])

    # Map region -> division
    base = base.merge(region_div, on="region", how="left")
    base["division"] = base["division"].fillna("england-and-wales")

    # Division-aware bank holiday flag
    def div_bh(row) -> int:
        if row["division"] == "scotland":
            return int(row.get("is_bank_holiday_scotland", 0))
        if row["division"] == "northern-ireland":
            return int(row.get("is_bank_holiday_northern_ireland", 0))
        return int(row.get("is_bank_holiday_england_wales", 0))

    base["is_bank_holiday_division"] = base.apply(div_bh, axis=1)

    # Closed days based on key holiday keywords
    closed_by_div = _build_closed_dates_by_division(bh)
    base["is_closed_day"] = base.apply(
        lambda r: int(pd.Timestamp(r["date"]).date() in closed_by_div.get(r["division"], set())),
        axis=1,
    )

    # Keep same schema (no target)
    keep = [
        "date_key",
        "date",
        "year",
        "quarter",
        "month",
        "iso_week",
        "day_name",
        "is_weekend",
        "season",
        "route_id",
        "region",
        "difficulty",
        "distance_km",
        "duration_hours",
        "is_bank_holiday_england_wales",
        "is_bank_holiday_scotland",
        "is_bank_holiday_northern_ireland",
        "is_bank_holiday_any",
        "is_bank_holiday_division",
        "is_closed_day",
        "division",
    ]
    base = base[[c for c in keep if c in base.columns]].copy()

    encoded, feature_cols = _encode_categoricals(base)
    encoded = encoded.drop(columns=["date", "division"], errors="ignore")
    feature_cols = [c for c in encoded.columns if c != "bookings_count"]
    return encoded, feature_cols
