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
    weather_daily: Path = PROCESSED_DIR / "fact_weather_daily_ukmo.csv"


CLOSED_HOLIDAY_KEYWORDS = (
    "Christmas Day",
    "Boxing Day",
    "New Year",
    "New Year's Day",
    "New Year’s Day",
    "Good Friday",
    "Easter Monday",
)


WEATHER_COLS = [
    "temp_mean",
    "temp_min",
    "temp_max",
    "precip_sum",
    "snowfall_sum",
    "windspeed_mean",
    "windgusts_max",
    "severity_index",
]


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


def _stable_unit(seed: str) -> float:
    """
    Deterministic float in [0,1) from a string.
    """
    h = 2166136261
    for b in seed.encode("utf-8"):
        h ^= b
        h = (h * 16777619) & 0xFFFFFFFF
    return (h % 10_000_000) / 10_000_000.0


def _seasonal_temp_baseline(season: str) -> float:
    return {
        "winter": 2.5,
        "spring": 8.0,
        "summer": 13.0,
        "autumn": 7.5,
    }.get(season, 8.0)


def _add_weather_features(
    df: pd.DataFrame,
    weather_daily_path: Path,
    use_fallback: bool = True,
) -> pd.DataFrame:
    """
    Left-join daily weather (route_id, date_key).
    If missing, optionally fill with deterministic seasonal synthetic weather.
    """
    out = df.copy()

    if weather_daily_path.exists():
        w = pd.read_csv(weather_daily_path)
        w["route_id"] = w["route_id"].astype(int)
        w["date_key"] = w["date_key"].astype(int)
        out = out.merge(w[["route_id", "date_key"] + WEATHER_COLS], on=["route_id", "date_key"], how="left")
    else:
        # Create empty cols so schema is stable
        for c in WEATHER_COLS:
            out[c] = np.nan

    if not use_fallback:
        return out

    # Fill missing weather with deterministic “synthetic meteorology”
    # based on season + stable noise per route_id/date_key.
    def fill_row(r):
        if pd.notna(r.get("temp_mean")):
            return r  # already has weather

        season = str(r.get("season", "spring"))
        base_t = _seasonal_temp_baseline(season)

        u = _stable_unit(f"{int(r['route_id'])}-{int(r['date_key'])}-u")
        v = _stable_unit(f"{int(r['route_id'])}-{int(r['date_key'])}-v")
        w = _stable_unit(f"{int(r['route_id'])}-{int(r['date_key'])}-w")

        # Temperature
        temp_mean = base_t + (u - 0.5) * 6.0
        temp_min = temp_mean - (1.5 + v * 3.0)
        temp_max = temp_mean + (1.0 + w * 3.0)

        # Precip/snow/wind (winter tends to be more severe)
        season_sev = {"winter": 1.25, "spring": 1.0, "summer": 0.85, "autumn": 1.1}.get(season, 1.0)
        precip_sum = max(0.0, (v * 8.0 - 2.0) * season_sev)  # mm/day-ish
        snowfall_sum = max(0.0, (u * 4.0 - 2.5) * (1.4 if season == "winter" else 0.3))
        windspeed_mean = 6.0 + (w * 10.0) * season_sev
        windgusts_max = windspeed_mean + 6.0 + (u * 14.0) * season_sev

        severity = np.clip((precip_sum * 4.0) + (snowfall_sum * 8.0) + (windgusts_max * 1.2), 0, 120)

        r["temp_mean"] = round(float(temp_mean), 2)
        r["temp_min"] = round(float(temp_min), 2)
        r["temp_max"] = round(float(temp_max), 2)
        r["precip_sum"] = round(float(precip_sum), 2)
        r["snowfall_sum"] = round(float(snowfall_sum), 2)
        r["windspeed_mean"] = round(float(windspeed_mean), 2)
        r["windgusts_max"] = round(float(windgusts_max), 2)
        r["severity_index"] = round(float(severity), 2)
        return r

    out = out.apply(fill_row, axis=1)
    return out


def _encode_categoricals(df: pd.DataFrame) -> Tuple[pd.DataFrame, list[str]]:
    cat_cols = ["region", "difficulty", "season", "day_name"]
    existing = [c for c in cat_cols if c in df.columns]
    df2 = pd.get_dummies(df, columns=existing, drop_first=False)
    feature_cols = [c for c in df2.columns if c not in ("bookings_count",)]
    return df2, feature_cols


def build_training_frame_v2(paths: Paths = Paths()) -> Tuple[pd.DataFrame, list[str]]:
    df = pd.read_csv(paths.fact_route_day)

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

    df["route_id"] = df["route_id"].astype(int)
    df["date_key"] = df["date_key"].astype(int)
    df["bookings_count"] = df["bookings_count"].astype(float)

    # region -> division
    region_div = pd.read_csv(paths.dim_region_division)
    df = df.merge(region_div, on="region", how="left")
    df["division"] = df["division"].fillna("england-and-wales")

    def div_bh(row) -> int:
        if row["division"] == "scotland":
            return int(row.get("is_bank_holiday_scotland", 0))
        if row["division"] == "northern-ireland":
            return int(row.get("is_bank_holiday_northern_ireland", 0))
        return int(row.get("is_bank_holiday_england_wales", 0))

    df["is_bank_holiday_division"] = df.apply(div_bh, axis=1)

    # closed day flag
    bh = pd.read_csv(paths.dim_bank_holiday, parse_dates=["date"])
    closed_by_div = _build_closed_dates_by_division(bh)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["is_closed_day"] = df.apply(
        lambda r: int(r["date"].date() in closed_by_div.get(r["division"], set())),
        axis=1,
    )

    # weather
    df = _add_weather_features(df, paths.weather_daily, use_fallback=True)

    # encode
    encoded, _ = _encode_categoricals(df)

    # drop helper cols
    encoded = encoded.drop(columns=[c for c in ["date", "division"] if c in encoded.columns], errors="ignore")

    feature_cols = [c for c in encoded.columns if c != "bookings_count"]
    return encoded, feature_cols


def build_scoring_frame_2026_v2(paths: Paths = Paths()) -> Tuple[pd.DataFrame, list[str]]:
    dim_date = pd.read_csv(paths.dim_date, parse_dates=["date"])
    dim_route = pd.read_csv(paths.dim_route)
    region_div = pd.read_csv(paths.dim_region_division)
    bh = pd.read_csv(paths.dim_bank_holiday, parse_dates=["date"])

    d26 = dim_date[(dim_date["date"] >= "2026-01-01") & (dim_date["date"] <= "2026-12-31")].copy()
    d26["date_key"] = d26["date_key"].astype(int)

    routes = dim_route[["route_id", "region", "difficulty", "distance_km", "duration_hours"]].copy()
    routes["route_id"] = routes["route_id"].astype(int)

    d26["key"] = 1
    routes["key"] = 1
    base = d26.merge(routes, on="key", how="outer").drop(columns=["key"])

    base = base.merge(region_div, on="region", how="left")
    base["division"] = base["division"].fillna("england-and-wales")

    def div_bh(row) -> int:
        if row["division"] == "scotland":
            return int(row.get("is_bank_holiday_scotland", 0))
        if row["division"] == "northern-ireland":
            return int(row.get("is_bank_holiday_northern_ireland", 0))
        return int(row.get("is_bank_holiday_england_wales", 0))

    base["is_bank_holiday_division"] = base.apply(div_bh, axis=1)

    closed_by_div = _build_closed_dates_by_division(bh)
    base["is_closed_day"] = base.apply(
        lambda r: int(pd.Timestamp(r["date"]).date() in closed_by_div.get(r["division"], set())),
        axis=1,
    )

    base = _add_weather_features(base, paths.weather_daily, use_fallback=True)

    encoded, _ = _encode_categoricals(base)
    encoded = encoded.drop(columns=[c for c in ["date", "division"] if c in encoded.columns], errors="ignore")

    feature_cols = [c for c in encoded.columns if c != "bookings_count"]
    return encoded, feature_cols
