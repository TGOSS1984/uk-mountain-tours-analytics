from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import INTERIM_DIR, PROCESSED_DIR


def _normalise_hourly_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Allow both old and new Open-Meteo column naming.
    Old: windspeed_10m, windgusts_10m, weathercode
    New: wind_speed_10m, wind_gusts_10m, weather_code
    """
    rename_map = {}

    if "windspeed_10m" in df.columns and "wind_speed_10m" not in df.columns:
        rename_map["windspeed_10m"] = "wind_speed_10m"
    if "windgusts_10m" in df.columns and "wind_gusts_10m" not in df.columns:
        rename_map["windgusts_10m"] = "wind_gusts_10m"
    if "weathercode" in df.columns and "weather_code" not in df.columns:
        rename_map["weathercode"] = "weather_code"

    if rename_map:
        df = df.rename(columns=rename_map)

    return df


def build_daily(
    hourly_path: Path | None = None,
    out_path: Path | None = None,
) -> Path:
    """
    Build daily weather features from hourly Open-Meteo (UKMO model) extracts.
    """
    hourly_path = hourly_path or (INTERIM_DIR / "weather_hourly_ukmo.csv")
    out_path = out_path or (PROCESSED_DIR / "weather_daily_ukmo.csv")

    df = pd.read_csv(hourly_path)

    # Normalise column names to the "new" schema
    df = _normalise_hourly_columns(df)

    # Parse datetime and derive date
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df = df.dropna(subset=["datetime"])
    df["date"] = df["datetime"].dt.date

    # Ensure numeric columns are numeric
    for c in ["temperature_2m", "precipitation", "snowfall", "wind_speed_10m", "wind_gusts_10m"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    def _mode_or_none(s: pd.Series):
        s2 = s.dropna()
        if s2.empty:
            return None
        m = s2.mode()
        return m.iloc[0] if not m.empty else None

    daily = (
        df.groupby(["route_id", "date"], as_index=False)
        .agg(
            temp_mean=("temperature_2m", "mean"),
            temp_min=("temperature_2m", "min"),
            temp_max=("temperature_2m", "max"),
            precip_sum=("precipitation", "sum"),
            snowfall_sum=("snowfall", "sum"),
            wind_speed_max=("wind_speed_10m", "max"),
            wind_gusts_max=("wind_gusts_10m", "max"),
            weather_code_mode=("weather_code", _mode_or_none),
        )
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    daily.to_csv(out_path, index=False)
    return out_path


if __name__ == "__main__":
    out = build_daily()
    print(f"Saved {out}")

