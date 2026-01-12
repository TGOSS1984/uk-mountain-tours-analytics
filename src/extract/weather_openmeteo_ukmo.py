from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List

import pandas as pd
import requests

from src.config import PROCESSED_DIR, RAW_DIR


# Open-Meteo Forecast endpoint (UKMO is selected via models=...)
OPEN_METEO_FORECAST_ENDPOINT = "https://api.open-meteo.com/v1/forecast"
UKMO_MODEL = "ukmo_seamless"


@dataclass(frozen=True)
class Paths:
    dim_route: Path = PROCESSED_DIR / "dim_route.csv"
    raw_dir: Path = RAW_DIR / "weather_ukmo"
    out_hourly_csv: Path = Path("data/interim/weather_hourly_ukmo.csv")


def fetch_ukmo_hourly(lat: float, lon: float) -> Dict[str, Any]:
    """
    Pull hourly forecast from Open-Meteo using the UK Met Office model.
    """
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": ",".join(
            [
                "temperature_2m",
                "precipitation",
                "snowfall",
                "wind_speed_10m",
                "wind_gusts_10m",
                "weather_code",
            ]
        ),
        "models": UKMO_MODEL,
        "timezone": "Europe/London",
    }

    r = requests.get(OPEN_METEO_FORECAST_ENDPOINT, params=params, timeout=30)
    r.raise_for_status()
    return r.json()


def pull_all_routes(paths: Paths = Paths()) -> Path:
    routes = pd.read_csv(paths.dim_route)

    paths.raw_dir.mkdir(parents=True, exist_ok=True)
    paths.out_hourly_csv.parent.mkdir(parents=True, exist_ok=True)

    rows: List[dict] = []

    for _, rt in routes.iterrows():
        route_id = int(rt["route_id"])
        lat = float(rt["route_lat"])
        lon = float(rt["route_lon"])

        data = fetch_ukmo_hourly(lat, lon)

        # Save raw JSON per route
        raw_path = paths.raw_dir / f"ukmo_route_{route_id}.json"
        raw_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

        hourly = data.get("hourly", {})
        times = hourly.get("time", [])

        # Flatten hourly to rows
        for i, t in enumerate(times):
            rows.append(
                {
                    "route_id": route_id,
                    "datetime": t,  # ISO string in Europe/London
                    "temperature_2m": hourly.get("temperature_2m", [None])[i],
                    "precipitation": hourly.get("precipitation", [None])[i],
                    "snowfall": hourly.get("snowfall", [None])[i],
                    "wind_speed_10m": hourly.get("wind_speed_10m", [None])[i],
                    "wind_gusts_10m": hourly.get("wind_gusts_10m", [None])[i],
                    "weather_code": hourly.get("weather_code", [None])[i],
                    "model": f"open-meteo-{UKMO_MODEL}",
                }
            )

    df = pd.DataFrame(rows)
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df = df.dropna(subset=["datetime"]).sort_values(["route_id", "datetime"])

    df.to_csv(paths.out_hourly_csv, index=False)
    return paths.out_hourly_csv


if __name__ == "__main__":
    out = pull_all_routes()
    print(f"Saved {out}")

