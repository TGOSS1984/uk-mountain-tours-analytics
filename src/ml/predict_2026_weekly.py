from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.config import PROCESSED_DIR


@dataclass(frozen=True)
class Paths:
    dim_date: Path = PROCESSED_DIR / "dim_date.csv"
    dim_route: Path = PROCESSED_DIR / "dim_route.csv"
    model_path: Path = Path("models/booking_forecast_weekly_xgb.joblib")
    out_csv: Path = PROCESSED_DIR / "fact_forecast_week_2026.csv"


def _build_week_scaffold_2026(dim_date: pd.DataFrame, dim_route: pd.DataFrame) -> pd.DataFrame:
    # 2026 dates only
    d = dim_date[(dim_date["date"] >= "2026-01-01") & (dim_date["date"] <= "2026-12-31")].copy()

    # Weekly counts
    d["is_weekend"] = d["day_name"].isin(["Saturday", "Sunday"]).astype(int)
    d["week_key"] = d["iso_year"].astype(str) + "-" + d["iso_week"].astype(str).str.zfill(2)

    weekly_cal = (
        d.groupby(["iso_year", "iso_week"], as_index=False)
        .agg(
            weekend_days=("is_weekend", "sum"),
            bank_holiday_days_any=("is_bank_holiday_any", "sum"),
        )
    )
    weekly_cal["week_start"] = pd.to_datetime(
        weekly_cal["iso_year"].astype(str) + "-W" + weekly_cal["iso_week"].astype(str).str.zfill(2) + "-1",
        format="%G-W%V-%u",
        errors="coerce",
    )

    routes = dim_route[["route_id", "region", "difficulty", "distance_km", "duration_hours"]].copy()
    routes["key"] = 1
    weekly_cal["key"] = 1

    scaffold = weekly_cal.merge(routes, on="key", how="outer").drop(columns=["key"])
    scaffold["route_id"] = scaffold["route_id"].astype(int)
    return scaffold


def predict(paths: Paths = Paths()) -> Path:
    dim_date = pd.read_csv(paths.dim_date, parse_dates=["date"])
    dim_route = pd.read_csv(paths.dim_route)

    scaffold = _build_week_scaffold_2026(dim_date, dim_route)

    bundle = joblib.load(paths.model_path)
    model = bundle["model"]
    feature_cols = bundle["feature_cols"]

    # Basic feature frame
    df = scaffold.copy()

    # One-hot encoding must match training
    df_enc = pd.get_dummies(df, columns=["region", "difficulty"], drop_first=False)

    # Align columns
    for c in feature_cols:
        if c not in df_enc.columns:
            df_enc[c] = 0
    X = df_enc[feature_cols]

    preds = np.clip(model.predict(X), 0, None)

    out = scaffold.copy()
    out["predicted_bookings_count"] = preds.round(3)
    out["prediction_version"] = "weekly_xgb_v1"
    out["year"] = 2026

    paths.out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(paths.out_csv, index=False)
    return paths.out_csv


if __name__ == "__main__":
    out = predict()
    print(f"Saved {out}")
