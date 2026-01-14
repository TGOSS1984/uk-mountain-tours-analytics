from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

from src.config import PROCESSED_DIR


@dataclass(frozen=True)
class Paths:
    weekly_fact: Path = PROCESSED_DIR / "fact_route_week_2024_2025.csv"
    out_model: Path = Path("models/booking_forecast_weekly_xgb.joblib")
    out_metrics: Path = PROCESSED_DIR / "ml_metrics_weekly_time_split.json"


def _encode(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    # One-hot encode categoricals
    cat_cols = ["region", "difficulty"]
    df2 = pd.get_dummies(df, columns=[c for c in cat_cols if c in df.columns], drop_first=False)

    # Target
    y = df2["bookings_count"].astype(float)

    # Drop non-features
    drop = ["bookings_count", "week_start"]
    X = df2.drop(columns=[c for c in drop if c in df2.columns], errors="ignore")

    feature_cols = list(X.columns)
    return df2, feature_cols


def train_time_split(paths: Paths = Paths()) -> tuple[Path, Path]:
    df = pd.read_csv(paths.weekly_fact, parse_dates=["week_start"])

    # Keep only the columns we need (and be tolerant if extras exist)
    keep = [
    "iso_year", "iso_week", "route_id",
    "region", "difficulty",
    "distance_km", "duration_hours",
    "bank_holiday_days_any", "weekend_days",
    "bookings_count",
    "week_start",
    ]

    df = df[[c for c in keep if c in df.columns]].copy()

    # Time-based split
    train_df = df[df["iso_year"] == 2024].copy()
    test_df = df[df["iso_year"] == 2025].copy()
    if train_df.empty or test_df.empty:
        raise ValueError("Weekly time split failed: expected iso_year 2024 and 2025 in weekly fact table.")

    train_enc, feature_cols = _encode(train_df)
    test_enc, _ = _encode(test_df)

    X_train = train_enc[feature_cols]
    y_train = train_enc["bookings_count"].astype(float)

    # Align test columns
    for c in feature_cols:
        if c not in test_enc.columns:
            test_enc[c] = 0
    X_test = test_enc[feature_cols]
    y_test = test_enc["bookings_count"].astype(float)

    model = XGBRegressor(
        n_estimators=800,
        learning_rate=0.04,
        max_depth=7,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="reg:squarederror",
        random_state=42,
    )
    model.fit(X_train, y_train)

    preds = np.clip(model.predict(X_test), 0, None)

    metrics = {
        "grain": "route-week",
        "split": "train=2024, test=2025",
        "mae": float(mean_absolute_error(y_test, preds)),
        "rmse": float(mean_squared_error(y_test, preds) ** 0.5),
        "r2": float(r2_score(y_test, preds)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "features": int(len(feature_cols)),
        "target": "bookings_count",
        "notes": "Weekly aggregation reduces daily Poisson noise and improves learnability. Features include route + calendar + weekly holiday/weekend counts + commercial signals.",
    }

    paths.out_model.parent.mkdir(parents=True, exist_ok=True)
    paths.out_metrics.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump({"model": model, "feature_cols": feature_cols}, paths.out_model)
    paths.out_metrics.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("Saved model:", paths.out_model)
    print("Saved metrics:", paths.out_metrics)
    print(metrics)

    return paths.out_model, paths.out_metrics


if __name__ == "__main__":
    train_time_split()
