from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.ml.features_v2 import build_scoring_frame_2026_v2


@dataclass(frozen=True)
class Paths:
    model_path: Path = Path("models/booking_forecast_xgb_v2.joblib")
    out_csv: Path = Path("data/processed/fact_forecast_2026_v2.csv")


def predict(paths: Paths = Paths()) -> Path:
    bundle = joblib.load(paths.model_path)
    model = bundle["model"]
    feature_cols = bundle["feature_cols"]

    df26, _ = build_scoring_frame_2026_v2()

    # Align columns with training feature list
    X = df26.copy()
    for c in feature_cols:
        if c not in X.columns:
            X[c] = 0
    X = X[feature_cols]

    preds = model.predict(X)
    preds = np.clip(preds, 0, None)

    out = df26.copy()
    out["predicted_bookings_count"] = preds.round(3)

    # Closed days forced to zero
    if "is_closed_day" in out.columns:
        out.loc[out["is_closed_day"] == 1, "predicted_bookings_count"] = 0.0

    out["prediction_version"] = "time_split_xgb_v2"
    out["year"] = 2026

    paths.out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(paths.out_csv, index=False)
    return paths.out_csv


if __name__ == "__main__":
    out = predict()
    print(f"Saved {out}")
