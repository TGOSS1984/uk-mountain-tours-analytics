from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from xgboost import XGBRegressor

from src.ml.features_v2 import build_training_frame_v2


@dataclass(frozen=True)
class Outputs:
    model_path: Path = Path("models/booking_forecast_xgb_v2.joblib")
    metrics_path: Path = Path("data/processed/ml_metrics_time_split_v2.json")


def train_time_split(outputs: Outputs = Outputs()) -> tuple[Path, Path]:
    df, feature_cols = build_training_frame_v2()

    # Need year for split; it exists in the encoded frame
    if "year" not in df.columns:
        raise ValueError("Expected 'year' column in training frame for time-based split")

    train_df = df[df["year"] == 2024].copy()
    test_df = df[df["year"] == 2025].copy()

    if train_df.empty or test_df.empty:
        raise ValueError("Time split failed: missing 2024 or 2025 rows")

    X_train = train_df[feature_cols]
    y_train = train_df["bookings_count"]

    X_test = test_df[feature_cols]
    y_test = test_df["bookings_count"]

    model = XGBRegressor(
        n_estimators=600,
        learning_rate=0.04,
        max_depth=7,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="reg:squarederror",
        random_state=42,
    )

    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    preds = np.clip(preds, 0, None)

    metrics = {
        "split": "train=2024, test=2025",
        "mae": float(mean_absolute_error(y_test, preds)),
        "rmse": float(mean_squared_error(y_test, preds) ** 0.5),
        "r2": float(r2_score(y_test, preds)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "features": int(len(feature_cols)),
        "target": "bookings_count",
        "notes": "v2 adds weather-ready features (joined if available; deterministic seasonal fallback otherwise) and uses a time-based split.",
    }

    outputs.model_path.parent.mkdir(parents=True, exist_ok=True)
    outputs.metrics_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump({"model": model, "feature_cols": feature_cols}, outputs.model_path)
    outputs.metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("Saved model:", outputs.model_path)
    print("Saved metrics:", outputs.metrics_path)
    print(metrics)

    return outputs.model_path, outputs.metrics_path


if __name__ == "__main__":
    train_time_split()
