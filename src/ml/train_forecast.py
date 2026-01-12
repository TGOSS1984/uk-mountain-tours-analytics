from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from xgboost import XGBRegressor

from src.ml.features import build_training_frame


@dataclass(frozen=True)
class Outputs:
    model_path: Path = Path("models/booking_forecast_xgb.joblib")
    metrics_path: Path = Path("data/processed/ml_metrics_baseline.json")


def train(outputs: Outputs = Outputs()) -> tuple[Path, Path]:
    df, feature_cols = build_training_frame()

    X = df[feature_cols].copy()
    y = df["bookings_count"].copy()

    # Basic split (portfolio-friendly). Later you can do time-based CV.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    model = XGBRegressor(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="reg:squarederror",
        random_state=42,
    )

    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    preds = np.clip(preds, 0, None)  # bookings can't be negative

    metrics = {
        "mae": float(mean_absolute_error(y_test, preds)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, preds))),
        "r2": float(r2_score(y_test, preds)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "features": int(len(feature_cols)),
        "target": "bookings_count",
        "notes": "Baseline model uses calendar + route attributes + bank holiday flags (no full-year historical weather yet).",
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
    train()
