import json
from pathlib import Path
from typing import Dict, Any, List

import pandas as pd

from src.config import RAW_DIR, PROCESSED_DIR


def load_bank_holidays_raw(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_dim_bank_holiday(raw_json: Path, out_csv: Path) -> Path:
    """
    Create a clean dimension table:
    bank_holiday_id, date, division, title, notes, bunting
    """
    data = load_bank_holidays_raw(raw_json)

    rows: List[Dict[str, Any]] = []
    for division_key, division in data.items():
        for ev in division.get("events", []):
            rows.append(
                {
                    "date": ev.get("date"),
                    "division": division_key,
                    "title": ev.get("title"),
                    "notes": ev.get("notes"),
                    "bunting": ev.get("bunting"),
                }
            )

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values(["date", "division", "title"])

    # Deterministic surrogate key (nice for BI)
    df["bank_holiday_id"] = (
        df["division"].astype(str)
        + "|"
        + df["date"].dt.strftime("%Y-%m-%d")
        + "|"
        + df["title"].astype(str)
    ).apply(lambda s: abs(hash(s)) % (10**12))

    df = df[
        ["bank_holiday_id", "date", "division", "title", "notes", "bunting"]
    ].drop_duplicates()

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    return out_csv


def build_bridge_bank_holiday_date(dim_csv: Path, out_csv: Path) -> Path:
    """
    Bridge table for modelling:
    date, division, bank_holiday_id
    (Lets you relate dim_date to holidays cleanly, even if multiple events exist.)
    """
    df = pd.read_csv(dim_csv, parse_dates=["date"])
    bridge = df[["date", "division", "bank_holiday_id"]].copy()
    bridge = bridge.sort_values(["date", "division"])
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    bridge.to_csv(out_csv, index=False)
    return out_csv


def build_dim_region_division(out_csv: Path) -> Path:
    """
    Map tour 'region' values to bank-holiday divisions.
    Most tour regions are England & Wales except Scotland routes.

    This becomes useful later for:
    - applying "is_bank_holiday" correctly to route/region
    - ML features (holiday uplift/closures)
    """
    rows = [
        {"region": "lake_district", "division": "england-and-wales"},
        {"region": "peak_district", "division": "england-and-wales"},
        {"region": "wales", "division": "england-and-wales"},
        {"region": "scotland", "division": "scotland"},
        # (If you ever add NI routes later, add: {"region":"northern_ireland","division":"northern-ireland"})
    ]
    df = pd.DataFrame(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    return out_csv


if __name__ == "__main__":
    raw = RAW_DIR / "bank_holidays.json"

    dim_out = PROCESSED_DIR / "dim_bank_holiday.csv"
    bridge_out = PROCESSED_DIR / "bridge_bank_holiday_date.csv"
    region_out = PROCESSED_DIR / "dim_region_division.csv"

    build_dim_bank_holiday(raw, dim_out)
    build_bridge_bank_holiday_date(dim_out, bridge_out)
    build_dim_region_division(region_out)

    print(f"Saved {dim_out}")
    print(f"Saved {bridge_out}")
    print(f"Saved {region_out}")
