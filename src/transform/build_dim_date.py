import json
from pathlib import Path
from typing import Dict, Any, List

import pandas as pd

from src.config import RAW_DIR, PROCESSED_DIR


def _load_bank_holidays(path: Path) -> pd.DataFrame:
    """
    Flatten GOV.UK bank holidays JSON into a table:
    date, division, title, notes, bunting
    """
    data: Dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))

    rows: List[Dict[str, Any]] = []
    for division_key, division in data.items():
        for ev in division.get("events", []):
            rows.append(
                {
                    "date": ev.get("date"),
                    "division": division_key,  # e.g. england-and-wales, scotland, northern-ireland
                    "title": ev.get("title"),
                    "notes": ev.get("notes"),
                    "bunting": ev.get("bunting"),
                }
            )

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values(["date", "division"])
    return df


def build_dim_date(
    start_date: str = "2024-01-01",
    end_date: str = "2026-12-31",
    bank_holidays_json: Path | None = None,
    out_csv: Path | None = None,
) -> Path:
    """
    Build a daily date dimension and enrich with UK bank holiday flags.
    """
    if bank_holidays_json is None:
        bank_holidays_json = RAW_DIR / "bank_holidays.json"
    if out_csv is None:
        out_csv = PROCESSED_DIR / "dim_date.csv"

    dates = pd.date_range(start=start_date, end=end_date, freq="D")
    df = pd.DataFrame({"date": dates})
    df["date_key"] = df["date"].dt.strftime("%Y%m%d").astype(int)

    # Date parts
    df["year"] = df["date"].dt.year
    df["quarter"] = df["date"].dt.quarter
    df["month"] = df["date"].dt.month
    df["month_name"] = df["date"].dt.strftime("%B")
    df["day"] = df["date"].dt.day
    df["day_name"] = df["date"].dt.strftime("%A")

    # ISO parts (useful for week-based reporting)
    iso = df["date"].dt.isocalendar()
    df["iso_year"] = iso["year"].astype(int)
    df["iso_week"] = iso["week"].astype(int)
    df["iso_day"] = iso["day"].astype(int)

    df["is_weekend"] = df["day_name"].isin(["Saturday", "Sunday"])

    # Simple season label (tweak later if you want meteorological seasons)
    def season(m: int) -> str:
        if m in (12, 1, 2):
            return "winter"
        if m in (3, 4, 5):
            return "spring"
        if m in (6, 7, 8):
            return "summer"
        return "autumn"

    df["season"] = df["month"].apply(season)

    # Bank holiday enrichment
    bh = _load_bank_holidays(bank_holidays_json)

    # Start with default flags
    df["is_bank_holiday_any"] = False
    df["is_bank_holiday_england_wales"] = False
    df["is_bank_holiday_scotland"] = False
    df["is_bank_holiday_northern_ireland"] = False

    if not bh.empty:
        # Map division to our columns
        division_map = {
            "england-and-wales": "is_bank_holiday_england_wales",
            "scotland": "is_bank_holiday_scotland",
            "northern-ireland": "is_bank_holiday_northern_ireland",
        }

        for div, col in division_map.items():
            dates_set = set(bh.loc[bh["division"] == div, "date"].dt.date)
            df[col] = df["date"].dt.date.isin(dates_set)

        df["is_bank_holiday_any"] = (
            df["is_bank_holiday_england_wales"]
            | df["is_bank_holiday_scotland"]
            | df["is_bank_holiday_northern_ireland"]
        )

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    return out_csv


if __name__ == "__main__":
    out = build_dim_date()
    print(f"Saved {out}")
