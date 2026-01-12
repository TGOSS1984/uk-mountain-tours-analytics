from pathlib import Path
import pandas as pd

from src.config import PROCESSED_DIR

PBI_DIR = Path("data/pbi")

TABLES = [
    "dim_date",
    "dim_route",
    "dim_guide",
    "fact_bookings_2024_2025",
    "fact_route_week_2024_2025",
    "fact_forecast_week_2026",
]


def export_excel():
    PBI_DIR.mkdir(parents=True, exist_ok=True)

    for name in TABLES:
        csv_path = PROCESSED_DIR / f"{name}.csv"
        if not csv_path.exists():
            print(f"Skipping {name} (not found)")
            continue

        df = pd.read_csv(csv_path)
        out_path = PBI_DIR / f"{name}.xlsx"
        df.to_excel(out_path, index=False)
        print(f"Exported {out_path}")


if __name__ == "__main__":
    export_excel()
