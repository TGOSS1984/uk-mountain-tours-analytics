from dataclasses import dataclass
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

# Bounding boxes for synthetic-but-realistic coordinates per region
# (Used for Power BI mapping visuals; these are approximations)
REGION_BOUNDS = {
    "lake_district": {"lat_min": 54.35, "lat_max": 54.70, "lon_min": -3.35, "lon_max": -2.80},
    "wales":         {"lat_min": 51.55, "lat_max": 53.35, "lon_min": -4.40, "lon_max": -2.70},
    "scotland":      {"lat_min": 56.60, "lat_max": 58.60, "lon_min": -6.30, "lon_max": -3.10},
    "peak_district": {"lat_min": 53.20, "lat_max": 54.15, "lon_min": -2.55, "lon_max": -1.55},
}

@dataclass(frozen=True)
class SeedFiles:
    routes_json: Path = RAW_DIR / "routes.json"
    guides_json: Path = RAW_DIR / "guides.json"

SEEDS = SeedFiles()

VAT_RATE = 0.20
RANDOM_SEED = 42
