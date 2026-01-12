import json
from pathlib import Path
from typing import Dict, Any, List, Tuple

import pandas as pd

from src.config import SEEDS, PROCESSED_DIR, REGION_BOUNDS


def _clean_text(s: str) -> str:
    """
    Normalise route names and fix common encoding artefacts.
    """
    replacements = {
        "�": "'",
        "Ar�te": "Arete",
        "M�r": "Mor",
    }
    for bad, good in replacements.items():
        s = s.replace(bad, good)
    return " ".join(s.split())


def _stable_hash_to_unit(x: str) -> float:
    """
    Convert string -> stable float in [0,1).
    Deterministic across runs (unlike Python's built-in hash).
    """
    h = 2166136261
    for ch in x.encode("utf-8"):
        h ^= ch
        h *= 16777619
        h &= 0xFFFFFFFF
    return (h % 10_000_000) / 10_000_000.0


def _make_lat_lon(region: str, key: str) -> Tuple[float, float]:
    """
    Generate deterministic lat/lon within a region bounding box.
    """
    if region not in REGION_BOUNDS:
        raise KeyError(
            f"Region '{region}' not found in REGION_BOUNDS. "
            f"Available regions: {list(REGION_BOUNDS.keys())}"
        )

    bounds = REGION_BOUNDS[region]
    u = _stable_hash_to_unit(key)
    v = _stable_hash_to_unit(key[::-1])

    lat = bounds["lat_min"] + u * (bounds["lat_max"] - bounds["lat_min"])
    lon = bounds["lon_min"] + v * (bounds["lon_max"] - bounds["lon_min"])

    return round(lat, 5), round(lon, 5)


def build_dim_route(routes_json: Path, out_csv: Path) -> Path:
    """
    Build the dim_route table from raw route seed data.
    """
    if not routes_json.exists():
        raise FileNotFoundError(f"Routes JSON not found: {routes_json}")

    # BOM-safe + whitespace-safe read
    text = routes_json.read_text(encoding="utf-8-sig").strip()
    if not text:
        raise ValueError(f"{routes_json} is empty. Expected JSON list of routes.")

    try:
        routes: List[Dict[str, Any]] = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {routes_json}") from exc

    rows = []

    for r in routes:
        route_id = int(r["route_id"])
        name = _clean_text(str(r["name"]))
        region = str(r["region"]).lower().strip()

        lat, lon = _make_lat_lon(region, f"{route_id}-{name}-{region}")

        rows.append(
            {
                "route_id": route_id,
                "route_name": name,
                "region": region,
                "gpx_path": str(r.get("gpx_path", "")),
                "distance_km": float(r["distance_km"]),
                "duration_hours": float(r["duration_hours"]),
                "difficulty": str(r["difficulty"]).lower().strip(),
                "route_lat": lat,
                "route_lon": lon,
            }
        )

    df = (
        pd.DataFrame(rows)
        .sort_values("route_id")
        .reset_index(drop=True)
    )

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    return out_csv


if __name__ == "__main__":
    out = build_dim_route(SEEDS.routes_json, PROCESSED_DIR / "dim_route.csv")
    print(f"Saved {out}")

