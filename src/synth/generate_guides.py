import json
from pathlib import Path
from typing import Dict, Any, List

import pandas as pd

from src.config import SEEDS, PROCESSED_DIR


def build_dim_guide(guides_json: Path, out_csv: Path) -> Path:
    """
    Build the dim_guide table from raw guide seed data.
    """
    if not guides_json.exists():
        raise FileNotFoundError(f"Guides JSON not found: {guides_json}")

    # BOM-safe + whitespace-safe read
    text = guides_json.read_text(encoding="utf-8-sig").strip()
    if not text:
        raise ValueError(f"{guides_json} is empty. Expected JSON list of guides.")

    try:
        guides: List[Dict[str, Any]] = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {guides_json}") from exc

    rows = []
    for g in guides:
        rows.append(
            {
                "guide_id": int(g["guide_id"]),
                "guide_name": str(g["name"]).strip(),
                "email": str(g.get("email", "")).strip(),
                "phone": str(g.get("phone", "")).strip(),
                "bio": str(g.get("bio", "")).strip(),
            }
        )

    df = (
        pd.DataFrame(rows)
        .sort_values("guide_id")
        .reset_index(drop=True)
    )

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    return out_csv


if __name__ == "__main__":
    out = build_dim_guide(SEEDS.guides_json, PROCESSED_DIR / "dim_guide.csv")
    print(f"Saved {out}")

