from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import pandas as pd

from src.config import PROCESSED_DIR, VAT_RATE


@dataclass(frozen=True)
class Files:
    dim_route: Path = PROCESSED_DIR / "dim_route.csv"
    dim_guide: Path = PROCESSED_DIR / "dim_guide.csv"
    dim_date: Path = PROCESSED_DIR / "dim_date.csv"
    fact_bookings: Path = PROCESSED_DIR / "fact_bookings_2024_2025.csv"


def _fail(msg: str) -> None:
    print(f"[FAIL] {msg}")
    raise SystemExit(1)


def _warn(msg: str) -> None:
    print(f"[WARN] {msg}")


def _ok(msg: str) -> None:
    print(f"[OK] {msg}")


def validate(files: Files = Files()) -> None:
    # Load tables
    routes = pd.read_csv(files.dim_route)
    guides = pd.read_csv(files.dim_guide)
    dates = pd.read_csv(files.dim_date)
    fact = pd.read_csv(files.fact_bookings)

    # --- Basic presence checks
    if routes.empty:
        _fail("dim_route is empty")
    if guides.empty:
        _fail("dim_guide is empty")
    if dates.empty:
        _fail("dim_date is empty")
    if fact.empty:
        _fail("fact_bookings is empty")

    _ok("All core tables loaded and non-empty")

    # --- Required columns
    required_fact_cols = [
        "booking_id",
        "booking_date",
        "date_key",
        "route_id",
        "guide_id",
        "party_size",
        "sales_ex_vat",
        "vat_amount",
        "sales_inc_vat",
        "staff_cost",
        "margin_amount",
        "margin_pct",
        "discount_flag",
    ]
    missing = [c for c in required_fact_cols if c not in fact.columns]
    if missing:
        _fail(f"Missing required fact columns: {missing}")
    _ok("Required fact columns present")

    # --- Key uniqueness
    if fact["booking_id"].duplicated().any():
        _fail("booking_id is not unique in fact_bookings")
    _ok("booking_id uniqueness passed")

    # --- Referential integrity
    route_ids = set(routes["route_id"].astype(int))
    guide_ids = set(guides["guide_id"].astype(int))
    date_keys = set(dates["date_key"].astype(int))

    bad_route = fact.loc[~fact["route_id"].astype(int).isin(route_ids), "route_id"].unique()
    bad_guide = fact.loc[~fact["guide_id"].astype(int).isin(guide_ids), "guide_id"].unique()
    bad_date = fact.loc[~fact["date_key"].astype(int).isin(date_keys), "date_key"].unique()

    if len(bad_route) > 0:
        _fail(f"Invalid route_id values found: {bad_route[:10]}")
    if len(bad_guide) > 0:
        _fail(f"Invalid guide_id values found: {bad_guide[:10]}")
    if len(bad_date) > 0:
        _fail(f"Invalid date_key values found: {bad_date[:10]}")
    _ok("Referential integrity passed (route_id, guide_id, date_key)")

    # --- Value checks
    if (fact["party_size"] <= 0).any():
        _fail("party_size contains non-positive values")
    if (fact["sales_ex_vat"] < 0).any():
        _fail("sales_ex_vat contains negative values")
    if (fact["sales_inc_vat"] < 0).any():
        _fail("sales_inc_vat contains negative values")

    # Margin bounds requested (30-50%)
    if (fact["margin_pct"] < 0.30).any() or (fact["margin_pct"] > 0.50).any():
        # For portfolio: allow warning instead of hard fail if you want
        min_m = fact["margin_pct"].min()
        max_m = fact["margin_pct"].max()
        _fail(f"margin_pct outside expected 0.30–0.50 range (min={min_m:.4f}, max={max_m:.4f})")

    _ok("Basic numeric range checks passed (party_size, sales, margin_pct)")

    # --- VAT consistency check (tolerance for rounding)
    # vat_amount ~= sales_ex_vat * VAT_RATE
    expected_vat = fact["sales_ex_vat"] * VAT_RATE
    vat_diff = (fact["vat_amount"] - expected_vat).abs()

    if vat_diff.max() > 0.03:  # allow rounding tolerance
        _warn(f"VAT calc diff max {vat_diff.max():.4f} exceeds tolerance (0.03). Check rounding.")
    else:
        _ok("VAT consistency check passed")

    # sales_inc_vat ~= sales_ex_vat + vat_amount
    inc_diff = (fact["sales_inc_vat"] - (fact["sales_ex_vat"] + fact["vat_amount"])).abs()
    if inc_diff.max() > 0.03:
        _warn(f"Inc VAT calc diff max {inc_diff.max():.4f} exceeds tolerance (0.03). Check rounding.")
    else:
        _ok("Inc VAT consistency check passed")

    print("\nAll validations complete ✅")


if __name__ == "__main__":
    validate()
