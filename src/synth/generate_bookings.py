from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd

from src.config import PROCESSED_DIR, VAT_RATE, RANDOM_SEED


CLOSED_HOLIDAY_KEYWORDS = (
    "Christmas Day",
    "Boxing Day",
    "New Year",
    "New Year's Day",
    "New Year’s Day",
    "Good Friday",
    "Easter Monday",
)

# Demand uplifts (tweak later)
REGION_UPLIFT = {
    "lake_district": 1.35,
    "wales": 1.15,
    "peak_district": 1.00,
    "scotland": 0.90,
}

SEASON_UPLIFT = {
    "winter": 1.30,
    "spring": 1.00,
    "summer": 0.85,
    "autumn": 1.10,
}

DIFFICULTY_UPLIFT = {
    "easy": 1.20,
    "moderate": 1.00,
    "hard": 0.90,
    "severe": 0.75,
}

WEEKEND_UPLIFT = 1.20
BANK_HOLIDAY_OPEN_UPLIFT = 1.40


@dataclass(frozen=True)
class Paths:
    dim_route: Path = PROCESSED_DIR / "dim_route.csv"
    dim_guide: Path = PROCESSED_DIR / "dim_guide.csv"
    dim_date: Path = PROCESSED_DIR / "dim_date.csv"
    dim_bank_holiday: Path = PROCESSED_DIR / "dim_bank_holiday.csv"
    bridge_bank_holiday_date: Path = PROCESSED_DIR / "bridge_bank_holiday_date.csv"
    dim_region_division: Path = PROCESSED_DIR / "dim_region_division.csv"
    out_fact: Path = PROCESSED_DIR / "fact_bookings_2024_2025.csv"


def _load_inputs(p: Paths) -> Dict[str, pd.DataFrame]:
    routes = pd.read_csv(p.dim_route)
    guides = pd.read_csv(p.dim_guide)
    dates = pd.read_csv(p.dim_date, parse_dates=["date"])
    bh = pd.read_csv(p.dim_bank_holiday, parse_dates=["date"])
    bridge = pd.read_csv(p.bridge_bank_holiday_date, parse_dates=["date"])
    region_div = pd.read_csv(p.dim_region_division)

    # keep only full year 2024 + 2025
    dates = dates[(dates["date"] >= "2024-01-01") & (dates["date"] <= "2025-12-31")].copy()

    return {
        "routes": routes,
        "guides": guides,
        "dates": dates,
        "bh": bh,
        "bridge": bridge,
        "region_div": region_div,
    }


def _build_closed_dates_by_division(bh: pd.DataFrame) -> Dict[str, set]:
    """
    Return division -> set(date) that are forced-closed days
    (Christmas/New Year/Easter rules).
    """
    bh = bh.copy()
    bh["title"] = bh["title"].astype(str)

    mask = False
    for kw in CLOSED_HOLIDAY_KEYWORDS:
        mask = mask | bh["title"].str.contains(kw, case=False, na=False)

    closed = bh[mask]
    out: Dict[str, set] = {}
    for division, grp in closed.groupby("division"):
        out[division] = set(grp["date"].dt.date)
    return out


def _price_per_person_ex_vat(duration_hours: float, difficulty: str, rng: np.random.Generator) -> float:
    """
    Base price per person (ex VAT). You can tweak these assumptions later.
    """
    base = 75.0 + (duration_hours * 8.0)  # longer route -> higher price

    # difficulty premium
    diff_mult = {
        "easy": 0.95,
        "moderate": 1.00,
        "hard": 1.10,
        "severe": 1.20,
    }.get(difficulty, 1.00)

    # small random noise +/- 10%
    noise = rng.normal(loc=1.0, scale=0.06)
    val = base * diff_mult * noise

    # clamp sensible bounds
    return float(np.clip(val, 60.0, 190.0))


def _party_size(difficulty: str, rng: np.random.Generator) -> int:
    """
    Larger parties more likely on easier/moderate routes.
    """
    if difficulty == "easy":
        probs = [0.10, 0.25, 0.25, 0.20, 0.12, 0.08]  # sizes 1..6
    elif difficulty == "moderate":
        probs = [0.15, 0.28, 0.25, 0.17, 0.10, 0.05]
    elif difficulty == "hard":
        probs = [0.22, 0.33, 0.23, 0.13, 0.07, 0.02]
    else:  # severe
        probs = [0.30, 0.38, 0.20, 0.08, 0.03, 0.01]

    return int(rng.choice(np.arange(1, 7), p=probs))


def _discount_flag_and_pct(party_size: int, season: str, is_weekend: bool, rng: np.random.Generator) -> Tuple[bool, float]:
    """
    Discount more common for bigger groups and off-peak periods.
    """
    base_prob = 0.10
    if party_size >= 4:
        base_prob += 0.10
    if season == "summer":
        base_prob += 0.05  # off-peak for winter tours
    if not is_weekend:
        base_prob += 0.03

    flag = rng.random() < min(base_prob, 0.35)
    if not flag:
        return False, 0.0

    pct = rng.uniform(0.05, 0.20)
    return True, float(pct)


def _margin_pct(discount_flag: bool, party_size: int, rng: np.random.Generator) -> float:
    """
    Target margin between 30–50%.
    Lower when discount.
    Slightly higher with more people (economies of scale).
    """
    m = rng.uniform(0.30, 0.50)

    if discount_flag:
        m -= rng.uniform(0.03, 0.08)

    # economies of scale for larger groups
    m += min(max(party_size - 2, 0) * 0.008, 0.03)

    return float(np.clip(m, 0.30, 0.50))


def _expected_bookings_per_route_day(region: str, season: str, difficulty: str,
                                    is_weekend: bool, is_bank_holiday_open: bool) -> float:
    """
    Mean for Poisson bookings count per route-day.
    """
    base = 0.55  # baseline mean bookings per route-day
    mu = base
    mu *= REGION_UPLIFT.get(region, 1.0)
    mu *= SEASON_UPLIFT.get(season, 1.0)
    mu *= DIFFICULTY_UPLIFT.get(difficulty, 1.0)

    if is_weekend:
        mu *= WEEKEND_UPLIFT
    if is_bank_holiday_open:
        mu *= BANK_HOLIDAY_OPEN_UPLIFT

    return float(np.clip(mu, 0.05, 3.0))


def generate_fact_bookings(paths: Paths = Paths()) -> Path:
    rng = np.random.default_rng(RANDOM_SEED)
    inp = _load_inputs(paths)

    routes = inp["routes"]
    guides = inp["guides"]
    dates = inp["dates"]
    bh = inp["bh"]
    region_div = inp["region_div"]

    # region -> bank holiday division mapping
    region_to_div = dict(zip(region_div["region"], region_div["division"]))

    # closed dates by division
    closed_dates_by_div = _build_closed_dates_by_division(bh)

    guide_ids = guides["guide_id"].astype(int).tolist()

    rows: List[dict] = []
    booking_id = 1

    # Iterate route x date (route-day), then generate 0..N bookings on that day
    # This gives realistic day-level seasonality and closures.
    for _, rt in routes.iterrows():
        route_id = int(rt["route_id"])
        region = str(rt["region"])
        difficulty = str(rt["difficulty"])
        duration_hours = float(rt["duration_hours"])

        division = region_to_div.get(region, "england-and-wales")
        closed_dates = closed_dates_by_div.get(division, set())

        for _, d in dates.iterrows():
            dt = pd.Timestamp(d["date"]).to_pydatetime().date()
            season = str(d["season"])
            is_weekend = bool(d["is_weekend"])

            # bank holiday flag (division-aware using dim_date columns)
            # (dim_date already has division flags)
            if division == "scotland":
                is_bh = bool(d["is_bank_holiday_scotland"])
            elif division == "northern-ireland":
                is_bh = bool(d["is_bank_holiday_northern_ireland"])
            else:
                is_bh = bool(d["is_bank_holiday_england_wales"])

            # closures: no bookings on key holidays (xmas/new year/easter)
            if dt in closed_dates:
                continue

            mu = _expected_bookings_per_route_day(
                region=region,
                season=season,
                difficulty=difficulty,
                is_weekend=is_weekend,
                is_bank_holiday_open=is_bh,
            )

            n = int(rng.poisson(mu))
            if n <= 0:
                continue

            for _ in range(n):
                party = _party_size(difficulty, rng)
                discount_flag, discount_pct = _discount_flag_and_pct(party, season, is_weekend, rng)

                ppp = _price_per_person_ex_vat(duration_hours, difficulty, rng)
                list_value_ex_vat = ppp * party

                discount_value = list_value_ex_vat * discount_pct if discount_flag else 0.0
                sales_ex_vat = max(list_value_ex_vat - discount_value, 0.0)

                margin_pct = _margin_pct(discount_flag, party, rng)
                margin_amount = sales_ex_vat * margin_pct

                # Costs: staff cost related to margin (higher margin -> lower costs)
                # We'll make staff cost ~ 55–70% of total costs for realism.
                total_cost = max(sales_ex_vat - margin_amount, 0.0)
                staff_share = float(np.clip(rng.normal(0.62, 0.06), 0.50, 0.75))
                staff_cost = total_cost * staff_share

                vat_amount = sales_ex_vat * VAT_RATE
                sales_inc_vat = sales_ex_vat + vat_amount

                guide_id = int(rng.choice(guide_ids))

                rows.append(
                    {
                        "booking_id": booking_id,
                        "booking_date": dt.isoformat(),
                        "date_key": int(pd.Timestamp(dt).strftime("%Y%m%d")),
                        "route_id": route_id,
                        "region": region,
                        "guide_id": guide_id,
                        "party_size": party,
                        "difficulty": difficulty,
                        "duration_hours": round(duration_hours, 2),

                        "discount_flag": int(discount_flag),
                        "discount_pct": round(discount_pct, 4),

                        "price_per_person_ex_vat": round(ppp, 2),
                        "sales_ex_vat": round(sales_ex_vat, 2),
                        "vat_amount": round(vat_amount, 2),
                        "sales_inc_vat": round(sales_inc_vat, 2),

                        "staff_cost": round(staff_cost, 2),
                        "margin_amount": round(margin_amount, 2),
                        "margin_pct": round(margin_pct, 4),

                        # Helpful flags for BI + later ML
                        "season": season,
                        "is_weekend": int(is_weekend),
                        "is_bank_holiday": int(is_bh),
                        "holiday_division": division,
                    }
                )
                booking_id += 1

    fact = pd.DataFrame(rows)
    paths.out_fact.parent.mkdir(parents=True, exist_ok=True)
    fact.to_csv(paths.out_fact, index=False)
    return paths.out_fact


if __name__ == "__main__":
    out = generate_fact_bookings()
    print(f"Saved {out}")
