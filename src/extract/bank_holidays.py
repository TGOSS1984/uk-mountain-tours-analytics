from pathlib import Path
import requests

from src.config import RAW_DIR

BANK_HOLIDAYS_URL = "https://www.gov.uk/bank-holidays.json"


def pull_bank_holidays(out_path: Path | None = None) -> Path:
    """
    Pull UK bank holidays JSON from GOV.UK and save as a raw file.
    """
    if out_path is None:
        out_path = RAW_DIR / "bank_holidays.json"

    out_path.parent.mkdir(parents=True, exist_ok=True)

    r = requests.get(BANK_HOLIDAYS_URL, timeout=30)
    r.raise_for_status()

    out_path.write_text(r.text, encoding="utf-8")
    return out_path


if __name__ == "__main__":
    p = pull_bank_holidays()
    print(f"Saved {p}")
