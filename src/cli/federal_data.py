"""Download OpenFEMA Disaster Declarations Summaries v2."""
from __future__ import annotations

import argparse
import json
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "data" / "fema" / "FEMA_Disaster_Declarations.csv"
API = "https://www.fema.gov/api/open/v2/DisasterDeclarationsSummaries"
REQUIRED = {"disasterNumber", "state", "incidentType", "incidentBeginDate", "incidentEndDate", "fipsStateCode", "fipsCountyCode", "designatedArea"}
MIN_INCIDENT_DATE = "2010-01-01T00:00:00.000Z"
MAX_INCIDENT_DATE = "2024-12-31T23:59:59.999Z"
EXCLUDED_INCIDENT_TYPES = (
    "Biological",
    "Dam/Levee Break",
    "Chemical",
    "Terrorist",
    "Other",
    "Toxic Substances",
)


def _api_filter() -> str:
    clauses = [
        "("
        "("
        f"incidentBeginDate ge '{MIN_INCIDENT_DATE}' and "
        f"incidentBeginDate le '{MAX_INCIDENT_DATE}'"
        ") or ("
        f"incidentEndDate ge '{MIN_INCIDENT_DATE}' and "
        f"incidentEndDate le '{MAX_INCIDENT_DATE}'"
        ")"
        ")"
    ]
    clauses.extend(f"incidentType ne '{value}'" for value in EXCLUDED_INCIDENT_TYPES)
    return " and ".join(clauses)


def _validate_scope(frame: pd.DataFrame) -> None:
    lower_bound = pd.Timestamp(MIN_INCIDENT_DATE)
    upper_bound = pd.Timestamp(MAX_INCIDENT_DATE)
    begin_dates = pd.to_datetime(frame["incidentBeginDate"], errors="coerce", utc=True)
    end_dates = pd.to_datetime(frame["incidentEndDate"], errors="coerce", utc=True)
    qualifying_begin = begin_dates.between(lower_bound, upper_bound, inclusive="both").fillna(False)
    qualifying_end = end_dates.between(lower_bound, upper_bound, inclusive="both").fillna(False)
    invalid_dates = ~(
        qualifying_begin | qualifying_end
    )
    excluded_types = frame["incidentType"].isin(EXCLUDED_INCIDENT_TYPES)
    if invalid_dates.any() or excluded_types.any():
        raise RuntimeError(
            "OpenFEMA response violated the requested scope: "
            f"{int(invalid_dates.sum())} rows without a qualifying begin/end date and "
            f"{int(excluded_types.sum())} excluded incident types"
        )


def download_fema_disaster_declarations(*, force: bool = False) -> Path:
    if OUTPUT.exists() and not force:
        existing = pd.read_csv(OUTPUT, dtype=str)
        missing = REQUIRED - set(existing.columns)
        if missing:
            raise RuntimeError(f"Existing FEMA file is missing columns: {sorted(missing)}")
        _validate_scope(existing)
        return OUTPUT
    rows, skip, page_size = [], 0, 5000
    while True:
        query = urllib.parse.urlencode({
            "$filter": _api_filter(),
            "$orderby": "disasterNumber",
            "$top": page_size,
            "$skip": skip,
        })
        request = urllib.request.Request(f"{API}?{query}", headers={"User-Agent": "housing-predict/fema-downloader"})
        with urllib.request.urlopen(request, timeout=300) as response:
            page = json.load(response).get("DisasterDeclarationsSummaries", [])
        if not page:
            break
        rows.extend(page)
        print(f"Fetched {len(rows):,} FEMA declaration rows...", flush=True)
        if len(page) < page_size:
            break
        skip += page_size
    if not rows:
        raise RuntimeError("OpenFEMA returned no disaster declarations")
    frame = pd.DataFrame(rows)
    _validate_scope(frame)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUTPUT, index=False)
    return OUTPUT


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", nargs="?", choices=["fema-declarations"], default="fema-declarations")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    print(download_fema_disaster_declarations(force=args.force))


if __name__ == "__main__":
    main()
