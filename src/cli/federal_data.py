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
REQUIRED = {"disasterNumber", "state", "incidentType", "incidentBeginDate", "fipsStateCode", "fipsCountyCode", "designatedArea"}


def download_fema_disaster_declarations(*, force: bool = False) -> Path:
    if OUTPUT.exists() and not force:
        missing = REQUIRED - set(pd.read_csv(OUTPUT, nrows=0).columns)
        if missing:
            raise RuntimeError(f"Existing FEMA file is missing columns: {sorted(missing)}")
        return OUTPUT
    rows, skip, page_size = [], 0, 5000
    while True:
        query = urllib.parse.urlencode({"$orderby": "disasterNumber", "$top": page_size, "$skip": skip})
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
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUTPUT, index=False)
    return OUTPUT


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", nargs="?", choices=["fema-declarations"], default="fema-declarations")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    print(download_fema_disaster_declarations(force=args.force))


if __name__ == "__main__":
    main()
