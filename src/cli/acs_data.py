"""Download the scoped ACS 5-year county profile/subject tables (2010-2024)."""
from __future__ import annotations

import argparse
import json
import os
import time
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
ACS_DIR = ROOT / "data" / "acs"
TABLES = {
    "DP02": ("acs/acs5/profile", "Selected Social Characteristics"),
    "DP03": ("acs/acs5/profile", "Selected Economic Characteristics"),
    "DP04": ("acs/acs5/profile", "Selected Housing Characteristics"),
    "S2503": ("acs/acs5/subject", "Financial Characteristics of Occupied Housing Units"),
}
SPECIAL_VALUES = {-222222222, -333333333, -555555555, -666666666, -888888888, -999999999}


def _get_json(url: str) -> object:
    request = urllib.request.Request(url, headers={"User-Agent": "housing-predict/acs1-downloader"})
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.load(response)


def _url(year: int, dataset: str, params: dict[str, str]) -> str:
    return f"https://api.census.gov/data/{year}/{dataset}?{urllib.parse.urlencode(params, safe=',:*')}"


def _variables(year: int, table: str, dataset: str, api_key: str | None) -> tuple[list[str], pd.DataFrame]:
    suffix = f"?key={urllib.parse.quote(api_key)}" if api_key else ""
    metadata = _get_json(f"https://api.census.gov/data/{year}/{dataset}/groups/{table}.json{suffix}")
    rows = []
    for variable, info in metadata["variables"].items():
        if variable.startswith(f"{table}_") and variable.endswith(("E", "M", "PE", "PM")) and not variable.endswith(("EA", "MA")):
            rows.append({"year": year, "variable": variable, "label": info.get("label", ""), "concept": info.get("concept", ""), "predicate_type": info.get("predicateType", "")})
    frame = pd.DataFrame(rows).sort_values("variable")
    return frame["variable"].tolist(), frame


def _chunks(values: list[str], size: int = 45):
    for start in range(0, len(values), size):
        yield values[start:start + size]


def download_table(table: str, start_year: int, end_year: int, output_dir: Path, *, api_key: str | None, force: bool = False) -> list[Path]:
    table = table.upper()
    dataset, description = TABLES[table]
    output = output_dir / f"census_acs5_county_{table.lower()}_{start_year}_{end_year}.csv"
    dictionary = output_dir / f"census_acs5_{table.lower()}_variable_dictionary_{start_year}_{end_year}.csv"
    failures = output_dir / f"census_acs5_{table.lower()}_failures_{start_year}_{end_year}.csv"
    if output.exists() and dictionary.exists() and not force:
        print(f"Skipping {table}: {output} exists")
        return [output, dictionary]

    frames, dictionaries, failed = [], [], []
    for year in range(start_year, end_year + 1):
        print(f"Downloading ACS 5-year {year} {table} county data...", flush=True)
        try:
            variables, variable_frame = _variables(year, table, dataset, api_key)
            # Census supports group(TABLE), avoiding dozens of variable-chunk
            # requests for each wide profile/subject table.
            params = {"get": f"NAME,group({table})", "for": "county:*", "in": "state:*"}
            if api_key:
                params["key"] = api_key
            payload = _get_json(_url(year, dataset, params))
            merged = pd.DataFrame(payload[1:], columns=payload[0])
            merged = merged.loc[:, ~merged.columns.duplicated()].copy()
            merged.insert(0, "year", year)
            merged["state"] = merged["state"].str.zfill(2)
            merged["county"] = merged["county"].str.zfill(3)
            merged.insert(1, "county_fips", merged["state"] + merged["county"])
            for column in variables:
                numeric = pd.to_numeric(merged[column], errors="coerce")
                merged[column] = numeric.mask(numeric.isin(SPECIAL_VALUES))
            frames.append(merged)
            dictionaries.append(variable_frame)
        except Exception as exc:
            failed.append({"year": year, "table": table, "error": str(exc)})

    if not frames:
        raise RuntimeError(f"No ACS 5-year data returned for {table}; failures: {failed}")
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.concat(frames, ignore_index=True, sort=False).sort_values(["year", "county_fips"]).to_csv(output, index=False)
    pd.concat(dictionaries, ignore_index=True).drop_duplicates().to_csv(dictionary, index=False)
    if failed:
        pd.DataFrame(failed).to_csv(failures, index=False)
    elif failures.exists():
        failures.unlink()
    print(f"Wrote {description}: {output}")
    return [output, dictionary]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-year", type=int, default=2010)
    parser.add_argument("--end-year", type=int, default=2024)
    parser.add_argument("--tables", nargs="+", choices=TABLES, default=list(TABLES))
    parser.add_argument("--output-dir", type=Path, default=ACS_DIR)
    parser.add_argument("--census-api-key", default=os.getenv("CENSUS_API_KEY") or os.getenv("CENSUS_KEY"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if not 2010 <= args.start_year <= args.end_year <= 2024:
        parser.error("the scoped ACS period must be within 2010-2024")
    for table in args.tables:
        download_table(table, args.start_year, args.end_year, args.output_dir, api_key=args.census_api_key, force=args.force)


if __name__ == "__main__":
    main()
