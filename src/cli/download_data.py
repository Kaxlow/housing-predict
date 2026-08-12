"""Download all public inputs in the 2010-2024 county modeling scope."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCES = {
    "acs": "acs_data.py",
    "fema": "federal_data.py",
    "storms": "climate_damage_data.py",
    "climate": "ncei_county_weather_data.py",
}


def _run(script: str, args: list[str]) -> None:
    subprocess.run([sys.executable, str(HERE / script), *args], check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", choices=[*SOURCES, "all"])
    parser.add_argument("--force", action="store_true")
    args, passthrough = parser.parse_known_args()
    force = ["--force"] if args.force else []
    common = ["--start-year", "2010", "--end-year", "2024", *force]
    selected = list(SOURCES) if args.source == "all" else [args.source]
    for source in selected:
        if source == "fema":
            _run(SOURCES[source], ["fema-declarations", *force])
        elif source in {"acs", "storms", "climate"}:
            _run(SOURCES[source], [*common, *passthrough])


if __name__ == "__main__":
    main()
