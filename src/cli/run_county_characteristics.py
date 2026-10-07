"""Run the associations-first portfolio study against existing DuckDB marts."""
from argparse import ArgumentParser
from pathlib import Path

from src.modeling.county_characteristics import run_limited

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--database', type=Path, default=ROOT / 'data/housing_predict.duckdb')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'data/modeling/county_characteristics')
    parser.add_argument('--quick', action='store_true', help='Smoke run, not portfolio evidence')
    args = parser.parse_args()
    output = args.output_dir / 'quick' if args.quick else args.output_dir
    run_limited(args.database, output, args.quick)


if __name__ == '__main__':
    main()
