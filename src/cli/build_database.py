"""Build a DuckDB backend containing only the county house-price modeling scope."""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
DATABASE_PATH = DATA_DIR / "housing_predict.duckdb"
START_YEAR, END_YEAR = 2010, 2024
EXCLUDED_FEMA_TYPES = ("Biological", "Dam/Levee Break", "Chemical", "Terrorist", "Other", "Toxic Substances")


def _q(value: str | Path) -> str:
    return "'" + str(value).replace("\\", "/").replace("'", "''") + "'"


def _latest(pattern: str) -> Path:
    paths = sorted(DATA_DIR.glob(pattern), key=lambda p: p.stat().st_mtime)
    if not paths:
        raise FileNotFoundError(f"Missing required scoped input: data/{pattern}")
    return paths[-1]


def _load_csv(con: duckdb.DuckDBPyConnection, schema: str, table: str, path: Path) -> None:
    con.execute(f'DROP TABLE IF EXISTS "{schema}"."{table}"')
    con.execute(f'CREATE TABLE "{schema}"."{table}" AS SELECT * FROM read_csv_auto({_q(path)}, header=true, all_varchar=true, sample_size=-1)')


def build_database(database_path: Path = DATABASE_PATH) -> None:
    required = {
        "dp02": _latest("acs/census_acs5_county_dp02_*.csv"),
        "dp03": _latest("acs/census_acs5_county_dp03_*.csv"),
        "dp04": _latest("acs/census_acs5_county_dp04_*.csv"),
        "s2503": _latest("acs/census_acs5_county_s2503_*.csv"),
        "fema": DATA_DIR / "fema" / "FEMA_Disaster_Declarations.csv",
        "noaa": DATA_DIR / "climate_damage" / "noaa_storm_events_county_damage.csv",
        "noaa_mapping": DATA_DIR / "climate_damage" / "noaa_storm_events_zone_county_mapping.csv",
        "climate": DATA_DIR / "climate" / "ncei_climate_at_a_glance_county_monthly.csv",
        "counties": DATA_DIR / "fipsgeo" / "fips_master_v2.csv",
    }
    missing = [str(path) for path in required.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required inputs:\n" + "\n".join(missing))
    database_path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(database_path))
    try:
        for schema in ("raw", "ref", "mart", "feature"):
            con.execute(f"CREATE SCHEMA IF NOT EXISTS {schema}")
        for name, path in required.items():
            _load_csv(con, "raw", name, path)

        con.execute("DROP TABLE IF EXISTS ref.counties")
        con.execute("""
            CREATE TABLE ref.counties AS SELECT lpad(fips, 5, '0') AS county_fips,
              county_name, state, state_long FROM raw.counties
        """)
        for table in ("dp02", "dp03", "dp04", "s2503"):
            con.execute(f"DROP TABLE IF EXISTS mart.acs_{table}")
            con.execute(f"""CREATE TABLE mart.acs_{table} AS
                SELECT * REPLACE (try_cast(year AS INTEGER) AS year, lpad(county_fips, 5, '0') AS county_fips)
                FROM raw.{table} WHERE try_cast(year AS INTEGER) BETWEEN {START_YEAR} AND {END_YEAR}""")

        # The median-value field shifted from DP04_0088 to DP04_0089 in 2015.
        con.execute("DROP TABLE IF EXISTS feature.county_house_price_target")
        con.execute("""CREATE TABLE feature.county_house_price_target AS
            SELECT year, county_fips, NAME AS county_name,
              try_cast(CASE WHEN year <= 2014 THEN DP04_0088E ELSE DP04_0089E END AS DOUBLE) AS median_owner_occupied_home_value_usd,
              try_cast(CASE WHEN year <= 2014 THEN DP04_0088M ELSE DP04_0089M END AS DOUBLE) AS median_owner_occupied_home_value_moe_usd
            FROM mart.acs_dp04
            WHERE try_cast(CASE WHEN year <= 2014 THEN DP04_0088E ELSE DP04_0089E END AS DOUBLE) IS NOT NULL""")

        excluded = ", ".join(_q(v) for v in EXCLUDED_FEMA_TYPES)
        con.execute("DROP TABLE IF EXISTS mart.fema_disaster_declarations_county")
        con.execute(f"""CREATE TABLE mart.fema_disaster_declarations_county AS
            WITH filtered AS (
              SELECT *, try_cast(substr(incidentBeginDate, 1, 4) AS INTEGER) AS event_year
              FROM raw.fema WHERE try_cast(substr(incidentBeginDate, 1, 4) AS INTEGER) BETWEEN {START_YEAR} AND {END_YEAR}
                AND incidentType NOT IN ({excluded})
            ), expanded AS (
              SELECT f.*, c.county_fips
              FROM filtered f JOIN ref.counties c ON lpad(f.fipsStateCode, 2, '0') = substr(c.county_fips, 1, 2)
              WHERE lower(trim(f.designatedArea)) IN ('statewide', 'state') OR trim(f.fipsCountyCode) IN ('000', '')
              UNION ALL
              SELECT f.*, lpad(f.fipsStateCode, 2, '0') || lpad(f.fipsCountyCode, 3, '0')
              FROM filtered f WHERE NOT (lower(trim(f.designatedArea)) IN ('statewide', 'state') OR trim(f.fipsCountyCode) IN ('000', ''))
            ) SELECT * FROM expanded QUALIFY row_number() OVER
              (PARTITION BY disasterNumber, county_fips ORDER BY lastRefresh DESC NULLS LAST) = 1""")

        con.execute("DROP TABLE IF EXISTS feature.county_fema_annual")
        con.execute("""CREATE TABLE feature.county_fema_annual AS
            SELECT event_year AS year, county_fips, count(DISTINCT disasterNumber)::INTEGER AS disaster_count
            FROM mart.fema_disaster_declarations_county GROUP BY 1, 2""")

        con.execute("DROP TABLE IF EXISTS mart.noaa_storm_events")
        con.execute(f"""CREATE TABLE mart.noaa_storm_events AS
            SELECT try_cast(n.year AS INTEGER) AS year,
              coalesce(nullif(lpad(n.county_fips, 5, '0'), '00000'), lpad(m.mapped_fips, 5, '0')) AS county_fips,
              n.event_id, n.event_type, try_cast(n.injuries_direct AS INTEGER) AS injuries_direct,
              try_cast(n.deaths_direct AS INTEGER) AS deaths_direct, try_cast(n.total_damage AS DOUBLE) AS total_damage_usd,
              try_strptime(n.begin_date_time, '%d-%b-%y %H:%M:%S') AS begin_timestamp
            FROM raw.noaa n LEFT JOIN raw.noaa_mapping m
              ON lpad(n.state_fips, 2, '0') = lpad(m.state_fips, 2, '0') AND lpad(n.cz_fips, 3, '0') = lpad(m.cz_fips, 3, '0')
            WHERE try_cast(n.year AS INTEGER) BETWEEN {START_YEAR} AND {END_YEAR}""")
        con.execute("DROP TABLE IF EXISTS feature.county_noaa_storm_annual")
        con.execute("""CREATE TABLE feature.county_noaa_storm_annual AS SELECT year, county_fips,
            count(DISTINCT event_id)::INTEGER AS storm_event_count, sum(total_damage_usd) AS storm_damage_usd,
            sum(injuries_direct)::INTEGER AS storm_injuries, sum(deaths_direct)::INTEGER AS storm_deaths
            FROM mart.noaa_storm_events WHERE county_fips IS NOT NULL GROUP BY 1, 2""")

        con.execute("DROP TABLE IF EXISTS feature.county_climate_monthly")
        con.execute(f"""CREATE TABLE feature.county_climate_monthly AS SELECT lpad(fips, 5, '0') AS county_fips,
            try_cast(year AS INTEGER) AS year, try_cast(month AS INTEGER) AS month,
            max(CASE WHEN parameter='tavg' THEN try_cast(value AS DOUBLE) END) AS avg_temperature_f,
            max(CASE WHEN parameter='tmin' THEN try_cast(value AS DOUBLE) END) AS min_temperature_f,
            max(CASE WHEN parameter='tmax' THEN try_cast(value AS DOUBLE) END) AS max_temperature_f,
            max(CASE WHEN parameter='pcp' THEN try_cast(value AS DOUBLE) END) AS precipitation_inches
            FROM raw.climate WHERE try_cast(year AS INTEGER) BETWEEN {START_YEAR} AND {END_YEAR} GROUP BY 1,2,3""")

        con.execute("DROP TABLE IF EXISTS feature.county_climate_annual")
        con.execute("""CREATE TABLE feature.county_climate_annual AS SELECT county_fips, year,
            avg(avg_temperature_f) AS avg_temperature_f, min(min_temperature_f) AS min_temperature_f,
            max(max_temperature_f) AS max_temperature_f, sum(precipitation_inches) AS precipitation_inches
            FROM feature.county_climate_monthly GROUP BY 1,2""")
        for table, cols in (("county_house_price_target", "county_fips, year"), ("county_fema_annual", "county_fips, year"), ("county_noaa_storm_annual", "county_fips, year"), ("county_climate_annual", "county_fips, year")):
            con.execute(f"CREATE INDEX IF NOT EXISTS idx_{table} ON feature.{table} ({cols})")
    finally:
        con.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=DATABASE_PATH)
    args = parser.parse_args()
    build_database(args.database)
    print(f"Built database: {args.database}")


if __name__ == "__main__":
    main()
