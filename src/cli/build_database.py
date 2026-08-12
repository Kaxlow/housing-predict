"""Build a DuckDB backend containing only the county house-price modeling scope."""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import duckdb

from .acs_feature_registry import ACS_FEATURES, ACSFeature

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
DATABASE_PATH = DATA_DIR / "housing_predict.duckdb"
START_YEAR, END_YEAR = 2010, 2024
EXCLUDED_FEMA_TYPES = ("Biological", "Dam/Levee Break", "Chemical", "Terrorist", "Other", "Toxic Substances")

# Annual CPI-U (U.S. city average, all items). ACS dollar estimates are expressed
# in each release year's dollars; these factors put them in constant 2024 dollars.
CPI_U = {
    2010: 218.056, 2011: 224.939, 2012: 229.594, 2013: 232.957,
    2014: 236.736, 2015: 237.017, 2016: 240.007, 2017: 245.120,
    2018: 251.107, 2019: 255.657, 2020: 258.811, 2021: 270.970,
    2022: 292.655, 2023: 304.702, 2024: 313.689,
}


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


def _resolve_acs_registry(con: duckdb.DuckDBPyConnection) -> dict[str, dict[int, str]]:
    """Resolve every canonical feature by meaning, failing on gaps or ambiguity."""
    resolved: dict[str, dict[int, str]] = {}
    for feature in ACS_FEATURES:
        rows = con.execute(
            f"SELECT try_cast(year AS INTEGER), variable, label FROM raw.{feature.table}_variable_dictionary "
            "WHERE try_cast(year AS INTEGER) BETWEEN ? AND ?",
            [feature.first_year, feature.last_year],
        ).fetchall()
        by_year: dict[int, str] = {}
        next_year_variable: str | None = None
        for year in range(feature.last_year, feature.first_year - 1, -1):
            matches = [(variable, label) for row_year, variable, label in rows
                       if row_year == year and re.fullmatch(feature.label_pattern, label)]
            if len(matches) > 1 and next_year_variable is not None:
                stable_match = [match for match in matches if match[0] == next_year_variable]
                if len(stable_match) == 1:
                    matches = stable_match
            if len(matches) != 1:
                observed = [f"{variable}: {label}" for variable, label in matches]
                raise RuntimeError(
                    "ACS canonical feature definition mismatch: "
                    f"feature={feature.name!r}, table={feature.table!r}, year={year}, "
                    f"expected one label matching {feature.label_pattern!r}, found {len(matches)}: {observed}. "
                    "Reconcile the definition in acs_feature_registry.py before rebuilding."
                )
            by_year[year] = matches[0][0]
            next_year_variable = matches[0][0]
        resolved[feature.name] = by_year
    return resolved


def _acs_case(feature: ACSFeature, mappings: dict[str, dict[int, str]]) -> str:
    clauses = [f'WHEN year = {year} THEN "{variable}"' for year, variable in mappings[feature.name].items()]
    return "CASE " + " ".join(clauses) + " END"


def _feature_select(feature: ACSFeature, mappings: dict[str, dict[int, str]]) -> str:
    value = f"try_cast({_acs_case(feature, mappings)} AS DOUBLE)"
    if feature.complement_of_percent:
        value = f"100.0 - ({value})"
    if feature.inflation_adjusted:
        value += " * c.inflation_factor_to_2024"
    return f'{value} AS "{feature.name}"'


def build_database(database_path: Path = DATABASE_PATH) -> None:
    required = {
        "dp02": _latest("acs/census_acs5_county_dp02_*.csv"),
        "dp03": _latest("acs/census_acs5_county_dp03_*.csv"),
        "dp04": _latest("acs/census_acs5_county_dp04_*.csv"),
        "s2503": _latest("acs/census_acs5_county_s2503_*.csv"),
        "dp02_variable_dictionary": _latest("acs/census_acs5_dp02_variable_dictionary_*.csv"),
        "dp03_variable_dictionary": _latest("acs/census_acs5_dp03_variable_dictionary_*.csv"),
        "dp04_variable_dictionary": _latest("acs/census_acs5_dp04_variable_dictionary_*.csv"),
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
        con.execute("BEGIN TRANSACTION")
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

        mappings = _resolve_acs_registry(con)
        con.execute("DROP TABLE IF EXISTS feature.canonical_feature_registry")
        registry_rows = [
            (feature.name, feature.table, feature.unit, year, variable, feature.label_pattern)
            for feature in ACS_FEATURES for year, variable in mappings[feature.name].items()
        ]
        con.execute("""CREATE TABLE feature.canonical_feature_registry (
            feature_name VARCHAR, acs_table VARCHAR, unit VARCHAR, year INTEGER,
            source_variable VARCHAR, expected_label_pattern VARCHAR
        )""")
        con.executemany("INSERT INTO feature.canonical_feature_registry VALUES (?, ?, ?, ?, ?, ?)", registry_rows)

        con.execute("DROP TABLE IF EXISTS ref.cpi_u_annual")
        con.execute("CREATE TABLE ref.cpi_u_annual (year INTEGER PRIMARY KEY, cpi_u DOUBLE, inflation_factor_to_2024 DOUBLE)")
        con.executemany(
            "INSERT INTO ref.cpi_u_annual VALUES (?, ?, ?)",
            [(year, cpi, CPI_U[2024] / cpi) for year, cpi in CPI_U.items()],
        )

        by_table = {table: [feature for feature in ACS_FEATURES if feature.table == table]
                    for table in ("dp02", "dp03", "dp04")}
        for old_table in ("county_house_price_target", "county_housing", "county_economic_annual", "county_social_annual"):
            con.execute(f"DROP TABLE IF EXISTS feature.{old_table}")

        economic_columns = ",\n              ".join(_feature_select(feature, mappings) for feature in by_table["dp03"])
        con.execute(f"""CREATE TABLE feature.county_economic_annual AS
            SELECT a.year, a.county_fips, a.NAME AS county_name,
              {economic_columns}
            FROM mart.acs_dp03 a JOIN ref.cpi_u_annual c USING (year)""")

        social_columns = ",\n              ".join(_feature_select(feature, mappings) for feature in by_table["dp02"])
        con.execute(f"""CREATE TABLE feature.county_social_annual AS
            SELECT a.year, a.county_fips, a.NAME AS county_name,
              {social_columns}
            FROM mart.acs_dp02 a JOIN ref.cpi_u_annual c USING (year)""")

        housing_columns = ",\n              ".join(_feature_select(feature, mappings) for feature in by_table["dp04"])
        con.execute("DROP TABLE IF EXISTS feature.county_house_price_target")
        con.execute(f"""CREATE TABLE feature.county_housing AS
            SELECT a.year, a.county_fips, a.NAME AS county_name,
              {housing_columns}
            FROM mart.acs_dp04 a JOIN ref.cpi_u_annual c USING (year)""")

        excluded = ", ".join(_q(v) for v in EXCLUDED_FEMA_TYPES)
        con.execute("DROP TABLE IF EXISTS mart.fema_disaster_declarations_county")
        con.execute(f"""CREATE TABLE mart.fema_disaster_declarations_county AS
            WITH parsed AS (
              SELECT *, try_cast(incidentBeginDate AS TIMESTAMP) AS incident_begin_timestamp,
                try_cast(incidentEndDate AS TIMESTAMP) AS reported_incident_end_timestamp
              FROM raw.fema
            ), duration_by_type AS (
              SELECT incidentType,
                round(avg(date_diff('day', incident_begin_timestamp, reported_incident_end_timestamp)))::INTEGER AS average_duration_days
              FROM parsed
              WHERE incident_begin_timestamp IS NOT NULL AND reported_incident_end_timestamp >= incident_begin_timestamp
              GROUP BY incidentType
            ), imputed AS (
              SELECT p.* EXCLUDE (incidentEndDate),
                coalesce(reported_incident_end_timestamp,
                  incident_begin_timestamp + d.average_duration_days * INTERVAL '1 day') AS incidentEndDate,
                reported_incident_end_timestamp IS NULL AND incident_begin_timestamp IS NOT NULL
                  AND d.average_duration_days IS NOT NULL AS incident_end_date_imputed
              FROM parsed p LEFT JOIN duration_by_type d USING (incidentType)
            ), filtered AS (
              SELECT *, CASE
                WHEN try_cast(substr(incidentBeginDate, 1, 4) AS INTEGER) BETWEEN {START_YEAR} AND {END_YEAR}
                  THEN try_cast(substr(incidentBeginDate, 1, 4) AS INTEGER)
                ELSE year(incidentEndDate)
              END AS event_year
              FROM imputed WHERE (
                try_cast(substr(incidentBeginDate, 1, 4) AS INTEGER) BETWEEN {START_YEAR} AND {END_YEAR}
                OR year(incidentEndDate) BETWEEN {START_YEAR} AND {END_YEAR}
              )
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
            WITH cleaned AS (SELECT try_cast(n.year AS INTEGER) AS year,
              coalesce(nullif(lpad(n.county_fips, 5, '0'), '00000'), lpad(m.mapped_fips, 5, '0')) AS county_fips,
              n.event_id, n.event_type, try_cast(n.injuries_direct AS INTEGER) AS injuries_direct,
              try_cast(n.deaths_direct AS INTEGER) AS deaths_direct, try_cast(n.total_damage AS DOUBLE) AS total_damage_usd,
              try_strptime(n.begin_date_time, '%d-%b-%y %H:%M:%S') AS begin_timestamp
            FROM raw.noaa n LEFT JOIN raw.noaa_mapping m
              ON lpad(n.state_fips, 2, '0') = lpad(m.state_fips, 2, '0')
             AND lpad(n.cz_fips, 3, '0') = lpad(m.cz_fips, 3, '0')
             AND trim(regexp_replace(
                   regexp_replace(upper(n.cz_name), '[^A-Z0-9]+', ' ', 'g'),
                   '\\s+(CITY AND BOROUGH|CENSUS AREA|MUNICIPALITY|BOROUGH|PARISH|COUNTY|CITY)$',
                   '', 'g'
                 )) = m.normalized_cz_name
            WHERE try_cast(n.year AS INTEGER) BETWEEN {START_YEAR} AND {END_YEAR})
            SELECT * FROM cleaned
            WHERE county_fips IS NOT NULL AND total_damage_usd IS NOT NULL""")
        con.execute("DROP TABLE IF EXISTS feature.county_noaa_storm_annual")
        con.execute("""CREATE TABLE feature.county_noaa_storm_annual AS SELECT year, county_fips,
            count(DISTINCT event_id)::INTEGER AS storm_event_count, sum(total_damage_usd) AS storm_damage_usd,
            sum(injuries_direct)::INTEGER AS storm_injuries, sum(deaths_direct)::INTEGER AS storm_deaths
            FROM mart.noaa_storm_events WHERE county_fips IS NOT NULL GROUP BY 1, 2""")

        con.execute("DROP TABLE IF EXISTS feature.county_climate_monthly")
        con.execute("DROP TABLE IF EXISTS feature.county_climate_annual")
        con.execute(f"""CREATE TABLE feature.county_climate_annual AS
            WITH monthly AS (
              SELECT lpad(fips, 5, '0') AS county_fips, try_cast(year AS INTEGER) AS year,
                try_cast(month AS INTEGER) AS month,
                max(CASE WHEN parameter='tavg' THEN try_cast(value AS DOUBLE) END) AS avg_temperature_f,
                max(CASE WHEN parameter='tmin' THEN try_cast(value AS DOUBLE) END) AS min_temperature_f,
                max(CASE WHEN parameter='tmax' THEN try_cast(value AS DOUBLE) END) AS max_temperature_f,
                max(CASE WHEN parameter='pcp' THEN try_cast(value AS DOUBLE) END) AS precipitation_inches
              FROM raw.climate
              WHERE try_cast(year AS INTEGER) BETWEEN {START_YEAR} AND {END_YEAR}
              GROUP BY 1, 2, 3
            )
            SELECT county_fips, year, avg(avg_temperature_f) AS avg_temperature_f,
              min(min_temperature_f) AS min_temperature_f,
              max(max_temperature_f) AS max_temperature_f,
              sum(precipitation_inches) AS precipitation_inches
            FROM monthly GROUP BY 1, 2""")
        for table, cols in (("county_housing", "county_fips, year"), ("county_economic_annual", "county_fips, year"), ("county_social_annual", "county_fips, year"), ("county_fema_annual", "county_fips, year"), ("county_noaa_storm_annual", "county_fips, year"), ("county_climate_annual", "county_fips, year")):
            con.execute(f"CREATE INDEX IF NOT EXISTS idx_{table} ON feature.{table} ({cols})")
        con.execute("COMMIT")
    except Exception:
        con.execute("ROLLBACK")
        raise
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
