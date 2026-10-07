from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from src.modeling.county_price import (
    TARGET,
    build_forecast_panel,
    choose_final_model,
    materialize_panel,
    regression_metrics,
)


class CountyPriceModelingTests(unittest.TestCase):
    def setUp(self) -> None:
        rows = []
        for county_index, county in enumerate(("01001", "01003")):
            for year in range(2010, 2015):
                step = year - 2010
                rows.append({
                    "year": year,
                    "county_fips": county,
                    "county_name": f"County {county_index}",
                    "state": "AL",
                    "msa_code": "C1000",
                    "msa_type": "Metro",
                    TARGET: 100_000 + county_index * 20_000 + step * 5_000,
                    "housing_units": 10_000 + step * 100,
                    "vacant_housing_units_pct": 10.0 - step * 0.2,
                    "median_household_income_2024_usd": 50_000 + step * 1_000,
                    "mean_household_earnings_2024_usd": 60_000 + step * 1_200,
                    "disaster_count": 1 if year == 2012 else 0,
                    "storm_event_count": step,
                    "storm_damage_2024_usd": step * 1_000.0,
                    "storm_injuries": 0,
                    "storm_deaths": 0,
                    "avg_temperature_f": 60.0 + step if county_index == 0 else np.nan,
                    "min_temperature_f": 40.0 + step if county_index == 0 else np.nan,
                    "max_temperature_f": 80.0 + step if county_index == 0 else np.nan,
                    "precipitation_inches": 30.0 + step if county_index == 0 else np.nan,
                })
        self.frame = pd.DataFrame(rows)
        self.acs = [
            "housing_units", "vacant_housing_units_pct",
            "median_household_income_2024_usd", "mean_household_earnings_2024_usd",
        ]

    def test_panel_uses_only_prior_year_information(self) -> None:
        panel, feature_sets = build_forecast_panel(self.frame, self.acs, self.acs)
        row = panel.loc[
            panel["county_fips"].eq("01001") & panel["forecast_year"].eq(2013)
        ].iloc[0]
        self.assertEqual(row["source_year"], 2012)
        self.assertEqual(row["price_lag1_2024_usd"], 110_000)
        self.assertEqual(row["target_price_2024_usd"], 115_000)
        self.assertEqual(row["lag1__disaster_count"], 1)
        self.assertEqual(row["sum3__disaster_count"], 1)
        self.assertNotIn(TARGET, panel.columns)
        self.assertIn("delta1__vacant_housing_units_pct", feature_sets["primary"])
        forecast = panel.loc[
            panel["county_fips"].eq("01001") & panel["forecast_year"].eq(2015)
        ].iloc[0]
        self.assertEqual(forecast["source_year"], 2014)
        self.assertTrue(pd.isna(forecast["target_price_2024_usd"]))

    def test_climate_is_flagged_and_never_zero_filled(self) -> None:
        panel, _ = build_forecast_panel(self.frame, self.acs, self.acs)
        covered = panel.loc[panel["county_fips"].eq("01001")]
        uncovered = panel.loc[panel["county_fips"].eq("01003")]
        self.assertTrue(covered["climate_available"].eq(1).all())
        self.assertTrue(uncovered["climate_available"].eq(0).all())
        self.assertTrue(uncovered["lag1__avg_temperature_f"].isna().all())

    def test_materialized_panel_has_unique_forecast_keys(self) -> None:
        panel, _ = build_forecast_panel(self.frame, self.acs, self.acs)
        with tempfile.TemporaryDirectory() as directory:
            con = duckdb.connect(str(Path(directory) / "test.duckdb"))
            con.execute("CREATE SCHEMA feature")
            materialize_panel(con, panel)
            rows, keys = con.execute(
                "SELECT count(*), count(DISTINCT (county_fips, forecast_year)) "
                "FROM feature.county_price_forecast_panel"
            ).fetchone()
            con.close()
        self.assertEqual(rows, keys)

    def test_metrics_and_selection_gate(self) -> None:
        metrics = regression_metrics([100.0, 200.0], [100.0, 220.0])
        self.assertGreater(metrics["rmsle"], 0)
        rows = []
        for year in ("all", 2023, 2024):
            rows.extend([
                {"model": "county_lag_baseline", "test_year": year, "rmsle": 0.10},
                {"model": "candidate", "test_year": year, "rmsle": 0.09},
            ])
        self.assertEqual(choose_final_model(pd.DataFrame(rows), ["candidate"]), "candidate")


if __name__ == "__main__":
    unittest.main()
