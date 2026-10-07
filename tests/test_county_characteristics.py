import unittest

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import GroupKFold

from src.cli.acs_feature_registry import ACS_FEATURES
from src.modeling.county_characteristics import (
    FE_FEATURES, HAZARD_COLUMNS, CLIMATE_COLUMNS, TARGET, assert_unique,
    build_study_panel, feature_manifest, fixed_effects, fit_model, study_candidates,
    screen_training_availability,
)
from src.modeling.county_price import select_forecast_from_validation


class CharacteristicsTests(unittest.TestCase):
    def registry(self):
        return pd.DataFrame([dict(feature_name=f.name, acs_table=f.table, unit=f.unit, year=y)
            for f in ACS_FEATURES for y in range(f.first_year, f.last_year + 1)])

    def annual(self):
        rows = []
        for fips in ['01001', '09110', '72001']:
            for year in range(2010, 2025):
                r = {'county_fips': fips, 'county_name': fips, 'year': year}
                r.update({f.name: 20.0 for f in ACS_FEATURES})
                r[TARGET] = 100000.0
                r.update({c: 1.0 for c in HAZARD_COLUMNS})
                r.update({c: 50.0 for c in CLIMATE_COLUMNS})
                rows.append(r)
        return pd.DataFrame(rows)

    def test_primary_features_exclude_value_and_affordability(self):
        manifest = feature_manifest(self.registry())
        primary = manifest.loc[manifest.role.eq('primary')]
        self.assertNotIn(TARGET, primary.source_feature.tolist())
        self.assertFalse(primary.group.eq('affordability').any())
        self.assertNotIn('renter_occupied_pct', primary.source_feature.tolist())
        self.assertNotIn('unemployment_rate_pct', primary.source_feature.tolist())

    def test_period_alignment_scope_and_geography(self):
        annual, panel, excluded = build_study_panel(self.annual(), feature_manifest(self.registry()))
        self.assertEqual(set(panel.year), {2014, 2019, 2024})
        self.assertEqual(set(panel.state), {'AL', 'CT'})
        self.assertTrue(excluded.county_fips.eq('72001').all())
        self.assertTrue(panel.loc[panel.state.eq('CT'), 'boundary_review'].all())
        self.assertTrue(panel.loc[panel.state.eq('AL'), 'balanced_geography'].all())
        self.assertTrue(np.allclose(panel['log1p__period_sum__disaster_count'], np.log1p(5)))
        self.assertTrue(panel['period_mean__avg_temperature_f'].eq(50).all())

    def test_missing_year_not_treated_as_zero_hazards(self):
        annual = self.annual()
        annual = annual.loc[~(annual.county_fips.eq('01001') & annual.year.eq(2011))]
        _, panel, _ = build_study_panel(annual, feature_manifest(self.registry()))
        row = panel.loc[panel.county_fips.eq('01001') & panel.year.eq(2014)].iloc[0]
        self.assertTrue(np.isnan(row['log1p__period_sum__disaster_count']))

    def test_invalid_percentage_and_target_excluded_but_cold_climate_retained(self):
        annual = self.annual()
        annual.loc[annual.county_fips.eq('01001'), 'vacant_housing_units_pct'] = -666666666.
        annual.loc[annual.county_fips.eq('01001'), 'min_temperature_f'] = -15.
        annual.loc[annual.county_fips.eq('09110') & annual.year.eq(2024), TARGET] = 0.
        _, panel, excluded = build_study_panel(annual, feature_manifest(self.registry()))
        al = panel.loc[panel.state.eq('AL')]
        self.assertTrue(al.vacant_housing_units_pct.isna().all())
        self.assertTrue(al.period_mean__min_temperature_f.eq(-15.).all())
        self.assertTrue(excluded.reason.eq('Missing or nonpositive target').any())

    def test_duplicate_keys_fail(self):
        with self.assertRaises(ValueError):
            frame = self.annual()
            assert_unique(pd.concat([frame, frame.iloc[:1]]), ['county_fips', 'year'])

    def test_empty_initial_features_excluded_without_consulting_test(self):
        manifest = feature_manifest(self.registry())
        _, panel, _ = build_study_panel(self.annual(), manifest)
        feature = 'households_without_computer_pct'
        panel.loc[panel.year.eq(2014), feature] = np.nan
        first = screen_training_availability(manifest, panel)
        panel.loc[panel.year.eq(2024), feature] = 1e6
        second = screen_training_availability(manifest, panel)
        self.assertEqual(first.loc[first.feature.eq(feature), 'role'].iloc[0], 'excluded')
        pd.testing.assert_frame_equal(first, second)

    def test_preprocessing_uses_training_only(self):
        tr = pd.DataFrame({'x': [1., 2., 3., np.nan], 'log_value': [10., 11., 12., 11.]})
        model = fit_model(study_candidates(True)[0], tr, ['x'])
        transformer = model.named_steps['preprocessor'].named_transformers_['numeric']
        self.assertEqual(transformer.named_steps['imputer'].statistics_[0], 2.)
        model.predict(pd.DataFrame({'x': [1e9, np.nan]}))
        self.assertEqual(transformer.named_steps['imputer'].statistics_[0], 2.)

    def test_hist_model_accepts_numeric_only_characteristics(self):
        candidate = next(c for c in study_candidates(True) if c.estimator == 'hist_gradient_boosting')
        tr = pd.DataFrame({'x': np.arange(50), 'log_value': np.arange(50) / 100 + 10})
        model = fit_model(candidate, tr, ['x'])
        self.assertEqual(len(model.predict(tr[['x']])), 50)
        repeated = fit_model(candidate, tr, ['x'])
        np.testing.assert_array_equal(model.predict(tr[['x']]), repeated.predict(tr[['x']]))

    def test_validation_selection_ignores_test_targets(self):
        cv = pd.DataFrame([dict(candidate='a', validation_year=2022, rmsle=.01,
                                mae_2024_usd=1., median_ape=.01, r2=.9)])
        panel = pd.DataFrame({'forecast_year': [2022, 2024], 'target_price_2024_usd': [200., 500.],
                              'price_lag1_2024_usd': [100., 500.]})
        first, _ = select_forecast_from_validation(cv, panel, ['a'])
        panel.loc[1, 'target_price_2024_usd'] = 1e9
        second, _ = select_forecast_from_validation(cv, panel, ['a'])
        self.assertEqual(first, 'a')
        self.assertEqual(first, second)

    def test_grouped_counties_do_not_cross_folds(self):
        frame = pd.DataFrame({'county_fips': np.repeat(np.arange(20), 2)})
        for train, test in GroupKFold(5).split(frame, groups=frame.county_fips):
            self.assertTrue(set(frame.iloc[train].county_fips).isdisjoint(frame.iloc[test].county_fips))

    def test_fixed_effects_recovers_known_within_coefficients(self):
        rng = np.random.default_rng(42)
        beta = np.array([.4, .01, -.02, .005, .002, -.1])
        rows = []
        for county in range(100):
            intercept = rng.normal(10, 1)
            for period, year in enumerate([2014, 2019, 2024]):
                x = rng.normal(size=6)
                row = dict(zip(FE_FEATURES, x))
                row.update(county_fips=str(county), year=year, balanced_geography=True,
                           log_value=intercept + period * .1 + x @ beta + rng.normal(0, .001))
                rows.append(row)
        frame = pd.DataFrame(rows)
        coefficients, summary = fixed_effects(frame)
        np.testing.assert_allclose(coefficients.coefficient, beta, atol=.001)
        self.assertTrue(coefficients.cluster_se.gt(0).all())
        self.assertEqual(summary['counties'], 100)
        # Check absorption and finite-sample covariance against explicit dummies.
        dummies = pd.get_dummies(frame[['county_fips', 'year']].astype(str), drop_first=True, dtype=float)
        design = sm.add_constant(pd.concat([frame[FE_FEATURES], dummies], axis=1))
        explicit = sm.OLS(frame.log_value, design).fit(cov_type='cluster', cov_kwds={'groups': frame.county_fips})
        np.testing.assert_allclose(coefficients.coefficient, explicit.params[FE_FEATURES], atol=1e-10)
        np.testing.assert_allclose(coefficients.cluster_se, explicit.bse[FE_FEATURES], rtol=1e-8)


if __name__ == '__main__':
    unittest.main()
