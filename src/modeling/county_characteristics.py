"""Reproducible associations-first county study; no price inputs in primary models."""
from __future__ import annotations

import hashlib
import json
import platform
import os
from pathlib import Path

os.environ.setdefault('LOKY_MAX_CPU_COUNT', str(os.cpu_count() or 1))

import duckdb
import numpy as np
import pandas as pd
import scipy.stats as stats
import sklearn
import statsmodels.api as sm
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from src.modeling.county_price import (
    TARGET, CLIMATE_COLUMNS, HAZARD_COLUMNS, Candidate, build_forecast_panel,
    default_candidates, export_elastic_coefficients, load_county_year_data,
    make_estimator, regression_metrics, save_model,
)

PERIODS = (2014, 2019, 2024)
STATE_CODES = dict(zip(
    '01 02 04 05 06 08 09 10 11 12 13 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 44 45 46 47 48 49 50 51 53 54 55 56'.split(),
    'AL AK AZ AR CA CO CT DE DC FL GA HI ID IL IN IA KS KY LA ME MD MA MI MN MS MO MT NE NV NH NJ NM NY NC ND OH OK OR PA RI SC SD TN TX UT VT VA WA WV WI WY'.split(),
))
# Conservative screen, not an areal crosswalk. Keep affected places in cross sections.
BOUNDARY_REVIEW = {'02105', '02195', '02198', '02261', '02063', '02066',
                   '02270', '02158', '46113', '46102', '51019', '51515'}
REDUNDANT = {'renter_occupied_pct', 'without_mortgage_pct'}
FE_FEATURES = ['log1p__median_household_income_2024_usd',
               'bachelors_degree_or_higher_pct', 'vacant_housing_units_pct',
               'detached_single_unit_pct', 'mean_commute_time_minutes',
               'average_household_size']
SOURCES = {
    'acs_comparisons': 'https://www.census.gov/programs-surveys/acs/guidance/comparing-acs-data.html',
    'county_changes': 'https://www.census.gov/programs-surveys/geography/technical-documentation/county-changes.2010.html',
    'geography_changes': 'https://www.census.gov/programs-surveys/acs/technical-documentation/table-and-geography-changes.html',
    'release_schedule': 'https://www.census.gov/programs-surveys/acs/news/data-releases/2024/release-schedule.html',
}


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False, default=str) + '\n', encoding='utf-8')


def assert_unique(frame: pd.DataFrame, keys: list[str]) -> None:
    if frame.duplicated(keys).any():
        raise ValueError(f'Duplicate keys: {keys}')


def feature_manifest(registry: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for feature, data in registry.groupby('feature_name', sort=True):
        source = data.iloc[0]
        group = {'dp02': 'social', 'dp03': 'economic', 'dp04': 'housing'}[source.acs_table]
        if feature.startswith(('owner_cost_', 'gross_rent_')) or feature in {'with_mortgage_pct', 'without_mortgage_pct'}:
            group = 'affordability'
        role, reason = 'primary', 'Available in all three study periods'
        if feature == TARGET:
            role, reason = 'target', 'Outcome; never a characteristics predictor'
        elif not set(PERIODS).issubset(set(data.year)):
            role, reason = 'excluded', 'Definition unavailable in one or more study periods'
        elif feature in REDUNDANT:
            role, reason = 'excluded', 'Complement of another included percentage'
        elif group == 'affordability':
            role, reason = 'sensitivity', 'May reflect housing costs; evaluated separately'
        transformed = feature
        if role not in {'target', 'excluded'} and (feature.endswith(('_usd', '_count')) or feature == 'housing_units'):
            transformed = 'log1p__' + feature
        rows.append(dict(source_feature=feature, feature=transformed, group=group, role=role,
                         unit=source.unit, transformation='log1p' if transformed != feature else 'identity',
                         reason=reason, first_year=int(data.year.min())))
    for feature in HAZARD_COLUMNS:
        rows.append(dict(source_feature=feature, feature='log1p__period_sum__' + feature,
                         group='hazards', role='primary', unit='five-year sum', transformation='log1p',
                         reason='Events summed over the same five-year ACS window', first_year=2010))
    for feature in CLIMATE_COLUMNS:
        rows.append(dict(source_feature=feature, feature='period_mean__' + feature,
                         group='climate', role='sensitivity', unit='five-year mean of annual measure',
                         transformation='mean', reason='Covered-county sensitivity only', first_year=2010))
    return pd.DataFrame(rows)


def build_study_panel(annual: pd.DataFrame, manifest: pd.DataFrame):
    assert_unique(annual, ['county_fips', 'year'])
    annual = annual.sort_values(['county_fips', 'year']).copy()
    annual['state'] = annual.county_fips.str[:2].map(STATE_CODES)
    excluded = annual.loc[annual.state.isna(), ['county_fips', 'county_name', 'year']].copy()
    excluded['reason'] = 'Outside 50 states and DC'
    annual = annual.loc[annual.state.notna()].copy()
    numeric = annual.select_dtypes(include='number').columns
    annual[numeric] = annual[numeric].replace([np.inf, -np.inf], np.nan)
    # Census special negative estimates and impossible percentages cannot enter a fit.
    for row in manifest.itertuples():
        col = row.source_feature
        if col in annual and col not in CLIMATE_COLUMNS:
            annual.loc[annual[col].lt(0), col] = np.nan
            if col.endswith('_pct'):
                annual.loc[annual[col].gt(100), col] = np.nan
    panel = annual.loc[annual.year.isin(PERIODS)].copy()
    for row in manifest.itertuples():
        if row.transformation == 'log1p' and row.group != 'hazards':
            panel[row.feature] = np.log1p(panel[row.source_feature])
    for year in PERIODS:
        window = annual.loc[annual.year.between(year - 4, year)].groupby('county_fips')
        mask = panel.year.eq(year)
        for col in HAZARD_COLUMNS:
            totals = window[col].sum(min_count=5)
            panel.loc[mask, 'log1p__period_sum__' + col] = np.log1p(panel.loc[mask, 'county_fips'].map(totals))
        for col in CLIMATE_COLUMNS:
            means = window[col].mean().where(window[col].count().eq(5))
            panel.loc[mask, 'period_mean__' + col] = panel.loc[mask, 'county_fips'].map(means)
    bad_target = panel[TARGET].isna() | panel[TARGET].le(0)
    dropped = panel.loc[bad_target, ['county_fips', 'county_name', 'year']].copy()
    dropped['reason'] = 'Missing or nonpositive target'
    exclusions = pd.concat([excluded, dropped], ignore_index=True)
    panel = panel.loc[~bad_target].copy()
    panel['log_value'] = np.log(panel[TARGET])
    panel['period_start'] = panel.year - 4
    presence = panel.groupby('county_fips').year.nunique()
    panel['boundary_review'] = panel.county_fips.isin(BOUNDARY_REVIEW) | panel.county_fips.str.startswith('09')
    panel['balanced_geography'] = panel.county_fips.map(presence).eq(3) & ~panel.boundary_review
    assert_unique(panel, ['county_fips', 'year'])
    return annual, panel.reset_index(drop=True), exclusions


def screen_training_availability(manifest, panel):
    """Exclude undefined inputs using the initial training period, never test values."""
    manifest = manifest.copy()
    training = panel.loc[panel.year.eq(2014)]
    for index, row in manifest.loc[manifest.role.isin(['primary', 'sensitivity'])].iterrows():
        if training[row.feature].notna().sum() == 0:
            manifest.loc[index, 'role'] = 'excluded'
            manifest.loc[index, 'reason'] = 'No observed values in initial 2014 training period'
    return manifest


def fit_model(candidate, train, features, target='log_value'):
    if not features or train.empty:
        raise ValueError('Empty training design')
    model = make_estimator(candidate, features)
    model.fit(train[features], train[target])
    return model


def value_prediction(model, frame, features):
    return np.exp(np.clip(model.predict(frame[features]), np.log(1000), np.log(10_000_000)))


def study_candidates(quick=False):
    return [c for c in default_candidates(quick) if c.target_kind == 'direct']


def select_candidates(train, validation, features, candidates):
    rows = []
    for candidate in candidates:
        model = fit_model(candidate, train, features)
        rows.append({'candidate': candidate.name, 'estimator': candidate.estimator,
                     **regression_metrics(validation[TARGET], value_prediction(model, validation, features))})
    scores = pd.DataFrame(rows).sort_values(['rmsle', 'candidate'])
    names = scores.drop_duplicates('estimator').candidate.tolist()
    return scores, [next(c for c in candidates if c.name == name) for name in names]


def fixed_effects(panel):
    complete = panel.loc[panel.balanced_geography].dropna(subset=FE_FEATURES + ['log_value']).copy()
    counts = complete.groupby('county_fips').year.transform('size')
    complete = complete.loc[counts.eq(3)].copy()
    if complete.county_fips.nunique() <= len(FE_FEATURES) + 2:
        raise ValueError('Insufficient balanced complete counties for fixed effects')
    cols = FE_FEATURES + ['log_value']
    demeaned = (complete[cols] - complete.groupby('county_fips')[cols].transform('mean')
                - complete.groupby('year')[cols].transform('mean') + complete[cols].mean())
    X = demeaned[FE_FEATURES]
    if np.linalg.matrix_rank(X) != len(FE_FEATURES):
        raise ValueError('Rank-deficient within-county design')
    result = sm.OLS(demeaned.log_value, X).fit()
    cov = sm.stats.sandwich_covariance.cov_cluster(result, complete.county_fips, use_correction=False)
    n, g, k = len(complete), complete.county_fips.nunique(), len(FE_FEATURES)
    # Account for absorbed county and period effects in finite-sample correction.
    cov *= g / (g - 1) * (n - 1) / (n - g - 2 - k)
    se = np.sqrt(np.diag(cov))
    critical = stats.t.ppf(.975, g - 1)
    rows = pd.DataFrame({'feature': FE_FEATURES, 'coefficient': result.params.to_numpy(),
                         'cluster_se': se, 'ci_low': result.params.to_numpy() - critical * se,
                         'ci_high': result.params.to_numpy() + critical * se})
    rows['p_value'] = 2 * stats.t.sf(np.abs(rows.coefficient / rows.cluster_se), g - 1)
    rows['interpretation'] = 'Log outcome per one predictor unit; income predictor is log1p dollars'
    return rows, {'rows': n, 'counties': int(g), 'within_r2': float(result.rsquared),
                  'excluded_rows': int(len(panel) - n), 'features': FE_FEATURES,
                  'uncertainty': 'County-clustered t intervals; not spatially clustered; no causal identification'}


def group_permutation(model, test, groups, features, repeats=5):
    base = regression_metrics(test[TARGET], value_prediction(model, test, features))['rmsle']
    rng = np.random.default_rng(42)
    rows = []
    for group, columns in groups.items():
        columns = [c for c in columns if c in features]
        if not columns:
            continue
        for repeat in range(repeats):
            changed = test[features].copy()
            changed[columns] = changed[columns].iloc[rng.permutation(len(changed))].to_numpy()
            score = regression_metrics(test[TARGET], value_prediction(model, changed, features))['rmsle']
            rows.append({'group': group, 'repeat': repeat, 'rmsle_increase': score - base})
    return pd.DataFrame(rows)


def secondary_forecast(annual, stable_acs, all_acs, output, quick=False):
    # Restrict to continuous annual histories and screen known changing geography.
    counts = annual.groupby('county_fips').year.transform('nunique')
    allowed = counts.eq(15) & ~annual.county_fips.isin(BOUNDARY_REVIEW) & ~annual.county_fips.str.startswith('09')
    forecast, _ = build_forecast_panel(annual.loc[allowed], stable_acs, all_acs)
    forecast = forecast.loc[forecast.target_price_2024_usd.notna()].copy()
    history = ['log_price_lag1', 'log_price_lag2', 'price_log_growth_1y', 'price_cagr_3y',
               'price_growth_mean_3y', 'price_growth_std_3y']
    # Use the same selected, non-price ACS characteristics as the main study.
    manifest = pd.read_csv(output / 'feature_manifest.csv')
    sources = manifest.loc[manifest.role.eq('primary') & manifest.group.ne('hazards'), 'source_feature']
    characteristics = ['lag1__' + name for name in sources if 'lag1__' + name in forecast]
    characteristics += ['lag1__' + name for name in HAZARD_COLUMNS]
    designs = {'history_only': history, 'history_plus_characteristics': history + characteristics}
    validation_rows, test_rows, selected = [], [], {}
    candidates = study_candidates(quick)
    years = [2022] if quick else list(range(2018, 2023))
    for label, features in designs.items():
        for candidate in candidates:
            for year in years:
                train = forecast.loc[forecast.forecast_year.lt(year)]
                val = forecast.loc[forecast.forecast_year.eq(year)]
                model = fit_model(candidate, train, features, 'target_log_price')
                score = regression_metrics(val.target_price_2024_usd, value_prediction(model, val, features))
                validation_rows.append({'design': label, 'candidate': candidate.name, 'year': year, **score})
        scores = pd.DataFrame(validation_rows)
        winner = scores.loc[scores.design.eq(label)].groupby('candidate').rmsle.mean().idxmin()
        selected[label] = winner
    train = forecast.loc[forecast.forecast_year.le(2022)]
    test = forecast.loc[forecast.forecast_year.isin([2023, 2024])]
    predictions = test[['county_fips', 'forecast_year', 'source_year', 'target_price_2024_usd']].copy()
    predictions['previous_value'] = test.price_lag1_2024_usd
    for label, features in designs.items():
        candidate = next(c for c in candidates if c.name == selected[label])
        model = fit_model(candidate, train, features, 'target_log_price')
        predictions[label] = value_prediction(model, test, features)
    for label in ['previous_value', *designs]:
        for year in ['all', 2023, 2024]:
            subset = predictions if year == 'all' else predictions.loc[predictions.forecast_year.eq(year)]
            test_rows.append({'design': label, 'year': year, 'rows': len(subset),
                              **regression_metrics(subset.target_price_2024_usd, subset[label])})
    pd.DataFrame(validation_rows).to_csv(output / 'forecast_validation.csv', index=False)
    predictions.to_csv(output / 'forecast_predictions.csv', index=False)
    metrics = pd.DataFrame(test_rows)
    metrics.to_csv(output / 'forecast_metrics.csv', index=False)
    return metrics, selected


def run_study(database: Path, output: Path, quick=False):
    output.mkdir(parents=True, exist_ok=True)
    print('Auditing marts and building period panel', flush=True)
    con = duckdb.connect(str(database), read_only=True)
    try:
        audits = []
        tables = ['county_housing', 'county_economic_annual', 'county_social_annual',
                  'county_fema_annual', 'county_noaa_storm_annual', 'county_climate_annual']
        for table in tables:
            duplicates = con.execute(f'SELECT count(*) - count(DISTINCT (county_fips, year)) FROM feature.{table}').fetchone()[0]
            audits.append({'table': table, 'duplicate_keys': duplicates})
            if duplicates:
                raise ValueError(f'Duplicate source keys in {table}')
        raw_reference = con.execute("SELECT lpad(fips, 5, '0') AS county_fips FROM raw.counties").df()
        assert_unique(raw_reference, ['county_fips'])
        registry = con.execute('SELECT * FROM feature.canonical_feature_registry').df()
        annual, stable_acs, all_acs = load_county_year_data(con)
        source_keys = con.execute('SELECT county_fips, year FROM feature.county_housing').df()
    finally:
        con.close()
    unmatched = source_keys.merge(annual[['county_fips', 'year']], how='left', indicator=True)
    unmatched = unmatched.loc[unmatched._merge.eq('left_only')].drop(columns='_merge')
    unmatched['reason'] = 'Housing row lost in economic/social inner join'
    annual.loc[~annual.county_fips.isin(raw_reference.county_fips), 'msa_type'] = 'Unknown'
    manifest = feature_manifest(registry)
    annual, panel, exclusions = build_study_panel(annual, manifest)
    manifest = screen_training_availability(manifest, panel)
    exclusions = pd.concat([exclusions, unmatched], ignore_index=True)
    groups = {group: data.feature.tolist() for group, data in manifest.loc[manifest.role.eq('primary')].groupby('group')}
    features = [feature for columns in groups.values() for feature in columns]
    assert TARGET not in features and not any('price' in f for f in features)
    manifest.to_csv(output / 'feature_manifest.csv', index=False)
    registry.to_csv(output / 'feature_definitions_by_year.csv', index=False)
    exclusions.to_csv(output / 'exclusions.csv', index=False)
    panel.to_parquet(output / 'county_period_panel.parquet', index=False)
    # Dedicated database avoids mutating or invalidating the source marts.
    with duckdb.connect(str(output / 'study.duckdb')) as study_db:
        study_db.register('panel_frame', panel)
        study_db.execute('CREATE OR REPLACE TABLE county_period_panel AS SELECT * FROM panel_frame')
        study_db.execute('CREATE UNIQUE INDEX IF NOT EXISTS county_period_key ON county_period_panel(county_fips, year)')
    missing = []
    for year, frame in panel.groupby('year'):
        for feature in features:
            missing.append({'year': year, 'feature': feature, 'rows': len(frame), 'missing_fraction': float(frame[feature].isna().mean())})
    pd.DataFrame(missing).to_csv(output / 'missingness.csv', index=False)
    geography = panel.groupby('county_fips').agg(county_name=('county_name', 'last'),
        periods=('year', 'nunique'), boundary_review=('boundary_review', 'max'),
        balanced_geography=('balanced_geography', 'all')).reset_index()
    geography.to_csv(output / 'geography_audit.csv', index=False)
    pd.DataFrame(audits).to_csv(output / 'source_key_audit.csv', index=False)
    coverage = panel.groupby(['year', 'state']).agg(rows=('county_fips', 'size'),
        longitudinal_eligible=('balanced_geography', 'sum')).reset_index()
    coverage.to_csv(output / 'coverage.csv', index=False)
    train, validation, test = [panel.loc[panel.year.eq(year)] for year in PERIODS]
    print('Selecting characteristics models on 2014 -> 2019 only', flush=True)
    candidates = study_candidates(quick)
    scores, winners = select_candidates(train, validation, features, candidates)
    scores.to_csv(output / 'validation_metrics.csv', index=False)
    best = winners[0]
    validation_baseline = regression_metrics(validation[TARGET], np.full(len(validation), train[TARGET].median()))
    selected_name = best.name if scores.iloc[0].rmsle < validation_baseline['rmsle'] else 'training_median'
    # The selection manifest is written before accessing held-out model scores.
    write_json(output / 'selection.json', {'selected': selected_name, 'best_ml': best.name,
        'selection_metric': '2019 RMSLE', 'validation_baseline': validation_baseline,
        'candidates': [dict(name=c.name, estimator=c.estimator, params=c.params) for c in candidates]})
    development = panel.loc[panel.year.lt(2024)]
    predictions = test[['county_fips', 'county_name', 'state', 'msa_type', 'year', TARGET]].copy()
    predictions['training_median'] = development[TARGET].median()
    metrics = [{'model': 'training_median', **regression_metrics(test[TARGET], predictions.training_median)}]
    models = {}
    for candidate in winners:
        model = fit_model(candidate, development, features)
        models[candidate.name] = model
        predictions[candidate.name] = value_prediction(model, test, features)
        metrics.append({'model': candidate.name, **regression_metrics(test[TARGET], predictions[candidate.name])})
    predictions['selected_prediction'] = predictions[selected_name]
    predictions['residual_usd'] = predictions[TARGET] - predictions.selected_prediction
    predictions.to_csv(output / 'test_predictions.csv', index=False)
    metrics = pd.DataFrame(metrics)
    metrics.to_csv(output / 'test_metrics.csv', index=False)
    save_model(output / 'characteristics_model.pkl', models[best.name], {'features': features,
        'target': TARGET, 'transform': 'log', 'training_periods': [2014, 2019],
        'selected': selected_name, 'best_ml': best.name, 'purpose': 'Held-out evaluation; not refitted on 2024'})
    print('Running five county-grouped development folds', flush=True)
    grouped_rows, fold_rows = [], []
    for fold, (ti, vi) in enumerate(GroupKFold(n_splits=5).split(development, groups=development.county_fips), 1):
        tr, va = development.iloc[ti], development.iloc[vi]
        assert set(tr.county_fips).isdisjoint(va.county_fips)
        # Retune inside the geographic fold; no held-out county enters selection.
        _, fold_winners = select_candidates(tr.loc[tr.year.eq(2014)], tr.loc[tr.year.eq(2019)], features, candidates)
        for candidate in fold_winners:
            model = fit_model(candidate, tr, features)
            grouped_rows.append({'fold': fold, 'model': candidate.estimator,
                'candidate': candidate.name, 'rows': len(va),
                **regression_metrics(va[TARGET], value_prediction(model, va, features))})
        grouped_rows.append({'fold': fold, 'model': 'training_median', 'candidate': 'training_median',
            'rows': len(va), **regression_metrics(va[TARGET], np.full(len(va), tr[TARGET].median()))})
        fold_rows.extend({'fold': fold, 'county_fips': fips} for fips in sorted(va.county_fips.unique()))
    pd.DataFrame(grouped_rows).to_csv(output / 'geographic_metrics.csv', index=False)
    pd.DataFrame(fold_rows).to_csv(output / 'geographic_fold_assignments.csv', index=False)
    print('Feature-group ablations, climate and affordability sensitivities', flush=True)
    ablations = []
    for group, columns in groups.items():
        remaining = [f for f in features if f not in columns]
        model = fit_model(best, development, remaining)
        ablations.append({'experiment': 'without_' + group, 'rows': len(test),
            **regression_metrics(test[TARGET], value_prediction(model, test, remaining))})
    ablations.append({'experiment': 'all_primary', 'rows': len(test),
        **regression_metrics(test[TARGET], predictions[best.name])})
    for group in ('affordability', 'climate'):
        extra = manifest.loc[manifest.group.eq(group) & manifest.role.eq('sensitivity'), 'feature'].tolist()
        tr, te = development, test
        if group == 'climate':
            tr, te = development.dropna(subset=extra), test.dropna(subset=extra)
        for label, cols in [('matched_primary', features), ('augmented', features + extra)]:
            model = fit_model(best, tr, cols)
            ablations.append({'experiment': group + '_' + label, 'rows': len(te),
                **regression_metrics(te[TARGET], value_prediction(model, te, cols))})
    pd.DataFrame(ablations).to_csv(output / 'feature_group_experiments.csv', index=False)
    breakdown = predictions.copy()
    breakdown['value_decile'] = pd.qcut(breakdown[TARGET], 10, labels=False, duplicates='drop')
    rows = []
    for dimension in ('state', 'msa_type', 'value_decile'):
        for value, subset in breakdown.groupby(dimension, dropna=False):
            rows.append({'dimension': dimension, 'value': value, 'rows': len(subset),
                **regression_metrics(subset[TARGET], subset.selected_prediction)})
    pd.DataFrame(rows).to_csv(output / 'error_breakdowns.csv', index=False)
    print('Estimating associations and explanations', flush=True)
    fe, fe_summary = fixed_effects(panel)
    fe.to_csv(output / 'fixed_effects.csv', index=False)
    write_json(output / 'fixed_effects_summary.json', fe_summary)
    association = []
    for year, subset in panel.groupby('year'):
        for feature in features:
            valid = subset[[feature, 'log_value']].dropna()
            association.append({'year': year, 'feature': feature, 'rows': len(valid),
                'spearman_rho': valid[feature].corr(valid.log_value, method='spearman')})
    pd.DataFrame(association).to_csv(output / 'associations_by_period.csv', index=False)
    importance_rows = []
    for candidate in winners:
        importance = group_permutation(models[candidate.name], test, groups, features)
        importance['model'] = candidate.name
        importance_rows.append(importance)
    pd.concat(importance_rows).to_csv(output / 'group_permutation.csv', index=False)
    elastic = next(c for c in winners if c.estimator == 'elastic_net')
    export_elastic_coefficients(models[elastic.name], output / 'standardized_coefficients.csv')
    # Period-specific coefficient stability is descriptive, never model selection.
    coefficient_rows = []
    for year, subset in panel.groupby('year'):
        model = fit_model(elastic, subset, features)
        coefficient_rows.extend({'year': year, 'feature': name, 'coefficient': float(value)}
            for name, value in zip(model.named_steps['preprocessor'].get_feature_names_out(), model.named_steps['model'].coef_))
    pd.DataFrame(coefficient_rows).to_csv(output / 'coefficient_stability.csv', index=False)
    # Local-bin response curves avoid replacing every county with an implausible value.
    curves = []
    for feature in FE_FEATURES[:3]:
        low, high = development[feature].quantile([.05, .95])
        edges = np.unique(np.quantile(development[feature].dropna().clip(low, high), np.linspace(0, 1, 11)))
        for left, right in zip(edges[:-1], edges[1:]):
            local = test.loc[test[feature].between(left, right)].copy()
            if len(local) < 20:
                continue
            at_low, at_high = local[features].copy(), local[features].copy()
            at_low[feature], at_high[feature] = left, right
            effect = models[best.name].predict(at_high) - models[best.name].predict(at_low)
            curves.append({'feature': feature, 'left': left, 'right': right, 'rows': len(local),
                           'local_log_prediction_change': float(effect.mean())})
    curves = pd.DataFrame(curves)
    if not curves.empty:
        curves['accumulated_log_effect'] = curves.groupby('feature').local_log_prediction_change.cumsum()
        curves['accumulated_log_effect'] -= curves.groupby('feature').accumulated_log_effect.transform('mean')
    curves.to_csv(output / 'local_response_curves.csv', index=False)
    print('Running annual retrospective forecasting comparison', flush=True)
    forecast_metrics, forecast_selected = secondary_forecast(annual, stable_acs, all_acs, output, quick)
    summary = {'question': 'How much can county characteristics explain county home values?',
        'target': TARGET, 'periods': list(PERIODS), 'rows': len(panel),
        'counties_by_period': {str(y): len(d) for y, d in panel.groupby('year')},
        'features': len(features), 'selected_model': selected_name, 'best_ml': best.name,
        'run_mode': 'quick' if quick else 'full', 'seed': 42, 'weighting': 'Equal county-period weights',
        'source_database': str(database.resolve()),
        'forecast_selected': forecast_selected, 'fixed_effects': fe_summary, 'sources': SOURCES,
        'panel_sha256': hashlib.sha256(pd.util.hash_pandas_object(panel, index=False).values.tobytes()).hexdigest(),
        'versions': {'python': platform.python_version(), 'sklearn': sklearn.__version__,
                     'pandas': pd.__version__, 'duckdb': duckdb.__version__},
        'limitations': ['Observational associations, not causal effects or individual-house appraisals.',
            'Same-period ACS characteristics estimate home values; this is not advance forecasting.',
            'Geographic folds are county-disjoint but neighboring counties can remain correlated.',
            'Fixed-effects sample excludes known changes and incomplete histories; no areal crosswalk.',
            'Minor boundary changes and ACS margins of error are not modeled.',
            'Annual forecast labels overlap four ACS survey years; release vintages are not reconstructed.',
            '2024 ACS five-year estimates were released January 29, 2026; no operational end-2024 forecast claim.',
            'Hazard absence is assumed zero in event marts; reporting and mapping gaps may remain.',
            'Metro labels come from a static reference and are used only for error breakdowns.',
            'Ablations and explanations are prespecified held-out diagnostics, not model-selection evidence.']}
    write_json(output / 'run_summary.json', summary)
    export_portfolio(output, panel, predictions, metrics, forecast_metrics, summary, fe)
    export_figures(output, predictions, metrics, fe)
    print(f'Completed study: {output}', flush=True)
    return summary


def export_portfolio(output, panel, predictions, metrics, forecast, summary, fe):
    web = output / 'website_data'
    web.mkdir(exist_ok=True)
    chart = panel[['county_fips', 'county_name', 'state', 'year', 'period_start', TARGET,
                   'balanced_geography', *FE_FEATURES]].copy()
    chart = chart.merge(predictions[['county_fips', 'year', 'selected_prediction', 'residual_usd']],
                        on=['county_fips', 'year'], how='left')
    chart.to_json(web / 'counties.json', orient='records', double_precision=4)
    for name in ['test_metrics', 'validation_metrics', 'geographic_metrics', 'error_breakdowns',
                 'feature_group_experiments', 'fixed_effects', 'group_permutation',
                 'associations_by_period', 'local_response_curves', 'forecast_metrics']:
        pd.read_csv(output / (name + '.csv')).to_json(web / (name + '.json'), orient='records', double_precision=6)
    write_json(web / 'summary.json', summary)
    winner = metrics.set_index('model').loc[summary['selected_model']]
    baseline = metrics.set_index('model').loc['training_median']
    fc = forecast.loc[forecast.year.eq('all')].set_index('design')
    improvement = 100 * (1 - winner.rmsle / baseline.rmsle)
    fc_gain = 100 * (1 - fc.loc['history_plus_characteristics', 'rmsle'] / fc.loc['history_only', 'rmsle'])
    rows = ['# County characteristics and US home values', '',
        '## Question and headline finding', '',
        f"The validation-selected model ({summary['selected_model']}) explains {winner.r2:.1%} of the "
        f"held-out 2024 cross-county variance (R²), with a ${winner.mae_2024_usd:,.0f} mean absolute error. "
        f"Its RMSLE is {improvement:.1f}% lower than the training-median baseline.", '',
        'These are out-of-period predictions using characteristics measured in the same ACS window as the outcome. '
        'They establish predictive association, not causation or advance forecasting.', '',
        '## Data journey', '',
        f"Existing Census, FEMA, NOAA and climate inputs feed DuckDB marts, then an audited panel of {summary['rows']:,} "
        f"county-period observations and {summary['features']} primary predictors. "
        'The periods are 2010–2014, 2015–2019 and 2020–2024; values are in 2024 dollars. '
        'The target is median owner-occupied home value, not transaction price.', '',
        'See coverage.csv, exclusions.csv, geography_audit.csv, missingness.csv and feature_manifest.csv for the exact sample.', '',
        '![Model performance and held-out predictions](figures/prediction_evidence.png)', '',
        '## Model evidence', '', '| Model | 2024 MAE | RMSLE | R² |', '|---|---:|---:|---:|']
    for r in metrics.itertuples():
        rows.append(f'| {r.model} | ${r.mae_2024_usd:,.0f} | {r.rmsle:.3f} | {r.r2:.3f} |')
    group_results = pd.read_csv(output / 'feature_group_experiments.csv').set_index('experiment')
    geographic = pd.read_csv(output / 'geographic_metrics.csv').groupby('model')[['rmsle', 'mae_2024_usd', 'r2']].mean()
    removals = group_results.loc[group_results.index.str.startswith('without_'), 'rmsle']
    strongest = removals.idxmax().removeprefix('without_')
    penalty = 100 * (removals.max() / group_results.loc['all_primary', 'rmsle'] - 1)
    rows += ['', f'Removing the {strongest} feature group increases 2024 RMSLE the most '
        f'({penalty:.1f}%). Group removal uses the fixed selected specification, '
        'so it measures reliance by this model rather than a causal contribution.', '',
        'The separate geographic evaluation gives these unweighted means across five county-disjoint folds:', '',
        '| Family | Geographic MAE | RMSLE | R² |', '|---|---:|---:|---:|']
    for label, r in geographic.iterrows():
        rows.append(f'| {label} | ${r.mae_2024_usd:,.0f} | {r.rmsle:.3f} | {r.r2:.3f} |')
    rows += ['', 'Parameters and model family were selected on 2014 → 2019 validation before evaluating 2024. '
        'The final evaluation fit uses 2014 and 2019. Geographic evaluation uses five county-disjoint folds '
        'with tuning repeated inside each fold on pre-2024 data.', '', '## Within-county associations', '',
        f"The fixed-effects regression uses {summary['fixed_effects']['counties']:,} balanced, complete counties. "
        'It controls for county and period effects. Intervals use county-clustered standard errors. '
        f"Its within R² is {summary['fixed_effects']['within_r2']:.3f}, so these six predictors account for "
        'a substantially smaller share of variation after removing persistent county differences and common period effects. '
        'This R² has a different denominator from the prediction benchmark above.', '',
        '| Predictor | Log-value coefficient | 95% interval |', '|---|---:|---:|']
    for r in fe.itertuples():
        rows.append(f'| {r.feature} | {r.coefficient:.4f} | [{r.ci_low:.4f}, {r.ci_high:.4f}] |')
    rows += ['', '![Within-county associations](figures/within_county_associations.png)', '',
        'A percentage predictor is measured in percentage points; log income is in log1p dollars. '
        'Cross-county relationships need not have the same sign as within-county changes. '
        'See associations_by_period.csv, coefficient_stability.csv and group_permutation.csv for stability diagnostics.', '',
        '## What characteristics add to forecasting', '',
        f'Adding characteristics changes held-out RMSLE by {fc_gain:.1f}% relative to the history-only model '
        '(positive means improvement). This separate experiment predicts the next overlapping ACS estimate.', '',
        '| Design | 2023–2024 MAE | RMSLE |', '|---|---:|---:|']
    for label, r in fc.iterrows():
        rows.append(f'| {label} | ${r.mae_2024_usd:,.0f} | {r.rmsle:.3f} |')
    rows += ['', '## Website handoff', '',
        'website_data/counties.json contains FIPS, observed period values, selected characteristics, predictions and residuals. '
        'Join it to matching-vintage county boundaries when implementing the county map; geometry is not bundled. '
        'The other JSON files support model comparisons, error charts, association plots and response curves. '
        'Host these static results; do not train in the browser.', '', '## Limitations', '']
    rows.extend('- ' + item for item in summary['limitations'])
    rows += ['', '## Reproduce', '', '```powershell', 'python -m src.cli.run_county_characteristics',
             'python -m unittest discover -s tests -v', '```', '', '## Sources', '']
    rows.extend(f'- [{name}]({url})' for name, url in SOURCES.items())
    (output / 'REPORT.md').write_text('\n'.join(rows) + '\n', encoding='utf-8')


def export_figures(output, predictions, metrics, fe):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    figures = output / 'figures'
    figures.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), layout='constrained')
    display = metrics.sort_values('mae_2024_usd', ascending=False)
    axes[0].barh(display.model, display.mae_2024_usd, color=['#9ba7b5' if m == 'training_median' else '#247e8a' for m in display.model])
    axes[0].xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'${x / 1000:.0f}k'))
    axes[0].set(title='2024 error by model', xlabel='Mean absolute error in 2024 dollars')
    axes[1].scatter(predictions[TARGET], predictions.selected_prediction, s=8, alpha=.3, color='#247e8a', rasterized=True)
    bounds = [min(predictions[TARGET].min(), predictions.selected_prediction.min()),
              max(predictions[TARGET].max(), predictions.selected_prediction.max())]
    axes[1].plot(bounds, bounds, color='#aa523b', linewidth=1)
    axes[1].set(xscale='log', yscale='log', xlabel='Observed median home value (log scale)',
                ylabel='Predicted value (log scale)', title='Validation-selected model: held-out 2024 period')
    for axis in [axes[1].xaxis, axes[1].yaxis]:
        axis.set_major_formatter(FuncFormatter(lambda x, _: f'${x / 1000:,.0f}k'))
    fig.suptitle('County characteristics predict ACS home values without price-history inputs', fontsize=14)
    fig.savefig(figures / 'prediction_evidence.png', dpi=150)
    plt.close(fig)
    # Convert coefficients to stated, readable increments before comparing effects.
    scale = np.array([.1, 10, 10, 10, 10, 1])
    labels = ['Log household income +0.1', 'Bachelor\'s attainment +10 pp',
              'Vacant housing +10 pp', 'Detached housing +10 pp', 'Commute time +10 min',
              'Household size +1 person']
    estimate, low, high = [100 * np.expm1(fe[column].to_numpy() * scale) for column in ['coefficient', 'ci_low', 'ci_high']]
    fig, ax = plt.subplots(figsize=(10, 5), layout='constrained')
    ax.errorbar(estimate, labels, xerr=[estimate - low, high - estimate], fmt='o', color='#247e8a', capsize=4)
    ax.axvline(0, color='#8c9099', linewidth=1)
    ax.set(title='Within-county associations, controlling for county and period',
           xlabel='Associated percent difference in home value (95% county-clustered interval)')
    fig.savefig(figures / 'within_county_associations.png', dpi=150)
    plt.close(fig)


def run_limited(database, output, quick=False):
    # Avoid nested BLAS/OpenMP oversubscription; tree parallelism remains explicit.
    with threadpool_limits(limits=2):
        return run_study(database, output, quick)
