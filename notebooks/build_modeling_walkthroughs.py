"""Build readable, executable notebooks from the completed county study.

Run from the repository root. This packages public county aggregates, never the
source database or a pickled estimator. Notebook execution is a separate step.
"""
from pathlib import Path
import hashlib
import importlib.metadata
import json
import shutil
import textwrap

import duckdb
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'notebooks/03_modeling'
EVIDENCE = OUT / 'evidence'
RESULTS = ROOT / 'data/modeling/county_characteristics'


def md(text):
    return nbf.v4.new_markdown_cell(textwrap.dedent(text).strip())


def code(text):
    return nbf.v4.new_code_cell(textwrap.dedent(text).strip())


SETUP = '''
from pathlib import Path
import sys, json, hashlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display

# Works from the notebook directory or the source/bundle root.
ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents]
            if (p / "src/modeling/county_characteristics.py").exists())
sys.path.insert(0, str(ROOT))
NB_DIR = ROOT / "notebooks/03_modeling"
EVIDENCE = NB_DIR / "evidence"
def read_csv(name):
    return pd.read_csv(EVIDENCE / name, dtype={"county_fips": str})
def read_json(name):
    return json.loads((EVIDENCE / name).read_text(encoding="utf-8"))

# Fail loudly if a packaged artifact has changed since publication.
provenance = read_json("snapshot.json")
for name, digest in provenance["sha256"].items():
    assert hashlib.sha256((EVIDENCE / name).read_bytes()).hexdigest() == digest, name
panel = pd.read_parquet(EVIDENCE / "county_period_panel.parquet")
manifest = read_csv("feature_manifest.csv")
groups = {group: data.feature.tolist() for group, data in
          manifest.loc[manifest.role.eq("primary")].groupby("group")}
features = [feature for columns in groups.values() for feature in columns]
from src.modeling.county_price import TARGET, regression_metrics
plt.rcParams.update({"figure.figsize": (9, 4), "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 110})
pd.set_option("display.max_colwidth", 90)
print("Evidence: completed full study, 2026-10-06; equal county-period weights.")
'''


def save(name, cells):
    notebook = nbf.v4.new_notebook(cells=cells, metadata={
        'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
        'language_info': {'name': 'python', 'version': importlib.metadata.version('ipykernel')},
    })
    # Let the executing kernel supply its actual Python version.
    notebook.metadata.language_info.pop('version')
    nbf.write(notebook, OUT / name)


def main():
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    for path in RESULTS.glob('*.csv'):
        if path.name not in {'forecast_predictions.csv', 'forecast_validation.csv'}:
            shutil.copy2(path, EVIDENCE / path.name)
    for name in ['county_period_panel.parquet', 'selection.json', 'fixed_effects_summary.json']:
        shutil.copy2(RESULTS / name, EVIDENCE / name)
    summary = json.loads((RESULTS / 'run_summary.json').read_text())
    summary.pop('source_database', None)
    (EVIDENCE / 'run_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    # A real, small annual sample makes aggregation code executable without DuckDB.
    import sys
    sys.path.insert(0, str(ROOT))
    from src.modeling.county_price import load_county_year_data
    with duckdb.connect(str(ROOT / 'data/housing_predict.duckdb'), read_only=True) as con:
        annual, _, _ = load_county_year_data(con)
    annual.loc[annual.county_fips.isin(['01001', '06037', '08031'])].to_parquet(
        EVIDENCE / 'annual_example.parquet', index=False)
    snapshot = {'study_date': '2026-10-06', 'description': 'Completed-study walkthrough, not original exploration.',
                'source': 'data/modeling/county_characteristics full run',
                'sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in sorted(EVIDENCE.iterdir()) if p.name != 'snapshot.json'},
                'source_sha256': {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                    for name in ['src/modeling/county_characteristics.py', 'src/modeling/county_price.py']}}
    (EVIDENCE / 'snapshot.json').write_text(json.dumps(snapshot, indent=2) + '\n')

    save('01_modeling_dataset.ipynb', [
        md('''# 01 · From DuckDB marts to a modeling dataset

        **Question:** Can county characteristics explain differences in home values across the United States?

        This is a reconstructed walkthrough of the full study completed October 6, 2026.
        It documents the final analysis; it is not a contemporaneous exploration log.
        The target is ACS median owner-occupied home value in **constant 2024 dollars**,
        not a sale price or an individual-home appraisal. Associations are not causal effects.

        Run all cells for the packaged evidence walkthrough. Original DuckDB marts are
        required only for the optional rebuild. See [README](README.md) for setup and provenance.
        Next: [training and validation](02_training_and_validation.ipynb).'''),
        code(SETUP),
        md('''## 1. Define the observation and the sample

        One row represents one county and one five-year ACS period. We use the 50 states
        and DC and give each county equal weight within a period. The three windows do
        not overlap: 2010–2014, 2015–2019, and 2020–2024. Adjacent annual ACS releases
        share four survey years ([Census guidance](https://www.census.gov/programs-surveys/acs/guidance/comparing-acs-data.html)).
        Cross sections retain eligible counties; the longitudinal analysis applies a
        stricter geography screen. County identifiers are strings so leading zeros survive.'''),
        code('''from src.modeling.county_characteristics import assert_unique, STATE_CODES
assert_unique(panel, ["county_fips", "year"])
assert panel.county_fips.str.fullmatch(r"\\d{5}").all()
assert set(panel.year) == {2014, 2019, 2024}
assert panel.county_fips.str[:2].isin(STATE_CODES).all()
assert panel[TARGET].gt(0).all()
np.testing.assert_allclose(panel.log_value, np.log(panel[TARGET]))
display(panel.groupby(["period_start", "year"]).agg(
    counties=("county_fips", "nunique"), median_value=(TARGET, "median")))
print(f"{len(panel):,} unique county-period observations")'''),
        md('''## 2. Audit the joins before modeling

        The production loader joins housing, economic and social marts, then attaches
        hazards, climate and county labels. Duplicate keys fail the run; rows lost in the
        inner join are recorded. Here we inspect the full-run audit, not just the small example.
        Missing metro labels remain `Unknown`. Exclusion counts below are source county-year
        rows by reason, not necessarily distinct study counties.'''),
        code('''audit = read_csv("source_key_audit.csv")
assert audit.duplicate_keys.eq(0).all()
display(audit)
exclusions = read_csv("exclusions.csv")
display(exclusions.groupby("reason").size().rename("source_county_year_rows").to_frame())
geography = read_csv("geography_audit.csv")
display(geography.groupby(["boundary_review", "balanced_geography"]).size().rename("counties").to_frame())'''),
        md('''## 3. Establish a feature contract

        Economic, social, housing structure and hazard variables form the primary design.
        Climate is a covered-county sensitivity analysis. Affordability and mortgage measures
        are separate because they can already reflect housing costs. Previous values,
        geographic price summaries, IDs and target-derived ratios are excluded.

        Definitions must exist in all three periods. Availability is then checked using
        **2014 only**: broadband/computer measures have no initial observed values.
        Redundant owner/renter and mortgage/no-mortgage categories are not both included.'''),
        code('''display(manifest.groupby(["role", "group"]).size().rename("features").to_frame())
display(manifest.loc[manifest.role.ne("primary"), ["source_feature", "role", "reason"]])
assert len(features) == 57
assert TARGET not in features
assert not set(features) & {"county_fips", "state", "msa_code", "log_value"}
assert not any("price" in f or "home_value" in f for f in features)
print("Primary model inputs:", len(features))'''),
        md('''## 4. Engineer features with the same code used in production

        Negative ACS sentinels and impossible percentages become missing. Percentages stay
        in percentage points; dollar and count predictors use `log1p`. The target uses `log`.
        Hazard counts/damage sum over the matching five years, requiring five observed years.
        Climate uses five-year means, also requiring five observations. Valid negative
        temperatures remain valid. Event marts already treat absent records as zero.

        The real annual sample below contains Autauga County, Los Angeles County and Denver
        County. It demonstrates the transformation; all headline results use the full panel.
        The function source is displayed to make the transformations reviewable.'''),
        code('''import inspect
from src.modeling.county_characteristics import (
    feature_manifest, build_study_panel, screen_training_availability)
print(inspect.getsource(build_study_panel))'''),
        code('''registry = read_csv("feature_definitions_by_year.csv")
annual_example = pd.read_parquet(EVIDENCE / "annual_example.parquet")
clean_example, engineered, _ = build_study_panel(annual_example, feature_manifest(registry))
income = "median_household_income_2024_usd"
display(engineered[["county_fips", "year", income, "log1p__" + income,
                    "storm_event_count", "log1p__period_sum__storm_event_count"]])
np.testing.assert_allclose(engineered["log1p__" + income], np.log1p(engineered[income]))
window = clean_example.loc[clean_example.year.between(2010, 2014)]
totals = window.groupby("county_fips").storm_event_count.sum(min_count=5)
first = engineered.loc[engineered.year.eq(2014)]
np.testing.assert_allclose(first.log1p__period_sum__storm_event_count,
                          np.log1p(first.county_fips.map(totals)), equal_nan=True)
original = panel.set_index(["county_fips", "year"])
rebuilt = engineered.set_index(["county_fips", "year"])
np.testing.assert_allclose(rebuilt[features], original.loc[rebuilt.index, features], equal_nan=True)
print("Example transformations match the published panel.")'''),
        md('''## 5. Check missingness without learning preprocessing from the test set

        Missingness is reported for all periods as a data diagnostic. Median imputation,
        missingness indicators and linear-model standardization are fitted separately
        inside every training fold. Nothing is imputed here.'''),
        code('''screened = screen_training_availability(feature_manifest(registry), panel)
assert set(screened.loc[screened.role.eq("primary"), "feature"]) == set(features)
missing = read_csv("missingness.csv").pivot(index="feature", columns="year", values="missing_fraction")
display(missing.loc[missing.max(axis=1).sort_values(ascending=False).head(12).index].round(3))
display(manifest.loc[manifest.reason.str.contains("No observed"), ["source_feature", "reason"]])'''),
        md('''## 6. Optional: rebuild the full panel from DuckDB

        This branch reads the original database without writing to it. It reproduces the
        join, geography-label correction, feature construction and training-availability
        screen. The complete CLI additionally exports the join/exclusion audits and all
        modeling experiments. Set the flag only when the marts are available locally.'''),
        code('''REBUILD_FROM_MARTS = False
DATABASE = ROOT / "data/housing_predict.duckdb"
if REBUILD_FROM_MARTS:
    import duckdb
    from src.modeling.county_price import load_county_year_data
    with duckdb.connect(str(DATABASE), read_only=True) as con:
        for table in read_csv("source_key_audit.csv")["table"]:
            assert con.execute(f"SELECT count(*) - count(DISTINCT (county_fips, year)) FROM feature.{table}").fetchone()[0] == 0
        reference = con.execute("SELECT lpad(fips, 5, '0') AS county_fips FROM raw.counties").df()
        assert_unique(reference, ["county_fips"])
        full_registry = con.execute("SELECT * FROM feature.canonical_feature_registry").df()
        annual, _, _ = load_county_year_data(con)
    annual.loc[~annual.county_fips.isin(reference.county_fips), "msa_type"] = "Unknown"
    _, rebuilt_panel, _ = build_study_panel(annual, feature_manifest(full_registry))
    rebuilt_manifest = screen_training_availability(feature_manifest(full_registry), rebuilt_panel)
    pd.testing.assert_frame_equal(rebuilt_panel, panel)
    print("Full panel matches the saved evidence.")
else:
    print("Using verified packaged evidence; the original database is not required.")'''),
        md('''## What this establishes

        The full panel contains 9,420 observations and 57 primary inputs. The annual sample
        reproduces the feature transformations. Known substantial county changes are
        flagged rather than resolved with an invented crosswalk; minor boundary changes
        and ACS sampling uncertainty remain limitations.

        [Continue to training and validation →](02_training_and_validation.ipynb)'''),
    ])

    save('02_training_and_validation.ipynb', [
        md('''# 02 · Train models without leaking the answer

        **Question:** Do nonlinear models explain county differences better than a regularized linear model?

        This completed-study walkthrough uses the same pipeline functions as the benchmark.
        Run all cells to audit the saved selection and folds. Set `RETRAIN = True` to rerun
        all six candidates and nested county-grouped validation from the packaged panel.
        It does not require DuckDB marts. No 2024 performance is used here.

        [Dataset](01_modeling_dataset.ipynb) · [Evaluation](03_evaluation_and_interpretation.ipynb) · [Setup](README.md)'''),
        code(SETUP),
        md('''## 1. Freeze the evaluation design

        Fit on 2014 and choose parameters using 2019 RMSLE. Compare the best candidate
        with a training-median baseline. After selection, refit on 2014 + 2019 and reserve
        2024 for final evaluation. Predictors are contemporaneous with the target; this
        tests whether associations transfer over time, not advance forecasting.

        Separate geographic validation holds out whole counties on pre-2024 data and
        repeats parameter selection using only each fold's training counties.'''),
        code('''train = panel.loc[panel.year.eq(2014)].copy()
validation = panel.loc[panel.year.eq(2019)].copy()
development = panel.loc[panel.year.lt(2024)].copy()
assert set(development.year) == {2014, 2019}
display(pd.DataFrame({"purpose": ["fit", "select", "final evaluation (unopened here)"],
                      "period_end": [2014, 2019, 2024],
                      "rows": [len(train), len(validation), int(panel.year.eq(2024).sum())]}))
baseline = regression_metrics(validation[TARGET], np.full(len(validation), train[TARGET].median()))
display(pd.Series(baseline, name="2019 training-median baseline"))'''),
        md('''## 2. Put preprocessing inside the estimator

        Elastic Net uses median imputation, missingness indicators, standardization and
        regularization. Extra Trees and histogram gradient boosting allow nonlinear
        relationships and interactions. They impute within each fit without standardizing.
        Targets are natural log values; scoring converts predictions back to dollars.
        The implementation clips back-transformed predictions to $1,000–$10,000,000.
        Model random states are 42.'''),
        code('''import inspect
from src.modeling.county_price import make_estimator
from src.modeling.county_characteristics import study_candidates, select_candidates, fit_model, value_prediction
from threadpoolctl import threadpool_limits
candidates = study_candidates(quick=False)
print(inspect.getsource(make_estimator))
display(pd.DataFrame([{"candidate": c.name, "family": c.estimator, "parameters": c.params} for c in candidates]))'''),
        md('''## 3. Select using 2019 only

        RMSLE assesses proportional-scale errors, MAE is in 2024 dollars, and R² is computed
        in dollar space. Lower RMSLE wins; ties between candidates use their names for
        deterministic ordering. A median baseline wins if the best model does not improve
        on its validation RMSLE. The saved decision precedes held-out test scoring.'''),
        code('''RETRAIN = False  # True reruns selection and geographic validation; allow several minutes.
saved_selection = read_json("selection.json")
saved_scores = read_csv("validation_metrics.csv")
if RETRAIN:
    with threadpool_limits(limits=2):
        scores, winners = select_candidates(train, validation, features, candidates)
else:
    scores = saved_scores.sort_values(["rmsle", "candidate"])
    winner_names = scores.drop_duplicates("estimator").candidate.tolist()
    winners = [next(c for c in candidates if c.name == name) for name in winner_names]
selected = winners[0].name if scores.iloc[0].rmsle < baseline["rmsle"] else "training_median"
assert selected == saved_selection["selected"], "Results changed: investigate before updating the study."
np.testing.assert_allclose(baseline["rmsle"], saved_selection["validation_baseline"]["rmsle"])
if RETRAIN:
    np.testing.assert_allclose(scores.set_index("candidate").sort_index().rmsle,
                              saved_scores.set_index("candidate").sort_index().rmsle, rtol=1e-6)
display(scores)
print("Frozen selection:", selected)'''),
        code('''ax = scores.sort_values("rmsle", ascending=False).plot.barh(x="candidate", y="rmsle", legend=False, color="#276454")
ax.axvline(baseline["rmsle"], color="#a55d38", linestyle="--", label="Training median")
ax.set(xlabel="2019 RMSLE · lower is better", ylabel="", title="Selection evidence uses 2014 → 2019 only")
ax.legend(); plt.tight_layout(); plt.show()'''),
        md('''**Reading the evidence:** histogram gradient boosting (`hist_direct_2`) is selected
        by validation RMSLE. This decision is not revisited after looking at 2024.
        The search is deliberately small; it is not proof that this is the best possible model.'''),
        md('''## 4. Check that each county stays in one geographic fold

        This measures generalization to held-out counties. It is not future forecasting
        and does not hold out entire states or spatial buffers. Neighboring counties can
        remain correlated across folds. The mean below averages fold scores; it is not
        a pooled out-of-fold R².'''),
        code('''from sklearn.model_selection import GroupKFold
assignments = read_csv("geographic_fold_assignments.csv")
assert not assignments.county_fips.duplicated().any()
assert assignments.fold.nunique() == 5
assert set(assignments.county_fips) == set(development.county_fips)
folds = list(GroupKFold(n_splits=5).split(development, groups=development.county_fips))
checks = []
for fold, (ti, vi) in enumerate(folds, 1):
    tr, va = development.iloc[ti], development.iloc[vi]
    assert set(tr.county_fips).isdisjoint(va.county_fips)
    assert set(va.county_fips) == set(assignments.loc[assignments.fold.eq(fold), "county_fips"])
    checks.append({"fold": fold, "training_rows": len(tr), "held_out_rows": len(va),
                   "held_out_counties": va.county_fips.nunique(), "overlapping_counties": 0})
display(pd.DataFrame(checks))'''),
        code('''if RETRAIN:
    geographic_rows = []
    with threadpool_limits(limits=2):
        for fold, (ti, vi) in enumerate(folds, 1):
            tr, va = development.iloc[ti], development.iloc[vi]
            # Tuning sees only training counties, separated into 2014 and 2019.
            _, fold_winners = select_candidates(tr.loc[tr.year.eq(2014)],
                tr.loc[tr.year.eq(2019)], features, candidates)
            for candidate in fold_winners:
                fitted = fit_model(candidate, tr, features)
                geographic_rows.append({"fold": fold, "model": candidate.estimator,
                    "candidate": candidate.name, "rows": len(va),
                    **regression_metrics(va[TARGET], value_prediction(fitted, va, features))})
            geographic_rows.append({"fold": fold, "model": "training_median",
                "candidate": "training_median", "rows": len(va),
                **regression_metrics(va[TARGET], np.full(len(va), tr[TARGET].median()))})
    geographic = pd.DataFrame(geographic_rows)
    saved_geo = read_csv("geographic_metrics.csv")
    np.testing.assert_allclose(geographic.sort_values(["fold", "model"]).rmsle,
                              saved_geo.sort_values(["fold", "model"]).rmsle, rtol=1e-6)
else:
    geographic = read_csv("geographic_metrics.csv")
display(geographic.groupby("model")[["rmsle", "mae_2024_usd", "r2"]].agg(["mean", "std"]).round(4))'''),
        md('''## 5. Handoff to final evaluation

        The selected name, candidate parameters and baseline are stored in `selection.json`.
        Every trained pipeline fits its preprocessing using only its own training rows.
        The next notebook evaluates frozen family winners on 2024 and uses those outcomes
        for reporting and diagnosis, never for a second selection round.

        [Continue to evaluation and interpretation →](03_evaluation_and_interpretation.ipynb)'''),
    ])

    save('03_evaluation_and_interpretation.ipynb', [
        md('''# 03 · Evaluate the frozen model and interpret associations

        **Question:** How well do the relationships generalize, where do predictions fail,
        and which associations are stable enough to discuss?

        This walkthrough recomputes metrics from saved predictions, draws diagnostics,
        and checks that the published findings match the evidence. Optional refitting
        uses the already-selected specification. All 2024 analyses here are reporting
        diagnostics; none changes model selection.

        [Dataset](01_modeling_dataset.ipynb) · [Training](02_training_and_validation.ipynb) · [Setup](README.md)'''),
        code(SETUP),
        md('''## 1. Recompute held-out metrics

        Refit the frozen family winners on 2014 + 2019; evaluate 2024. The baseline now
        uses the median of those same development rows. MAE and R² use dollars; RMSLE
        uses differences in `log1p` dollar values (distinct from the natural-log training target).
        All models are compared on the identical 3,139 counties.'''),
        code('''selection = read_json("selection.json")
predictions = read_csv("test_predictions.csv")
saved_metrics = read_csv("test_metrics.csv")
assert predictions.year.eq(2024).all() and predictions.county_fips.is_unique
np.testing.assert_allclose(predictions.selected_prediction, predictions[selection["selected"]])
metrics = pd.DataFrame([{"model": name, **regression_metrics(predictions[TARGET], predictions[name])}
                        for name in saved_metrics.model])
pd.testing.assert_frame_equal(metrics, saved_metrics, check_exact=False, rtol=1e-10)
display(metrics.round(4))
indexed = metrics.set_index("model")
chosen = indexed.loc[selection["selected"]]
improvement = 1 - chosen.rmsle / indexed.loc["training_median", "rmsle"]
print(f"Selected model: R² {chosen.r2:.3f}; MAE ${chosen.mae_2024_usd:,.0f}; RMSLE improvement {improvement:.1%}")'''),
        md('''## 2. Optional refit, with the choice already frozen

        This code calls the shared training pipeline. It never searches over 2024 results.
        The check compares a fresh development fit with published predictions. Minor
        numerical differences across software versions can occur; use the recorded environment.'''),
        code('''REFIT_SELECTED = False
if REFIT_SELECTED:
    from threadpoolctl import threadpool_limits
    from src.modeling.county_characteristics import study_candidates, fit_model, value_prediction
    development = panel.loc[panel.year.lt(2024)]
    test = panel.loc[panel.year.eq(2024)]
    if selection["selected"] == "training_median":
        predicted = np.full(len(test), development[TARGET].median())
    else:
        candidate = next(c for c in study_candidates() if c.name == selection["selected"])
        with threadpool_limits(limits=2):
            model = fit_model(candidate, development, features)
            predicted = value_prediction(model, test, features)
    saved = predictions.set_index("county_fips").loc[test.county_fips, "selected_prediction"]
    np.testing.assert_allclose(predicted, saved, rtol=1e-6)
    print("Refitted predictions match the study.")
else:
    print("Metrics recomputed from saved predictions; refit is optional.")'''),
        code('''fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].scatter(predictions[TARGET] / 1000, predictions.selected_prediction / 1000,
                s=8, alpha=.3, color="#276454")
limit = max(predictions[TARGET].max(), predictions.selected_prediction.max()) / 1000
axes[0].plot([0, limit], [0, limit], "--", color="#a55d38")
axes[0].set(xlabel="Observed value · $1,000", ylabel="Predicted value · $1,000", title="2024 held-out counties")
axes[1].scatter(predictions[TARGET] / 1000, predictions.residual_usd / 1000,
                s=8, alpha=.3, color="#276454")
axes[1].axhline(0, color="#a55d38", linestyle="--")
axes[1].set(xlabel="Observed value · $1,000", ylabel="Observed − predicted · $1,000", title="Positive residual = underprediction")
plt.tight_layout(); plt.show()'''),
        md('''## 3. Find where the model is less reliable

        State, metro status and observed-value deciles describe errors; they were not used
        to select a model. Small state samples can make R² unstable. Dollar errors usually
        grow with the scale of home values, so inspect proportional errors too. Deciles use
        observed 2024 outcomes and are diagnostic groups, not deployable inputs.'''),
        code('''breakdowns = read_csv("error_breakdowns.csv")
display(breakdowns.loc[breakdowns.dimension.eq("state")].sort_values("mae_2024_usd", ascending=False).head(10))
display(breakdowns.loc[breakdowns.dimension.eq("msa_type")])
deciles = breakdowns.loc[breakdowns.dimension.eq("value_decile")].copy()
deciles["decile"] = deciles.value.astype(int) + 1
ax = deciles.plot.bar(x="decile", y="mae_2024_usd", color="#276454", legend=False)
ax.set(xlabel="Observed-value decile · 1 = lowest", ylabel="MAE · 2024 dollars", title="Error across the value distribution")
plt.tight_layout(); plt.show()'''),
        md('''## 4. Test feature groups on identical rows

        Removing one group at a time asks how much predictive information the remaining
        groups can recover under the fixed model specification. It is not a causal test.
        Affordability and climate sensitivities stay separate; climate comparisons use
        only their matched covered counties.'''),
        code('''experiments = read_csv("feature_group_experiments.csv")
primary = experiments.loc[experiments.experiment.eq("all_primary") | experiments.experiment.str.startswith("without_")].copy()
assert primary.rows.nunique() == 1
base = primary.loc[primary.experiment.eq("all_primary"), "rmsle"].iloc[0]
primary["rmsle_change_pct"] = 100 * (primary.rmsle / base - 1)
display(primary[["experiment", "rows", "rmsle", "rmsle_change_pct"]].round(3))
display(experiments.loc[experiments.experiment.str.startswith(("climate_", "affordability_"))])'''),
        md('''Removing the economic group raises RMSLE by about **24.6%**, the largest increase
        among the primary groups. Correlated social and housing features can substitute
        for some economic information, so this does not allocate unique causal contributions.'''),
        md('''## 5. Compare linear coefficients and grouped permutation importance

        Coefficients are from the standardized development-fit Elastic Net. They are
        conditional associations in log target units per transformed-feature standard
        deviation, not dollar effects. Group permutation shuffles related columns
        together and measures the increase in held-out RMSLE. The repeat spread measures
        permutation variability, not sampling uncertainty.'''),
        code('''coefficients = read_csv("standardized_coefficients.csv")
top = coefficients.loc[~coefficients.feature.str.contains("missingindicator")].nlargest(12, "absolute_coefficient")
display(top)
importance = read_csv("group_permutation.csv")
display(importance.groupby(["group", "model"]).rmsle_increase.agg(["mean", "std"]).round(4))
stability = read_csv("coefficient_stability.csv")
display(stability.loc[stability.feature.isin(top.feature.head(6))].pivot(index="feature", columns="year", values="coefficient").round(4))'''),
        md('''Income and educational attainment are positive associations in the reported
        study; vacancy is negative. Coefficient magnitudes need caution when predictors
        overlap. Period-specific fits are descriptive checks, including 2024, and never
        feed back into model selection. Inspect changes in sign and magnitude rather
        than presenting any one ranking as definitive.'''),
        md('''## 6. Restrict response curves to supported ranges

        The study's accumulated local log effects use development-derived bins within
        the 5th–95th percentiles. Each displayed bin contains at least 20 test counties.
        Differences compare bin-edge predictions only for counties inside that bin.
        Missing bins are not interpolated. These model responses are not interventions.'''),
        code('''curves = read_csv("local_response_curves.csv")
assert curves.rows.ge(20).all()
fig, axes = plt.subplots(1, 3, figsize=(12, 3.7))
labels = ["Household income · 2024 dollars", "Bachelor's attainment · %", "Vacant units · %"]
for ax, (feature, group), label in zip(axes, curves.groupby("feature", sort=False), labels):
    for row in group.itertuples():
        x = np.array([row.left, row.right])
        if feature.startswith("log1p__"):
            x = np.expm1(x)
        ax.plot(x, [row.accumulated_log_effect] * 2, color="#276454", linewidth=3)
    ax.axhline(0, color="gray", linewidth=.7)
    ax.set(xlabel=label, ylabel="Centered accumulated log effect")
    ax.tick_params(axis="x", rotation=25)
plt.tight_layout(); plt.show()
display(curves.groupby("feature").agg(bins=("rows", "size"), smallest_bin=("rows", "min")))'''),
        md('''## 7. Keep the secondary questions distinct

        County-and-period fixed effects study changes within stable counties. Their
        within R² has a different denominator than the cross-county prediction R².
        Intervals use county-clustered uncertainty. The annual forecast experiment
        predicts the next ACS estimate; overlapping survey windows and unreconstructed
        release/CPI vintages prevent an operational forecasting claim.
        These saved summaries provide context; full implementation is in
        `src/modeling/county_characteristics.py` and [COUNTY_STUDY.md](../../COUNTY_STUDY.md).'''),
        code('''display(pd.Series(read_json("fixed_effects_summary.json")))
display(read_csv("fixed_effects.csv"))
display(read_csv("forecast_metrics.csv"))'''),
        md('''## What the evidence supports

        Characteristics explain a substantial share of **between-county** variation:
        the selected model reaches 2024 R² **0.801**, MAE **$37,198**, and **63.5%** lower
        RMSLE than the training median. Economic information contributes the largest
        ablation difference. Within-county and next-release analyses answer separate
        questions and offer more modest results.

        The design does not identify causality, individual-home prices, or spatially
        independent effects. ACS smoothing and sampling uncertainty, boundary changes,
        correlated features and incomplete climate coverage constrain interpretation.

        [Return to the interactive study](https://kennethlow.com/housing-predict/)'''),
    ])


if __name__ == '__main__':
    main()
