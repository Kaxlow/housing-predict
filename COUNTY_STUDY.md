# County characteristics and US home values

The study asks how county characteristics relate to median owner-occupied home
values and whether those relationships generalize across periods and counties.
It estimates observational associations, not individual-home prices or causal
effects.

## Run and reproduce

For an executed, step-by-step walkthrough, start with
[Explore the analysis](notebooks/03_modeling/README.md). The three notebooks cover
the modeling dataset, training and validation, and evaluation and interpretation.
They include a compact evidence snapshot for running without the original marts,
plus optional full rebuild and retraining paths.

```powershell
python -m src.cli.run_county_characteristics
python -m unittest discover -s tests -v
```

Optional arguments are `--database PATH`, `--output-dir PATH`, and `--quick`.
Quick mode uses fewer candidates and one annual forecast validation year; its
outputs go into a `quick/` subdirectory. Full mode uses six characteristics
candidates and all five annual forecast validation years. Neither mode changes
the source DuckDB marts. Model artifacts are trusted local pickle files; load
only files you generated yourself.

Results are in `data/modeling/county_characteristics/`. Start with `REPORT.md`
and its figures. This directory is not part of the Git repository because it is expected to be created by running the workflow locally. The results can be viewed on the [project website](https://kennethlow.com/housing-predict/).

## Dataset and feature contract

- `study.duckdb`, table `county_period_panel`, and `county_period_panel.parquet`
  contain the same analytical panel with unique `(county_fips, year)` keys.
  FIPS is a five-character string. `year` is the ACS period end, `period_start`
  is four years earlier. The periods end in 2014, 2019 and 2024.
- The outcome is `target_median_owner_occupied_home_value_2024_usd` and its natural
  log `log_value`. Data preparation excludes missing/nonpositive targets,
  territories, negative Census sentinel estimates and impossible percentages.
  Negative climate temperatures remain valid. Models give equal weight to each
  county-period row; complete longitudinal counties contribute three rows each.
- Primary predictors are economic, social, housing structure and hazard
  characteristics. Geography identifiers, price history, geographic price
  summaries and value-derived ratios never enter the primary design. Percentages
  remain in percentage points; dollar estimates and counts use `log1p`.
- The registry must define a feature in all three periods. A feature with no
  observed values in 2014 is excluded using training data alone. In these marts,
  computer/broadband measures fail this observed-availability check despite
  registry labels existing. Other missing predictors are median-imputed inside
  each training fit, with missingness indicators; Elastic Net also standardizes
  its predictors inside each fit.
- Affordability and mortgage measures are a separate sensitivity analysis.
  Complementary renter/owner and mortgage/no-mortgage percentages are not both
  included. Percentage groups may still correlate; coefficients and importance
  are interpreted jointly rather than as independent causal effects.
- Hazard event totals cover the same five-year ACS window. A complete five-year
  history is required for each sum. The existing event marts assume absent
  events are zero; reporting/mapping gaps may remain. Climate is a five-year
  mean of annual measures, requiring five observed annual values, and is tested
  only on covered counties. Base and augmented climate models use identical
  training and test rows.

`feature_manifest.csv` gives each source, transformed predictor, group, role,
unit and inclusion/exclusion reason. `feature_definitions_by_year.csv` preserves
the actual ACS variable mappings. `source_key_audit.csv`, `coverage.csv`,
`missingness.csv` and `exclusions.csv` document the sample and joins. Duplicate
source or output keys cause the run to fail.

## Geographic comparability

State labels are derived from official FIPS prefixes, including newer county
equivalents missing from the static county reference. Missing metro labels are
`Unknown`, not assumed nonmetro. Metro status is a static descriptive label used
only for error breakdowns.

Cross sections retain all eligible counties in each period. The longitudinal
analysis requires all three periods, complete values for its six specified
predictors, and excludes Connecticut and known affected Alaska, South Dakota
and Virginia codes. This conservatively excludes name/code-only changes too;
no approximate crosswalk is invented. `geography_audit.csv` reports eligibility.
This is a substantial-change screen, not proof that all retained boundaries are
identical. Minor boundary changes and ACS sampling margins of error are not
modeled. The forecast sensitivity also requires continuous 2010–2024 histories.

## Experiments and selection

1. **Characteristics-only prediction.** Fit on 2014, select parameters within
   each family on 2019 RMSLE, and select the overall model against the 2014
   training-median baseline using the same validation criterion. Freeze that
   decision in `selection.json`. Refit family winners on 2014 + 2019 and evaluate
   2024 once per specified experiment. Predictors and targets are contemporaneous:
   this measures transferable associations, not an advance forecast.
2. **Geographic generalization.** Five county-disjoint folds use only 2014 and
   2019. Each fold repeats parameter selection on its training counties (2014
   to 2019), refits on both periods of those counties, and scores the excluded
   counties. Thus a county cannot influence preprocessing or tuning for its own
   geographic validation predictions. Adjacent counties may still be correlated;
   this is not a spatially buffered evaluation.
3. **Within-county associations.** A balanced three-period regression removes
   county and period means before fitting log value on log1p household income,
   bachelor's attainment, vacancy, detached-unit share, commute time and household
   size. County-clustered covariance accounts for absorbed effects in its degrees
   of freedom; confidence intervals use a t distribution with counties minus one
   degrees of freedom. No time-invariant geography coefficients are estimated.
4. **Retrospective annual forecasting.** Compare previous value with a learned
   price-history-only model and history plus characteristics. Independently tune
   each learned design on expanding 2018–2022 validation folds, fit through 2022,
   and evaluate 2023–2024 using each row's observed prior-year features. This is
   a sequence of one-release-ahead predictions with fixed fitted parameters,
   not a two-step recursive forecast. It uses equal comparison rows. Adjacent
   ACS windows overlap and release/CPI vintages are not reconstructed.

The candidate grid reuses the existing Elastic Net, Extra Trees and histogram
gradient boosting implementations. `selection.json` records parameters;
`run_summary.json` records software versions, seed, sample, limitations and a
panel fingerprint. The selected ML artifact retains its development fit. If the
median baseline wins, the manifest says so and the ML artifact is explicitly
labeled `best_ml`, not claimed to be the selected predictor.

## Interpretation and website exports

MAE is in 2024 dollars; RMSLE measures proportional-scale error; R² uses dollar
values. There is no R²-based model selection. `error_breakdowns.csv` reports the
selected model's performance by state, metro status and observed value decile.

`feature_group_experiments.csv` removes each primary group using the fixed
validation-selected ML specification on the same rows, plus affordability and
matched climate sensitivities. These are diagnostics, not another selection
round. `group_permutation.csv` jointly permutes all columns in a group (five
repetitions) for each model family; its spread is permutation variability, not
a confidence interval. Correlated groups can substitute for each other.

`standardized_coefficients.csv` contains development-fit Elastic Net coefficients;
`coefficient_stability.csv` contains descriptive fits for each period using that
fixed specification. `associations_by_period.csv` contains pairwise Spearman
correlations. These descriptive 2024 fits never change the selected predictor.

`local_response_curves.csv` supplies accumulated local log effects for income,
education and vacancy: training-derived bins lie within the 5th–95th percentiles,
and test bins require at least 20 rows. Empty bins are omitted, not interpolated.
Binwise effects compare predictions at each bin's edges only for counties inside
that bin; these remain model associations, not interventions.

`website_data/` contains JSON for the comparison tables and charts.
`counties.json` has FIPS, labels, periods, observed values, six interpretable
characteristics, 2024 predictions and residuals; missing values are JSON null.
Join to matching-period geometry for a county map. `summary.json` includes
provenance and limitations. The report follows the presentation's question,
data journey, evidence, interpretation, forecasting and limitations sequence.

## Source guidance

- [Comparing ACS periods](https://www.census.gov/programs-surveys/acs/guidance/comparing-acs-data.html)
- [Substantial county changes](https://www.census.gov/programs-surveys/geography/technical-documentation/county-changes.2010.html)
- [ACS table and geography changes](https://www.census.gov/programs-surveys/acs/technical-documentation/table-and-geography-changes.html)
- [2024 ACS release schedule](https://www.census.gov/programs-surveys/acs/news/data-releases/2024/release-schedule.html):
  the five-year release was January 29, 2026, not the end of 2024.
