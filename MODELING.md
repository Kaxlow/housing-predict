# County home-value modeling

## Associations-first portfolio study

Run the complete study against the existing marts (no rebuild required):

```powershell
python -m src.cli.run_county_characteristics
python -m unittest discover -s tests -v
```

Read `data/modeling/county_characteristics/REPORT.md` for the results and
`COUNTY_STUDY.md` for the experiment design and output contracts. The study uses
non-overlapping ACS periods ending in 2014, 2019 and 2024, excludes price history
from characteristics models, and treats findings as observational associations.
It reads the source database without modifying it and creates a separate
`study.duckdb` beside its exports. `--quick` writes into a separate `quick/`
subdirectory and is a smoke run, not portfolio evidence.

## Existing annual forecast benchmark

The modeling pipeline predicts the next ACS release's county median value of
owner-occupied housing in constant 2024 dollars. Adjacent ACS five-year estimates share survey years, so persistence is a
strong baseline and all reported model results are compared with it.

Run the full benchmark after building the database:

```powershell
python -m src.cli.build_database
python -m src.cli.train_county_price_models
```

For a faster pipeline check, add `--quick`; it uses one temporal validation year,
one parameter set per estimator, fewer trees, and three grouped folds. The full
run uses every 2018–2022 validation fold. The trainer materializes
`feature.county_price_forecast_panel` and writes the selected model, validation
metrics, held-out predictions, grouped-county results, challenger metrics,
coefficients, permutation importance, and partial-dependence plots under
`data/modeling/county_price/`. After evaluation, the selected estimator is
refitted through 2024 and `forecast_2025.csv` is generated from 2024 features.
Rebuilding the source database invalidates and removes the materialized forecast
panel, so rerun the trainer after each database rebuild.

The feature row for `forecast_year` contains only values observed in
`forecast_year - 1`. Model selection uses expanding validation for 2018–2022;
2023–2024 remain held out. FEMA and NOAA absences are zero-filled because their
tables record observed events, while climate is evaluated only in a separate
covered-county challenger.

Final model selection and the persistence improvement gate now use temporal
validation only, before held-out scores are computed. The gate requires a
0.5% relative improvement in mean validation RMSLE and improvement in every
validation year. `selection_validation_metrics.csv` records that decision.
Previously generated artifacts are not automatically updated by changing code;
rerun this legacy trainer to refresh its output directory.

These annual experiments predict the next ACS release retrospectively. End-year
labels are not publication dates: the 2024 five-year release was published on
January 29, 2026. Available-at-time release vintages and contemporaneous CPI
vintages are not reconstructed, so these are not operational end-year forecasts.
