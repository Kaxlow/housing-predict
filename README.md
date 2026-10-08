# A Dissection of Home Values

How much can county characteristics explain differences in home values across the US, and how
well do those relationships generalize across places and time?

[Explore the interactive study](https://kennethlow.com/housing-predict/) ·
[Read the portfolio case study](https://kennethlow.com/projects/housing-predict/)

This observational study uses Census ACS median owner-occupied home values in
constant 2024 dollars. It studies county aggregates, not individual-home sale
prices, and does not identify causal effects.

## Explore the analysis

Read the executed notebooks directly on GitHub:

1. [Modeling dataset](notebooks/03_modeling/01_modeling_dataset.ipynb): source
   audits, non-overlapping ACS periods, feature engineering and inclusion rules.
2. [Training and validation](notebooks/03_modeling/02_training_and_validation.ipynb):
   preprocessing, baseline comparisons, model selection and county-grouped folds.
3. [Evaluation and interpretation](notebooks/03_modeling/03_evaluation_and_interpretation.ipynb):
   final temporal evaluation, error patterns, feature groups and associations.

The notebooks include saved outputs and compact public county-level evidence.
Default execution does not require the original database. See the
[notebook setup guide](notebooks/03_modeling/README.md) for dependencies and
optional dataset rebuild/retraining. These are walkthroughs of the completed
October 6, 2026 analysis, not original exploratory logs.

Earlier notebooks cover [data quality](notebooks/01_data_quality/) and
[feature exploration](notebooks/02_feature_exploration/).

## Main results

- 9,420 county-period observations across ACS periods ending in 2014, 2019 and
  2024, with 57 primary characteristics.
- Histogram gradient boosting selected using 2014-to-2019 validation.
- Final 2024 evaluation: R² **0.801**, MAE **$37,198**, and RMSLE **0.209** across
  3,139 counties. RMSLE is 63.5% below the training-median baseline.
- Removing economic features raises RMSLE by **24.6%** on the same test rows.

Same-period predictors make this a study of transferable associations, not an
advance forecast. Within-county fixed effects and next-ACS-release prediction
are separate secondary analyses. See [methods and limitations](COUNTY_STUDY.md).

## Reproduce the complete study

With the cleaned source marts at `data/housing_predict.duckdb`:

```sh
python -m pip install -r requirements.txt
python -m src.cli.run_county_characteristics
python -m unittest discover -s tests -v
```

The source marts and full generated results are excluded from Git. A compact
snapshot under `notebooks/03_modeling/evidence/` supports the notebook walkthroughs.
The CLI does not modify the source marts. See [COUNTY_STUDY.md](COUNTY_STUDY.md)
for the analytical contract and [MODELING.md](MODELING.md) for modeling commands.

## Repository layout

- `src/cli/`: data ingestion, mart building and study entry points.
- `src/modeling/`: shared feature construction, models and evaluation functions.
- `notebooks/`: data quality, exploration and modeling walkthroughs.
- `tests/`: alignment, selection, folds and statistical calculation checks.

The website is deployed separately through the
[portfolio repository](https://github.com/Kaxlow/kaxlow.github.io/tree/main/housing-predict).
Model training stays offline.
