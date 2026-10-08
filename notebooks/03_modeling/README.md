# Explore the analysis

The `03_modeling` notebooks show the modeling pipeline used for the county housing
study. Outputs are retained for reading on GitHub.

1. [Modeling dataset](01_modeling_dataset.ipynb): observations, source audits,
   feature definitions, exclusions, five-year transformations and missingness.
2. [Training and validation](02_training_and_validation.ipynb): the median
   baseline, fold-fitted preprocessing, six candidates, temporal selection and
   nested county-grouped validation.
3. [Evaluation and interpretation](03_evaluation_and_interpretation.ipynb):
   recomputed held-out metrics, geographic errors, feature-group comparisons,
   coefficients, stability and supported response curves.

## Run locally

From the repository or standalone analysis bundle root:

```sh
python -m pip install -r requirements.txt
python -m jupyterlab notebooks/03_modeling
```

Run each notebook from top to bottom in its own kernel. Default execution uses
the bundled `evidence/` snapshot and requires no database or pickle. It performs
real transformations on a small annual example, validates fold separation,
recomputes metrics from predictions, and builds plots. Optional flags:

- Notebook 1: `REBUILD_FROM_MARTS = True` rebuilds from
  `data/housing_predict.duckdb`, which is not distributed here.
- Notebook 2: `RETRAIN = True` reruns all candidates and nested geographic folds
  from the included panel. Allow several minutes.
- Notebook 3: `REFIT_SELECTED = True` checks a fresh development fit against the
  published predictions. It does not reselect a model.

The full original study, including secondary analyses, can be run with:

```sh
python -m src.cli.run_county_characteristics
```

That command requires the original marts. See [COUNTY_STUDY.md](../../COUNTY_STUDY.md)
for the data contract, limitations and complete workflow.

## Evidence and provenance

`evidence/` contains public county aggregates: the analytical panel (about 2.3 MB),
a three-county annual transformation example, audit tables, predictions and
model summaries. It does not contain the original database, fitted pickle,
credentials, or local source paths. No synthetic observations are used.

`snapshot.json` records SHA-256 hashes for artifacts and the two shared modeling
modules. Every notebook verifies evidence hashes before proceeding.
`run_summary.json` records the original run's software versions, seed, sample,
panel fingerprint and limitations. The root [requirements.txt](../../requirements.txt) pins the notebook execution
environment. Numerical reproduction may vary on other software/platform versions.

The final holdout is 2024. Historical saved outcomes are exposed here for
transparency; they are not a new unseen test set for subsequent experimentation.
If new modeling decisions are made after inspecting them, that would constitute a
new exploratory analysis rather than an untouched final evaluation.
