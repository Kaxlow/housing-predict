"""Leakage-safe county house-price panel construction and model benchmarking."""
from __future__ import annotations

import json
import math
import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import PartialDependenceDisplay, permutation_importance
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_log_error, median_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET = "target_median_owner_occupied_home_value_2024_usd"
PRIMARY_CATEGORICAL = ["state", "msa_code", "msa_type"]
HIST_CATEGORICAL = ["state", "msa_type"]
HAZARD_COLUMNS = [
    "disaster_count",
    "storm_event_count",
    "storm_damage_2024_usd",
    "storm_injuries",
    "storm_deaths",
]
CLIMATE_COLUMNS = [
    "avg_temperature_f",
    "min_temperature_f",
    "max_temperature_f",
    "precipitation_inches",
]


@dataclass(frozen=True)
class Candidate:
    name: str
    estimator: str
    target_kind: str
    feature_set: str
    params: dict[str, Any]


class LagPriceBaseline(BaseEstimator, RegressorMixin):
    """Serializable last-observation forecast used when ML cannot beat persistence."""

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "LagPriceBaseline":
        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return X["price_lag1_2024_usd"].to_numpy(dtype=float)


def load_county_year_data(con: duckdb.DuckDBPyConnection) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Load aligned county-year features and identify stable/all ACS predictors."""
    frame = con.execute(
        """
        SELECT h.*,
          e.* EXCLUDE (year, county_fips, county_name),
          s.* EXCLUDE (year, county_fips, county_name),
          coalesce(f.disaster_count, 0) AS disaster_count,
          coalesce(n.storm_event_count, 0) AS storm_event_count,
          coalesce(n.storm_damage_usd, 0) * c.inflation_factor_to_2024 AS storm_damage_2024_usd,
          coalesce(n.storm_injuries, 0) AS storm_injuries,
          coalesce(n.storm_deaths, 0) AS storm_deaths,
          cl.avg_temperature_f, cl.min_temperature_f, cl.max_temperature_f,
          cl.precipitation_inches,
          coalesce(r.state, CASE WHEN substr(h.county_fips, 1, 2) = '72' THEN 'PR' ELSE 'UNKNOWN' END) AS state,
          coalesce(nullif(r.msa_code, ''), 'NO_MSA') AS msa_code,
          coalesce(nullif(r.msa_type, ''), 'Nonmetro') AS msa_type
        FROM feature.county_housing h
        JOIN feature.county_economic_annual e USING (county_fips, year)
        JOIN feature.county_social_annual s USING (county_fips, year)
        LEFT JOIN feature.county_fema_annual f USING (county_fips, year)
        LEFT JOIN feature.county_noaa_storm_annual n USING (county_fips, year)
        LEFT JOIN feature.county_climate_annual cl USING (county_fips, year)
        JOIN ref.cpi_u_annual c USING (year)
        LEFT JOIN raw.counties r ON lpad(r.fips, 5, '0') = h.county_fips
        ORDER BY h.county_fips, h.year
        """
    ).df()
    registry = con.execute(
        """SELECT feature_name, min(year) AS first_year
           FROM feature.canonical_feature_registry
           GROUP BY feature_name"""
    ).df()
    all_acs = registry.loc[registry["feature_name"].ne(TARGET), "feature_name"].tolist()
    stable_acs = registry.loc[
        registry["feature_name"].ne(TARGET) & registry["first_year"].le(2010), "feature_name"
    ].tolist()
    missing = sorted(set(all_acs + [TARGET]) - set(frame.columns))
    if missing:
        raise RuntimeError(f"Feature registry columns missing from joined county data: {missing}")
    return frame, stable_acs, all_acs


def _rolling_sum(frame: pd.DataFrame, column: str, window: int = 3) -> pd.Series:
    return frame.groupby("county_fips", sort=False)[column].transform(
        lambda values: values.rolling(window, min_periods=1).sum()
    )


def _trailing_climatology(frame: pd.DataFrame, column: str) -> pd.Series:
    return frame.groupby("county_fips", sort=False)[column].transform(
        lambda values: values.rolling(3, min_periods=2).mean().shift(1)
    )


def build_forecast_panel(
    county_year: pd.DataFrame,
    stable_acs: list[str],
    all_acs: list[str],
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """Engineer predictors available at t for the ACS target at t+1."""
    frame = county_year.sort_values(["county_fips", "year"]).copy()
    groups = frame.groupby("county_fips", sort=False)
    next_year = groups["year"].shift(-1)
    frame["forecast_year"] = frame["year"] + 1
    frame["source_year"] = frame["year"]
    frame["target_price_2024_usd"] = groups[TARGET].shift(-1)
    frame["target_log_price"] = np.log(frame["target_price_2024_usd"])
    frame["price_lag1_2024_usd"] = frame[TARGET]
    frame["price_lag2_2024_usd"] = groups[TARGET].shift(1)
    frame["log_price_lag1"] = np.log(frame["price_lag1_2024_usd"])
    frame["log_price_lag2"] = np.log(frame["price_lag2_2024_usd"])
    frame["price_log_growth_1y"] = frame["log_price_lag1"] - frame["log_price_lag2"]
    price_lag4 = groups[TARGET].shift(3)
    frame["price_cagr_3y"] = np.exp((frame["log_price_lag1"] - np.log(price_lag4)) / 3.0) - 1.0
    frame["price_growth_mean_3y"] = frame.groupby("county_fips", sort=False)["price_log_growth_1y"].transform(
        lambda values: values.rolling(3, min_periods=2).mean()
    )
    frame["price_growth_std_3y"] = frame.groupby("county_fips", sort=False)["price_log_growth_1y"].transform(
        lambda values: values.rolling(3, min_periods=2).std()
    )
    frame["state_median_price_lag1"] = frame.groupby(["state", "year"])[TARGET].transform("median")

    engineered_by_source: dict[str, list[str]] = {}
    acs_engineered: dict[str, pd.Series] = {}
    for column in all_acs:
        lag_name = f"lag1__{column}"
        acs_engineered[lag_name] = frame[column]
        engineered_by_source.setdefault(column, []).append(lag_name)
        availability = f"available__{column}"
        acs_engineered[availability] = frame[column].notna().astype("int8")
        engineered_by_source[column].append(availability)
        if column.endswith("_pct"):
            delta = f"delta1__{column}"
            acs_engineered[delta] = groups[column].diff()
            engineered_by_source[column].append(delta)
        if column.endswith("_usd") or column.endswith("_count") or column == "housing_units":
            logged = f"log1p__{column}"
            acs_engineered[logged] = np.log1p(frame[column].clip(lower=0))
            engineered_by_source[column].append(logged)
    frame = pd.concat([frame, pd.DataFrame(acs_engineered, index=frame.index)], axis=1)

    frame["median_income_to_price"] = (
        frame["median_household_income_2024_usd"] / frame["price_lag1_2024_usd"]
    )
    frame["mean_earnings_to_price"] = (
        frame["mean_household_earnings_2024_usd"] / frame["price_lag1_2024_usd"]
    )

    hazard_features: list[str] = []
    hazard_engineered: dict[str, pd.Series] = {}
    for column in HAZARD_COLUMNS:
        current = f"lag1__{column}"
        trailing = f"sum3__{column}"
        hazard_engineered[current] = frame[column].fillna(0)
        hazard_engineered[trailing] = _rolling_sum(frame, column)
        hazard_features.extend([current, trailing])
    frame = pd.concat([frame, pd.DataFrame(hazard_engineered, index=frame.index)], axis=1)
    frame["log1p__storm_damage_2024_usd"] = np.log1p(frame["storm_damage_2024_usd"].clip(lower=0))
    frame["log1p__sum3_storm_damage_2024_usd"] = np.log1p(
        frame["sum3__storm_damage_2024_usd"].clip(lower=0)
    )
    hazard_features.extend(["log1p__storm_damage_2024_usd", "log1p__sum3_storm_damage_2024_usd"])

    frame["climate_available"] = frame[CLIMATE_COLUMNS].notna().all(axis=1).astype("int8")
    climate_features = ["climate_available"]
    climate_engineered: dict[str, pd.Series] = {}
    for column in CLIMATE_COLUMNS:
        current = f"lag1__{column}"
        anomaly = f"anomaly__{column}"
        climate_engineered[current] = frame[column]
        climate_engineered[anomaly] = frame[column] - _trailing_climatology(frame, column)
        climate_features.extend([current, anomaly])
    frame = pd.concat([frame, pd.DataFrame(climate_engineered, index=frame.index)], axis=1)

    price_features = [
        "price_lag1_2024_usd", "price_lag2_2024_usd", "log_price_lag1", "log_price_lag2",
        "price_log_growth_1y", "price_cagr_3y", "price_growth_mean_3y", "price_growth_std_3y",
        "state_median_price_lag1", "median_income_to_price", "mean_earnings_to_price",
    ]
    common = ["forecast_year", *price_features, *hazard_features]
    stable_engineered = [name for source in stable_acs for name in engineered_by_source[source]]
    full_engineered = [name for source in all_acs for name in engineered_by_source[source]]
    feature_sets = {
        "primary": list(dict.fromkeys([*common, *stable_engineered, *PRIMARY_CATEGORICAL])),
        "climate": list(dict.fromkeys([*common, *stable_engineered, *climate_features, *PRIMARY_CATEGORICAL])),
        "post2020": list(dict.fromkeys([*common, *full_engineered, *PRIMARY_CATEGORICAL])),
    }

    last_source_year = int(frame["year"].max())
    panel = frame.loc[
        frame["price_lag1_2024_usd"].gt(0)
        & (
            (next_year.eq(frame["forecast_year"]) & frame["target_price_2024_usd"].gt(0))
            | frame["year"].eq(last_source_year)
        )
    ].copy()
    panel["target_log_growth"] = panel["target_log_price"] - panel["log_price_lag1"]
    panel = panel.drop(columns=[*all_acs, TARGET, *HAZARD_COLUMNS, *CLIMATE_COLUMNS], errors="ignore")
    if panel.duplicated(["county_fips", "forecast_year"]).any():
        raise RuntimeError("Forecast panel is not unique by county_fips and forecast_year")
    if not panel["source_year"].eq(panel["forecast_year"] - 1).all():
        raise RuntimeError("Forecast panel contains a feature timestamp that is not exactly one year before its target")
    return panel.reset_index(drop=True), feature_sets


def materialize_panel(
    con: duckdb.DuckDBPyConnection,
    panel: pd.DataFrame,
    table: str = "feature.county_price_forecast_panel",
) -> None:
    con.register("county_price_forecast_panel_frame", panel)
    try:
        con.execute(f"CREATE OR REPLACE TABLE {table} AS SELECT * FROM county_price_forecast_panel_frame")
        con.execute(
            "CREATE INDEX IF NOT EXISTS idx_county_price_forecast_panel "
            f"ON {table} (county_fips, forecast_year)"
        )
    finally:
        con.unregister("county_price_forecast_panel_frame")


def regression_metrics(actual: Iterable[float], predicted: Iterable[float]) -> dict[str, float]:
    actual_array = np.asarray(actual, dtype=float)
    predicted_array = np.clip(np.asarray(predicted, dtype=float), 1.0, None)
    return {
        "rmsle": float(math.sqrt(mean_squared_log_error(actual_array, predicted_array))),
        "mae_2024_usd": float(mean_absolute_error(actual_array, predicted_array)),
        "median_ape": float(np.median(np.abs(predicted_array - actual_array) / actual_array)),
        "r2": float(r2_score(actual_array, predicted_array)) if len(actual_array) > 1 else float("nan"),
    }


def make_estimator(candidate: Candidate, features: list[str], random_state: int = 42) -> Pipeline:
    categorical = [column for column in PRIMARY_CATEGORICAL if column in features]
    numeric = [column for column in features if column not in categorical]
    if candidate.estimator == "elastic_net":
        preprocessor = ColumnTransformer([
            ("numeric", Pipeline([
                ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                ("scaler", StandardScaler()),
            ]), numeric),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical),
        ])
        model = ElasticNet(max_iter=5_000, tol=1e-3, selection="cyclic", **candidate.params)
    elif candidate.estimator == "hist_gradient_boosting":
        hist_features = [column for column in features if column != "msa_code"]
        numeric = [column for column in hist_features if column not in HIST_CATEGORICAL]
        preprocessor = ColumnTransformer([
            ("numeric", SimpleImputer(strategy="median", add_indicator=True), numeric),
            ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
             [column for column in HIST_CATEGORICAL if column in hist_features]),
        ], sparse_threshold=0.0)
        model = HistGradientBoostingRegressor(
            random_state=random_state, early_stopping=True, **candidate.params
        )
    elif candidate.estimator == "extra_trees":
        preprocessor = ColumnTransformer([
            ("numeric", SimpleImputer(strategy="median", add_indicator=True), numeric),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical),
        ])
        model = ExtraTreesRegressor(n_jobs=-1, random_state=random_state, **candidate.params)
    else:
        raise ValueError(f"Unknown estimator: {candidate.estimator}")
    return Pipeline([("preprocessor", preprocessor), ("model", model)])


def candidate_features(candidate: Candidate, feature_sets: dict[str, list[str]]) -> list[str]:
    features = feature_sets[candidate.feature_set]
    if candidate.estimator in {"hist_gradient_boosting", "extra_trees"}:
        return [column for column in features if column != "msa_code"]
    return features


def model_target(frame: pd.DataFrame, target_kind: str) -> pd.Series:
    if target_kind == "direct":
        return frame["target_log_price"]
    if target_kind == "growth":
        return frame["target_log_growth"]
    raise ValueError(f"Unknown target kind: {target_kind}")


def predict_prices(model: Pipeline, frame: pd.DataFrame, features: list[str], target_kind: str) -> np.ndarray:
    prediction = model.predict(frame[features])
    if target_kind == "growth":
        prediction = prediction + frame["log_price_lag1"].to_numpy()
    return np.exp(np.clip(prediction, math.log(1_000), math.log(10_000_000)))


def default_candidates(quick: bool = False) -> list[Candidate]:
    elastic = [
        {"alpha": 0.0003, "l1_ratio": 0.1},
        {"alpha": 0.003, "l1_ratio": 0.5},
    ]
    hist = [
        {"max_iter": 80, "learning_rate": 0.05, "max_leaf_nodes": 15, "l2_regularization": 0.1},
        {"max_iter": 80, "learning_rate": 0.05, "max_leaf_nodes": 31, "l2_regularization": 1.0},
    ]
    extra = [
        {"n_estimators": 40, "max_features": 0.5, "min_samples_leaf": 2},
        {"n_estimators": 40, "max_features": 0.5, "min_samples_leaf": 5},
    ]
    if quick:
        elastic, hist, extra = elastic[:1], hist[:1], extra[:1]
        hist[0] = {**hist[0], "max_iter": 60}
        extra[0] = {**extra[0], "n_estimators": 15}
    candidates: list[Candidate] = []
    for index, params in enumerate(elastic):
        candidates.append(Candidate(f"elastic_net_{index + 1}", "elastic_net", "direct", "primary", params))
    for index, params in enumerate(hist):
        candidates.append(Candidate(f"hist_direct_{index + 1}", "hist_gradient_boosting", "direct", "primary", params))
        candidates.append(Candidate(f"hist_growth_{index + 1}", "hist_gradient_boosting", "growth", "primary", params))
    for index, params in enumerate(extra):
        candidates.append(Candidate(f"extra_trees_{index + 1}", "extra_trees", "direct", "primary", params))
    return candidates


def cross_validate_candidates(
    panel: pd.DataFrame,
    feature_sets: dict[str, list[str]],
    candidates: list[Candidate],
    validation_years: Iterable[int] = range(2018, 2023),
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        features = candidate_features(candidate, feature_sets)
        for validation_year in validation_years:
            print(f"CV {candidate.name}: validation year {validation_year}", flush=True)
            train = panel.loc[panel["forecast_year"].lt(validation_year)]
            validation = panel.loc[panel["forecast_year"].eq(validation_year)]
            model = make_estimator(candidate, features)
            model.fit(train[features], model_target(train, candidate.target_kind))
            predicted = predict_prices(model, validation, features, candidate.target_kind)
            rows.append({
                "candidate": candidate.name,
                "validation_year": validation_year,
                "train_rows": len(train),
                "validation_rows": len(validation),
                **regression_metrics(validation["target_price_2024_usd"], predicted),
            })
    return pd.DataFrame(rows)


def select_cv_winners(cv_results: pd.DataFrame, candidates: list[Candidate]) -> list[Candidate]:
    mean_scores = cv_results.groupby("candidate", as_index=False)["rmsle"].mean().sort_values("rmsle")
    by_name = {candidate.name: candidate for candidate in candidates}
    winners: list[Candidate] = []
    for estimator in ("elastic_net", "hist_gradient_boosting", "extra_trees"):
        eligible = [name for name, candidate in by_name.items() if candidate.estimator == estimator]
        best_name = mean_scores.loc[mean_scores["candidate"].isin(eligible), "candidate"].iloc[0]
        winners.append(by_name[best_name])
    best_growth = mean_scores.loc[
        mean_scores["candidate"].map(lambda name: by_name[name].target_kind == "growth"), "candidate"
    ].iloc[0]
    if best_growth not in {candidate.name for candidate in winners}:
        winners.append(by_name[best_growth])
    return winners


def select_forecast_from_validation(
    cv_results: pd.DataFrame, panel: pd.DataFrame, candidate_names: list[str],
    minimum_improvement: float = 0.005,
) -> tuple[str, pd.DataFrame]:
    """Apply the persistence gate on validation folds only, before test evaluation."""
    rows = cv_results.loc[cv_results.candidate.isin(candidate_names)].rename(
        columns={'candidate': 'model', 'validation_year': 'test_year'}
    ).to_dict('records')
    for year in sorted(cv_results.validation_year.unique()):
        validation = panel.loc[panel.forecast_year.eq(year)]
        rows.append({'model': 'county_lag_baseline', 'test_year': int(year),
                     **regression_metrics(validation.target_price_2024_usd, validation.price_lag1_2024_usd)})
    metrics = pd.DataFrame(rows)
    aggregate = metrics.groupby('model', as_index=False)[['rmsle', 'mae_2024_usd', 'median_ape', 'r2']].mean()
    aggregate['test_year'] = 'all'
    metrics = pd.concat([metrics, aggregate], ignore_index=True)
    return choose_final_model(metrics, candidate_names, minimum_improvement), metrics.rename(columns={'test_year': 'validation_year'})


def evaluate_test_models(
    panel: pd.DataFrame,
    feature_sets: dict[str, list[str]],
    candidates: list[Candidate],
    test_years: Iterable[int] = (2023, 2024),
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Pipeline]]:
    test_years = tuple(test_years)
    train = panel.loc[panel["forecast_year"].lt(min(test_years))]
    test = panel.loc[panel["forecast_year"].isin(test_years)].copy()
    metric_rows: list[dict[str, Any]] = []
    predictions = test[["county_fips", "county_name", "state", "msa_type", "forecast_year", "climate_available", "target_price_2024_usd"]].copy()
    models: dict[str, Pipeline] = {}

    baseline_predictions = {
        "county_lag_baseline": test["price_lag1_2024_usd"].to_numpy(),
        "state_lag_baseline": test["state_median_price_lag1"].to_numpy(),
    }
    for name, predicted in baseline_predictions.items():
        predictions[name] = predicted
        metric_rows.append({"model": name, "test_year": "all", **regression_metrics(test["target_price_2024_usd"], predicted)})
        for year in test_years:
            mask = test["forecast_year"].eq(year)
            metric_rows.append({"model": name, "test_year": year, **regression_metrics(test.loc[mask, "target_price_2024_usd"], predicted[mask])})

    for candidate in candidates:
        print(f"Held-out fit: {candidate.name}", flush=True)
        features = candidate_features(candidate, feature_sets)
        model = make_estimator(candidate, features)
        model.fit(train[features], model_target(train, candidate.target_kind))
        predicted = predict_prices(model, test, features, candidate.target_kind)
        predictions[candidate.name] = predicted
        models[candidate.name] = model
        metric_rows.append({"model": candidate.name, "test_year": "all", **regression_metrics(test["target_price_2024_usd"], predicted)})
        for year in test_years:
            mask = test["forecast_year"].eq(year)
            metric_rows.append({"model": candidate.name, "test_year": year, **regression_metrics(test.loc[mask, "target_price_2024_usd"], predicted[mask])})
    return pd.DataFrame(metric_rows), predictions, models


def choose_final_model(test_metrics: pd.DataFrame, candidate_names: list[str], minimum_improvement: float = 0.005) -> str:
    overall = test_metrics.loc[test_metrics["test_year"].eq("all")].set_index("model")
    baseline = float(overall.loc["county_lag_baseline", "rmsle"])
    eligible: list[str] = []
    for name in candidate_names:
        aggregate_ok = float(overall.loc[name, "rmsle"]) <= baseline * (1.0 - minimum_improvement)
        yearly = test_metrics.loc[test_metrics["model"].isin([name, "county_lag_baseline"]) & ~test_metrics["test_year"].eq("all")]
        pivot = yearly.pivot(index="test_year", columns="model", values="rmsle")
        yearly_ok = (pivot[name] < pivot["county_lag_baseline"]).all()
        if aggregate_ok and yearly_ok:
            eligible.append(name)
    if not eligible:
        return "county_lag_baseline"
    return min(eligible, key=lambda name: float(overall.loc[name, "rmsle"]))


def breakdown_metrics(predictions: pd.DataFrame, prediction_column: str) -> pd.DataFrame:
    frame = predictions.copy()
    frame["price_decile"] = pd.qcut(frame["target_price_2024_usd"], 10, labels=False, duplicates="drop")
    rows: list[dict[str, Any]] = []
    for dimension in ("forecast_year", "state", "msa_type", "climate_available", "price_decile"):
        for value, group in frame.groupby(dimension, dropna=False):
            rows.append({
                "dimension": dimension,
                "value": value,
                "rows": len(group),
                **regression_metrics(group["target_price_2024_usd"], group[prediction_column]),
            })
    return pd.DataFrame(rows)


def county_grouped_validation(
    panel: pd.DataFrame,
    feature_sets: dict[str, list[str]],
    candidate: Candidate,
    folds: int = 5,
) -> pd.DataFrame:
    from sklearn.model_selection import GroupKFold

    development = panel.loc[panel["forecast_year"].le(2022)].copy()
    features = candidate_features(candidate, feature_sets)
    rows: list[dict[str, Any]] = []
    splitter = GroupKFold(n_splits=folds)
    for fold, (train_index, validation_index) in enumerate(
        splitter.split(development, groups=development["county_fips"]), start=1
    ):
        train = development.iloc[train_index]
        validation = development.iloc[validation_index]
        model = make_estimator(candidate, features, random_state=42 + fold)
        model.fit(train[features], model_target(train, candidate.target_kind))
        predicted = predict_prices(model, validation, features, candidate.target_kind)
        rows.append({"fold": fold, "train_rows": len(train), "validation_rows": len(validation), **regression_metrics(validation["target_price_2024_usd"], predicted)})
    return pd.DataFrame(rows)


def diagnostic_challenger(
    panel: pd.DataFrame,
    feature_sets: dict[str, list[str]],
    base_candidate: Candidate,
    feature_set: str,
) -> tuple[dict[str, float], Pipeline]:
    candidate = Candidate(
        f"{base_candidate.name}_{feature_set}", base_candidate.estimator,
        base_candidate.target_kind, feature_set, base_candidate.params,
    )
    subset = panel.loc[panel["forecast_year"].ge(2021)].copy()
    if feature_set == "climate":
        subset = panel.loc[panel["climate_available"].eq(1)].copy()
    train = subset.loc[subset["forecast_year"].le(2022)]
    test = subset.loc[subset["forecast_year"].isin([2023, 2024])]
    features = candidate_features(candidate, feature_sets)
    model = make_estimator(candidate, features)
    model.fit(train[features], model_target(train, candidate.target_kind))
    predicted = predict_prices(model, test, features, candidate.target_kind)
    return {"model": candidate.name, "rows": len(test), **regression_metrics(test["target_price_2024_usd"], predicted)}, model


def export_explanations(
    model: Pipeline,
    candidate: Candidate,
    panel: pd.DataFrame,
    feature_sets: dict[str, list[str]],
    output_dir: Path,
) -> None:
    test = panel.loc[panel["forecast_year"].isin([2023, 2024])].copy()
    features = candidate_features(candidate, feature_sets)
    if len(test) > 4_000:
        test = test.sample(4_000, random_state=42)

    def scorer(estimator: Pipeline, X: pd.DataFrame, y: np.ndarray) -> float:
        predicted_component = estimator.predict(X)
        if candidate.target_kind == "growth":
            actual_log = y + X["log_price_lag1"].to_numpy()
            predicted_log = predicted_component + X["log_price_lag1"].to_numpy()
        else:
            actual_log = y
            predicted_log = predicted_component
        return -float(np.sqrt(np.mean((actual_log - predicted_log) ** 2)))

    importance = permutation_importance(
        model, test[features], model_target(test, candidate.target_kind),
        scoring=scorer, n_repeats=3, random_state=42, n_jobs=1,
    )
    pd.DataFrame({
        "feature": features,
        "importance_mean": importance.importances_mean,
        "importance_std": importance.importances_std,
    }).sort_values("importance_mean", ascending=False).to_csv(output_dir / "permutation_importance.csv", index=False)

    numeric_top = [
        feature for feature in np.asarray(features)[np.argsort(importance.importances_mean)[::-1]]
        if feature not in PRIMARY_CATEGORICAL
    ][:5]
    if numeric_top:
        fig, axes = plt.subplots(len(numeric_top), 1, figsize=(9, 3.2 * len(numeric_top)))
        axes = np.atleast_1d(axes)
        PartialDependenceDisplay.from_estimator(model, test[features], numeric_top, ax=axes)
        fig.tight_layout()
        fig.savefig(output_dir / "partial_dependence.png", dpi=160, bbox_inches="tight")
        plt.close(fig)


def export_elastic_coefficients(
    model: Pipeline,
    output_path: Path,
) -> None:
    preprocessor = model.named_steps["preprocessor"]
    names = preprocessor.get_feature_names_out()
    coefficients = model.named_steps["model"].coef_
    pd.DataFrame({"feature": names, "coefficient": coefficients}).assign(
        absolute_coefficient=lambda frame: frame["coefficient"].abs()
    ).sort_values("absolute_coefficient", ascending=False).to_csv(output_path, index=False)


def save_summary(path: Path, summary: dict[str, Any]) -> None:
    path.write_text(json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8")


def save_model(path: Path, model: BaseEstimator, metadata: dict[str, Any]) -> None:
    with path.open("wb") as output:
        pickle.dump({"model": model, "metadata": metadata}, output)
