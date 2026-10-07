"""Build the county forecast panel, benchmark models, and save the selected forecaster."""
from __future__ import annotations

import argparse
from pathlib import Path

import duckdb

from src.modeling.county_price import (
    Candidate,
    LagPriceBaseline,
    breakdown_metrics,
    build_forecast_panel,
    candidate_features,
    select_forecast_from_validation,
    county_grouped_validation,
    cross_validate_candidates,
    default_candidates,
    diagnostic_challenger,
    evaluate_test_models,
    export_elastic_coefficients,
    export_explanations,
    load_county_year_data,
    make_estimator,
    materialize_panel,
    model_target,
    predict_prices,
    save_model,
    save_summary,
    select_cv_winners,
)

ROOT = Path(__file__).resolve().parents[2]
DATABASE_PATH = ROOT / "data" / "housing_predict.duckdb"
OUTPUT_DIR = ROOT / "data" / "modeling" / "county_price"


def train(database: Path, output_dir: Path, quick: bool = False) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(database))
    try:
        county_year, stable_acs, all_acs = load_county_year_data(con)
        panel, feature_sets = build_forecast_panel(county_year, stable_acs, all_acs)
        materialize_panel(con, panel)
    finally:
        con.close()

    labeled_panel = panel.loc[panel["target_price_2024_usd"].notna()].copy()
    forecast_panel = panel.loc[panel["target_price_2024_usd"].isna()].copy()
    candidates = default_candidates(quick)
    validation_years = (2022,) if quick else range(2018, 2023)
    cv_results = cross_validate_candidates(labeled_panel, feature_sets, candidates, validation_years)
    cv_results.to_csv(output_dir / "cv_results.csv", index=False)
    winners = select_cv_winners(cv_results, candidates)

    selected_name, selection_metrics = select_forecast_from_validation(
        cv_results, labeled_panel, [candidate.name for candidate in winners]
    )
    selection_metrics.to_csv(output_dir / 'selection_validation_metrics.csv', index=False)
    cv_scores = cv_results.groupby('candidate').rmsle.mean()
    best_ml = min(winners, key=lambda candidate: float(cv_scores.loc[candidate.name]))

    test_metrics, predictions, fitted_models = evaluate_test_models(labeled_panel, feature_sets, winners)
    test_metrics.to_csv(output_dir / "test_metrics.csv", index=False)
    predictions.to_csv(output_dir / "test_predictions.csv", index=False)
    selected_candidate: Candidate | None = next(
        (candidate for candidate in winners if candidate.name == selected_name), None
    )
    explanation_candidate = selected_candidate or best_ml
    explanation_model = fitted_models[explanation_candidate.name]
    export_explanations(explanation_model, explanation_candidate, labeled_panel, feature_sets, output_dir)

    elastic = next(candidate for candidate in winners if candidate.estimator == "elastic_net")
    export_elastic_coefficients(fitted_models[elastic.name], output_dir / "elastic_net_coefficients.csv")
    grouped = county_grouped_validation(labeled_panel, feature_sets, best_ml, folds=3 if quick else 5)
    grouped.to_csv(output_dir / "county_grouped_validation.csv", index=False)

    diagnostics = []
    for feature_set in ("climate", "post2020"):
        metrics, _ = diagnostic_challenger(labeled_panel, feature_sets, best_ml, feature_set)
        diagnostics.append(metrics)
    import pandas as pd
    pd.DataFrame(diagnostics).to_csv(output_dir / "challenger_metrics.csv", index=False)

    breakdown_column = selected_name if selected_name in predictions else "county_lag_baseline"
    breakdown_metrics(predictions, breakdown_column).to_csv(output_dir / "breakdown_metrics.csv", index=False)

    if selected_candidate is None:
        selected_model = LagPriceBaseline().fit(labeled_panel[["price_lag1_2024_usd"]])
        selected_features = ["price_lag1_2024_usd"]
        target_kind = "price"
        future_prediction = selected_model.predict(forecast_panel[selected_features])
    else:
        selected_features = candidate_features(selected_candidate, feature_sets)
        target_kind = selected_candidate.target_kind
        selected_model = make_estimator(selected_candidate, selected_features)
        selected_model.fit(
            labeled_panel[selected_features], model_target(labeled_panel, selected_candidate.target_kind)
        )
        future_prediction = predict_prices(
            selected_model, forecast_panel, selected_features, selected_candidate.target_kind
        )

    future = forecast_panel[["county_fips", "county_name", "state", "msa_type", "forecast_year"]].copy()
    future["predicted_home_value_2024_usd"] = future_prediction
    future.to_csv(output_dir / "forecast_2025.csv", index=False)

    metadata = {
        "selected_model": selected_name,
        "features": selected_features,
        "target_kind": target_kind,
        "target": "target_median_owner_occupied_home_value_2024_usd",
        "evaluation_training_target_years": [int(labeled_panel["forecast_year"].min()), 2022],
        "refit_target_years": [int(labeled_panel["forecast_year"].min()), int(labeled_panel["forecast_year"].max())],
        "test_target_years": [2023, 2024],
        "minimum_validation_rmsle_improvement": 0.005,
        "selection_source": "expanding temporal validation only",
        "run_mode": "quick" if quick else "full",
        "temporal_validation_years": list(validation_years),
    }
    save_model(output_dir / "selected_model.pkl", selected_model, metadata)
    summary = {
        **metadata,
        "materialized_panel_rows": len(panel),
        "labeled_panel_rows": len(labeled_panel),
        "forecast_rows": len(forecast_panel),
        "panel_counties": int(panel["county_fips"].nunique()),
        "stable_acs_feature_count": len(stable_acs),
        "all_acs_feature_count": len(all_acs),
        "best_ml_challenger": best_ml.name,
        "artifacts": [
            "breakdown_metrics.csv", "challenger_metrics.csv", "county_grouped_validation.csv", "selection_validation_metrics.csv",
            "cv_results.csv", "elastic_net_coefficients.csv", "forecast_2025.csv",
            "partial_dependence.png", "permutation_importance.csv", "run_summary.json",
            "selected_model.pkl", "test_metrics.csv", "test_predictions.csv",
        ],
    }
    save_summary(output_dir / "run_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=DATABASE_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--quick", action="store_true", help="Use one parameter set per model and three grouped folds.")
    args = parser.parse_args()
    summary = train(args.database, args.output_dir, args.quick)
    print(f"Selected model: {summary['selected_model']}")
    print(f"Artifacts: {args.output_dir}")


if __name__ == "__main__":
    main()
