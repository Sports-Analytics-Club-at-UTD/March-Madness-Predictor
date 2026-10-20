"""Empirical residual simulations of team tournament-success scores."""

from collections.abc import Mapping

import numpy as np
import pandas as pd


def simulate_team_success(raw_team_features, ridge_bundle, n_simulations=10000, seed=42):
    """Predict one team and add independently resampled held-out errors.

    Accept a feature mapping, Series, or one-row DataFrame and a loaded bundle.
    Missing features/NaNs become zero, matching training; extra columns are
    ignored. Values remain continuous and unbounded. Threshold frequencies
    describe this score simulation, not calibrated bracket probabilities.
    std is the population standard deviation (ddof=0) of the simulated values.
    """
    if isinstance(n_simulations, (bool, np.bool_)) or not isinstance(n_simulations, (int, np.integer)):
        raise ValueError("n_simulations must be a positive integer.")
    if n_simulations <= 0:
        raise ValueError("n_simulations must be a positive integer.")
    required = {"model", "scaler", "feature_names", "validation_residuals"}
    if not isinstance(ridge_bundle, Mapping) or not required.issubset(ridge_bundle):
        raise ValueError("Supply a loaded Ridge inference bundle with held-out residuals.")
    names = list(ridge_bundle["feature_names"])
    if not names or len(set(names)) != len(names):
        raise ValueError("Bundle feature names must be nonempty and unique.")
    if any(name in names for name in ("team", "season", "target_wins")):
        raise ValueError("Bundle cannot include metadata or target columns as features.")
    for fitted in (ridge_bundle["scaler"], ridge_bundle["model"]):
        if getattr(fitted, "n_features_in_", None) != len(names):
            raise ValueError("Bundle model/scaler dimensions do not match its feature names.")
    scaler_names = getattr(ridge_bundle["scaler"], "feature_names_in_", names)
    if list(scaler_names) != names:
        raise ValueError("Bundle feature order differs from its fitted scaler.")

    if isinstance(raw_team_features, pd.Series):
        frame = raw_team_features.to_frame().T
    elif isinstance(raw_team_features, Mapping):
        frame = pd.DataFrame([raw_team_features])
    elif isinstance(raw_team_features, pd.DataFrame):
        frame = raw_team_features
    else:
        raise TypeError("Supply named raw features as a mapping, Series, or DataFrame.")
    if len(frame) != 1 or not frame.columns.is_unique:
        raise ValueError("Supply exactly one team row with unique feature names.")
    if not any(name in frame.columns for name in names):
        raise ValueError("Input contains none of the bundle's model features.")
    aligned = frame.reindex(columns=names, fill_value=0).fillna(0).astype(float)
    if not np.isfinite(aligned.to_numpy()).all():
        raise ValueError("Raw model features must be finite numeric values.")

    residuals = np.asarray(ridge_bundle["validation_residuals"], dtype=float)
    if residuals.ndim != 1 or not residuals.size or not np.isfinite(residuals).all():
        raise ValueError("Bundle must contain a nonempty finite held-out residual vector.")
    scaled = ridge_bundle["scaler"].transform(aligned)
    predictions = np.asarray(ridge_bundle["model"].predict(scaled), dtype=float)
    if predictions.shape != (1,) or not np.isfinite(predictions).all():
        raise ValueError("Ridge must produce one finite tournament-win prediction.")
    baseline = float(predictions[0])
    rng = np.random.default_rng(seed)
    values = baseline + rng.choice(residuals, size=n_simulations, replace=True)
    return {
        "baseline_prediction": baseline,
        "simulated_values": values,
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "std": float(np.std(values)),
        "percentile_5": float(np.percentile(values, 5)),
        "percentile_95": float(np.percentile(values, 95)),
        "probabilities": {wins: float(np.mean(values >= wins)) for wins in range(1, 7)},
    }


def simulate_season_success(
    raw_team_features, ridge_bundle, season, n_simulations=10000, seed=42,
    expected_teams=None,
):
    """Return a sorted summary table for an entire season.

    Independent random streams are assigned in alphabetical team order, so
    a fixed seed and roster reproduce results regardless of input row order.
    If this season has validation records, its roster is also checked. A
    held-out-season bundle must reproduce its saved predictions; master
    bundles retain errors from different historical fits. Otherwise
    expected_teams must be supplied explicitly.
    """
    if not isinstance(raw_team_features, pd.DataFrame):
        raise TypeError("Season features must be a DataFrame with team and season columns.")
    if not raw_team_features.columns.is_unique:
        raise ValueError("Season feature names must be unique.")
    if not {"team", "season"}.issubset(raw_team_features.columns):
        raise ValueError("Season features must contain team and season columns.")
    if raw_team_features.empty or raw_team_features[["team", "season"]].isna().any().any():
        raise ValueError("Season features must have nonempty team and season values.")
    if not raw_team_features["season"].eq(season).all():
        raise ValueError("All feature rows must belong to the selected season.")
    if not raw_team_features["team"].map(lambda name: isinstance(name, str) and bool(name.strip())).all():
        raise ValueError("Team names must be nonempty strings.")
    if raw_team_features["team"].duplicated().any():
        raise ValueError("Season features contain duplicate teams.")

    labels = np.asarray(ridge_bundle.get("validation_seasons", []))
    teams = np.asarray(ridge_bundle.get("validation_teams", []))
    predictions = np.asarray(ridge_bundle.get("validation_predicted_wins", []), dtype=float)
    if labels.ndim != 1 or teams.shape != labels.shape or predictions.shape != labels.shape:
        raise ValueError("Bundle validation teams, seasons, and predictions must match.")
    selected = labels == season
    validation_predictions = None
    if selected.any():
        if len(set(teams[selected])) != int(selected.sum()):
            raise ValueError("Bundle has duplicate validation teams for the selected season.")
        if set(teams[selected]) != set(raw_team_features["team"]):
            raise ValueError("Season roster differs from the saved validation roster.")
        if (ridge_bundle.get("residual_source") == "held_out_season"
                and ridge_bundle.get("validation_season") == season):
            validation_predictions = dict(zip(teams[selected], predictions[selected]))
        if expected_teams is None:
            expected_teams = int(selected.sum())
    if expected_teams is None:
        raise ValueError("Supply expected_teams when the bundle has no records for this season.")
    if isinstance(expected_teams, (bool, np.bool_)) or not isinstance(expected_teams, (int, np.integer)):
        raise ValueError("expected_teams must be a positive integer.")
    if expected_teams <= 0 or len(raw_team_features) != expected_teams:
        raise ValueError(f"Expected {expected_teams} teams, found {len(raw_team_features)}.")

    ordered = raw_team_features.sort_values("team").reset_index(drop=True)
    streams = np.random.SeedSequence(seed).spawn(len(ordered))
    rows = []
    for index, stream in enumerate(streams):
        team_row = ordered.iloc[[index]]
        team = team_row["team"].iloc[0]
        result = simulate_team_success(team_row, ridge_bundle, n_simulations, stream)
        if validation_predictions is not None:
            if not np.isclose(result["baseline_prediction"], validation_predictions[team], rtol=1e-10, atol=1e-10):
                raise ValueError(f"Rebuilt features do not reproduce the saved prediction for {team}.")
        summary = {
            "team": team,
            "season": season,
            "ridge_baseline_prediction": result["baseline_prediction"],
            "simulated_mean": result["mean"],
            "simulated_median": result["median"],
            "simulated_std": result["std"],
            "percentile_5": result["percentile_5"],
            "percentile_95": result["percentile_95"],
        }
        summary.update({f"probability_ge_{wins}_win{'s' if wins != 1 else ''}": result["probabilities"][wins]
                        for wins in range(1, 7)})
        rows.append(summary)
    table = pd.DataFrame(rows).sort_values(
        ["simulated_mean", "team"], ascending=[False, True]
    ).reset_index(drop=True)
    if table.isna().any().any() or not np.isfinite(table.drop(columns="team").to_numpy()).all():
        raise ValueError("Simulation summary contains nonfinite values.")
    probability_columns = [name for name in table.columns if name.startswith("probability_ge_")]
    probabilities = table[probability_columns].to_numpy()
    if not ((probabilities >= 0) & (probabilities <= 1)).all() or (np.diff(probabilities, axis=1) > 0).any():
        raise ValueError("Threshold frequencies must be in [0, 1] and nonincreasing.")
    return table
