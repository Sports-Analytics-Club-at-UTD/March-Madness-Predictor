"""Export fitted Ridge inference state and its held-out calibration errors."""

from pathlib import Path

import joblib
import numpy as np


def save_ridge_bundle(
    path, *, model, scaler, feature_names, training_seasons, validation_season,
    validation_actual_wins, validation_predicted_wins, validation_teams,
    validation_seasons=None, residual_source="held_out_season",
):
    """Save one joblib dictionary; predictions must come from held-out fits.

    A master model can use historical walk-forward errors, with
    validation_season=None and a season label for every validation record.
    Those errors belong to earlier fold models, not to the final master fit.
    """
    feature_names = list(feature_names)
    if not feature_names or len(set(feature_names)) != len(feature_names):
        raise ValueError("Feature names must be nonempty and unique.")
    if any(name in feature_names for name in ("team", "season", "target_wins")):
        raise ValueError("Metadata and target columns cannot be model features.")
    for fitted in (model, scaler):
        if getattr(fitted, "n_features_in_", None) != len(feature_names):
            raise ValueError("Fitted model/scaler dimensions must match feature names.")
    if hasattr(scaler, "feature_names_in_"):
        if list(scaler.feature_names_in_) != feature_names:
            raise ValueError("Feature order differs from the fitted scaler.")

    actual = np.asarray(validation_actual_wins, dtype=float).copy()
    predicted = np.asarray(validation_predicted_wins, dtype=float).copy()
    teams = list(validation_teams)
    if actual.ndim != 1 or predicted.ndim != 1 or actual.shape != predicted.shape:
        raise ValueError("Validation actuals and predictions must be matching vectors.")
    if len(teams) != len(actual):
        raise ValueError("Validation teams must match the validation vectors.")
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError("Validation actuals and predictions must be finite.")

    training_seasons = sorted(set(int(year) for year in training_seasons))
    if not training_seasons:
        raise ValueError("Training seasons must be recorded.")
    if residual_source not in ("held_out_season", "walk_forward"):
        raise ValueError("Residuals must come from held-out season or walk-forward fits.")
    if residual_source == "held_out_season":
        if validation_season is None or max(training_seasons) >= validation_season:
            raise ValueError("Validation season must be later than all training seasons.")
    if validation_seasons is None:
        validation_seasons = np.full(len(actual), validation_season, dtype=int)
    else:
        validation_seasons = np.asarray(validation_seasons, dtype=int).copy()
    if validation_seasons.shape != actual.shape:
        raise ValueError("Validation season labels must match the validation vectors.")
    if residual_source == "held_out_season":
        if not np.all(validation_seasons == validation_season):
            raise ValueError("Validation records must belong to the held-out season.")

    bundle = {
        "bundle_version": 1,
        "target": "target_wins",
        "model": model,
        "scaler": scaler,
        "feature_names": feature_names,
        "training_seasons": training_seasons,
        "validation_season": validation_season,
        "validation_seasons": validation_seasons,
        "validation_teams": teams,
        "validation_actual_wins": actual,
        "validation_predicted_wins": predicted,
        "validation_residuals": actual - predicted,
        "residual_source": residual_source,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)
    return bundle
