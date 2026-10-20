"""Run season-wide tournament-success score simulations without retraining."""

import argparse
import contextlib
import io
import json
from pathlib import Path
import runpy

import joblib
import pandas as pd

from monte_carlo import simulate_season_success


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--n-simulations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--expected-teams", type=int)
    parser.add_argument("--features-csv", type=Path, help="Already engineered team-season features.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--training-output-dir", type=Path, default=Path("model_outputs"))
    parser.add_argument("--output-dir", type=Path, default=Path("model_outputs/monte_carlo"))
    args = parser.parse_args()
    bundle = joblib.load(args.bundle)
    if args.features_csv:
        features = pd.read_csv(args.features_csv)
    else:
        cache = args.training_output_dir / "tournament_wins" / f"tournament_wins_{args.season}.csv"
        if not cache.exists():
            parser.error(f"Target cache missing: {cache}. Supply --features-csv or generate the cache first.")
        # Reuse exactly the training feature builder; no training main or API
        # call is needed because the target cache was verified above.
        with contextlib.redirect_stdout(io.StringIO()):
            training = runpy.run_path(str(Path(__file__).parent / "models" / "ridge_shap.py"))
            features = training["build_team_features"](
                str(args.data_dir), args.season, str(args.training_output_dir)
            )
    table = simulate_season_success(
        features, bundle, args.season, n_simulations=args.n_simulations,
        seed=args.seed, expected_teams=args.expected_teams,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / f"team_success_scores_{args.season}.csv"
    table.to_csv(path, index=False)
    # Keep the CSV's requested schema; describe interpretation and provenance
    # alongside it so they accompany any shared results.
    metadata = {
        "interpretation": "Continuous tournament-success scores and threshold frequencies; not calibrated matchup or bracket probabilities.",
        "season": args.season,
        "teams_processed": len(table),
        "n_simulations_per_team": args.n_simulations,
        "seed": args.seed,
        "random_streams": "Independent SeedSequence child streams in alphabetical team order.",
        "bundle_path": str(args.bundle),
        "training_seasons": bundle["training_seasons"],
        "validation_season": bundle["validation_season"],
        "residual_source": bundle["residual_source"],
        "residual_count": len(bundle["validation_residuals"]),
        "feature_count": len(bundle["feature_names"]),
        "missing_features_zero_filled": [name for name in bundle["feature_names"] if name not in features.columns],
        "validation_checks": [
            "Expected team count and unique team names",
            "All rows belong to the selected season",
            "Feature order and fitted model/scaler dimensions",
            "No NaN or infinite summary values",
            "Threshold frequencies are bounded and nonincreasing",
        ],
        "assumptions": [
            "The same empirical residual pool applies to each team.",
            "Scores are continuous, without rounding or clipping.",
            "The bundle's held-out residual pool calibrates these outputs; it is not an independent probability-validation set.",
            "Player-season totals require verification of pre-tournament provenance.",
        ],
    }
    if args.season in bundle["validation_seasons"]:
        metadata["validation_checks"].append("Complete saved validation roster")
    if bundle["residual_source"] == "held_out_season" and bundle["validation_season"] == args.season:
        metadata["validation_checks"].append("Matching saved baseline prediction for every team")
    path.with_suffix(".metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Saved {len(table)} teams to {path}")
    print("Validation passed. Results are tournament-success scores / threshold frequencies.")
    if metadata["missing_features_zero_filled"]:
        print(f"Zero-filled absent features: {metadata['missing_features_zero_filled']}")
    print(table[["team", "ridge_baseline_prediction", "simulated_mean"]].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
