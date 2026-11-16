"""Simulate a pre-tournament bracket from Bayesian ratings and export forecast JSON.

Optional and experimental; the default pipeline is unchanged. Each simulated
tournament uses one joint posterior rating draw for all of its games.
"""

import argparse
import json
from pathlib import Path
from uuid import uuid4

from bayesian_monte_carlo import (
    build_forecast_document, file_artifact, load_bracket_schedule, save_forecast_document,
    simulate_bracket, validate_forecast_document,
)
from bayesian_ratings import fit_ratings, load_games, load_ratings, save_ratings


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--bracket-schedule", type=Path, required=True,
                        help="Explicit pre-tournament schedule JSON (see docs/bayesian-monte-carlo.md)")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--ratings", type=Path, help="Saved ratings artifact from run_bayesian.py fit")
    source.add_argument("--cutoff", help="Fit new ratings from regular-season games strictly before this "
                                         "timezone-aware instant")
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="Game caches, used with --cutoff")
    parser.add_argument("--draws", type=int, default=4000, help="Posterior draws, used with --cutoff")
    parser.add_argument("--n-simulations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("model_outputs/bayesian_monte_carlo"))
    args = parser.parse_args(argv)
    try:
        schedule = load_bracket_schedule(args.bracket_schedule)
        if schedule.season != args.season:
            raise ValueError(f"Schedule is for season {schedule.season}, not {args.season}")
        run_id = str(uuid4())
        if args.ratings:
            ratings_path = args.ratings
            posterior = load_ratings(ratings_path)
        else:
            data = load_games(args.data_dir, args.season, args.cutoff)
            posterior = fit_ratings(data, season=args.season, cutoff=args.cutoff, n_draws=args.draws, seed=args.seed)
            ratings_path = args.output_dir / f"bayesian_ratings_{args.season}_{run_id}.npz"
        simulation = simulate_bracket(posterior, schedule, n_simulations=args.n_simulations, seed=args.seed)
        if not args.ratings:
            save_ratings(posterior, ratings_path)
        document = build_forecast_document(posterior, simulation, run_id=run_id, artifacts=[
            file_artifact(ratings_path, "application/x-npz", "Joint posterior rating draws"),
            file_artifact(args.bracket_schedule, "application/json", "Pre-tournament bracket schedule"),
        ])
        path = save_forecast_document(document, args.output_dir)
        champion = max(document["team_predictions"], key=lambda row: row["statistics"]["champion_probability"])
        names = {team["id"]: team["name"] for team in document["teams"]}
        print(json.dumps({
            "forecast": str(path), "ratings": str(ratings_path), "run_id": run_id,
            "teams": len(document["teams"]), "games": len(document["games"]),
            "n_simulations": args.n_simulations,
            "most_likely_champion": names[champion["team_id"]],
            "champion_probability": champion["statistics"]["champion_probability"],
            "contract_validation": validate_forecast_document(document),
        }, indent=2))
    except (ValueError, TypeError, KeyError, OSError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
