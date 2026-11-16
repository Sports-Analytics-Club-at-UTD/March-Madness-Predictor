"""Fit, inspect, predict with, or evaluate the standalone Bayesian team-rating baseline."""

import argparse
import json
from pathlib import Path

from bayesian_ratings import evaluate_games, fit_ratings, load_games, load_ratings, save_ratings, utc_timestamp


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    fit = commands.add_parser("fit", help="Fit one season using regular-season games strictly before cutoff")
    fit.add_argument("--data-dir", type=Path, default=Path("data"))
    fit.add_argument("--season", type=int, required=True)
    fit.add_argument("--cutoff", required=True, help="Exclusive timezone-aware cutoff")
    fit.add_argument("--draws", type=int, default=4000)
    fit.add_argument("--seed", type=int, default=42)
    fit.add_argument("--rating-prior-sd", type=float, default=1.5)
    fit.add_argument("--home-prior-sd", type=float, default=1.0)
    fit.add_argument("--output", type=Path, required=True, help="New .npz artifact path; existing files are never replaced")
    fit.add_argument("--summary", type=Path, help="Optional new human-readable JSON summary")
    predict = commands.add_parser("predict", help="Predict a matchup from a saved fit")
    predict.add_argument("artifact", type=Path)
    predict.add_argument("--team-a", required=True, help="Exact team name or cbbd:<ID>")
    predict.add_argument("--team-b", required=True)
    predict.add_argument("--venue", choices=("neutral", "a_home", "b_home"), default="neutral")
    summary = commands.add_parser("summary", help="Print ratings, uncertainty and fitting diagnostics")
    summary.add_argument("artifact", type=Path)
    evaluate = commands.add_parser("evaluate", help="Score held-out NCAA games from the fit cutoff to an exclusive end")
    evaluate.add_argument("artifact", type=Path)
    evaluate.add_argument("--data-dir", type=Path, default=Path("data"))
    evaluate.add_argument("--until", required=True, help="Exclusive timezone-aware end for evaluation games")
    args = parser.parse_args()
    try:
        if args.command == "fit":
            if args.output.exists() or (args.summary and args.summary.exists()):
                raise ValueError("Output already exists; choose a new run filename")
            if args.summary and args.summary.resolve() == args.output.resolve():
                raise ValueError("Artifact and summary must have different paths")
            data = load_games(args.data_dir, args.season, args.cutoff)
            posterior = fit_ratings(data, season=args.season, cutoff=args.cutoff, n_draws=args.draws,
                                    seed=args.seed, rating_prior_sd=args.rating_prior_sd, home_prior_sd=args.home_prior_sd)
            save_ratings(posterior, args.output)
            if args.summary:
                args.summary.parent.mkdir(parents=True, exist_ok=True)
                with args.summary.open("x", encoding="utf-8") as handle:
                    json.dump(posterior.summary(), handle, indent=2, allow_nan=False)
                    handle.write("\n")
            result = {"artifact": str(args.output), "games": len(data.games), "teams": len(posterior.team_ids),
                      "inference": "laplace", "data_report": data.report, "diagnostics": posterior.metadata["diagnostics"]}
        else:
            posterior = load_ratings(args.artifact)
            if args.command == "predict":
                result = posterior.predict(args.team_a, args.team_b, venue=args.venue)
            elif args.command == "summary":
                result = posterior.summary()
            else:
                cutoff = utc_timestamp(posterior.metadata["cutoff"])
                if utc_timestamp(args.until) <= cutoff:
                    raise ValueError("--until must be later than the fitting cutoff")
                data = load_games(args.data_dir, posterior.metadata["season"], args.until, regular_only=False)
                held_out = [game for game in data.games if game.start >= cutoff and game.tournament.casefold() == "ncaa"]
                result = evaluate_games(posterior, held_out)
        print(json.dumps(result, indent=2, allow_nan=False))
    except (ValueError, TypeError, KeyError, OSError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
