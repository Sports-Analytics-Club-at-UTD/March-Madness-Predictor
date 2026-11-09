"""Audit and evaluate historical Bayesian probabilities against earlier-trained baselines."""

import argparse
from pathlib import Path

import requests

from bayesian_evaluation import run_evaluation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True,
                        help="JSON with seasons, explicit cutoffs/ends, expected game counts and evaluation_seasons")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--results-dir", type=Path, help="Root with <season>/tournament_games.csv; defaults to --data-dir")
    parser.add_argument("--output-dir", type=Path, required=True, help="New run directory; never overwrites a run")
    parser.add_argument("--fetch-results", action="store_true", help="Fetch missing tournament CSVs using CBBD_API_KEY")
    parser.add_argument("--matchup-scores", type=Path, help="Existing ensemble scores with pre-cutoff provenance; see docs")
    parser.add_argument("--draws", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument("--min-games", type=int, default=10, help="Coverage warning threshold; teams are never silently dropped")
    parser.add_argument("--rating-prior-sd", type=float, default=1.5)
    parser.add_argument("--home-prior-sd", type=float, default=1.0)
    args = parser.parse_args()
    try:
        result = run_evaluation(args.config, args.output_dir, data_dir=args.data_dir,
                                results_dir=args.results_dir, matchup_scores=args.matchup_scores,
                                fetch_results=args.fetch_results, draws=args.draws, seed=args.seed,
                                bins=args.bins, min_games=args.min_games,
                                rating_prior_sd=args.rating_prior_sd, home_prior_sd=args.home_prior_sd)
    except (ValueError, KeyError, TypeError, OSError, requests.RequestException) as error:
        parser.exit(1, f"Error: {error}\n")
    print(f"Saved evaluation to {args.output_dir / 'report.md'}")
    for baseline, comparison in result["comparisons"].items():
        print(f"{baseline}: {comparison['n_games']} shared games, {comparison['excluded_games']} excluded")


if __name__ == "__main__":
    main()
