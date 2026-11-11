"""Run single-factor tuning, lock selection, then evaluate later seasons once."""
import argparse
from pathlib import Path
from bayesian_experiments import run_experiments


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--data-dir", type=Path)
    source.add_argument("--kaggle-dir", type=Path)
    parser.add_argument("--draws", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    result = run_experiments(args.config, args.plan, args.output_dir, data_dir=args.data_dir,
                            kaggle_dir=args.kaggle_dir, draws=args.draws, seed=args.seed)
    print(f"Selected {result['selected']}; results: {args.output_dir / 'results.json'}")


if __name__ == "__main__":
    main()
