"""Export cutoff-filtered historical ensemble scores from earlier walk-forward SHAP folds."""

import argparse
from pathlib import Path

from historical_matchup_scores import export_historical_scores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True, help="New export directory")
    parser.add_argument("--members", nargs="+", required=True, help="Fixed model directory names, e.g. ridge")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--training-output-dir", type=Path, default=Path("model_outputs"))
    parser.add_argument("--manifest-dir", type=Path, default=Path("data/brackets"))
    args = parser.parse_args()
    try:
        path = export_historical_scores(args.config, args.output_dir, members=args.members,
                                        data_dir=args.data_dir, training_output_dir=args.training_output_dir,
                                        manifest_dir=args.manifest_dir)
    except (ValueError, KeyError, TypeError, OSError) as error:
        parser.exit(1, f"Error: {error}\n")
    print(f"Saved historical scores to {path}")


if __name__ == "__main__":
    main()
