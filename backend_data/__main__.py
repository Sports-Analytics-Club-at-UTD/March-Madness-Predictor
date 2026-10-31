"""Commands: export existing results, generate schema, validate, or merge runs."""

import argparse
import json
from pathlib import Path

import pandas as pd

from .export import export_forecast, save_forecast
from .kalshi import source_artifact
from .models import ForecastDocument


def load_document(path):
    return ForecastDocument.model_validate_json(Path(path).read_text(encoding="utf-8"))


def merge_documents(documents):
    if not documents:
        raise ValueError("At least one document is required")
    data = documents[0].model_dump(mode="json")
    teams = {team["id"]: team for team in data["teams"]}
    for document in documents[1:]:
        other = document.model_dump(mode="json")
        for key in ("id", "season", "competition", "format", "includes_first_four"):
            if data["tournament"][key] != other["tournament"][key]:
                raise ValueError("Cannot merge different tournaments or formats")
        if data["games"] and other["games"] and data["games"] != other["games"]:
            raise ValueError("Cannot merge conflicting tournament topologies")
        if other["games"]:
            data["games"] = other["games"]
            data["tournament"] = other["tournament"]
        for team in other["teams"]:
            if team["id"] in teams and teams[team["id"]] != team:
                raise ValueError("Cannot merge conflicting team identities")
            teams[team["id"]] = team
        for key in ("prediction_runs", "team_predictions", "brackets"):
            data[key].extend(other[key])
        if data["market_data"] is not None and other["market_data"] is not None:
            first = {k: v for k, v in data["market_data"].items() if k != "imported_at"}
            second = {k: v for k, v in other["market_data"].items() if k != "imported_at"}
            if first != second:
                raise ValueError("Cannot merge conflicting market datasets; choose one before merging")
        elif other["market_data"] is not None:
            data["market_data"] = other["market_data"]
    data["teams"] = sorted(teams.values(), key=lambda t: t["id"])
    return ForecastDocument.model_validate(data)


def json_schema():
    return {"$schema": "https://json-schema.org/draft/2020-12/schema",
            **ForecastDocument.model_json_schema(mode="serialization")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    schema = commands.add_parser("schema", help="Generate JSON Schema from Pydantic models")
    schema.add_argument("--output", type=Path, default=Path("schemas/forecast.schema.json"))
    validate = commands.add_parser("validate", help="Validate JSON, including cross-record consistency")
    validate.add_argument("path", type=Path)
    export = commands.add_parser("export", help="Convert saved outputs without training")
    export.add_argument("--team-scores", required=True, type=Path)
    export.add_argument("--metadata", required=True, type=Path)
    export.add_argument("--mode", required=True, choices=("ridge", "shap"))
    export.add_argument("--bracket-csv", type=Path)
    export.add_argument("--topology", type=Path, help="JSON with teams64, round1_pairs, tree_edges, slot_rounds, and optional actual_slot_winners")
    export.add_argument("--kalshi-csv", type=Path)
    export.add_argument("--kalshi-mapping", type=Path)
    export.add_argument("--output-dir", type=Path, default=Path("model_outputs/monte_carlo"))
    merge = commands.add_parser("merge", help="Combine distinct runs for one tournament")
    merge.add_argument("paths", nargs="+", type=Path)
    merge.add_argument("--output-dir", type=Path, default=Path("model_outputs/monte_carlo"))
    args = parser.parse_args()
    try:
        if args.command == "schema":
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(json_schema(), indent=2) + "\n", encoding="utf-8")
            print(args.output)
        elif args.command == "validate":
            document = load_document(args.path)
            print(f"Valid schema {document.schema_version}: {len(document.prediction_runs)} runs, {len(document.brackets)} brackets")
        elif args.command == "merge":
            print(save_forecast(merge_documents([load_document(p) for p in args.paths]), args.output_dir))
        else:
            if args.kalshi_mapping and not args.kalshi_csv:
                parser.error("--kalshi-mapping requires --kalshi-csv")
            read_json = lambda path: json.loads(path.read_text(encoding="utf-8")) if path else None
            artifacts = [source_artifact(args.team_scores, "text/csv"),
                         source_artifact(args.metadata, "application/json")]
            for path, media_type in ((args.bracket_csv, "text/csv"), (args.topology, "application/json"),
                                     (args.kalshi_mapping, "application/json")):
                if path:
                    artifacts.append(source_artifact(path, media_type))
            print(export_forecast(
                pd.read_csv(args.team_scores), read_json(args.metadata), args.mode, args.output_dir,
                bracket_df=pd.read_csv(args.bracket_csv) if args.bracket_csv else None,
                parsed_tree=read_json(args.topology), kalshi_csv=args.kalshi_csv,
                kalshi_mapping=read_json(args.kalshi_mapping),
                source_artifacts=artifacts,
            ))
    except (ValueError, TypeError, OSError, KeyError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
