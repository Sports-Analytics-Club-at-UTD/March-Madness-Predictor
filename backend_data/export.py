"""Adapt existing pipeline outputs to the shared contract and immutable storage."""

import hashlib
import json
import math
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd

from .kalshi import COMPETITION, read_kalshi
from .models import ForecastDocument


def normalize(value):
    """Missing numpy/pandas scalars become null; infinity and unknown objects fail."""
    if isinstance(value, dict):
        return {str(k): normalize(v) for k, v in value.items()}
    if isinstance(value, (tuple, list, np.ndarray)):
        return [normalize(v) for v in value]
    if isinstance(value, np.generic):
        return normalize(value.item())
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, float):
        if math.isnan(value):
            return None
        if not math.isfinite(value):
            raise ValueError("Infinite values cannot be exported")
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError(f"Unsupported JSON value: {type(value).__name__}")


def stable_team_id(name, competition=COMPETITION):
    return "team-" + hashlib.sha256(f"{competition}|{name}".encode()).hexdigest()[:24]


def build_forecast(table, metadata, mode, bracket_df=None, parsed_tree=None,
                   kalshi_csv=None, kalshi_mapping=None, run_id=None, source_artifacts=None):
    """Export current producers without recalculating or relabeling probabilities."""
    if mode not in ("ridge", "shap"):
        raise ValueError("Current adapters accept ridge or shap mode")
    metadata = normalize(metadata)
    season = int(metadata["season"])
    if "team" not in table or table.empty or table["team"].duplicated().any():
        raise ValueError("Predictions require unique, nonempty team names")
    if not table["team"].map(lambda name: isinstance(name, str) and bool(name.strip())).all():
        raise ValueError("Predictions require unique, nonempty team names")
    if "season" not in table or not table["season"].eq(season).all():
        raise ValueError("Prediction seasons do not match metadata")
    tournament_id = f"ncaa-men-{season}"
    run_id = run_id or str(uuid4())
    names = set(table["team"])
    if parsed_tree is not None:
        names.update(parsed_tree["teams64"])
    ids = {name: stable_team_id(name) for name in sorted(names)}
    topology = parsed_tree is not None
    model_name = str(metadata.get("ensemble_name") or metadata.get("bundle_path") or mode)
    payload = {
        "tournament": {"id": tournament_id, "season": season, "competition": COMPETITION,
                       "topology_status": "available" if topology else "unavailable",
                       "bracket_team_count": len(parsed_tree["teams64"]) if topology else None},
        "teams": [{"id": ids[name], "name": name} for name in sorted(names)],
        "games": [],
        "prediction_runs": [{
            "id": run_id, "tournament_id": tournament_id,
            "method": "ridge_residual_bootstrap" if mode == "ridge" else "shap_ensemble",
            "model_name": model_name, "generated_at": datetime.now(timezone.utc),
            "n_simulations": metadata.get("n_simulations_per_team"), "seed": metadata.get("seed"),
            "source_metadata": metadata,
            "artifacts": source_artifacts or [],
        }],
        "team_predictions": [], "brackets": [], "market_data": None,
    }
    prefixes = {"success_score_threshold": "probability_ge_",
                "calibrated_success_score_threshold": "calibrated_prob_ge_", "bracket_tree": "tree_prob_ge_"}
    for row in normalize(table.to_dict(orient="records")):
        probabilities, columns = {}, set()
        for group, prefix in prefixes.items():
            values = []
            for wins in range(1, 7):
                column = f"{prefix}{wins}_win{'s' if wins != 1 else ''}"
                if column in row:
                    columns.add(column)
                    values.append({"minimum_wins": wins, "probability": row[column]})
            probabilities[group] = values
        intervals = []
        if row.get("percentile_5") is not None and row.get("percentile_95") is not None:
            intervals.append({"kind": "empirical_prediction", "target": "tournament_success_score",
                              "level": .9, "lower": row["percentile_5"], "upper": row["percentile_95"],
                              "method": "residual_bootstrap_percentiles"})
        payload["team_predictions"].append({
            "run_id": run_id, "team_id": ids[row["team"]], "probabilities": probabilities,
            "statistics": {k: v for k, v in row.items() if k not in columns | {"team", "season"}},
            "intervals": intervals,
        })
    if topology:
        teams = parsed_tree["teams64"]
        pairs, edges = parsed_tree["round1_pairs"], parsed_tree["tree_edges"]
        rounds = parsed_tree["slot_rounds"]
        if len(teams) != 64 or len(pairs) != 32 or len(edges) != 31 or len(rounds) != 63:
            raise ValueError("Current SHAP adapter requires a 64-team, 63-game tree")
        actuals = parsed_tree.get("actual_slot_winners", [-1] * 63)
        schedule = parsed_tree.get("scheduled_starts_utc")
        if schedule is None:
            frame = parsed_tree.get("df_63")
            schedule = list(frame["startDate"]) if frame is not None and "startDate" in frame else [None] * 63
        if len(schedule) != 63 or len(actuals) != 63:
            raise ValueError("Schedule and actual winners must match the 63 game slots")
        def game_id(slot):
            return f"{tournament_id}-g{slot}"
        for index in range(63):
            if index < 32:
                participants = [{"team_id": ids[teams[int(i)]]} for i in pairs[index]]
            else:
                participants = [{"source_game_id": game_id(int(i) + 1)} for i in edges[index - 32]]
            actual = int(actuals[index])
            scheduled = schedule[index]
            if scheduled is not None and pd.isna(scheduled):
                scheduled = None
            if scheduled is not None and hasattr(scheduled, "isoformat"):
                scheduled = scheduled.isoformat()
            payload["games"].append({
                "id": game_id(index + 1), "slot": index + 1, "round_number": int(rounds[index]),
                "round_points": 10 * 2 ** (int(rounds[index]) - 1), "participants": participants,
                "actual_winner_team_id": ids[teams[actual]] if 0 <= actual < len(teams) else None,
                "scheduled_start_utc": scheduled,
            })
    if bracket_df is not None:
        if not topology:
            raise ValueError("Bracket picks require structured topology")
        rows = normalize(bracket_df.to_dict(orient="records"))
        if sorted(row["game_slot"] for row in rows) != list(range(1, 64)):
            raise ValueError("Bracket export requires slots 1 through 63 exactly once")
        rows.sort(key=lambda row: row["game_slot"])
        summary = metadata.get("bracket_pool_optimization") or {}
        for strategy in ("max_ev", "leverage"):
            picks = []
            for row in rows:
                slot = int(row["game_slot"])
                game = payload["games"][slot - 1]
                if row["round_number"] != game["round_number"] or row["round_points"] != game["round_points"]:
                    raise ValueError("Bracket rounds/points disagree with topology")
                winner = ids[row[f"{strategy}_pick"]]
                actual = game["actual_winner_team_id"]
                picks.append({
                    "game_id": game["id"], "winner_team_id": winner,
                    "confidence": {"value": row[f"{strategy}_win_prob"], "method": "bracket_tree_monte_carlo"},
                    "public_pick_probability": row[f"{strategy}_public_pct"],
                    "expected_points": row[f"{strategy}_expected_pts"],
                    "leverage_ratio": row.get("leverage_ratio") if strategy == "leverage" else None,
                    "actual_correct": winner == actual if actual else None,
                })
            payload["brackets"].append({
                "id": f"{run_id}-{strategy}", "run_id": run_id, "tournament_id": tournament_id,
                "strategy": strategy, "picks": picks,
                "summary": {
                    "champion_team_id": picks[-1]["winner_team_id"],
                    "expected_points": summary.get(f"{strategy}_expected_points"),
                    "pool_win_probability": summary.get(f"{strategy}_pool_win_prob"),
                    "top_10_percent_probability": summary.get(f"{strategy}_top10pct_prob"),
                    "actual_points": summary.get(f"actual_{strategy}_points"),
                    "pool_size": summary.get("pool_size"),
                    "extensions": {"leverage_alpha": summary["leverage_alpha"]} if "leverage_alpha" in summary else {},
                },
            })
    if kalshi_csv is not None:
        payload["market_data"] = read_kalshi(kalshi_csv, season, ids, kalshi_mapping)
    return ForecastDocument.model_validate(payload)


def save_forecast(document, output_dir):
    """Publish a complete validated file atomically; never overwrite a prior export."""
    document = ForecastDocument.model_validate_json(document.model_dump_json())
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # Bundle ID is independent of run IDs because a document may contain several runs.
    path = output_dir / f"forecast_{document.tournament.season}_{uuid4()}.json"
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=output_dir, delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(document.model_dump_json(indent=2) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        # Hard-link publication is atomic and fails rather than overwriting a name.
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return path


def export_forecast(table, metadata, mode, output_dir, **kwargs):
    return save_forecast(build_forecast(table, metadata, mode, **kwargs), output_dir)
