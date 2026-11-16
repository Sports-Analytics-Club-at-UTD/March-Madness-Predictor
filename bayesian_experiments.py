"""Single-factor Bayesian experiments with earlier-season selection and a locked holdout."""
import csv
from dataclasses import replace
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import re
import platform

import numpy as np
import scipy
import sklearn

from bayesian_evaluation import (audit_coverage, baseline_models, digest, load_config,
    load_tournament_results, probability_metrics, write_csv, write_json)
from bayesian_ratings import Game, GameDataset, fit_ratings, load_games, save_ratings

BASE = {"rating_prior_sd": 1.5, "home_prior_sd": 1.0, "coverage": "full"}


def validate_plan(plan, targets):
    tuning, held = plan["tuning_seasons"], plan["held_out_seasons"]
    if (not tuning or not held or len(set(tuning + held)) != len(tuning + held)
            or set(tuning + held) != set(targets) or max(tuning) >= min(held)):
        raise ValueError("Need disjoint chronological tuning/held-out seasons partitioning evaluation_seasons")
    candidates = plan["candidates"]
    if candidates.get("baseline") != BASE:
        raise ValueError("baseline must match the unchanged model defaults and full coverage")
    for name, settings in candidates.items():
        if not re.fullmatch(r"[a-z][a-z0-9_-]*", name) or set(settings) != set(BASE):
            raise ValueError("Invalid candidate name or settings")
        if name != "baseline" and sum(settings[k] != BASE[k] for k in BASE) != 1:
            raise ValueError("Each ablation must change exactly one baseline setting")
        if settings["coverage"] not in ("full", "entrants"):
            raise ValueError("Invalid coverage")
        for key in ("rating_prior_sd", "home_prior_sd"):
            value = settings[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 < value < float("inf"):
                raise ValueError("Prior scales must be positive and finite")
    return sorted(tuning), sorted(held)


def kaggle_season(root, spec, *, training=True):
    """Keep Kaggle IDs separate from CBBD; timestamps have day resolution only."""
    root = Path(root)
    def read(name):
        with (root / name).open(newline="", encoding="utf-8-sig") as f:
            return list(csv.DictReader(f))
    teams = {f"kaggle:{r['TeamID']}": r["TeamName"] for r in read("MTeams.csv")}
    season_rows = [r for r in read("MSeasons.csv") if int(r["Season"]) == spec.season]
    if len(season_rows) != 1:
        raise ValueError("Need one Kaggle season calendar")
    day_zero = datetime.strptime(season_rows[0]["DayZero"], "%m/%d/%Y").replace(tzinfo=timezone.utc)
    spec = replace(spec, cutoff=spec.cutoff.replace(hour=0, minute=0, second=0, microsecond=0))
    def games(name, tournament):
        result = {}
        for r in read(name):
            if int(r["Season"]) != spec.season:
                continue
            start = day_zero + timedelta(days=int(r["DayNum"]))
            if not tournament and start >= spec.cutoff:
                continue
            if tournament and not spec.cutoff <= start < spec.until:
                raise ValueError("Kaggle tournament outside configured window")
            a, b = f"kaggle:{r['WTeamID']}", f"kaggle:{r['LTeamID']}"
            sa, sb = int(r["WScore"]), int(r["LScore"])
            if a == b or a not in teams or b not in teams or not sa > sb >= 0:
                raise ValueError("Invalid Kaggle participants or scores")
            home = {"H": 1, "A": -1, "N": 0}[r.get("WLoc", "N")]
            if a > b:
                a, b, sa, sb, home = b, a, sb, sa, -home
            key = f"kaggle:{spec.season}:{r['DayNum']}:{a}:{b}"
            g = Game(key, spec.season, start, a, b, sa, sb, home, "NCAA" if tournament else "")
            if key in result:
                raise ValueError("Duplicate Kaggle game key")
            result[key] = g
        return sorted(result.values(), key=lambda g: (g.start, g.game_id))
    seeds = {}
    for r in read("MNCAATourneySeeds.csv"):
        if int(r["Season"]) == spec.season:
            key = f"kaggle:{r['TeamID']}"
            if key in seeds or not re.fullmatch(r"[A-Z](0[1-9]|1[0-6])[ab]?", r["Seed"]):
                raise ValueError("Invalid or duplicate Kaggle seed")
            seeds[key] = int(r["Seed"][1:3])
    tournament = games("MNCAATourneyCompactResults.csv", True)
    if len(tournament) != spec.expected_games:
        raise ValueError(f"{spec.season}: expected {spec.expected_games} games, found {len(tournament)}")
    entrants = {t for g in tournament for t in (g.team_a, g.team_b)}
    if not entrants <= seeds.keys():
        raise ValueError("Missing tournament seeds")
    data = dict(spec=spec, games=tournament, registry={t: teams[t] for t in entrants}, seeds=seeds)
    if training:
        regular = games("MRegularSeasonCompactResults.csv", False)
        observed = {t for g in regular for t in (g.team_a, g.team_b)}
        data["dataset"] = GameDataset(regular, {t: teams[t] for t in observed}, {
            "source": "Kaggle compact results", "games": len(regular),
            "coverage_note": "Day resolution; no independently verified point-in-time archive or complete league membership."})
    return data


def run_experiments(config_path, plan_path, output_dir, *, data_dir=None, kaggle_dir=None,
                    draws=4000, seed=42):
    specs, targets = load_config(config_path)
    plan = json.loads(Path(plan_path).read_text())
    tuning, held = validate_plan(plan, targets)
    if (data_dir is None) == (kaggle_dir is None):
        raise ValueError("Supply exactly one of data_dir or kaggle_dir")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    root = Path(kaggle_dir if kaggle_dir is not None else data_dir)
    if kaggle_dir is not None:
        sources = [root / name for name in ("MTeams.csv", "MSeasons.csv", "MNCAATourneySeeds.csv",
                    "MRegularSeasonCompactResults.csv", "MNCAATourneyCompactResults.csv")]
    else:
        sources = [root / str(s.season) / "tournament_games.csv" for s in specs]
        sources += [p for y in targets for p in sorted((root / str(y)).glob("*/*_game_stats.csv"))]
    write_json(output / "protocol.json", {"plan": plan, "config_sha256": digest(config_path),
        "config": json.loads(Path(config_path).read_text()),
        "plan_sha256": digest(plan_path), "draws": draws, "seed": seed,
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "scipy": scipy.__version__, "scikit_learn": sklearn.__version__},
        "code_sha256": {name: digest(Path(__file__).with_name(name)) for name in
                        ("bayesian_experiments.py", "bayesian_ratings.py", "bayesian_evaluation.py")},
        "source": "kaggle" if kaggle_dir else "cbbd", "selection_metric": "pooled tuning Brier; baseline wins ties",
        "inputs": [{"path": str(p.resolve()), "sha256": digest(p)} for p in sources]})
    loaded = {}
    def load(spec):
        if kaggle_dir is not None:
            return kaggle_season(root, spec, training=spec.season in targets)
        games, registry, seeds, _ = load_tournament_results(root / str(spec.season) / "tournament_games.csv", spec)
        d = dict(spec=spec, games=games, registry=registry, seeds=seeds)
        if spec.season in targets:
            d["dataset"] = load_games(root, spec.season, spec.cutoff)
        return d
    # No held-out outcomes are loaded into model selection.
    for spec in specs:
        if spec.season < min(held):
            loaded[spec.season] = load(spec)
    def evaluate(name, years, phase):
        settings = plan["candidates"][name]
        directory = output / phase / name
        directory.mkdir(parents=True)
        rows, audits = [], {}
        for year in years:
            d = loaded[year]
            dataset = d["dataset"]
            if settings["coverage"] == "entrants":
                subset = [g for g in dataset.games if g.team_a in d["registry"] or g.team_b in d["registry"]]
                dataset = GameDataset(subset, dataset.teams, dict(dataset.report, coverage="entrant schedules only"))
            audit = audit_coverage(dataset, d["games"], d["registry"])
            if audit["missing_tournament_teams"] or audit["overlapping_game_ids"]:
                raise ValueError(f"{year}: missing entrants or overlapping training IDs")
            audits[year] = audit
            posterior = fit_ratings(dataset, season=year, cutoff=d["spec"].cutoff, n_draws=draws, seed=seed,
                rating_prior_sd=settings["rating_prior_sd"], home_prior_sd=settings["home_prior_sd"])
            save_ratings(posterior, directory / f"ratings_{year}.npz")
            for g in d["games"]:
                p = posterior.predict(g.team_a, g.team_b, venue={0: "neutral", 1: "a_home", -1: "b_home"}[g.home])
                rows.append(dict(season=year, game_id=g.game_id, outcome=int(g.score_a > g.score_b),
                                 probability=p["win_probability_a"]))
        metrics = {}
        for scope in [*years, "pooled"]:
            subset = rows if scope == "pooled" else [r for r in rows if r["season"] == scope]
            metrics[str(scope)] = probability_metrics([r["outcome"] for r in subset], [r["probability"] for r in subset])[0]
        write_csv(directory / "predictions.csv", rows)
        write_json(directory / "metrics.json", metrics)
        write_json(directory / "coverage.json", audits)
        return metrics
    tuned = {name: evaluate(name, tuning, "tuning") for name in plan["candidates"]}
    winner = min(tuned, key=lambda name: (tuned[name]["pooled"]["brier_score"], name != "baseline", name))
    selection = {"selected": winner, "settings": plan["candidates"][winner], "tuning_metrics": tuned}
    write_json(output / "selection.json", selection)
    for spec in specs:
        if spec.season in held:
            loaded[spec.season] = load(spec)
    evaluated = {name: evaluate(name, held, "held_out") for name in dict.fromkeys(("baseline", winner))}
    # Frozen earlier history: no held-out results train the seed comparator.
    earlier = {y: d for y, d in loaded.items() if y < min(held)}
    seed_rows = []
    for year in held:
        d = loaded[year]
        model, _, _ = baseline_models(d["spec"], earlier, {})
        if model is None:
            raise ValueError("Seed comparator requires earlier tournament history")
        for g in d["games"]:
            p = float(model.predict_proba([[d["seeds"][g.team_b] - d["seeds"][g.team_a]]])[0, 1])
            seed_rows.append(dict(season=year, game_id=g.game_id, outcome=int(g.score_a > g.score_b), probability=p))
    write_csv(output / "held_out_seed_predictions.csv", seed_rows)
    result = dict(selected=winner, held_out=evaluated,
        seed=probability_metrics([r["outcome"] for r in seed_rows], [r["probability"] for r in seed_rows])[0],
        seed_training_seasons=sorted(earlier), limitations="Retrospective experiment; not an independent future confirmation. No Ridge comparison. Coverage source and historical revisions differ from the documented benchmark.")
    write_json(output / "results.json", result)
    return result
