"""Walk-forward evaluation of Bayesian NCAA game probabilities, without simulation."""

from collections import Counter
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

from bayesian_ratings import (
    Game, _boolean, _provider_id, fit_ratings, load_games, positive_integer,
    save_ratings, utc_timestamp,
)
from matchup_probability import fit_symmetric_matchup, matchup_probability


@dataclass(frozen=True)
class SeasonSpec:
    season: int
    cutoff: datetime
    until: datetime
    expected_games: int


def load_config(path):
    config = json.loads(Path(path).read_text())
    specs = []
    for item in config["seasons"]:
        positive_integer(item["season"], "season")
        positive_integer(item["expected_games"], "expected_games")
        cutoff, until = utc_timestamp(item["cutoff"]), utc_timestamp(item["until"])
        if until <= cutoff:
            raise ValueError("Each evaluation end must be later than its cutoff")
        specs.append(SeasonSpec(item["season"], cutoff, until, item["expected_games"]))
    years = [s.season for s in specs]
    targets = config["evaluation_seasons"]
    if not years or len(set(years)) != len(years):
        raise ValueError("Configure nonempty, unique seasons")
    if not targets or len(set(targets)) != len(targets) or not set(targets) <= set(years):
        raise ValueError("evaluation_seasons must be nonempty, unique configured seasons")
    # Historical labels alone are insufficient if date windows overlap.
    specs.sort(key=lambda s: s.season)
    if any(a.until > b.cutoff for a, b in zip(specs, specs[1:])):
        raise ValueError("Configured season windows must be chronological and nonoverlapping")
    return specs, sorted(targets)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def fetch_tournament_results(spec, path):
    """Explicit, read-only API request; preserve the response in a new local cache."""
    import os
    import requests
    from dotenv import load_dotenv

    path = Path(path)
    if path.exists():
        raise FileExistsError(f"Results cache already exists: {path}")
    load_dotenv()
    key = os.getenv("CBBD_API_KEY")
    if not key:
        raise ValueError("--fetch-results requires CBBD_API_KEY in the environment or .env")
    response = requests.get(
        "https://api.collegebasketballdata.com/games",
        params={"season": spec.season, "seasonType": "postseason", "tournament": "NCAA"},
        headers={"Authorization": f"Bearer {key}"}, timeout=45,
    )
    response.raise_for_status()
    rows = response.json()
    if not isinstance(rows, list) or not rows or any(not isinstance(row, dict) for row in rows):
        raise ValueError(f"Expected a nonempty list of games for {spec.season}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted(set().union(*(row.keys() for row in rows))))
        writer.writeheader()
        writer.writerows(rows)
    write_json(path.with_suffix(".source.json"), {
        "url": response.url, "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "sha256": digest(path), "rows": len(rows),
    })


def _integer(value, field, low=0, high=None):
    number = float(value)
    if not np.isfinite(number) or not number.is_integer() or number < low or (high is not None and number > high):
        raise ValueError(f"Invalid {field}: {value!r}")
    return int(number)


def load_tournament_results(path, spec):
    """Read CBBD /games CSVs by provider IDs, retaining First Four games."""
    records, registry, seeds = {}, {}, {}
    excluded = Counter()
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        for line, row in enumerate(csv.DictReader(handle), 2):
            try:
                if _integer(row["season"], "season") != spec.season:
                    raise ValueError("Tournament row belongs to a different season")
                if row.get("tournament", "NCAA").strip().casefold() != "ncaa":
                    excluded["non_ncaa"] += 1
                    continue
                start = utc_timestamp(row["startDate"])
                if not spec.cutoff <= start < spec.until:
                    raise ValueError("Tournament game is outside the configured held-out window")
                status = row.get("status", "").strip().lower()
                if status and status not in ("final", "completed"):
                    excluded["unfinished"] += 1
                    continue
                a = _provider_id(row.get("homeId", row.get("homeTeamId", "")))
                b = _provider_id(row.get("awayId", row.get("awayTeamId", "")))
                if a == b:
                    raise ValueError("Tournament participants must differ")
                sa = _integer(row["homePoints"], "homePoints")
                sb = _integer(row["awayPoints"], "awayPoints")
                if sa == sb:
                    raise ValueError("Tournament game must have a completed, non-tied score")
                home = 0 if _boolean(row["neutralSite"], "neutralSite") else 1
                for team, name, seed in ((a, row["homeTeam"], row["homeSeed"]),
                                         (b, row["awayTeam"], row["awaySeed"])):
                    name = name.strip()
                    seed = _integer(seed, "seed", 1, 16)
                    if not name or (team in registry and registry[team] != name):
                        raise ValueError(f"Missing or conflicting name for {team}")
                    if team in seeds and seeds[team] != seed:
                        raise ValueError(f"Conflicting seed for {team}")
                    registry[team], seeds[team] = name, seed
                # Stable orientation independent of the game outcome and file ordering.
                if a > b:
                    a, b, sa, sb, home = b, a, sb, sa, -home
                game = Game(_provider_id(row.get("id", row.get("gameId", ""))),
                            spec.season, start, a, b, sa, sb, home, "NCAA")
                if game.game_id in records:
                    if records[game.game_id] != game:
                        raise ValueError("Conflicting duplicate tournament game")
                    excluded["duplicate"] += 1
                records[game.game_id] = game
            except (ValueError, KeyError, TypeError) as error:
                raise ValueError(f"{path}:{line}: {error}") from error
    games = sorted(records.values(), key=lambda g: (g.start, g.game_id))
    if len(games) != spec.expected_games:
        raise ValueError(f"{spec.season}: expected {spec.expected_games} unique completed NCAA games, found {len(games)}")
    return games, registry, seeds, dict(excluded)


def audit_coverage(dataset, games, registry, min_games=10):
    counts = Counter(t for g in dataset.games for t in (g.team_a, g.team_b))
    graph = {t: set() for t in counts}
    for game in dataset.games:
        graph[game.team_a].add(game.team_b)
        graph[game.team_b].add(game.team_a)
    component, sizes = {}, []
    for team in graph:
        if team in component:
            continue
        pending, size = [team], 0
        component[team] = len(sizes)
        while pending:
            current = pending.pop()
            size += 1
            for other in graph[current]:
                if other not in component:
                    component[other] = len(sizes)
                    pending.append(other)
        sizes.append(size)
    missing = sorted(set(registry) - counts.keys())
    shared_ids = sorted({g.game_id for g in games} & {g.game_id for g in dataset.games})
    return {
        "loader": dataset.report, "training_games": len(dataset.games),
        "training_teams": len(counts), "tournament_games": len(games),
        "tournament_teams": len(registry), "missing_tournament_teams": missing,
        "overlapping_game_ids": shared_ids, "connected_component_sizes": sorted(sizes, reverse=True),
        "min_games_warning_threshold": min_games,
        "teams_below_min_games": sum(n < min_games for n in counts.values()),
        "tournament_team_coverage": [
            {"team_id": t, "team": registry[t], "training_games": counts[t],
             "component": component.get(t), "below_min_games": counts[t] < min_games}
            for t in sorted(registry)
        ],
        "all_team_coverage": [
            {"team_id": t, "team": dataset.teams[t], "training_games": counts[t],
             "component": component[t], "below_min_games": counts[t] < min_games}
            for t in sorted(counts)
        ],
        "limitations": "Counts describe supplied caches, not complete Division I schedules; opponent schedules and division membership are not verified.",
    }


def load_matchup_scores(path, specs):
    """Require explicit pre-cutoff, earlier-trained provenance for ensemble scores."""
    by_year = {spec.season: spec for spec in specs}
    scores = {}
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"season", "team_id", "ensemble_score", "as_of", "trained_through_season"}
        if not required <= set(reader.fieldnames or []):
            raise ValueError(f"Matchup scores require columns: {sorted(required)}")
        for line, row in enumerate(reader, 2):
            try:
                year = _integer(row["season"], "season", 1)
                if year not in by_year:
                    continue
                team = row["team_id"].strip()
                if not team.startswith("cbbd:") or _provider_id(team[5:]) != team:
                    raise ValueError("team_id must be a canonical cbbd:<ID>")
                value = float(row["ensemble_score"])
                if not np.isfinite(value):
                    raise ValueError("ensemble_score must be finite")
                if utc_timestamp(row["as_of"]) >= by_year[year].cutoff:
                    raise ValueError("Matchup score must be available strictly before the season cutoff")
                if _integer(row["trained_through_season"], "trained_through_season", 1) >= year:
                    raise ValueError("Matchup score producer must be trained only on earlier seasons")
                key = (year, team)
                if key in scores:
                    raise ValueError("Duplicate season/team matchup score")
                scores[key] = value
            except (ValueError, KeyError, TypeError) as error:
                raise ValueError(f"{path}:{line}: {error}") from error
    return scores


def probability_metrics(outcomes, probabilities, bins=10):
    positive_integer(bins, "bins")
    y, p = np.asarray(outcomes, dtype=float), np.asarray(probabilities, dtype=float)
    if y.ndim != 1 or p.shape != y.shape or not len(y):
        raise ValueError("Need aligned nonempty prediction/outcome vectors")
    if not np.isfinite(p).all() or not np.isin(y, [0, 1]).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("Need binary outcomes and finite probabilities in [0, 1]")
    indices = np.minimum((p * bins).astype(int), bins - 1)
    reliability = []
    for index in range(bins):
        mask = indices == index
        n = int(mask.sum())
        reliability.append({
            "bin_lower": index / bins, "bin_upper": (index + 1) / bins, "n_games": n,
            "mean_probability": float(p[mask].mean()) if n else None,
            "observed_win_rate": float(y[mask].mean()) if n else None,
        })
    clipped = np.clip(p, 1e-12, 1 - 1e-12)
    metrics = {
        "n_games": len(y), "brier_score": float(np.mean((p - y) ** 2)),
        "log_loss": float(-np.mean(y * np.log(clipped) + (1 - y) * np.log1p(-clipped))),
        "accuracy": float(np.mean((p >= .5) == y)),
        "ece": float(sum(b["n_games"] * abs(b["mean_probability"] - b["observed_win_rate"])
                         for b in reliability if b["n_games"]) / len(y)),
    }
    return metrics, reliability


MODELS = ("bayesian", "seed", "matchup")


def summarize_predictions(rows, bins):
    metrics, calibration = {}, []
    for scope in [str(year) for year in sorted({r["season"] for r in rows})] + ["pooled"]:
        selected = rows if scope == "pooled" else [r for r in rows if str(r["season"]) == scope]
        metrics[scope] = {}
        for model in MODELS:
            available = [r for r in selected if r[f"{model}_probability"] is not None]
            if not available:
                metrics[scope][model] = {"n_games": 0, "unavailable_games": len(selected)}
                continue
            result, reliability = probability_metrics(
                [r["outcome_a"] for r in available], [r[f"{model}_probability"] for r in available], bins)
            metrics[scope][model] = dict(result, unavailable_games=len(selected) - len(available))
            calibration.extend(dict(scope=scope, model=model, **b) for b in reliability)
    comparisons = {}
    for baseline in ("seed", "matchup"):
        available = [r for r in rows if r[f"{baseline}_probability"] is not None]
        result = {"n_games": len(available), "excluded_games": len(rows) - len(available),
                  "game_keys": [[r["season"], r["game_id"]] for r in available]}
        if available:
            y = [r["outcome_a"] for r in available]
            bayes, _ = probability_metrics(y, [r["bayesian_probability"] for r in available], bins)
            other, _ = probability_metrics(y, [r[f"{baseline}_probability"] for r in available], bins)
            result.update(bayesian=bayes, baseline=other,
                          brier_difference=bayes["brier_score"] - other["brier_score"],
                          log_loss_difference=bayes["log_loss"] - other["log_loss"])
        comparisons[f"bayesian_vs_{baseline}"] = result
    return {"metrics": metrics, "comparisons": comparisons}, calibration


def baseline_models(spec, loaded, scores):
    seed_x, seed_y, matchup_x, matchup_y = [], [], [], []
    years = []
    missing_scores = 0
    for previous in sorted(loaded):
        data = loaded[previous]
        if previous >= spec.season or data["spec"].until > spec.cutoff:
            continue
        years.append(previous)
        for g in data["games"]:
            dseed = data["seeds"][g.team_b] - data["seeds"][g.team_a]
            outcome = int(g.score_a > g.score_b)
            seed_x.append([dseed])
            seed_y.append(outcome)
            if (previous, g.team_a) in scores and (previous, g.team_b) in scores:
                matchup_x.append([scores[previous, g.team_a] - scores[previous, g.team_b], dseed])
                matchup_y.append(outcome)
            else:
                missing_scores += 1
    seed_model = fit_symmetric_matchup(seed_x, seed_y) if seed_x else None
    # The existing model requires >=20 symmetric rows (=10 original games).
    model = fit_symmetric_matchup(matchup_x, matchup_y) if len(matchup_x) >= 10 else None
    provenance = {
        "training_seasons": years, "seed_training_games": len(seed_x),
        "seed_coefficient": float(seed_model.coef_[0, 0]) if seed_model is not None else None,
        "matchup_training_games": len(matchup_x), "matchup_missing_training_scores": missing_scores,
        "matchup_coefficients": model.coef_[0].tolist() if model is not None else None,
        "seed_status": "fitted on earlier tournaments" if seed_model is not None else "unavailable: no earlier configured tournament",
        "matchup_status": "fitted on earlier tournaments" if model is not None else "unavailable: fewer than 10 earlier games with supplied ensemble scores",
    }
    return seed_model, model, provenance


def render_report(summary, audits, baseline_info):
    lines = ["# Bayesian tournament probability evaluation", "",
             "This evaluates individual held-out NCAA game probabilities, including First Four games. "
             "It does not evaluate championship or advancement probabilities.", "",
             "## Results", "",
             "| Season | Model | Games | Missing | Brier ↓ | Log loss ↓ | Accuracy ↑ | ECE ↓ |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for season, models in summary["metrics"].items():
        for model, values in models.items():
            stats = " | ".join(f"{values[k]:.4f}" for k in ("brier_score", "log_loss", "accuracy", "ece")) if values["n_games"] else "— | — | — | —"
            lines.append(f"| {season} | {model} | {values['n_games']} | {values['unavailable_games']} | {stats} |")
    lines += ["", "## Comparisons on identical games", ""]
    for name, result in summary["comparisons"].items():
        baseline = name.removeprefix("bayesian_vs_")
        if not result["n_games"]:
            lines.append(f"- {baseline}: unavailable; no comparison can be made.")
            continue
        lines.append(f"- {baseline}: {result['n_games']} shared games; {result['excluded_games']} excluded. "
                     f"Bayesian minus baseline Brier = {result['brier_difference']:+.4f}; "
                     f"log loss = {result['log_loss_difference']:+.4f}. Negative differences favor Bayesian.")
    seed_result = summary["comparisons"]["bayesian_vs_seed"]
    if seed_result["n_games"]:
        directions = [seed_result[k] for k in ("brier_difference", "log_loss_difference")]
        if all(d > 0 for d in directions):
            finding = "Bayesian probabilities scored worse than the seed baseline on both pooled Brier score and log loss in this run."
        elif all(d < 0 for d in directions):
            finding = "Bayesian probabilities scored better than the seed baseline on both pooled Brier score and log loss in this run."
        else:
            finding = "Pooled Brier score and log loss do not show a consistent improvement over the seed baseline in this run."
        lines += ["", finding]
    lines += ["", "These are descriptive comparisons, not significance tests. They do not establish "
              "that the Bayesian model improves future forecasts or is ready to become the default.", "",
              "## Coverage and fitting boundaries", ""]
    for year, audit in audits.items():
        counts = [r["training_games"] for r in audit["tournament_team_coverage"]]
        lines.append(f"- {year}: {audit['training_games']} unique training games, {audit['training_teams']} teams, "
                     f"{audit['tournament_games']} tournament games, {audit['tournament_teams']} entrants. "
                     f"Entrant training-game counts: {min(counts)}–{max(counts)}. "
                     f"{audit['teams_below_min_games']} fitted teams have fewer than "
                     f"{audit['min_games_warning_threshold']} observed games.")
    lines += ["", "## Baseline provenance", ""]
    for year, info in baseline_info.items():
        lines.append(f"- {year}: earlier tournament seasons {info['training_seasons']}; "
                     f"seed: {info['seed_status']}; matchup: {info['matchup_status']}.")
    lines += ["", "## Calibration", "",
              "`calibration.csv` contains equal-width probability bins, counts, mean predicted "
              "probability and observed win rate for every season and pooled model. Empty bins "
              "are retained. ECE weights absolute bin gaps by game count; its value depends on "
              "binning and sample size and is not a calibration guarantee. Probabilities and outcomes "
              "refer to team A, ordered by provider ID independently of the outcome.", "",
              "## Limitations and interpretation", "",
              "- Regular-season caches focus on tournament entrants. Opponent schedules may be incomplete; "
              "Division I membership and full-league coverage are not established.",
              "- Training uses regular-season games strictly before each explicit cutoff. Start times "
              "do not establish completion or historical data availability; later corrections remain possible.",
              "- Bayesian uncertainty uses a Laplace approximation and fixed priors. Probability credible "
              "intervals are saved but interval coverage is not evaluated against binary game outcomes.",
              "- Seed coefficients and the existing score-plus-seed logistic model are trained only on "
              "earlier configured tournaments. Future-season results are never baseline training data.",
              "- Existing matchup comparison requires actual ensemble scores with declared pre-cutoff, "
              "earlier-trained provenance. Declarations cannot independently prove upstream feature or "
              "ensemble-selection provenance. Missing scores are reported; no scores are invented.",
              "- Missing baseline predictions change the comparison sample. Use the saved identical-game "
              "comparisons rather than comparing model-wide metrics with different coverage.",
              "- Games within a tournament share teams and are dependent. Multiple seasons, uncertainty "
              "analysis and settings fixed before evaluation are needed for stronger conclusions.",
              "- Accuracy chooses team A at exactly 0.5. Log loss clips probabilities to [1e-12, 1-1e-12].",
              "", "## Artifacts", "",
              "`predictions.csv`, `metrics.json`, `calibration.csv`, `coverage.json`, `baseline_training.json`, "
              "`run.json`, and per-season posterior NPZ artifacts preserve the inputs, fitted state, "
              "predictions and diagnostics. `run.json` records file hashes and configuration.", ""]
    return "\n".join(lines)


def run_evaluation(config_path, output_dir, *, data_dir=Path("data"), results_dir=None,
                   matchup_scores=None, fetch_results=False, draws=4000, seed=42, bins=10,
                   min_games=10, rating_prior_sd=1.5, home_prior_sd=1.0):
    specs, targets = load_config(config_path)
    positive_integer(draws, "draws")
    positive_integer(bins, "bins")
    positive_integer(min_games, "min_games")
    if draws < 2:
        raise ValueError("At least two posterior draws are required")
    if any(not np.isfinite(sd) or sd <= 0 for sd in (rating_prior_sd, home_prior_sd)):
        raise ValueError("Prior standard deviations must be positive and finite")
    data_dir, output_dir = Path(data_dir), Path(output_dir)
    results_dir = Path(results_dir) if results_dir is not None else data_dir
    scores = load_matchup_scores(matchup_scores, specs) if matchup_scores else {}
    output_dir.mkdir(parents=True, exist_ok=False)
    loaded, audits, inputs = {}, {}, []
    for spec in specs:
        path = results_dir / str(spec.season) / "tournament_games.csv"
        if not path.exists():
            if not fetch_results:
                raise ValueError(f"Missing {path}; supply a CBBD /games CSV or use --fetch-results")
            fetch_tournament_results(spec, path)
        games, registry, seeds, excluded = load_tournament_results(path, spec)
        loaded[spec.season] = dict(spec=spec, games=games, registry=registry, seeds=seeds)
        inputs.append({"path": str(path.resolve()), "sha256": digest(path)})
        if spec.season in targets:
            dataset = load_games(data_dir, spec.season, spec.cutoff)
            audit = audit_coverage(dataset, games, registry, min_games)
            audit["result_exclusions"] = excluded
            audits[str(spec.season)] = audit
            loaded[spec.season]["dataset"] = dataset
            inputs.extend({"path": str(p.resolve()), "sha256": digest(p)}
                          for p in sorted((data_dir / str(spec.season)).glob("*/*_game_stats.csv")))
    write_json(output_dir / "coverage.json", audits)
    for year, audit in audits.items():
        if audit["missing_tournament_teams"] or audit["overlapping_game_ids"]:
            raise ValueError(f"{year}: missing entrant ratings or overlapping training/evaluation game IDs; see coverage.json")
    rows, baseline_info = [], {}
    for spec in specs:
        if spec.season not in targets:
            continue
        data = loaded[spec.season]
        posterior = fit_ratings(data["dataset"], season=spec.season, cutoff=spec.cutoff,
                                n_draws=draws, seed=seed, rating_prior_sd=rating_prior_sd,
                                home_prior_sd=home_prior_sd)
        save_ratings(posterior, output_dir / f"ratings_{spec.season}.npz")
        write_json(output_dir / f"ratings_{spec.season}.summary.json", posterior.summary())
        seed_model, model, info = baseline_models(spec, loaded, scores)
        baseline_info[str(spec.season)] = info
        for game in data["games"]:
            venue = {0: "neutral", 1: "a_home", -1: "b_home"}[game.home]
            prediction = posterior.predict(game.team_a, game.team_b, venue=venue)
            dseed = data["seeds"][game.team_b] - data["seeds"][game.team_a]
            seed_p = float(seed_model.predict_proba([[dseed]])[0, 1]) if seed_model is not None else None
            score_a, score_b = scores.get((spec.season, game.team_a)), scores.get((spec.season, game.team_b))
            matchup_p, reason = None, ""
            if model is None:
                reason = info["matchup_status"]
            elif score_a is None or score_b is None:
                reason = "missing target-season ensemble score"
            else:
                matchup_p = float(matchup_probability(score_a - score_b, dseed, model))
            rows.append({
                "season": spec.season, "game_id": game.game_id, "start": game.start.isoformat(),
                "cutoff": spec.cutoff.isoformat(), "team_a_id": game.team_a, "team_b_id": game.team_b,
                "team_a": data["registry"][game.team_a], "team_b": data["registry"][game.team_b],
                "seed_a": data["seeds"][game.team_a], "seed_b": data["seeds"][game.team_b],
                "score_a": game.score_a, "score_b": game.score_b, "venue": venue,
                "outcome_a": int(game.score_a > game.score_b),
                "bayesian_probability": prediction["win_probability_a"],
                "bayesian_interval_lower": prediction["probability_interval_90"][0],
                "bayesian_interval_upper": prediction["probability_interval_90"][1],
                "seed_probability": seed_p, "matchup_probability": matchup_p,
                "seed_unavailable_reason": "" if seed_model is not None else info["seed_status"],
                "matchup_unavailable_reason": reason,
            })
    summary, calibration = summarize_predictions(rows, bins)
    write_csv(output_dir / "predictions.csv", rows)
    write_csv(output_dir / "calibration.csv", calibration)
    write_json(output_dir / "metrics.json", summary)
    write_json(output_dir / "baseline_training.json", baseline_info)
    if matchup_scores:
        inputs.append({"path": str(Path(matchup_scores).resolve()), "sha256": digest(matchup_scores)})
    import platform
    import scipy
    import sklearn

    write_json(output_dir / "run.json", {
        "schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
        "config": json.loads(Path(config_path).read_text()), "config_sha256": digest(config_path),
        "draws": draws, "seed": seed, "bins": bins, "min_games": min_games,
        "rating_prior_sd": rating_prior_sd, "home_prior_sd": home_prior_sd,
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "scipy": scipy.__version__, "scikit_learn": sklearn.__version__},
        "inputs": inputs,
    })
    (output_dir / "report.md").write_text(render_report(summary, audits, baseline_info))
    return summary
