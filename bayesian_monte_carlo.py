"""Optional Bayesian bracket Monte Carlo and forecast-contract export.

Every simulated tournament uses ONE joint posterior draw of all team ratings,
shared by all of its games. Ratings are never drawn independently per team or
redrawn per game. The bracket comes from an explicit pre-tournament schedule:
later rounds reference source slots, never realized matchups or winners.

The exported document follows the forecast JSON contract 1.0.0 defined by
``backend_data.models.ForecastDocument`` on main. That package is validated
against when importable; this module does not depend on it.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from uuid import uuid4

import numpy as np
from scipy.special import expit

from bayesian_ratings import positive_integer, utc_timestamp

SCHEMA_VERSION = "1.0.0"
COMPETITION = "Men's college basketball"  # same constant as backend_data.kalshi on main
METHOD = "bayesian_bradley_terry_joint_draw_monte_carlo"
SEED_ORDER = (1, 16, 8, 9, 5, 12, 4, 13, 6, 11, 3, 14, 7, 10, 2, 15)
_TOP_KEYS = {"season", "as_of", "source", "notes", "games", "regions", "final_four"}
_GAME_KEYS = {"slot", "round_number", "participants", "play_in", "round_points", "scheduled_start_utc"}
_IDENTIFIER = re.compile(r"^[A-Za-z0-9_.:-]+$")


@dataclass(frozen=True)
class ScheduledGame:
    slot: int
    round_number: int
    participants: tuple  # two of ("team", entrant index) or ("slot", zero-based source game index)
    play_in: bool
    round_points: float
    scheduled_start: datetime | None


@dataclass(frozen=True)
class BracketSchedule:
    season: int
    as_of: datetime
    source: str | None
    team_refs: tuple  # provider IDs or exact names, in order of entry
    seeds: dict  # entrant index -> seed, where supplied
    games: tuple
    main_rounds: int
    sha256: str | None = None


def _only(mapping, allowed, where):
    if not isinstance(mapping, dict):
        raise ValueError(f"{where} must be a JSON object")
    extra = sorted(set(mapping) - allowed)
    if extra:
        raise ValueError(f"{where} has unsupported fields {extra}. A pre-tournament schedule lists scheduled "
                         "slots only; winners, scores and other results are not accepted.")


def _integer(value, name):
    positive_integer(value, name)
    return int(value)


def _expand_regions(regions, final_four):
    """Expand four seeded regions into explicit games; a two-team seed entry is a play-in."""
    if (not isinstance(final_four, list) or len(final_four) != 2
            or any(not isinstance(pair, list) or len(pair) != 2 for pair in final_four)):
        raise ValueError("final_four must be two pairs of region names, e.g. [[\"East\", \"West\"], [\"South\", \"Midwest\"]]")
    order = [name for pair in final_four for name in pair]
    if not isinstance(regions, dict) or len(set(order)) != 4 or set(order) != set(regions):
        raise ValueError("regions must contain exactly the four regions named once each in final_four")
    for name in order:
        if not isinstance(regions[name], dict) or set(regions[name]) != {str(seed) for seed in SEED_ORDER}:
            raise ValueError(f"Region {name!r} must map every seed \"1\" to \"16\" to a team (or two play-in teams)")
    play_ins = any(isinstance(entry, list) for name in order for entry in regions[name].values())
    games = []

    def add(round_number, participants, play_in=False):
        games.append({"slot": len(games) + 1, "round_number": round_number,
                      "participants": participants, "play_in": play_in})
        return {"source_slot": len(games)}

    live = {}
    for name in order:
        live[name] = []
        for seed in SEED_ORDER:
            entry = regions[name][str(seed)]
            if isinstance(entry, list):
                if len(entry) != 2:
                    raise ValueError(f"Play-in entry for {name} seed {seed} needs exactly two teams")
                entry = add(1, [{"team": team, "seed": seed} for team in entry], play_in=True)
            else:
                entry = {"team": entry, "seed": seed}
            live[name].append(entry)
    first = 2 if play_ins else 1
    for depth in range(4):
        for name in order:
            live[name] = [add(first + depth, live[name][i:i + 2]) for i in range(0, len(live[name]), 2)]
    add(first + 5, [add(first + 4, [live[a][0], live[b][0]]) for a, b in final_four])
    return games


def parse_bracket_schedule(raw, *, sha256=None):
    """Validate an explicit pre-tournament schedule and return a BracketSchedule.

    Supply either ``games`` (explicit slots) or ``regions`` + ``final_four``
    (standard seeded regions). ``as_of`` records when the bracket was captured.
    """
    _only(raw, _TOP_KEYS, "Bracket schedule")
    season = _integer(raw.get("season"), "Schedule season")
    if "as_of" not in raw:
        raise ValueError("Schedule needs as_of: the timezone-aware time the pre-tournament bracket was captured")
    as_of = utc_timestamp(raw["as_of"])
    if ("games" in raw) == ("regions" in raw):
        raise ValueError("Supply exactly one of games or regions")
    if "regions" in raw:
        raw_games = _expand_regions(raw["regions"], raw.get("final_four"))
    elif "final_four" in raw:
        raise ValueError("final_four applies only to the regions form")
    else:
        raw_games = raw["games"]
    if not isinstance(raw_games, list) or not raw_games or any(not isinstance(game, dict) for game in raw_games):
        raise ValueError("games must be a nonempty list of objects")
    try:
        raw_games = sorted(raw_games, key=lambda game: _integer(game.get("slot"), "slot"))
    except ValueError as error:
        raise ValueError(f"Every game needs a positive integer slot: {error}") from error

    refs, seeds, parsed, consumed, rounds = [], {}, [], set(), {}
    for position, item in enumerate(raw_games, 1):
        where = f"Game slot {item['slot']}"
        _only(item, _GAME_KEYS, where)
        if item["slot"] != position:
            raise ValueError("Game slots must be 1..N without gaps or duplicates")
        round_number = _integer(item.get("round_number"), f"{where} round_number")
        play_in = item.get("play_in", False)
        if not isinstance(play_in, bool):
            raise ValueError(f"{where} play_in must be true or false")
        entries = item.get("participants")
        if not isinstance(entries, list) or len(entries) != 2:
            raise ValueError(f"{where} needs exactly two participants")
        participants = []
        for entry in entries:
            _only(entry, {"team", "seed", "source_slot"}, f"{where} participant")
            if ("team" in entry) == ("source_slot" in entry):
                raise ValueError(f"{where} participant must reference exactly one team or source_slot")
            if "team" in entry:
                ref = entry["team"].strip() if isinstance(entry["team"], str) else ""
                if not ref:
                    raise ValueError(f"{where} team must be a provider ID or exact team name")
                if ref in refs:
                    raise ValueError(f"{ref!r} enters the bracket more than once. Later rounds must reference "
                                     "source slots, not realized matchups.")
                if "seed" in entry:
                    seed = _integer(entry["seed"], f"{where} seed")
                    if seed > 16:
                        raise ValueError(f"{where} seed must be between 1 and 16")
                    seeds[len(refs)] = seed
                participants.append(("team", len(refs)))
                refs.append(ref)
            else:
                if "seed" in entry:
                    raise ValueError(f"{where}: seed applies only to team participants")
                source = entry["source_slot"]
                if play_in:
                    raise ValueError(f"{where} is a play-in game and must list two teams")
                if isinstance(source, bool) or not isinstance(source, int) or not 1 <= source < position:
                    raise ValueError(f"{where} source_slot must reference an earlier slot")
                if source in consumed:
                    raise ValueError(f"Slot {source} feeds more than one game")
                if rounds[source] >= round_number:
                    raise ValueError(f"{where} must be in a later round than its source slot {source}")
                consumed.add(source)
                participants.append(("slot", source - 1))
        start = item.get("scheduled_start_utc")
        if start is not None:
            start = utc_timestamp(start)
            if start < as_of:
                raise ValueError(f"{where} starts before the schedule's as_of time; a pre-tournament schedule "
                                 "cannot be captured after its games begin")
        points = item.get("round_points")
        if points is not None and (isinstance(points, bool) or not isinstance(points, (int, float))
                                   or not np.isfinite(points) or points < 0):
            raise ValueError(f"{where} round_points must be a finite nonnegative number")
        rounds[position] = round_number
        parsed.append((position, round_number, tuple(participants), play_in, points, start))
    if consumed != set(range(1, len(parsed))):
        raise ValueError("Schedule must be one connected single-elimination tree: every game except the "
                         "last must feed exactly one later game")
    main = [game[1] for game in parsed if not game[3]]
    if parsed[-1][3] or not main:
        raise ValueError("The final game cannot be a play-in game")
    first = min(main)
    games = tuple(ScheduledGame(slot, number, participants, play_in,
                                float(points) if points is not None else (0. if play_in else 10. * 2 ** (number - first)),
                                start)
                  for slot, number, participants, play_in, points, start in parsed)
    source = raw.get("source")
    if source is not None and not isinstance(source, str):
        raise ValueError("source must be text")
    return BracketSchedule(season, as_of, source, tuple(refs), seeds, games, max(main) - first + 1, sha256)


def load_bracket_schedule(path):
    data = Path(path).read_bytes()
    try:
        raw = json.loads(data)
    except json.JSONDecodeError as error:
        raise ValueError(f"Bracket schedule {path} is not valid JSON: {error}") from error
    return parse_bracket_schedule(raw, sha256=hashlib.sha256(data).hexdigest())


@dataclass
class BracketSimulation:
    schedule: BracketSchedule
    team_ids: list  # provider IDs in entrant order
    team_names: list
    posterior_indices: list
    draw_indices: np.ndarray  # (n_simulations,) joint posterior draw used by each tournament
    slot_winners: np.ndarray  # (games, n_simulations) entrant indices
    win_counts: np.ndarray  # (entrants, n_simulations), play-in wins excluded
    n_simulations: int
    seed: int

    def slot_probabilities(self):
        """(games, entrants) frequency that each entrant wins each slot, including reaching it."""
        entrants = len(self.team_ids)
        return np.stack([np.bincount(row, minlength=entrants) for row in self.slot_winners]) / self.n_simulations


def simulate_bracket(posterior, schedule, *, n_simulations=10000, seed=42):
    """Simulate the scheduled bracket with one joint rating draw per tournament.

    Draw indices are sampled uniformly with replacement. All games are treated
    as neutral-site, so the posterior home-advantage draws are unused.
    """
    positive_integer(n_simulations, "n_simulations")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    posterior.validate()
    if posterior.metadata.get("season") != schedule.season:
        raise ValueError(f"Ratings are for season {posterior.metadata.get('season')}, schedule is for {schedule.season}")
    cutoff = utc_timestamp(posterior.metadata["cutoff"])
    starts = [game.scheduled_start for game in schedule.games if game.scheduled_start is not None]
    if starts and min(starts) < cutoff:
        raise ValueError("The rating cutoff is later than a scheduled tournament game; fit ratings with a "
                         "cutoff at or before the first scheduled start")
    indices = [posterior.resolve_team(ref) for ref in schedule.team_refs]
    if len(set(indices)) != len(indices):
        raise ValueError("Two schedule entries resolve to the same rated team")
    names = [posterior.team_names[i] for i in indices]
    if len(set(names)) != len(names):
        raise ValueError("Bracket entrants need distinct display names")

    rng = np.random.default_rng(seed)
    draw_indices = rng.integers(0, len(posterior.home_draws), size=n_simulations)
    ratings = posterior.rating_draws[np.ix_(draw_indices, indices)]  # row s: the joint draw for tournament s
    simulations = np.arange(n_simulations)
    slot_winners = np.empty((len(schedule.games), n_simulations), dtype=np.int32)
    win_counts = np.zeros((len(indices), n_simulations), dtype=np.int32)
    for g, game in enumerate(schedule.games):
        a, b = (np.full(n_simulations, ref, dtype=np.int32) if kind == "team" else slot_winners[ref]
                for kind, ref in game.participants)
        probability = expit(ratings[simulations, a] - ratings[simulations, b])
        winners = np.where(rng.random(n_simulations) < probability, a, b)
        slot_winners[g] = winners
        if not game.play_in:
            win_counts[winners, simulations] += 1
    return BracketSimulation(schedule, [posterior.team_ids[i] for i in indices], names, indices,
                             draw_indices, slot_winners, win_counts, int(n_simulations), int(seed))


def max_expected_points_bracket(schedule, slot_probabilities):
    """Exact dynamic program for the consistent bracket with the highest expected points.

    Returns (picks, expected_points) with one entrant index per game. Ties go
    to the entrant more likely to win its previous slot, then the earlier entrant.
    """
    best, rival = [], []
    for g, game in enumerate(schedule.games):
        branches = [({ref: 0.}, None) if kind == "team" else (best[ref], ref) for kind, ref in game.participants]
        best.append({})
        rival.append({})
        for side in (0, 1):
            values, source = branches[1 - side]
            opponent = max(values, key=lambda t: (values[t], 1. if source is None else slot_probabilities[source, t], -t))
            for team, value in branches[side][0].items():
                best[g][team] = game.round_points * slot_probabilities[g, team] + value + values[opponent]
                rival[g][team] = opponent
    final = len(schedule.games) - 1
    champion = max(best[final], key=lambda t: (best[final][t], slot_probabilities[final, t], -t))
    picks = [None] * len(schedule.games)
    pending = [(final, champion)]
    while pending:
        g, team = pending.pop()
        picks[g] = int(team)
        for kind, ref in schedule.games[g].participants:
            if kind == "slot":
                pending.append((ref, team if team in best[ref] else rival[g][team]))
    return picks, float(best[final][champion])


def stable_team_id(name, competition=COMPETITION):
    """Same name-derived team identity as backend_data.export.stable_team_id on main."""
    return "team-" + hashlib.sha256(f"{competition}|{name}".encode()).hexdigest()[:24]


def _wire_time(value):
    """UTC with a Z suffix, the form main's Pydantic models serialize."""
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def file_artifact(path, media_type, description=None):
    path = Path(path)
    return {"uri": str(path), "media_type": media_type, "description": description,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def build_forecast_document(posterior, simulation, *, run_id=None, generated_at=None, artifacts=None):
    """Return a complete contract-1.0.0 forecast document as plain JSON-ready data."""
    schedule = simulation.schedule
    run_id = run_id or str(uuid4())
    generated_at = generated_at or datetime.now(timezone.utc)
    tournament_id = f"ncaa-men-{schedule.season}"
    team_ids = [stable_team_id(name) for name in simulation.team_names]
    if len(set(team_ids)) != len(team_ids):
        raise ValueError("Team identity collision")
    probabilities = simulation.slot_probabilities()
    final = len(schedule.games) - 1

    def game_id(index):
        return f"{tournament_id}-g{index + 1}"

    games = [{
        "id": game_id(g), "slot": game.slot, "round_number": game.round_number, "round_points": game.round_points,
        "participants": [{"team_id": team_ids[ref], "source_game_id": None} if kind == "team"
                         else {"team_id": None, "source_game_id": game_id(ref)} for kind, ref in game.participants],
        "scheduled_start_utc": _wire_time(game.scheduled_start) if game.scheduled_start else None,
        "actual_winner_team_id": None,  # pre-tournament export: results are never included
    } for g, game in enumerate(schedule.games)]

    play_in_slot = {ref: g for g, game in enumerate(schedule.games) if game.play_in
                    for kind, ref in game.participants}
    predictions = []
    for t, (team_id, column) in enumerate(zip(team_ids, simulation.posterior_indices)):
        wins = simulation.win_counts[t]
        draws = posterior.rating_draws[:, column]
        low, high = (float(value) for value in np.quantile(draws, [.05, .95]))
        target = "team_rating_log_odds"
        statistics = {"tree_expected_wins": float(wins.mean()), "tree_std_wins": float(wins.std()),
                      "champion_probability": float(probabilities[final, t])}
        if t in schedule.seeds:
            statistics["seed"] = float(schedule.seeds[t])
        if t in play_in_slot:
            statistics["play_in_win_probability"] = float(probabilities[play_in_slot[t], t])
        predictions.append({
            "run_id": run_id, "team_id": team_id, "statistics": statistics,
            "probabilities": {
                "success_score_threshold": [], "calibrated_success_score_threshold": [],
                "bracket_tree": [{"minimum_wins": k, "probability": float(np.mean(wins >= k))}
                                 for k in range(1, schedule.main_rounds + 1)],
            },
            "intervals": [],
            "posterior": {
                "target": target, "mean": float(draws.mean()), "median": float(np.median(draws)),
                "std": float(draws.std()),
                "intervals": [{"kind": "credible", "target": target, "level": .9, "lower": low, "upper": high,
                               "method": "laplace_posterior_draw_quantiles"}],
                "diagnostics": {"inference": "laplace", "n_draws": int(len(draws)),
                                "centered_across_fitted_teams": True},
            },
        })

    picks, expected_points = max_expected_points_bracket(schedule, probabilities)
    bracket = {
        "id": f"{run_id}-max_ev", "run_id": run_id, "tournament_id": tournament_id, "strategy": "max_ev",
        "picks": [{
            "game_id": game_id(g), "winner_team_id": team_ids[team],
            "confidence": {"value": float(probabilities[g, team]), "meaning": "probability_team_wins_slot",
                           "method": "bayesian_joint_draw_bracket_monte_carlo", "intervals": []},
            "public_pick_probability": None,
            "expected_points": float(schedule.games[g].round_points * probabilities[g, team]),
            "leverage_ratio": None, "actual_correct": None,
        } for g, team in enumerate(picks)],
        "summary": {"champion_team_id": team_ids[picks[final]], "expected_points": expected_points,
                    "pool_win_probability": None, "top_10_percent_probability": None, "actual_points": None,
                    "pool_size": None, "extensions": {}},
    }

    starts = [game.scheduled_start for game in schedule.games if game.scheduled_start is not None]
    metadata = {key: value for key, value in posterior.metadata.items() if key != "training_game_ids"}
    metadata.update({
        "status": "Experimental optional producer; not the pipeline default. See docs/bayesian-evaluation-expanded.md. "
                  "Advancement and championship probabilities have not been evaluated against outcomes.",
        "simulation": {
            "design": "One joint posterior draw of all team ratings per simulated tournament, shared by every "
                      "game in that tournament. Ratings are not drawn independently per team or redrawn per game.",
            "draw_sampling": "uniform with replacement from saved posterior draws",
            "n_posterior_draws": int(len(posterior.home_draws)),
            "distinct_draws_used": int(len(np.unique(simulation.draw_indices))),
            "venue": "every game neutral-site; home-advantage draws unused",
            "wins": "bracket_tree thresholds count main-bracket wins; play-in wins are reported separately",
            "bracket_strategy": "max_ev maximizes expected points from simulated slot-win frequencies; "
                                "no public-pick or pool-leverage model",
        },
        "bracket_schedule": {
            "kind": "explicit pre-tournament schedule; later rounds reference source slots, no results",
            "as_of": schedule.as_of.isoformat(), "source": schedule.source, "sha256": schedule.sha256,
            "games": len(schedule.games), "teams": len(team_ids),
            "play_in_games": sum(game.play_in for game in schedule.games),
            "earliest_scheduled_start": min(starts).isoformat() if starts else None,
            "cutoff_check": "rating cutoff is at or before every supplied scheduled start" if starts
                            else "not checkable: schedule has no scheduled starts",
        },
        "team_registry": {team_id: {"provider_id": provider, "name": name, "seed": schedule.seeds.get(t)}
                          for t, (team_id, provider, name)
                          in enumerate(zip(team_ids, simulation.team_ids, simulation.team_names))},
    })
    document = {
        "schema_version": SCHEMA_VERSION,
        "tournament": {"id": tournament_id, "season": schedule.season, "competition": COMPETITION,
                       "format": "single_elimination", "topology_status": "available",
                       "bracket_team_count": len(team_ids),
                       "includes_first_four": any(game.play_in for game in schedule.games)},
        "teams": sorted(({"id": team_id, "name": name} for team_id, name in zip(team_ids, simulation.team_names)),
                        key=lambda team: team["name"]),
        "games": games,
        "prediction_runs": [{
            "id": run_id, "tournament_id": tournament_id, "method": METHOD,
            "model_name": str(posterior.metadata.get("model", "bayesian_bradley_terry")),
            "model_version": str(posterior.metadata["model_version"]),
            "generated_at": _wire_time(generated_at),
            "n_simulations": simulation.n_simulations, "seed": simulation.seed,
            "source_metadata": metadata, "artifacts": list(artifacts or []),
        }],
        "team_predictions": predictions, "brackets": [bracket], "market_data": None,
    }
    validate_forecast_document(document)
    return document


def validate_forecast_document(document):
    """Validate against main's Pydantic contract when installed; always run local checks.

    Returns the name of the strongest validation performed.
    """
    json.dumps(document, allow_nan=False)  # finite numbers and JSON types only
    identifiers = [document["tournament"]["id"]]
    for key in ("teams", "games", "prediction_runs", "brackets"):
        ids = [item["id"] for item in document[key]]
        if len(set(ids)) != len(ids):
            raise ValueError(f"Duplicate {key} IDs")
        identifiers += ids
    if any(not _IDENTIFIER.fullmatch(identifier) for identifier in identifiers):
        raise ValueError("Identifiers may contain only letters, digits and _ . : -")
    if document["schema_version"] != SCHEMA_VERSION:
        raise ValueError("Unsupported schema version")
    try:
        from backend_data.models import ForecastDocument
    except ImportError:
        return "local structural checks only (backend_data contract package is not installed)"
    ForecastDocument.model_validate(document)
    return "backend_data.models.ForecastDocument"


def save_forecast_document(document, output_dir):
    """Publish forecast_{season}_{uuid}.json atomically; never replace an existing file."""
    validate_forecast_document(document)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"forecast_{document['tournament']['season']}_{uuid4()}.json"
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=output_dir, delete=False) as handle:
            temporary = Path(handle.name)
            json.dump(document, handle, indent=2, allow_nan=False)
            handle.write("\n")
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return path
