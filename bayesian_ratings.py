"""Bayesian Bradley–Terry ratings with a Gaussian (Laplace) posterior approximation.

This is an independent game-level baseline, not a Ridge score or SHAP feature.
Team priors are N(0, rating_prior_sd²); home advantage is N(0, home_prior_sd²).
"""

import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np
from scipy.linalg import cholesky, solve_triangular
from scipy.optimize import minimize
from scipy.sparse import csr_matrix
from scipy.special import expit


def utc_timestamp(value):
    result = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if result.tzinfo is None or result.utcoffset() is None:
        raise ValueError("Timestamps must include a timezone, e.g. 2025-03-18T00:00:00Z")
    return result.astimezone(timezone.utc)


def positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True)
class Game:
    game_id: str
    season: int
    start: datetime
    team_a: str
    team_b: str
    score_a: int
    score_b: int
    home: int  # +1 if A is home, -1 if B is home, 0 if neutral
    tournament: str = ""


@dataclass
class GameDataset:
    games: list[Game]
    teams: dict[str, str]
    report: dict


def _boolean(value, field):
    if str(value).strip().lower() not in ("true", "false"):
        raise ValueError(f"{field} must be True or False")
    return str(value).strip().lower() == "true"


def _provider_id(value):
    text = str(value).strip()
    if not text.isdecimal():
        raise ValueError(f"Expected integer provider ID, got {text!r}")
    return f"cbbd:{int(text)}"


def load_games(data_dir, season, cutoff, *, regular_only=True):
    """Read team-perspective CBBD caches, deduplicating opposite perspectives.

    Only starts strictly before cutoff are included. Default: regular season.
    All opponents in those files are retained, including non-tournament teams.
    """
    positive_integer(season, "season")
    cutoff = utc_timestamp(cutoff)
    files = sorted((Path(data_dir) / str(season)).glob("*/*_game_stats.csv"))
    if not files:
        raise ValueError(f"No team game CSVs found for season {season}")
    games, teams = {}, {}
    report = dict(files=len(files), rows_read=0, excluded_season=0, excluded_type=0,
                  excluded_cutoff=0, excluded_unfinished=0, duplicates=0)
    required = {"gameId", "season", "seasonType", "startDate", "teamId", "team",
                "opponentId", "opponent", "neutralSite", "isHome",
                "teamStats.points.total", "opponentStats.points.total"}
    for path in files:
        with path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            if not required <= set(reader.fieldnames or []):
                raise ValueError(f"Missing game columns in {path}: {sorted(required - set(reader.fieldnames or []))}")
            for line, row in enumerate(reader, 2):
                report["rows_read"] += 1
                try:
                    if int(row["season"]) != season:
                        report["excluded_season"] += 1
                        continue
                    if regular_only and row["seasonType"].strip().lower() != "regular":
                        report["excluded_type"] += 1
                        continue
                    start = utc_timestamp(row["startDate"])
                    if start >= cutoff:
                        report["excluded_cutoff"] += 1
                        continue
                    status = row.get("status", "").strip().lower()
                    scores = [row[key].strip() for key in ("teamStats.points.total", "opponentStats.points.total")]
                    if (status and status not in ("final", "completed")) or any(not value for value in scores):
                        report["excluded_unfinished"] += 1
                        continue
                    numbers = [float(value) for value in scores]
                    if any(not np.isfinite(value) or value < 0 or not value.is_integer() for value in numbers):
                        raise ValueError("Scores must be finite nonnegative integers")
                    if numbers == [0., 0.]:
                        report["excluded_unfinished"] += 1
                        continue
                    if numbers[0] == numbers[1]:
                        raise ValueError("A completed basketball game cannot be tied")
                    ids = [_provider_id(row[key]) for key in ("teamId", "opponentId")]
                    names = [row[key].strip() for key in ("team", "opponent")]
                    if ids[0] == ids[1] or not all(names):
                        raise ValueError("Game needs two different teams with names")
                    for team_id, name in zip(ids, names):
                        if team_id in teams and teams[team_id] != name:
                            raise ValueError(f"Conflicting names for {team_id}: {teams[team_id]!r} / {name!r}")
                        teams[team_id] = name
                    home = 0 if _boolean(row["neutralSite"], "neutralSite") else (1 if _boolean(row["isHome"], "isHome") else -1)
                    if ids[0] > ids[1]:
                        ids.reverse()
                        numbers.reverse()
                        home = -home
                    game = Game(_provider_id(row["gameId"]), season, start, *ids,
                                int(numbers[0]), int(numbers[1]), home, row.get("tournament", "").strip())
                    if game.game_id in games:
                        if games[game.game_id] != game:
                            raise ValueError(f"Conflicting duplicate game {game.game_id}")
                        report["duplicates"] += 1
                    games[game.game_id] = game
                except (ValueError, TypeError) as error:
                    raise ValueError(f"{path}:{line}: {error}") from error
    ordered = sorted(games.values(), key=lambda game: (game.start, game.game_id))
    if not ordered:
        raise ValueError("No completed eligible games before cutoff")
    report["games"] = len(ordered)
    report["coverage_note"] = "Only games present in the supplied caches; opponents may have incomplete schedules."
    return GameDataset(ordered, dict(sorted(teams.items())), report)


@dataclass
class RatingPosterior:
    team_ids: list[str]
    team_names: list[str]
    rating_draws: np.ndarray  # joint draws, centered across teams within each draw
    home_draws: np.ndarray
    metadata: dict

    def validate(self):
        n = len(self.team_ids)
        if n < 2 or len(set(self.team_ids)) != n or len(self.team_names) != n:
            raise ValueError("Invalid team registry")
        if self.rating_draws.ndim != 2 or self.rating_draws.shape[1] != n or self.rating_draws.shape[0] < 2:
            raise ValueError("Invalid posterior rating dimensions")
        if self.home_draws.shape != (self.rating_draws.shape[0],):
            raise ValueError("Invalid posterior home dimensions")
        if not np.isfinite(self.rating_draws).all() or not np.isfinite(self.home_draws).all():
            raise ValueError("Posterior draws must be finite")
        if not np.allclose(self.rating_draws.mean(axis=1), 0, atol=1e-10):
            raise ValueError("Rating draws must be centered")
        if self.metadata.get("model_version") != 1 or self.metadata.get("inference") != "laplace":
            raise ValueError("Unsupported rating artifact version or inference method")
        utc_timestamp(self.metadata["cutoff"])
        return self

    def resolve_team(self, team):
        if team in self.team_ids:
            return self.team_ids.index(team)
        matches = [i for i, name in enumerate(self.team_names) if name.casefold() == str(team).strip().casefold()]
        if len(matches) != 1:
            raise ValueError(f"Unknown or ambiguous team {team!r}; use a provider ID from the artifact")
        return matches[0]

    def matchup_draws(self, team_a, team_b, *, venue="neutral"):
        a, b = self.resolve_team(team_a), self.resolve_team(team_b)
        if a == b:
            raise ValueError("Matchup requires two different teams")
        if venue not in ("neutral", "a_home", "b_home"):
            raise ValueError("venue must be neutral, a_home or b_home")
        home = {"neutral": 0, "a_home": 1, "b_home": -1}[venue]
        return expit(self.rating_draws[:, a] - self.rating_draws[:, b] + home * self.home_draws)

    def predict(self, team_a, team_b, *, venue="neutral"):
        probabilities = self.matchup_draws(team_a, team_b, venue=venue)
        return {"team_a": self.team_ids[self.resolve_team(team_a)], "team_b": self.team_ids[self.resolve_team(team_b)],
                "venue": venue, "win_probability_a": float(probabilities.mean()),
                "probability_interval_90": np.quantile(probabilities, [.05, .95]).tolist(),
                "interval_meaning": "approximate credible interval for the win probability, not game outcome"}

    def probability_matrix(self, teams, *, draw_index=None):
        """Neutral-site matrix in caller order, usable by simulate_bracket_tree.

        For shared strength uncertainty in a tournament use one draw_index for
        the entire tournament, rather than drawing independently for each game.
        """
        indices = [self.resolve_team(team) for team in teams]
        if len(set(indices)) != len(indices) or not indices:
            raise ValueError("Supply a nonempty list of unique teams")
        if draw_index is not None:
            if isinstance(draw_index, bool) or not isinstance(draw_index, (int, np.integer)) or not 0 <= draw_index < len(self.home_draws):
                raise ValueError("draw_index is outside the posterior")
            ratings = self.rating_draws[draw_index, indices]
            return expit(ratings[:, None] - ratings[None, :])
        matrix = np.full((len(indices), len(indices)), .5)
        for a in range(len(indices)):
            for b in range(a + 1, len(indices)):
                p = expit(self.rating_draws[:, indices[a]] - self.rating_draws[:, indices[b]]).mean()
                matrix[a, b], matrix[b, a] = p, 1 - p
        return matrix

    def summary(self):
        records = []
        for i, (team_id, name) in enumerate(zip(self.team_ids, self.team_names)):
            values = self.rating_draws[:, i]
            records.append({"team_id": team_id, "team": name, "rating_mean": float(values.mean()),
                            "rating_std": float(values.std()),
                            "approximate_credible_interval_90": np.quantile(values, [.05, .95]).tolist()})
        return {"metadata": self.metadata, "teams": sorted(records, key=lambda row: -row["rating_mean"]),
                "home_advantage_log_odds": {"mean": float(self.home_draws.mean()),
                    "approximate_credible_interval_90": np.quantile(self.home_draws, [.05, .95]).tolist()}}


def fit_ratings(dataset, *, season, cutoff, n_draws=4000, seed=42,
                rating_prior_sd=1.5, home_prior_sd=1.0):
    """Find the posterior mode, then sample N(mode, inverse posterior Hessian)."""
    positive_integer(n_draws, "n_draws")
    if n_draws < 2:
        raise ValueError("At least two posterior draws are required")
    positive_integer(season, "season")
    cutoff = utc_timestamp(cutoff)
    if any(isinstance(sd, bool) or not np.isfinite(sd) or sd <= 0 for sd in (rating_prior_sd, home_prior_sd)):
        raise ValueError("Prior standard deviations must be positive and finite")
    games = sorted(dataset.games, key=lambda game: (game.start, game.game_id))
    if not games or len({game.game_id for game in games}) != len(games):
        raise ValueError("Need nonempty unique games")
    for game in games:
        if game.season != season or utc_timestamp(game.start) >= cutoff:
            raise ValueError("Training games must belong to the selected season and precede cutoff")
        if game.team_a == game.team_b or game.home not in (-1, 0, 1):
            raise ValueError("Invalid participants or venue")
        if any(isinstance(score, bool) or not isinstance(score, (int, np.integer)) or score < 0 for score in (game.score_a, game.score_b)) or game.score_a == game.score_b:
            raise ValueError("Training requires nonnegative integer scores and a non-tied result")
    ids = sorted({team for game in games for team in (game.team_a, game.team_b)})
    if any(team not in dataset.teams for team in ids):
        raise ValueError("Missing names in team registry")
    lookup = {team: i for i, team in enumerate(ids)}
    n, m = len(ids), len(games)
    rows = np.repeat(np.arange(m), 3)
    columns = np.array([[lookup[g.team_a], lookup[g.team_b], n] for g in games]).ravel()
    values = np.array([[1., -1., g.home] for g in games]).ravel()
    design = csr_matrix((values, (rows, columns)), shape=(m, n + 1))
    outcomes = np.array([g.score_a > g.score_b for g in games], dtype=float)
    precision = np.r_[np.full(n, 1 / rating_prior_sd ** 2), 1 / home_prior_sd ** 2]

    def objective(theta):
        logits = design @ theta
        loss = np.sum(np.logaddexp(0, logits) - outcomes * logits) + .5 * np.dot(theta * precision, theta)
        gradient = np.asarray(design.T @ (expit(logits) - outcomes)).ravel() + precision * theta
        return float(loss), gradient

    optimized = minimize(objective, np.zeros(n + 1), jac=True, method="L-BFGS-B",
                         options={"maxiter": 2000, "ftol": 1e-14, "gtol": 1e-7})
    gradient_norm = float(np.max(np.abs(objective(optimized.x)[1])))
    if not optimized.success or not np.isfinite(optimized.x).all() or gradient_norm > 1e-4:
        raise ValueError(f"Rating fit failed convergence checks: {optimized.message}; gradient={gradient_norm}")
    p = expit(design @ optimized.x)
    hessian = (design.T @ design.multiply((p * (1 - p))[:, None])).toarray() + np.diag(precision)
    lower = cholesky(hessian, lower=True)
    rng = np.random.default_rng(seed)
    draws = optimized.x + solve_triangular(lower.T, rng.standard_normal((n + 1, n_draws)), lower=False).T
    centered = draws[:, :n] - draws[:, :n].mean(axis=1, keepdims=True)
    serialized = [[g.game_id, g.season, g.start.isoformat(), g.team_a, g.team_b, g.score_a, g.score_b, g.home] for g in games]
    metadata = {
        "model_version": 1, "model": "bayesian_bradley_terry", "inference": "laplace",
        "season": season, "cutoff": cutoff.isoformat(), "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_draws": n_draws, "seed": seed, "n_games": m, "n_teams": n,
        "rating_prior_sd": rating_prior_sd, "home_prior_sd": home_prior_sd,
        "training_games_sha256": hashlib.sha256(json.dumps(serialized, separators=(",", ":")).encode()).hexdigest(),
        "training_game_ids": [g.game_id for g in games], "data_report": dataset.report,
        "diagnostics": {"optimizer_converged": True, "max_abs_gradient": gradient_norm,
                        "iterations": int(optimized.nit), "hessian_positive_definite": True},
        "limitations": "Gaussian posterior approximation with fixed prior scales; no MCMC or learned hyperparameters. Ratings are season-static log-odds, not expected wins. Cache coverage may be incomplete.",
    }
    return RatingPosterior(ids, [dataset.teams[team] for team in ids], centered, draws[:, n], metadata).validate()


def save_ratings(posterior, path):
    """Publish a validated, pickle-free artifact without overwriting an existing run."""
    posterior.validate()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as handle:
            temporary = Path(handle.name)
            np.savez_compressed(handle, team_ids=np.array(posterior.team_ids), team_names=np.array(posterior.team_names),
                                rating_draws=posterior.rating_draws, home_draws=posterior.home_draws,
                                metadata=np.array(json.dumps(posterior.metadata, allow_nan=False)))
        load_ratings(temporary)
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def load_ratings(path):
    with np.load(path, allow_pickle=False) as artifact:
        result = RatingPosterior(artifact["team_ids"].tolist(), artifact["team_names"].tolist(),
                                 artifact["rating_draws"].copy(), artifact["home_draws"].copy(),
                                 json.loads(str(artifact["metadata"])))
    return result.validate()


def evaluate_games(posterior, games):
    """Score held-out games only; refuse unknown teams instead of silently dropping them."""
    if not games:
        raise ValueError("No held-out games supplied")
    training_ids = set(posterior.metadata["training_game_ids"])
    cutoff = utc_timestamp(posterior.metadata["cutoff"])
    probabilities, outcomes = [], []
    seen = set()
    for game in games:
        if game.game_id in training_ids or game.start < cutoff or game.season != posterior.metadata["season"]:
            raise ValueError("Evaluation games must be held out, in the fitted season, at or after cutoff")
        if game.game_id in seen or game.score_a == game.score_b or game.home not in (-1, 0, 1):
            raise ValueError("Invalid or duplicate evaluation game")
        seen.add(game.game_id)
        venue = {0: "neutral", 1: "a_home", -1: "b_home"}[game.home]
        probabilities.append(posterior.matchup_draws(game.team_a, game.team_b, venue=venue).mean())
        outcomes.append(float(game.score_a > game.score_b))
    p, y = np.array(probabilities), np.array(outcomes)
    bounded = np.clip(p, 1e-12, 1 - 1e-12)
    return {"n_games": len(games), "brier_score": float(np.mean((p - y) ** 2)),
            "log_loss": float(-np.mean(y * np.log(bounded) + (1 - y) * np.log1p(-bounded))),
            "accuracy": float(np.mean((p >= .5) == y))}
