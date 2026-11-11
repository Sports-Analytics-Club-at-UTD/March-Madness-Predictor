import contextlib
import csv
import io
import itertools
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scipy.special import expit

from bayesian_monte_carlo import (
    METHOD, build_forecast_document, load_bracket_schedule, max_expected_points_bracket,
    parse_bracket_schedule, save_forecast_document, simulate_bracket, stable_team_id,
    validate_forecast_document,
)
from bayesian_ratings import RatingPosterior, save_ratings
import run_bayesian_monte_carlo
import run_pipeline

CUTOFF = "2025-03-18T00:00:00Z"
AS_OF = "2025-03-16T23:00:00Z"


def posterior(draws, cutoff=CUTOFF, season=2025):
    draws = np.asarray(draws, dtype=float)
    draws = draws - draws.mean(axis=1, keepdims=True)
    count = draws.shape[1]
    return RatingPosterior(
        [f"cbbd:{i + 1}" for i in range(count)], [f"Team {i + 1}" for i in range(count)], draws,
        np.zeros(len(draws)),
        {"model_version": 1, "model": "bayesian_bradley_terry", "inference": "laplace", "season": season,
         "cutoff": cutoff, "training_game_ids": ["cbbd:900"], "n_games": 1},
    ).validate()


def four_team_schedule(**updates):
    raw = {"season": 2025, "as_of": AS_OF, "games": [
        {"slot": 1, "round_number": 1, "participants": [{"team": "cbbd:1"}, {"team": "cbbd:2"}]},
        {"slot": 2, "round_number": 1, "participants": [{"team": "cbbd:3"}, {"team": "cbbd:4"}]},
        {"slot": 3, "round_number": 2, "participants": [{"source_slot": 1}, {"source_slot": 2}]},
    ]}
    raw.update(updates)
    return raw


def regions_schedule(play_ins=True):
    ids = iter(range(1, 69))
    regions = {}
    for name in ("East", "West", "South", "Midwest"):
        double = ("16" if name in ("East", "West") else "11") if play_ins else None
        regions[name] = {str(seed): [f"cbbd:{next(ids)}", f"cbbd:{next(ids)}"] if str(seed) == double
                         else f"cbbd:{next(ids)}" for seed in range(1, 17)}
    return {"season": 2025, "as_of": AS_OF, "source": "synthetic test bracket", "regions": regions,
            "final_four": [["East", "West"], ["South", "Midwest"]]}


def check_contract_relationships(test, document):
    """Independent re-statement of the main contract's cross-record rules."""
    teams = {team["id"] for team in document["teams"]}
    games = {game["id"]: game for game in document["games"]}
    test.assertEqual(document["schema_version"], "1.0.0")
    test.assertEqual(document["tournament"]["topology_status"], "available")
    eligible, consumed, opening = {}, [], []
    for game in sorted(games.values(), key=lambda g: g["slot"]):
        branches = []
        for participant in game["participants"]:
            test.assertEqual(participant["team_id"] is None, participant["source_game_id"] is not None)
            if participant["team_id"] is not None:
                test.assertIn(participant["team_id"], teams)
                opening.append(participant["team_id"])
                branches.append({participant["team_id"]})
            else:
                source = games[participant["source_game_id"]]
                test.assertIn(source["id"], eligible)
                test.assertLess(source["round_number"], game["round_number"])
                consumed.append(source["id"])
                branches.append(eligible[source["id"]])
        test.assertFalse(branches[0] & branches[1])
        eligible[game["id"]] = branches[0] | branches[1]
        test.assertIsNone(game["actual_winner_team_id"])
    test.assertEqual(len(consumed), len(set(consumed)))
    finals = set(games) - set(consumed)
    test.assertEqual(len(finals), 1)
    test.assertEqual(len(opening), len(set(opening)))
    test.assertEqual(len(games), len(opening) - 1)
    test.assertEqual(document["tournament"]["bracket_team_count"], len(opening))
    runs = {run["id"] for run in document["prediction_runs"]}
    for prediction in document["team_predictions"]:
        test.assertIn(prediction["run_id"], runs)
        test.assertIn(prediction["team_id"], teams)
        values = [row["probability"] for row in prediction["probabilities"]["bracket_tree"]]
        test.assertTrue(all(0 <= value <= 1 for value in values))
        test.assertEqual(values, sorted(values, reverse=True))
    for bracket in document["brackets"]:
        picks = {pick["game_id"]: pick for pick in bracket["picks"]}
        test.assertEqual(set(picks), set(games))
        for game_id, pick in picks.items():
            allowed = {p["team_id"] or picks[p["source_game_id"]]["winner_team_id"]
                       for p in games[game_id]["participants"]}
            test.assertIn(pick["winner_team_id"], allowed)
        test.assertEqual(bracket["summary"]["champion_team_id"], picks[next(iter(finals))]["winner_team_id"])


class JointDrawSimulationTests(unittest.TestCase):
    def test_every_game_in_a_tournament_uses_the_same_joint_draw(self):
        # Draw 0 makes teams 1 and 3 certain winners; draw 1 reverses both games.
        # Independent or per-game redraws would mix the two worlds in about half of tournaments.
        ratings = posterior([[30, 10, -10, -30], [-30, -10, 10, 30]])
        schedule = parse_bracket_schedule(four_team_schedule())
        result = simulate_bracket(ratings, schedule, n_simulations=4000, seed=7)
        self.assertEqual(result.draw_indices.shape, (4000,))
        self.assertEqual(set(result.draw_indices.tolist()), {0, 1})
        first = result.draw_indices == 0
        for mask, expected in ((first, (0, 2, 0)), (~first, (1, 3, 3))):
            for slot, winner in enumerate(expected):
                self.assertTrue((result.slot_winners[slot, mask] == winner).all())
        mixed = (result.slot_winners[0] == 0) & (result.slot_winners[1] == 3)
        self.assertEqual(int(mixed.sum()), 0)
        # A posterior-mean matrix would give team 4 about 25%; joint draws give about 50%.
        self.assertAlmostEqual(result.slot_probabilities()[2, 3], .5, delta=.03)

    def test_each_tournament_matches_the_posterior_draw_matrix_it_was_assigned(self):
        rng = np.random.default_rng(3)
        ratings = posterior(rng.normal(0, 1.2, size=(6, 4)))
        schedule = parse_bracket_schedule(four_team_schedule())
        result = simulate_bracket(ratings, schedule, n_simulations=60000, seed=11)
        for draw in range(6):
            mask = result.draw_indices == draw
            matrix = ratings.probability_matrix(list(schedule.team_refs), draw_index=draw)
            both = mask & (result.slot_winners[0] == 0) & (result.slot_winners[1] == 2)
            for subset, slot, expected in ((mask, 0, matrix[0, 1]), (both, 2, matrix[0, 2])):
                tolerance = 5 * np.sqrt(expected * (1 - expected) / subset.sum()) + 1e-3
                self.assertAlmostEqual(np.mean(result.slot_winners[slot, subset] == 0), expected, delta=tolerance)

    def test_reproducible_and_counts_are_coherent_with_play_ins(self):
        rng = np.random.default_rng(5)
        ratings = posterior(rng.normal(0, 1, size=(40, 68)))
        schedule = parse_bracket_schedule(regions_schedule())
        self.assertEqual((len(schedule.team_refs), len(schedule.games), schedule.main_rounds), (68, 67, 6))
        self.assertEqual(sum(game.play_in for game in schedule.games), 4)
        self.assertEqual([game.round_points for game in schedule.games[:5]], [0., 0., 0., 0., 10.])
        self.assertEqual(schedule.games[-1].round_points, 320.)
        a = simulate_bracket(ratings, schedule, n_simulations=500, seed=1)
        b = simulate_bracket(ratings, schedule, n_simulations=500, seed=1)
        c = simulate_bracket(ratings, schedule, n_simulations=500, seed=2)
        np.testing.assert_array_equal(a.slot_winners, b.slot_winners)
        self.assertFalse(np.array_equal(a.slot_winners, c.slot_winners))
        # Play-in wins are excluded: 63 main-bracket wins per tournament, one six-win champion.
        np.testing.assert_array_equal(a.win_counts.sum(axis=0), np.full(500, 63))
        np.testing.assert_array_equal((a.win_counts == 6).sum(axis=0), np.ones(500))
        np.testing.assert_allclose(a.slot_probabilities().sum(axis=1), 1)

    def test_rejects_mismatched_ratings(self):
        schedule = parse_bracket_schedule(four_team_schedule())
        draws = [[1, 0, -1, 0], [0, 1, 0, -1]]
        with self.assertRaisesRegex(ValueError, "season"):
            simulate_bracket(posterior(draws, season=2024), schedule)
        with self.assertRaisesRegex(ValueError, "Unknown or ambiguous"):
            simulate_bracket(posterior([[1, 0, -1], [0, 1, -1]]), schedule)
        timed = four_team_schedule()
        timed["games"][0]["scheduled_start_utc"] = "2025-03-18T16:00:00Z"
        with self.assertRaisesRegex(ValueError, "cutoff is later"):
            simulate_bracket(posterior(draws, cutoff="2025-03-19T00:00:00Z"), parse_bracket_schedule(timed))
        simulate_bracket(posterior(draws), parse_bracket_schedule(timed), n_simulations=5)
        alias = four_team_schedule()
        alias["games"][1]["participants"][0] = {"team": "team 1"}  # name of cbbd:1, already entered by ID
        with self.assertRaisesRegex(ValueError, "same rated team"):
            simulate_bracket(posterior(draws), parse_bracket_schedule(alias))
        with self.assertRaisesRegex(ValueError, "n_simulations"):
            simulate_bracket(posterior(draws), schedule, n_simulations=0)


class PreTournamentScheduleTests(unittest.TestCase):
    def rejects(self, pattern, raw):
        with self.assertRaisesRegex(ValueError, pattern):
            parse_bracket_schedule(raw)

    def test_results_and_realized_matchups_are_rejected(self):
        raw = four_team_schedule()
        raw["games"][0]["winner"] = "cbbd:1"
        self.rejects("results are not accepted", raw)
        raw = four_team_schedule()
        raw["games"][0]["participants"][0]["score"] = 80
        self.rejects("results are not accepted", raw)
        self.rejects("results are not accepted", four_team_schedule(actual_slot_winners=[0, 2, 0]))
        raw = four_team_schedule()
        raw["games"][2]["participants"] = [{"team": "cbbd:1"}, {"team": "cbbd:3"}]  # realized final
        self.rejects("realized matchups", raw)
        raw = four_team_schedule()
        raw["games"][2]["participants"] = [{"team": "cbbd:5"}, {"team": "cbbd:6"}]  # disconnected
        self.rejects("connected single-elimination tree", raw)

    def test_structure_and_timing_checks(self):
        raw = four_team_schedule()
        raw["games"][2]["participants"][1] = {"source_slot": 1}
        self.rejects("feeds more than one game", raw)
        raw = four_team_schedule()
        raw["games"][2]["round_number"] = 1
        self.rejects("later round", raw)
        raw = four_team_schedule()
        raw["games"][1]["participants"][0] = {"source_slot": 3}
        self.rejects("earlier slot", raw)
        raw = four_team_schedule()
        raw["games"][2]["slot"] = 5
        self.rejects("without gaps", raw)
        raw = four_team_schedule()
        raw["games"][2]["play_in"] = True
        self.rejects("play-in", raw)
        raw = four_team_schedule()
        raw["games"][0]["scheduled_start_utc"] = "2025-03-16T12:00:00Z"
        self.rejects("before the schedule's as_of", raw)
        self.rejects("timezone", four_team_schedule(as_of="2025-03-16T23:00:00"))
        raw = four_team_schedule()
        del raw["as_of"]
        self.rejects("as_of", raw)
        self.rejects("exactly one of games or regions", four_team_schedule(regions={}))
        broken = regions_schedule()
        del broken["regions"]["East"]["7"]
        self.rejects("every seed", broken)

    def test_regions_form_builds_standard_64_team_bracket_and_loads_from_file(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "schedule.json"
            path.write_text(json.dumps(regions_schedule(play_ins=False)))
            schedule = load_bracket_schedule(path)
        self.assertEqual(len(schedule.sha256), 64)
        self.assertEqual((len(schedule.team_refs), len(schedule.games)), (64, 63))
        self.assertEqual([game.round_number for game in schedule.games],
                         [1] * 32 + [2] * 16 + [3] * 8 + [4] * 4 + [5] * 2 + [6])
        self.assertEqual(schedule.team_refs[:4], ("cbbd:1", "cbbd:16", "cbbd:8", "cbbd:9"))  # 1v16, 8v9
        self.assertEqual((schedule.seeds[0], schedule.seeds[1]), (1, 16))
        self.assertEqual(schedule.games[62].participants, (("slot", 60), ("slot", 61)))


class ForecastExportTests(unittest.TestCase):
    def build(self, n_simulations=400):
        rng = np.random.default_rng(9)
        ratings = posterior(rng.normal(0, 1, size=(50, 68)))
        raw = regions_schedule()
        raw["regions"]["East"]["1"] = "Team 1"  # exact names resolve like provider IDs
        schedule = parse_bracket_schedule(raw, sha256="a" * 64)
        simulation = simulate_bracket(ratings, schedule, n_simulations=n_simulations, seed=4)
        return ratings, simulation, build_forecast_document(ratings, simulation, run_id="run-1")

    def test_document_follows_contract_relationships(self):
        ratings, simulation, document = self.build()
        check_contract_relationships(self, document)
        self.assertEqual(json.loads(json.dumps(document, allow_nan=False)), document)
        self.assertEqual(document["tournament"], {
            "id": "ncaa-men-2025", "season": 2025, "competition": "Men's college basketball",
            "format": "single_elimination", "topology_status": "available", "bracket_team_count": 68,
            "includes_first_four": True})
        self.assertEqual((len(document["games"]), len(document["team_predictions"])), (67, 68))
        run = document["prediction_runs"][0]
        self.assertEqual((run["method"], run["n_simulations"], run["seed"]), (METHOD, 400, 4))
        self.assertNotIn("training_game_ids", run["source_metadata"])
        self.assertIn("One joint posterior draw", run["source_metadata"]["simulation"]["design"])
        self.assertEqual(run["source_metadata"]["bracket_schedule"]["sha256"], "a" * 64)
        team_id = stable_team_id("Team 1")
        self.assertEqual(run["source_metadata"]["team_registry"][team_id]["provider_id"], "cbbd:1")
        prediction = next(row for row in document["team_predictions"] if row["team_id"] == team_id)
        self.assertEqual([row["minimum_wins"] for row in prediction["probabilities"]["bracket_tree"]], list(range(1, 7)))
        self.assertEqual(prediction["statistics"]["seed"], 1.)
        self.assertEqual(prediction["statistics"]["champion_probability"],
                         prediction["probabilities"]["bracket_tree"][-1]["probability"])
        draws = ratings.rating_draws[:, 0]
        self.assertAlmostEqual(prediction["posterior"]["mean"], draws.mean())
        interval = prediction["posterior"]["intervals"][0]
        self.assertEqual((interval["kind"], interval["target"]), ("credible", prediction["posterior"]["target"]))
        self.assertEqual(prediction["intervals"], [])
        champions = sum(row["statistics"]["champion_probability"] for row in document["team_predictions"])
        self.assertAlmostEqual(champions, 1)
        play_in = [row for row in document["team_predictions"] if "play_in_win_probability" in row["statistics"]]
        self.assertEqual(len(play_in), 8)
        bracket = document["brackets"][0]
        self.assertEqual(bracket["strategy"], "max_ev")
        self.assertAlmostEqual(bracket["summary"]["expected_points"],
                               sum(pick["expected_points"] for pick in bracket["picks"]))

    def test_matches_main_pydantic_contract_when_installed(self):
        try:
            from backend_data.models import ForecastDocument
        except ImportError:
            self.skipTest("backend_data (forecast contract on main) is not installed on this branch")
        _, _, document = self.build()
        self.assertEqual(validate_forecast_document(document), "backend_data.models.ForecastDocument")
        parsed = ForecastDocument.model_validate(document)
        # The file is already in main's canonical serialized form, including defaults.
        self.assertEqual(json.loads(parsed.model_dump_json()), document)
        broken = json.loads(json.dumps(document))
        broken["brackets"][0]["summary"]["champion_team_id"] = document["teams"][0]["id"]
        broken["brackets"][0]["picks"][-1]["winner_team_id"] = "team-unknown"
        with self.assertRaises(ValueError):
            validate_forecast_document(broken)

    def test_local_validation_and_immutable_storage(self):
        _, _, document = self.build(50)
        broken = json.loads(json.dumps(document))
        broken["team_predictions"][0]["statistics"]["tree_expected_wins"] = float("nan")
        with self.assertRaises(ValueError):
            validate_forecast_document(broken)
        broken = json.loads(json.dumps(document))
        broken["prediction_runs"][0]["id"] = "bad id"
        with self.assertRaisesRegex(ValueError, "Identifiers"):
            validate_forecast_document(broken)
        with tempfile.TemporaryDirectory() as temp:
            first = save_forecast_document(document, temp)
            second = save_forecast_document(document, temp)
            self.assertNotEqual(first, second)
            self.assertRegex(first.name, r"^forecast_2025_[0-9a-f-]{36}\.json$")
            self.assertEqual(json.loads(first.read_text()), document)
            self.assertEqual(len(list(Path(temp).iterdir())), 2)

    def test_max_expected_points_bracket_is_optimal(self):
        rng = np.random.default_rng(21)
        raw = {"season": 2025, "as_of": AS_OF, "games": [
            {"slot": 1, "round_number": 1, "play_in": True, "participants": [{"team": "cbbd:1"}, {"team": "cbbd:2"}]},
            {"slot": 2, "round_number": 2, "participants": [{"source_slot": 1}, {"team": "cbbd:3"}]},
            {"slot": 3, "round_number": 2, "participants": [{"team": "cbbd:4"}, {"team": "cbbd:5"}]},
            {"slot": 4, "round_number": 3, "participants": [{"source_slot": 2}, {"source_slot": 3}]},
        ]}
        schedule = parse_bracket_schedule(raw)
        eligible = [(0, 1), (0, 1, 2), (3, 4), (0, 1, 2, 3, 4)]
        for _ in range(20):
            probabilities = np.zeros((4, 5))
            for g, teams in enumerate(eligible):
                probabilities[g, list(teams)] = rng.dirichlet(np.ones(len(teams)))
            picks, value = max_expected_points_bracket(schedule, probabilities)
            best = -1.
            for candidate in itertools.product(*eligible):
                consistent = (candidate[1] in (candidate[0], 2) and candidate[3] in (candidate[1], candidate[2]))
                if consistent:
                    best = max(best, sum(schedule.games[g].round_points * probabilities[g, t]
                                         for g, t in enumerate(candidate)))
            self.assertAlmostEqual(value, best)
            self.assertAlmostEqual(value, sum(schedule.games[g].round_points * probabilities[g, t]
                                              for g, t in enumerate(picks)))
            self.assertIn(picks[1], (picks[0], 2))
            self.assertIn(picks[3], (picks[1], picks[2]))


class CommandTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.schedule = self.root / "schedule.json"
        self.schedule.write_text(json.dumps(four_team_schedule()))

    def run_command(self, *arguments):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            run_bayesian_monte_carlo.main([str(a) for a in arguments])
        return json.loads(output.getvalue())

    def write_game_caches(self):
        fields = ["gameId", "season", "seasonType", "tournament", "startDate", "teamId", "team", "opponentId",
                  "opponent", "neutralSite", "isHome", "teamStats.points.total", "opponentStats.points.total"]
        rng = np.random.default_rng(0)
        rows, strengths = [], [1.5, .5, -.5, -1.5]
        for a, b in itertools.combinations(range(4), 2):
            for repeat in range(12):
                win = rng.random() < expit(strengths[a] - strengths[b])
                rows.append({"gameId": len(rows) + 1, "season": 2025, "seasonType": "regular", "tournament": "",
                             "startDate": f"2025-01-{1 + repeat:02d}T00:00:00Z", "teamId": a + 1, "team": f"Team {a + 1}",
                             "opponentId": b + 1, "opponent": f"Team {b + 1}", "neutralSite": "True", "isHome": "False",
                             "teamStats.points.total": 80 if win else 70, "opponentStats.points.total": 70 if win else 80})
        # A post-cutoff "tournament" result that must never reach the fit.
        rows.append({**rows[0], "gameId": 999, "seasonType": "postseason", "tournament": "NCAA",
                     "startDate": "2025-03-20T00:00:00Z"})
        directory = self.root / "data" / "2025" / "team-1"
        directory.mkdir(parents=True)
        with (directory / "team-1_game_stats.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    def test_fit_then_simulate_and_reuse_saved_ratings(self):
        self.write_game_caches()
        out = self.root / "out"
        fitted = self.run_command("--season", 2025, "--bracket-schedule", self.schedule, "--cutoff", CUTOFF,
                                  "--data-dir", self.root / "data", "--draws", 200, "--n-simulations", 300,
                                  "--output-dir", out)
        document = json.loads(Path(fitted["forecast"]).read_text())
        check_contract_relationships(self, document)
        self.assertEqual(fitted["most_likely_champion"], "Team 1")
        run = document["prediction_runs"][0]
        self.assertEqual(run["source_metadata"]["n_games"], 72)
        self.assertEqual([a["media_type"] for a in run["artifacts"]], ["application/x-npz", "application/json"])
        self.assertTrue(all(len(a["sha256"]) == 64 for a in run["artifacts"]))
        reused = self.run_command("--season", 2025, "--bracket-schedule", self.schedule,
                                  "--ratings", fitted["ratings"], "--n-simulations", 300, "--output-dir", out)
        again = json.loads(Path(reused["forecast"]).read_text())
        self.assertEqual([row["statistics"] for row in again["team_predictions"]],
                         [row["statistics"] for row in document["team_predictions"]])
        self.assertEqual(len(list(out.glob("forecast_2025_*.json"))), 2)
        self.assertEqual(len(list(out.glob("bayesian_ratings_2025_*.npz"))), 1)

    def test_errors_exit_without_writing_outputs(self):
        ratings = self.root / "ratings.npz"
        save_ratings(posterior([[1, 0, -1, 0], [0, 1, 0, -1]], season=2024), ratings)
        out = self.root / "out"
        for arguments in (("--season", 2025, "--ratings", ratings),  # ratings from another season
                          ("--season", 2024, "--ratings", ratings)):  # schedule from another season
            with self.assertRaises(SystemExit) as raised, contextlib.redirect_stderr(io.StringIO()):
                self.run_command("--bracket-schedule", self.schedule, "--output-dir", out, *arguments)
            self.assertEqual(raised.exception.code, 1)
        self.assertFalse(out.exists())


class PipelineStageTests(unittest.TestCase):
    def run_pipeline(self, *arguments):
        output = io.StringIO()
        with patch.object(sys, "argv", ["run_pipeline.py", *arguments]), contextlib.redirect_stdout(output), \
                patch("run_pipeline.subprocess.run") as run:
            run_pipeline.main()
        return output.getvalue(), run

    def test_default_pipeline_does_not_include_bayesian_stage(self):
        output, run = self.run_pipeline("--all", "--dry-run", "--skip-model-training")
        self.assertNotIn("run_bayesian_monte_carlo.py", output)
        self.assertNotIn("bayesian_monte_carlo", output)
        self.assertIn("run_monte_carlo.py", output)
        run.assert_not_called()

    def test_optional_stage_builds_command(self):
        output, run = self.run_pipeline(
            "--stages", "bayesian-monte-carlo", "--season", "2025", "--bayesian-cutoff", CUTOFF,
            "--n-simulations", "250", "--seed", "9", "--bayesian-output-dir", "/tmp/unused-bayes-out")
        self.assertIn("SUCCESS", output)
        command = run.call_args.args[0]
        self.assertTrue(command[1].endswith("run_bayesian_monte_carlo.py"))
        self.assertTrue(command[command.index("--bracket-schedule") + 1].endswith("data/2025/bracket_schedule.json"))
        for flag, value in (("--season", "2025"), ("--cutoff", CUTOFF), ("--n-simulations", "250"),
                            ("--seed", "9"), ("--draws", "4000"), ("--output-dir", "/tmp/unused-bayes-out")):
            self.assertEqual(command[command.index(flag) + 1], value)
        self.assertNotIn("--ratings", command)
        self.assertNotIn("run_monte_carlo.py", output)

    def test_optional_stage_requires_one_rating_source(self):
        output, run = self.run_pipeline("--run-bayesian-monte-carlo", "--season", "2025")
        self.assertIn("exactly one of --bayesian-ratings or --bayesian-cutoff", output)
        self.assertIn("FAILED", output)
        output, run = self.run_pipeline("--run-bayesian-monte-carlo", "--season", "2024", "2025",
                                        "--bayesian-ratings", "ratings/{season}.npz")
        self.assertEqual(run.call_count, 2)
        self.assertTrue(run.call_args.args[0][run.call_args.args[0].index("--ratings") + 1].endswith("ratings/2025.npz"))
        run.assert_called()


if __name__ == "__main__":
    unittest.main()
