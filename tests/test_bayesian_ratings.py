import contextlib
import csv
from datetime import datetime, timedelta, timezone
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scipy.special import expit

from bayesian_ratings import (
    Game, GameDataset, evaluate_games, fit_ratings, load_games, load_ratings,
    save_ratings, utc_timestamp,
)
from run_bayesian import main


START = datetime(2024, 11, 1, tzinfo=timezone.utc)
CUTOFF = "2025-03-18T00:00:00Z"


def synthetic_games(repeats=80):
    rng = np.random.default_rng(123)
    strengths = [1.0, 0., -1.0]
    games = []
    for a, b in ((0, 1), (0, 2), (1, 2)):
        for i in range(repeats):
            home = (-1, 0, 1)[i % 3]
            win = rng.random() < expit(strengths[a] - strengths[b] + .5 * home)
            games.append(Game(f"g{len(games)}", 2025, START + timedelta(hours=len(games)),
                              str(a), str(b), 80 if win else 70, 70 if win else 80, home))
    return GameDataset(games, {"0": "Strong", "1": "Middle", "2": "Weak"}, {})


def row(**updates):
    result = {
        "gameId": "1", "season": "2025", "seasonType": "regular", "tournament": "",
        "startDate": "2024-11-05T00:00:00Z", "teamId": "10", "team": "A",
        "opponentId": "20", "opponent": "B", "neutralSite": "False", "isHome": "True",
        "teamStats.points.total": "80", "opponentStats.points.total": "70",
    }
    result.update(updates)
    return result


class BayesianRatingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def write_rows(self, team, rows):
        directory = self.root / "2025" / team
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{team}_game_stats.csv"
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row()))
            writer.writeheader()
            writer.writerows(rows)
        return path

    def test_loader_deduplicates_and_filters_without_future_score_leakage(self):
        self.write_rows("a", [row(), row(gameId="2", startDate=CUTOFF, **{"teamStats.points.total": "bad future score"}),
                              row(gameId="3", seasonType="postseason"),
                              row(gameId="4", **{"teamStats.points.total": ""})])
        reverse = row(teamId="20", team="B", opponentId="10", opponent="A", isHome="False",
                      **{"teamStats.points.total": "70", "opponentStats.points.total": "80"})
        self.write_rows("b", [reverse])
        data = load_games(self.root, 2025, CUTOFF)
        self.assertEqual(len(data.games), 1)
        self.assertEqual(data.games[0].home, 1)
        self.assertEqual(data.report["duplicates"], 1)
        self.assertEqual(data.report["excluded_cutoff"], 1)
        self.assertEqual(data.report["excluded_type"], 1)
        self.assertEqual(data.report["excluded_unfinished"], 1)
        self.assertEqual(data.teams, {"cbbd:10": "A", "cbbd:20": "B"})

    def test_conflicting_duplicate_is_rejected(self):
        self.write_rows("a", [row(), row(**{"teamStats.points.total": "81"})])
        with self.assertRaisesRegex(ValueError, "Conflicting duplicate"):
            load_games(self.root, 2025, CUTOFF)

    def test_invalid_csv_inputs_are_rejected(self):
        cases = [row(startDate="2024-11-05"), row(neutralSite="unknown"),
                 row(teamId="20"), row(**{"teamStats.points.total": "nan"}),
                 row(**{"teamStats.points.total": "70"})]
        for value in cases:
            with self.subTest(value=value):
                self.write_rows("a", [value])
                with self.assertRaises(ValueError):
                    load_games(self.root, 2025, CUTOFF)

    def test_synthetic_strength_recovery_and_centered_joint_draws(self):
        posterior = fit_ratings(synthetic_games(), season=2025, cutoff=CUTOFF, n_draws=4000)
        means = posterior.rating_draws.mean(axis=0)
        self.assertGreater(means[0], means[1])
        self.assertGreater(means[1], means[2])
        self.assertGreater(posterior.home_draws.mean(), 0)
        np.testing.assert_allclose(posterior.rating_draws.sum(axis=1), 0, atol=1e-12)
        self.assertTrue(posterior.metadata["diagnostics"]["optimizer_converged"])
        self.assertLess(posterior.metadata["diagnostics"]["max_abs_gradient"], 1e-4)

    def test_reproducible_fit_independent_of_input_order(self):
        data = synthetic_games(20)
        first = fit_ratings(data, season=2025, cutoff=CUTOFF, n_draws=100, seed=7)
        data.games.reverse()
        second = fit_ratings(data, season=2025, cutoff=CUTOFF, n_draws=100, seed=7)
        np.testing.assert_array_equal(first.rating_draws, second.rating_draws)
        np.testing.assert_array_equal(first.home_draws, second.home_draws)
        self.assertEqual(first.metadata["training_games_sha256"], second.metadata["training_games_sha256"])

    def test_matchup_symmetry_venue_and_ordered_matrices(self):
        posterior = fit_ratings(synthetic_games(), season=2025, cutoff=CUTOFF, n_draws=1000)
        forward = posterior.matchup_draws("Strong", "Weak", venue="a_home")
        reverse = posterior.matchup_draws("Weak", "Strong", venue="b_home")
        np.testing.assert_allclose(forward + reverse, 1)
        prediction = posterior.predict("Strong", "Weak")
        self.assertGreater(prediction["win_probability_a"], .5)
        self.assertTrue(0 < prediction["probability_interval_90"][0] < prediction["probability_interval_90"][1] < 1)
        for index in (None, 0):
            matrix = posterior.probability_matrix(["Weak", "Strong", "Middle"], draw_index=index)
            np.testing.assert_allclose(matrix + matrix.T, 1)
            np.testing.assert_array_equal(np.diag(matrix), .5)
            self.assertTrue(np.all((matrix >= 0) & (matrix <= 1)))
        self.assertAlmostEqual(posterior.probability_matrix(["Weak", "Strong"])[1, 0], prediction["win_probability_a"])
        for a, b in (("Strong", "Unknown"), ("Strong", "Strong")):
            with self.assertRaises(ValueError):
                posterior.predict(a, b)
        with self.assertRaises(ValueError):
            posterior.probability_matrix(["Strong", "Strong"])

    def test_more_evidence_reduces_rating_difference_uncertainty(self):
        narrow = fit_ratings(synthetic_games(100), season=2025, cutoff=CUTOFF, n_draws=3000)
        broad = fit_ratings(synthetic_games(4), season=2025, cutoff=CUTOFF, n_draws=3000)
        self.assertLess(np.std(narrow.rating_draws[:, 0] - narrow.rating_draws[:, 1]),
                        np.std(broad.rating_draws[:, 0] - broad.rating_draws[:, 1]))

    def test_laplace_prediction_agrees_with_numerical_two_team_posterior(self):
        # Independently integrate the exact one-dimensional posterior for a
        # neutral matchup. The rating difference prior has variance 2 * 1.5².
        games = [Game(str(i), 2025, START, "a", "b", 80 if i < 21 else 70,
                      70 if i < 21 else 80, 0) for i in range(30)]
        posterior = fit_ratings(GameDataset(games, {"a": "A", "b": "B"}, {}),
                                season=2025, cutoff=CUTOFF, n_draws=20000)
        difference = np.linspace(-12, 12, 24001)
        log_density = 21 * difference - 30 * np.logaddexp(0, difference) - difference ** 2 / (4 * 1.5 ** 2)
        weight = np.exp(log_density - log_density.max())
        expected = np.sum(expit(difference) * weight) / weight.sum()
        self.assertAlmostEqual(posterior.predict("A", "B")["win_probability_a"], expected, delta=.015)

    def test_priors_keep_undefeated_disconnected_teams_finite(self):
        games = [Game("1", 2025, START, "a", "b", 80, 70, 0),
                 Game("2", 2025, START, "c", "d", 80, 70, 0)]
        posterior = fit_ratings(GameDataset(games, {x: x for x in "abcd"}, {}), season=2025, cutoff=CUTOFF, n_draws=100)
        self.assertTrue(np.isfinite(posterior.rating_draws).all())
        self.assertAlmostEqual(posterior.home_draws.std(), 1., delta=.2)

    def test_fit_rejects_leakage_and_invalid_configuration(self):
        data = synthetic_games(4)
        for options in ({"n_draws": 1}, {"n_draws": True}, {"rating_prior_sd": 0}, {"home_prior_sd": np.nan}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                fit_ratings(data, season=2025, cutoff=CUTOFF, **options)
        with self.assertRaisesRegex(ValueError, "precede cutoff"):
            fit_ratings(data, season=2025, cutoff=START.isoformat())
        data.games.append(data.games[0])
        with self.assertRaisesRegex(ValueError, "unique"):
            fit_ratings(data, season=2025, cutoff=CUTOFF)
        with self.assertRaisesRegex(ValueError, "timezone"):
            utc_timestamp("2025-03-18")

    def test_artifact_roundtrip_is_pickle_free_and_does_not_overwrite(self):
        posterior = fit_ratings(synthetic_games(8), season=2025, cutoff=CUTOFF, n_draws=200)
        path = self.root / "fit.npz"
        save_ratings(posterior, path)
        loaded = load_ratings(path)
        np.testing.assert_array_equal(loaded.rating_draws, posterior.rating_draws)
        self.assertEqual(loaded.predict("Strong", "Weak"), posterior.predict("Strong", "Weak"))
        with self.assertRaises(FileExistsError):
            save_ratings(posterior, path)
        with np.load(path, allow_pickle=False) as artifact:
            self.assertTrue(all(artifact[key].dtype.kind != "O" for key in artifact.files))

    def test_heldout_metrics_and_training_rejection(self):
        data = synthetic_games(15)
        posterior = fit_ratings(data, season=2025, cutoff=CUTOFF, n_draws=1000)
        heldout = [Game("future", 2025, utc_timestamp(CUTOFF), "0", "2", 80, 70, 0, "NCAA")]
        metrics = evaluate_games(posterior, heldout)
        p = posterior.predict("Strong", "Weak")["win_probability_a"]
        self.assertAlmostEqual(metrics["brier_score"], (1 - p) ** 2)
        self.assertAlmostEqual(metrics["log_loss"], -np.log(p))
        with self.assertRaisesRegex(ValueError, "held out"):
            evaluate_games(posterior, data.games)
        with self.assertRaises(ValueError):
            evaluate_games(posterior, heldout * 2)

    def test_cli_fit_predict_and_summary_without_training_on_predict(self):
        self.write_rows("a", [row(), row(gameId="2", neutralSite="True", **{"teamStats.points.total": "70", "opponentStats.points.total": "80"})])
        artifact = self.root / "fit.npz"
        summary = self.root / "summary.json"
        argv = ["run_bayesian", "fit", "--data-dir", str(self.root), "--season", "2025", "--cutoff", CUTOFF,
                "--draws", "100", "--output", str(artifact), "--summary", str(summary)]
        with patch("sys.argv", argv), contextlib.redirect_stdout(io.StringIO()):
            main()
        self.assertEqual(json.loads(summary.read_text())["metadata"]["n_games"], 2)
        with patch("sys.argv", ["run_bayesian", "predict", str(artifact), "--team-a", "A", "--team-b", "B"]), patch("run_bayesian.fit_ratings") as fitter, contextlib.redirect_stdout(io.StringIO()) as output:
            main()
        fitter.assert_not_called()
        self.assertIn("win_probability_a", json.loads(output.getvalue()))
        self.write_rows("a", [row(), row(gameId="3", seasonType="postseason", tournament="NCAA", startDate="2025-03-19T00:00:00Z"),
                              row(gameId="4", seasonType="postseason", tournament="NIT", startDate="2025-03-19T00:00:00Z")])
        argv = ["run_bayesian", "evaluate", str(artifact), "--data-dir", str(self.root), "--until", "2025-07-01T00:00:00Z"]
        with patch("sys.argv", argv), contextlib.redirect_stdout(io.StringIO()) as output:
            main()
        self.assertEqual(json.loads(output.getvalue())["n_games"], 1)


if __name__ == "__main__":
    unittest.main()
