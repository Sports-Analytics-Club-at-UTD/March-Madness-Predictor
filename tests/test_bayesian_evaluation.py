import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np

from bayesian_evaluation import (
    SeasonSpec, digest, fetch_tournament_results, load_config, load_matchup_scores,
    load_tournament_results, probability_metrics, run_evaluation,
    season_bootstrap, summarize_predictions,
)
from bayesian_ratings import utc_timestamp


class HistoricalEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.data = self.root / "data"
        self.config = self.root / "config.json"
        self.score_path = self.root / "scores.csv"
        self.specs = []
        self.results = {}
        self.scores = []
        for year in (2023, 2024, 2025):
            cutoff = f"{year}-03-01T00:00:00Z"
            self.specs.append(dict(season=year, cutoff=cutoff, until=f"{year}-05-01T00:00:00Z", expected_games=12))
            # A small synthetic schedule; expected count is explicit, never a claimed real bracket.
            self.results[year] = [dict(
                id=str(year * 100 + i), season=year, startDate=f"{year}-03-02T{i:02d}:00:00Z",
                tournament="NCAA", seasonType="postseason", status="final", neutralSite="True",
                homeTeamId=str(10 + i % 3), awayTeamId=str(11 + i % 3),
                homeTeam=f"Team {10 + i % 3}", awayTeam=f"Team {11 + i % 3}",
                homeSeed=1 + i % 3, awaySeed=2 + i % 3,
                homePoints=80 if i % 4 else 70, awayPoints=70 if i % 4 else 80,
            ) for i in range(12)]
            self.write_results(year)
            training = [dict(
                gameId=str(year * 1000 + i), season=year, seasonType="regular",
                startDate=f"{year - 1}-11-02T{i:02d}:00:00Z",
                teamId=str(10 + i % 3), opponentId=str(11 + i % 3),
                team=f"Team {10 + i % 3}", opponent=f"Team {11 + i % 3}",
                neutralSite="True", isHome="False",
                **{"teamStats.points.total": 80 if i % 4 else 70,
                   "opponentStats.points.total": 70 if i % 4 else 80},
            ) for i in range(12)]
            self.write_csv(self.data / str(year) / "cache" / "cache_game_stats.csv", training)
            self.scores.extend(dict(season=year, team_id=f"cbbd:{team}", ensemble_score=(14 - team) / 4,
                                    as_of=f"{year}-02-28T00:00:00Z", trained_through_season=year - 1)
                               for team in range(10, 14))
        self.config.write_text(json.dumps({"seasons": self.specs, "evaluation_seasons": [2024, 2025]}))
        self.write_csv(self.score_path, self.scores)

    @staticmethod
    def write_csv(path, rows):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    def write_results(self, year):
        self.write_csv(self.data / str(year) / "tournament_games.csv", self.results[year])

    def run_eval(self, name="run", **kwargs):
        return run_evaluation(self.config, self.root / name, data_dir=self.data,
                              draws=100, seed=7, **kwargs)

    def test_end_to_end_artifacts_and_earlier_only_baseline_training(self):
        result = self.run_eval(matchup_scores=self.score_path)
        for name in ("predictions.csv", "metrics.json", "calibration.csv", "coverage.json", "run.json", "report.md",
                     "ratings_2024.npz", "ratings_2025.npz", "baseline_training.json"):
            self.assertTrue((self.root / "run" / name).is_file(), name)
        self.assertEqual(result["comparisons"]["bayesian_vs_matchup"]["n_games"], 24)
        provenance = json.loads((self.root / "run" / "baseline_training.json").read_text())
        self.assertEqual(provenance["2024"]["training_seasons"], [2023])
        self.assertEqual(provenance["2025"]["training_seasons"], [2023, 2024])
        with (self.root / "run" / "predictions.csv").open() as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 24)
        y = [int(r["outcome_a"]) for r in rows]
        p = [float(r["bayesian_probability"]) for r in rows]
        self.assertEqual(probability_metrics(y, p)[0],
                         {k: v for k, v in result["metrics"]["pooled"]["bayesian"].items() if k != "unavailable_games"})
        with self.assertRaises(FileExistsError):
            self.run_eval()

    def test_target_and_future_outcomes_do_not_change_earlier_predictions(self):
        self.run_eval("before", matchup_scores=self.score_path)
        for year in (2024, 2025):
            for row in self.results[year]:
                row["homePoints"], row["awayPoints"] = row["awayPoints"], row["homePoints"]
            self.write_results(year)
        self.run_eval("after", matchup_scores=self.score_path)
        predictions = []
        for folder in ("before", "after"):
            with (self.root / folder / "predictions.csv").open() as handle:
                predictions.append([r for r in csv.DictReader(handle) if r["season"] == "2024"])
        for before, after in zip(*predictions):
            for model in ("bayesian", "seed", "matchup"):
                self.assertEqual(before[f"{model}_probability"], after[f"{model}_probability"])
            self.assertNotEqual(before["outcome_a"], after["outcome_a"])

    def test_missing_matchup_scores_are_explicit_and_never_filled(self):
        result = self.run_eval()
        self.assertEqual(result["comparisons"]["bayesian_vs_matchup"]["n_games"], 0)
        self.assertEqual(result["metrics"]["pooled"]["matchup"]["unavailable_games"], 24)
        self.assertIn("no comparison can be made", (self.root / "run" / "report.md").read_text())

    def test_complete_comparison_requirement_preserves_incomplete_report(self):
        with self.assertRaisesRegex(ValueError, "Incomplete baseline"):
            self.run_eval(require_complete_comparisons=True)
        self.assertTrue((self.root / "run" / "report.md").exists())
        result = self.run_eval("complete", matchup_scores=self.score_path, require_complete_comparisons=True)
        self.assertEqual(result["comparisons"]["bayesian_vs_matchup"]["excluded_games"], 0)

    def test_export_provenance_must_match_configuration_and_score_file(self):
        path = self.root / "provenance.json"
        metadata = dict(scores_sha256=digest(self.score_path), config_sha256=digest(self.config),
                        members=["ridge"], comparison_label="single-member Ridge SHAP + seed")
        path.write_text(json.dumps(metadata))
        self.run_eval(matchup_scores=self.score_path, matchup_provenance=path)
        self.assertEqual(json.loads((self.root / "run" / "matchup_score_provenance.json").read_text()), metadata)
        metadata["config_sha256"] = "wrong"
        path.write_text(json.dumps(metadata))
        with self.assertRaisesRegex(ValueError, "does not match"):
            self.run_eval("invalid", matchup_scores=self.score_path, matchup_provenance=path)

    def test_missing_entrant_fails_with_saved_coverage_audit(self):
        for row in self.results[2024]:
            if row["awayTeamId"] == "13":
                row["awayTeamId"], row["awayTeam"] = "99", "Unknown"
        self.write_results(2024)
        with self.assertRaisesRegex(ValueError, "missing entrant"):
            self.run_eval()
        audit = json.loads((self.root / "run" / "coverage.json").read_text())
        self.assertEqual(audit["2024"]["missing_tournament_teams"], ["cbbd:99"])

    def test_overlapping_training_game_ids_are_rejected(self):
        self.results[2024][0]["id"] = "2024000"
        self.write_results(2024)
        with self.assertRaisesRegex(ValueError, "overlapping"):
            self.run_eval()

    def test_incomplete_or_conflicting_tournament_data_is_rejected(self):
        spec = load_config(self.config)[0][1]
        path = self.data / "2024" / "tournament_games.csv"
        original = self.results[2024].copy()
        self.results[2024] = original[:-1]
        self.write_results(2024)
        with self.assertRaisesRegex(ValueError, "expected 12"):
            load_tournament_results(path, spec)
        self.results[2024] = original + [dict(original[0], homePoints=99)]
        self.write_results(2024)
        with self.assertRaisesRegex(ValueError, "Conflicting duplicate"):
            load_tournament_results(path, spec)

    def test_tournament_cutoff_violation_and_naive_dates_are_rejected(self):
        spec = load_config(self.config)[0][1]
        for date in ("2024-02-28T00:00:00Z", "2024-03-02"):
            self.results[2024][0]["startDate"] = date
            self.write_results(2024)
            with self.assertRaises(ValueError):
                load_tournament_results(self.data / "2024" / "tournament_games.csv", spec)

    def test_score_provenance_rejects_target_training_and_future_features(self):
        specs = load_config(self.config)[0]
        for change in ({"trained_through_season": 2023}, {"as_of": self.specs[0]["cutoff"]},
                       {"ensemble_score": "nan"}):
            rows = [dict(self.scores[0], **change)]
            self.write_csv(self.score_path, rows)
            with self.assertRaises(ValueError):
                load_matchup_scores(self.score_path, specs)

    def test_config_rejects_repeated_seasons_and_overlapping_windows(self):
        config = json.loads(self.config.read_text())
        config["seasons"].append(config["seasons"][0])
        self.config.write_text(json.dumps(config))
        with self.assertRaisesRegex(ValueError, "unique"):
            load_config(self.config)
        config["seasons"] = self.specs
        config["seasons"][0]["until"] = "2024-03-02T00:00:00Z"
        self.config.write_text(json.dumps(config))
        with self.assertRaisesRegex(ValueError, "nonoverlapping"):
            load_config(self.config)

    def test_fetch_is_explicit_and_uses_expected_api_filters(self):
        response = Mock()
        response.json.return_value = self.results[2024]
        response.url = "https://api.collegebasketballdata.com/games?season=2024"
        path = self.root / "fetched" / "tournament_games.csv"
        spec = load_config(self.config)[0][1]
        with patch.dict("os.environ", {"CBBD_API_KEY": "test-key"}), patch("requests.get", return_value=response) as get:
            fetch_tournament_results(spec, path)
        self.assertEqual(get.call_args.kwargs["params"],
                         {"season": 2024, "seasonType": "postseason", "tournament": "NCAA"})
        self.assertEqual(len(load_tournament_results(path, spec)[0]), 12)
        self.assertTrue(path.with_suffix(".source.json").exists())
        self.assertNotIn("test-key", path.with_suffix(".source.json").read_text())


class ProbabilityMetricTests(unittest.TestCase):
    def test_season_bootstrap_preserves_known_paired_difference(self):
        rows = [dict(season=year, outcome_a=1, bayesian_probability=.8, seed_probability=.6)
                for year in (2023, 2024)]
        result = season_bootstrap(rows, "seed")
        np.testing.assert_allclose(result["brier_difference_interval_95"], [-.12, -.12])
        self.assertEqual(result["seasons_with_lower_bayesian_brier"], 2)
        self.assertEqual(result, season_bootstrap(rows, "seed"))
        self.assertIsNone(season_bootstrap(rows[:1], "seed"))

    def test_calibration_includes_empty_bins_and_probability_boundaries(self):
        metrics, bins = probability_metrics([0, 1, 0, 1], [0, 1, .25, .75], bins=4)
        self.assertEqual([b["n_games"] for b in bins], [1, 1, 0, 2])
        self.assertEqual(bins[2]["mean_probability"], None)
        self.assertAlmostEqual(metrics["brier_score"], .03125)
        self.assertAlmostEqual(metrics["ece"], .125)
        self.assertTrue(np.isfinite(metrics["log_loss"]))
        for p in ([float("nan")], [-.1], [1.1]):
            with self.assertRaises(ValueError):
                probability_metrics([1], p)

    def test_comparison_uses_identical_games_instead_of_all_bayesian_rows(self):
        rows = [dict(season=2024, game_id="a", outcome_a=1, bayesian_probability=.8,
                     seed_probability=.6, matchup_probability=None),
                dict(season=2024, game_id="b", outcome_a=0, bayesian_probability=.99,
                     seed_probability=None, matchup_probability=None)]
        summary, _ = summarize_predictions(rows, bins=10)
        comparison = summary["comparisons"]["bayesian_vs_seed"]
        self.assertEqual(comparison["game_keys"], [[2024, "a"]])
        self.assertAlmostEqual(comparison["bayesian"]["brier_score"], .04)
        self.assertAlmostEqual(comparison["baseline"]["brier_score"], .16)
        self.assertEqual(summary["metrics"]["pooled"]["bayesian"]["n_games"], 2)


if __name__ == "__main__":
    unittest.main()
