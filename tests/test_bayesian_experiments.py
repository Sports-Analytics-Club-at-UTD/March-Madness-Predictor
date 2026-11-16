import copy
import csv
import json
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

from bayesian_experiments import BASE, kaggle_season, run_experiments, validate_plan
from dataclasses import replace
from run_schedule_coverage import acquire_season
from bayesian_evaluation import load_config
import test_bayesian_evaluation as evaluation_tests


class ExperimentTests(unittest.TestCase):
    def setUp(self):
        self.fixture = evaluation_tests.HistoricalEvaluationTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root = self.fixture.root
        self.plan = dict(tuning_seasons=[2024], held_out_seasons=[2025], candidates={
            "baseline": BASE.copy(), "prior": dict(BASE, rating_prior_sd=0.75)})
        self.plan_path = self.root / "plan.json"
        self.plan_path.write_text(json.dumps(self.plan))

    def test_plan_rejects_leakage_and_combined_changes(self):
        for tuning, held in (([2025], [2024]), ([2024], [2024, 2025]), ([], [2024, 2025])):
            with self.assertRaises(ValueError):
                validate_plan(dict(self.plan, tuning_seasons=tuning, held_out_seasons=held), [2024, 2025])
        broken = copy.deepcopy(self.plan)
        broken["candidates"]["prior"]["home_prior_sd"] = 0.5
        with self.assertRaisesRegex(ValueError, "exactly one"):
            validate_plan(broken, [2024, 2025])

    def run_experiment(self, name):
        return run_experiments(self.fixture.config, self.plan_path, self.root / name,
                               data_dir=self.fixture.data, draws=40)

    def test_selection_precedes_holdout_and_ignores_its_outcomes(self):
        from bayesian_experiments import load_tournament_results
        def checked(path, spec):
            if spec.season == 2025:
                self.assertTrue((self.root / "one" / "selection.json").exists())
            return load_tournament_results(path, spec)
        with patch("bayesian_experiments.load_tournament_results", side_effect=checked):
            self.run_experiment("one")
        for r in self.fixture.results[2025]:
            r["homePoints"], r["awayPoints"] = r["awayPoints"], r["homePoints"]
        self.fixture.write_results(2025)
        self.run_experiment("two")
        self.assertEqual((self.root / "one/selection.json").read_text(),
                         (self.root / "two/selection.json").read_text())
        def probabilities(run):
            with (self.root / run / "held_out/baseline/predictions.csv").open() as f:
                return [r["probability"] for r in csv.DictReader(f)]
        self.assertEqual(probabilities("one"), probabilities("two"))
        result = json.loads((self.root / "one/results.json").read_text())
        self.assertEqual(result["seed_training_seasons"], [2023, 2024])
        with self.assertRaises(FileExistsError):
            self.run_experiment("one")

    def test_acquisition_validates_before_publish_and_never_overwrites(self):
        spec = load_config(self.fixture.config)[0][0]
        rows = [dict(r, seasonType="regular", startDate="2022-11-02T00:00:00Z")
                for r in self.fixture.results[2023]]
        response = Mock()
        response.json.return_value = rows
        out = self.root / "coverage"
        with patch.dict("os.environ", {"CBBD_API_KEY": "test"}), patch("requests.get", return_value=response) as get:
            result = acquire_season(spec, out)
            self.assertEqual(result["games"], 12)
            self.assertEqual(get.call_args.kwargs["params"], {"season": 2023, "seasonType": "regular"})
            with self.assertRaises(FileExistsError):
                acquire_season(spec, out)
            self.assertEqual(get.call_count, 1)
            rows.append(dict(rows[0], homePoints=99))
            with self.assertRaisesRegex(ValueError, "Conflicting duplicate"):
                acquire_season(spec, self.root / "bad")
            self.assertFalse((self.root / "bad/2023").exists())

    def test_kaggle_namespace_orientation_and_conservative_cutoff(self):
        root = self.root / "kaggle"
        write = self.fixture.write_csv
        write(root / "MTeams.csv", [dict(TeamID=1, TeamName="A"), dict(TeamID=2, TeamName="B")])
        write(root / "MSeasons.csv", [dict(Season=2023, DayZero="02/28/2023")])
        write(root / "MNCAATourneySeeds.csv", [dict(Season=2023, Seed="W01", TeamID=1),
                                               dict(Season=2023, Seed="W16a", TeamID=2)])
        game = dict(Season=2023, DayNum=0, WTeamID=2, WScore=80, LTeamID=1, LScore=70, WLoc="H")
        write(root / "MRegularSeasonCompactResults.csv", [game, dict(game, DayNum=1)])
        write(root / "MNCAATourneyCompactResults.csv", [dict(game, DayNum=2, WLoc="N")])
        spec = replace(load_config(self.fixture.config)[0][0], expected_games=1)
        data = kaggle_season(root, spec)
        self.assertEqual(len(data["dataset"].games), 1)
        g = data["dataset"].games[0]
        self.assertEqual((g.team_a, g.team_b, g.score_a, g.home), ("kaggle:1", "kaggle:2", 70, -1))
        self.assertEqual(data["seeds"]["kaggle:2"], 16)
        with self.assertRaisesRegex(ValueError, "expected 2 games"):
            kaggle_season(root, replace(spec, expected_games=2))


if __name__ == "__main__":
    unittest.main()
