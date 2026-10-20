import ast
import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, ndcg_score, r2_score
from sklearn.model_selection import train_test_split
from scipy.stats import spearmanr

from monte_carlo import simulate_season_success, simulate_team_success
from ridge_bundle import save_ridge_bundle
from run_monte_carlo import main as run_season


class MonteCarloTests(unittest.TestCase):
    def setUp(self):
        self.train = pd.DataFrame({"elo": [1000., 1500., 2000.], "pace": [50., 60., 80.]})
        self.scaler = StandardScaler().fit(self.train)
        self.model = Ridge(alpha=1).fit(self.scaler.transform(self.train), [0., 1., 3.])
        self.team = {"elo": 1600., "pace": 65.}
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "ridge_bundle.joblib"
        self.bundle = save_ridge_bundle(
            self.path, model=self.model, scaler=self.scaler,
            feature_names=["elo", "pace"], training_seasons=[2023, 2024],
            validation_season=2025, validation_actual_wins=[0., 3., 6.],
            validation_predicted_wins=[1., 2., 4.], validation_teams=["A", "B", "C"],
        )

    def test_round_trip_alignment_and_scaling(self):
        loaded = joblib.load(self.path)
        result = simulate_team_success({"pace": 65., "target_wins": 999., "elo": 1600.}, loaded)
        expected = self.model.predict(self.scaler.transform(pd.DataFrame([self.team])))[0]
        self.assertAlmostEqual(result["baseline_prediction"], expected)
        np.testing.assert_array_equal(loaded["validation_residuals"], [-1., 1., 2.])
        self.assertEqual(loaded["training_seasons"], [2023, 2024])
        np.testing.assert_array_equal(loaded["validation_seasons"], [2025]*3)

    def test_empirical_sampling_reproducibility_and_statistics(self):
        result = simulate_team_success(self.team, self.bundle, n_simulations=5000, seed=7)
        repeat = simulate_team_success(self.team, self.bundle, n_simulations=5000, seed=7)
        values = result["simulated_values"]
        np.testing.assert_array_equal(values, repeat["simulated_values"])
        expected = result["baseline_prediction"] + np.random.default_rng(7).choice(
            [-1., 1., 2.], size=5000, replace=True
        )
        np.testing.assert_array_equal(values, expected)
        for key, stat in (("mean", np.mean), ("median", np.median), ("std", np.std)):
            self.assertAlmostEqual(result[key], stat(values))
        self.assertEqual(result["percentile_5"], np.percentile(values, 5))
        self.assertEqual(result["percentile_95"], np.percentile(values, 95))
        for threshold in range(1, 7):
            self.assertEqual(result["probabilities"][threshold], np.mean(values >= threshold))

    def test_missing_features_and_nan_match_training_zero_fill(self):
        missing = simulate_team_success({"elo": 1600.}, self.bundle, seed=1)
        nan = simulate_team_success(pd.Series({"elo": 1600., "pace": np.nan}), self.bundle, seed=1)
        explicit = simulate_team_success({"elo": 1600., "pace": 0.}, self.bundle, seed=1)
        np.testing.assert_array_equal(missing["simulated_values"], explicit["simulated_values"])
        np.testing.assert_array_equal(nan["simulated_values"], explicit["simulated_values"])

    def test_values_are_not_clipped_and_threshold_is_inclusive(self):
        baseline = simulate_team_success(self.team, self.bundle, n_simulations=1)["baseline_prediction"]
        # All samples are exactly at the >= 1 boundary.
        self.bundle["validation_residuals"] = np.array([1. - baseline])
        result = simulate_team_success(self.team, self.bundle, n_simulations=10)
        self.assertEqual(result["probabilities"][1], 1.)
        self.assertEqual(result["probabilities"][2], 0.)
        for residual in (-100., 100.):
            self.bundle["validation_residuals"] = np.array([residual])
            result = simulate_team_success(self.team, self.bundle, n_simulations=10)
            np.testing.assert_array_equal(result["simulated_values"], np.full(10, baseline + residual))

    def test_invalid_inputs_and_residual_pools(self):
        for count in (0, -1, True, 1.5):
            with self.subTest(count=count), self.assertRaises(ValueError):
                simulate_team_success(self.team, self.bundle, n_simulations=count)
        for errors in ([], [np.nan], [np.inf], [[1., 2.]]):
            with self.subTest(errors=errors), self.assertRaises(ValueError):
                simulate_team_success(self.team, dict(self.bundle, validation_residuals=errors))
        for features in ({"unrelated": 1}, {"elo": np.inf}, pd.DataFrame([self.team]*2)):
            with self.subTest(features=features), self.assertRaises(ValueError):
                simulate_team_success(features, self.bundle)
        with self.assertRaises(ValueError):
            simulate_team_success(self.team, dict(self.bundle, feature_names=["pace", "elo"]))

    def test_export_rejects_overlapping_holdout(self):
        with self.assertRaisesRegex(ValueError, "later than"):
            save_ridge_bundle(
                self.path, model=self.model, scaler=self.scaler, feature_names=["elo", "pace"],
                training_seasons=[2024, 2025], validation_season=2025,
                validation_actual_wins=[1.], validation_predicted_wins=[.5], validation_teams=["A"],
            )

    def test_master_retains_walk_forward_error_provenance(self):
        bundle = save_ridge_bundle(
            self.path, model=self.model, scaler=self.scaler, feature_names=["elo", "pace"],
            training_seasons=[2023, 2024, 2025], validation_season=None,
            validation_seasons=[2024, 2025], residual_source="walk_forward",
            validation_actual_wins=[1., 3.], validation_predicted_wins=[.5, 2.],
            validation_teams=["A", "B"],
        )
        self.assertEqual(bundle["residual_source"], "walk_forward")
        self.assertIsNone(bundle["validation_season"])
        np.testing.assert_array_equal(bundle["validation_seasons"], [2024, 2025])
        np.testing.assert_array_equal(bundle["validation_residuals"], [.5, 1.])

    def test_season_summary_and_seed_are_independent_of_row_order(self):
        features = pd.DataFrame([
            dict(self.team, team="B", season=2026),
            {"elo": 1900., "pace": 75., "team": "A", "season": 2026},
        ])
        table = simulate_season_success(features, self.bundle, 2026, 1000, 42, expected_teams=2)
        reordered = simulate_season_success(features.iloc[::-1], self.bundle, 2026, 1000, 42, expected_teams=2)
        pd.testing.assert_frame_equal(table, reordered)
        self.assertEqual(len(table.columns), 14)
        self.assertFalse(table.isna().any().any())
        self.assertTrue(table["simulated_mean"].is_monotonic_decreasing)
        stream = np.random.SeedSequence(42).spawn(2)[0]
        expected = simulate_team_success(features.loc[features["team"] == "A"], self.bundle, 1000, stream)
        self.assertEqual(table.loc[table["team"] == "A", "simulated_mean"].iloc[0], expected["mean"])

    def test_season_rejects_invalid_roster_and_count(self):
        features = pd.DataFrame([dict(self.team, team="A", season=2026)])
        for invalid in (pd.concat([features, features]), features.assign(team=None), features.assign(season=2025)):
            with self.subTest(), self.assertRaises(ValueError):
                simulate_season_success(invalid, self.bundle, 2026, expected_teams=1)
        with self.assertRaisesRegex(ValueError, "Expected 2 teams"):
            simulate_season_success(features, self.bundle, 2026, expected_teams=2)
        with self.assertRaisesRegex(ValueError, "expected_teams"):
            simulate_season_success(features, self.bundle, 2026)

    def test_season_checks_saved_baselines_and_master_provenance(self):
        features = pd.DataFrame([dict(self.team, team=team, season=2025) for team in ["A", "B", "C"]])
        baseline = self.model.predict(self.scaler.transform(features[["elo", "pace"]]))
        bundle = dict(self.bundle, validation_predicted_wins=baseline)
        table = simulate_season_success(features, bundle, 2025, n_simulations=10)
        self.assertEqual(len(table), 3)
        with self.assertRaisesRegex(ValueError, "saved prediction"):
            simulate_season_success(features.assign(elo=9999.), bundle, 2025, n_simulations=10)
        with self.assertRaisesRegex(ValueError, "roster"):
            simulate_season_success(features.iloc[:2], bundle, 2025, n_simulations=10)
        # Master baselines are not compared with predictions made by earlier folds.
        master = dict(self.bundle, residual_source="walk_forward", validation_season=None)
        self.assertEqual(len(simulate_season_success(features, master, 2025, n_simulations=10)), 3)

    def test_cli_exports_csv_and_correct_residual_provenance_for_new_season(self):
        features_path = Path(self.temp.name) / "features.csv"
        output = Path(self.temp.name) / "results"
        pd.DataFrame([dict(self.team, team="A", season=2026)]).to_csv(features_path, index=False)
        arguments = [
            "run_monte_carlo.py", "--season", "2026", "--bundle", str(self.path),
            "--features-csv", str(features_path), "--output-dir", str(output),
            "--n-simulations", "10", "--seed", "42", "--expected-teams", "1",
        ]
        with patch.object(sys, "argv", arguments), contextlib.redirect_stdout(io.StringIO()):
            run_season()
        table = pd.read_csv(output / "team_success_scores_2026.csv")
        metadata = json.loads((output / "team_success_scores_2026.metadata.json").read_text())
        self.assertEqual(table["team"].tolist(), ["A"])
        self.assertEqual(metadata["season"], 2026)
        self.assertEqual(metadata["validation_season"], 2025)
        self.assertIn(
            "The bundle's held-out residual pool calibrates these outputs; it is not an independent probability-validation set.",
            metadata["assumptions"],
        )

    def test_training_pipeline_exports_holdout_and_master_bundles(self):
        # Exercise the real orchestration on tiny synthetic seasons without
        # importing the optional plotting/SHAP stack or running API calls.
        source = Path(__file__).resolve().parents[1] / "models" / "ridge_shap.py"
        node = next(n for n in ast.parse(source.read_text()).body
                    if isinstance(n, ast.FunctionDef) and n.name == "main")
        early = pd.DataFrame({
            "team": [f"A{i}" for i in range(10)], "season": 2023,
            "target_wins": np.arange(10) % 4, "elo": np.arange(10) + 1000.,
        })
        later = pd.DataFrame({
            "team": [f"B{i}" for i in range(10)], "season": 2024,
            "target_wins": np.arange(10) % 5, "elo": np.arange(10) + 2000.,
            "future_only": np.arange(10) + 100.,
        })
        namespace = dict(
            os=os, np=np, pd=pd, joblib=joblib, Ridge=Ridge,
            StandardScaler=StandardScaler, save_ridge_bundle=save_ridge_bundle,
            mean_absolute_error=mean_absolute_error, mean_squared_error=mean_squared_error,
            ndcg_score=ndcg_score, r2_score=r2_score, spearmanr=spearmanr,
            train_test_split=train_test_split,
            build_team_features=lambda data, year, out: {2023: early, 2024: later}[year].copy(),
            compute_recency_weights=lambda seasons, **kwargs: np.ones(len(seasons)),
            optimize_ridge=lambda *args, **kwargs: {"alpha": 1.},
            generate_eda=lambda *args, **kwargs: None,
            generate_3d_umap=lambda *args: None,
            plot_accuracy_over_time=lambda *args: None,
            compute_shap=lambda *args: None,
        )
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), namespace)
        with contextlib.chdir(self.temp.name), contextlib.redirect_stdout(io.StringIO()):
            Path("data/2023").mkdir(parents=True)
            Path("data/2024").mkdir()
            namespace["main"]()
            fold = joblib.load("model_outputs/ridge/Train_2023_to_2023_Val_2024/ridge_bundle.joblib")
            master = joblib.load("model_outputs/ridge/all_years/ridge_master_bundle.joblib")
        self.assertEqual(fold["feature_names"], ["elo"])
        np.testing.assert_allclose(fold["scaler"].mean_, [1004.5])
        np.testing.assert_array_equal(fold["validation_actual_wins"], later["target_wins"])
        expected = fold["model"].predict(fold["scaler"].transform(later[["elo"]]))
        np.testing.assert_array_equal(fold["validation_predicted_wins"], expected)
        self.assertEqual(master["feature_names"], ["elo", "future_only"])
        np.testing.assert_array_equal(master["validation_residuals"], fold["validation_residuals"])
        self.assertEqual(master["residual_source"], "walk_forward")


if __name__ == "__main__":
    unittest.main()
