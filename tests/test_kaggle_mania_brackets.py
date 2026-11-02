"""Unit tests for kaggle_mania_brackets.py."""

import unittest
from pathlib import Path
import numpy as np
import pandas as pd

from kaggle_mania_brackets import (
    extract_seed_number,
    get_slot_round,
    get_game_round,
    sort_slots_topologically,
    solve_dp_optimal_bracket,
    simulate_stochastic_bracket,
    generate_projected_2027_field,
    CalibratedMatchupModel,
    apply_sharpening_scalar,
    apply_sharpening_vector,
    compute_calibration_metrics,
    evaluate_kaggle_submission_file,
)


class TestKaggleManiaBrackets(unittest.TestCase):

    def test_extract_seed_number(self):
        self.assertEqual(extract_seed_number("W01"), 1)
        self.assertEqual(extract_seed_number("X16a"), 16)
        self.assertEqual(extract_seed_number("Y11b"), 11)
        self.assertEqual(extract_seed_number("Z08"), 8)
        self.assertEqual(extract_seed_number("None"), 8)

    def test_get_slot_round(self):
        self.assertEqual(get_slot_round("X16"), 0)
        self.assertEqual(get_slot_round("Y11"), 0)
        self.assertEqual(get_slot_round("R1W1"), 1)
        self.assertEqual(get_slot_round("R2X3"), 2)
        self.assertEqual(get_slot_round("R3Y1"), 3)
        self.assertEqual(get_slot_round("R4Z1"), 4)
        self.assertEqual(get_slot_round("R5WX"), 5)
        self.assertEqual(get_slot_round("R6CH"), 6)

    def test_projected_2027_field_structure(self):
        teams_df = pd.DataFrame({
            "TeamID": list(range(1101, 1101 + 100)),
            "TeamName": [f"Team_{i}" for i in range(1101, 1101 + 100)],
        })
        stats_df = pd.DataFrame({
            "Season": [2026] * 100,
            "TeamID": list(range(1101, 1101 + 100)),
            "NetRtg": np.linspace(25, -10, 100),
            "OffRtg": np.linspace(115, 95, 100),
            "DefRtg": np.linspace(90, 105, 100),
        })

        seeds_27, slots_27, stats_27 = generate_projected_2027_field(teams_df, stats_df)
        self.assertEqual(len(seeds_27), 68)
        self.assertEqual(len(slots_27), 67)
        self.assertEqual(len(stats_27), 68)

        # Check topological sorting works cleanly
        ordered = sort_slots_topologically(slots_27, set(seeds_27["Seed"]))
        self.assertEqual(len(ordered), 67)

    def test_dp_optimizer_and_anchoring(self):
        # Build mini 3-game tournament: 4 teams, 2 semifinals, 1 championship
        # Seeds: W01 (101), W04 (104), W02 (102), W03 (103)
        seed_team = {"W01": 101, "W04": 104, "W02": 102, "W03": 103}
        slots_df = pd.DataFrame([
            {"Season": 2026, "Slot": "R1W1", "StrongSeed": "W01", "WeakSeed": "W04"},
            {"Season": 2026, "Slot": "R1W2", "StrongSeed": "W02", "WeakSeed": "W03"},
            {"Season": 2026, "Slot": "R6CH", "StrongSeed": "R1W1", "WeakSeed": "R1W2"},
        ])

        # Team 101 is heavy favorite, but team 102 can be anchored
        p_matrix = {
            (101, 104): 0.90, (104, 101): 0.10,
            (102, 103): 0.70, (103, 102): 0.30,
            (101, 102): 0.65, (102, 101): 0.35,
            (101, 103): 0.80, (103, 101): 0.20,
            (104, 102): 0.25, (102, 104): 0.75,
            (104, 103): 0.35, (103, 104): 0.65,
        }

        # 1. Unanchored Chalk DP
        picks_chalk = solve_dp_optimal_bracket(
            ordered_slots=slots_df,
            seed_team_map=seed_team,
            p_matrix=p_matrix,
        )
        self.assertEqual(picks_chalk["R6CH"]["winner_id"], 101)

        # 2. Anchored on Team 102 as Champion
        picks_anchored = solve_dp_optimal_bracket(
            ordered_slots=slots_df,
            seed_team_map=seed_team,
            p_matrix=p_matrix,
            anchor_champion=102,
        )
        self.assertEqual(picks_anchored["R6CH"]["winner_id"], 102)

        # 3. Stochastic simulation produces valid slots
        rng = np.random.default_rng(42)
        picks_sim = simulate_stochastic_bracket(
            ordered_slots=slots_df,
            seed_team_map=seed_team,
            p_matrix=p_matrix,
            rng=rng,
        )
        self.assertIn("R6CH", picks_sim)
        self.assertIn(picks_sim["R6CH"]["winner_id"], [101, 102, 103, 104])

    def test_get_game_round(self):
        self.assertEqual(get_game_round(134), 0)
        self.assertEqual(get_game_round(135), 0)
        self.assertEqual(get_game_round(136), 1)
        self.assertEqual(get_game_round(137), 1)
        self.assertEqual(get_game_round(138), 2)
        self.assertEqual(get_game_round(140), 2)
        self.assertEqual(get_game_round(143), 3)
        self.assertEqual(get_game_round(144), 3)
        self.assertEqual(get_game_round(145), 4)
        self.assertEqual(get_game_round(148), 4)
        self.assertEqual(get_game_round(152), 5)
        self.assertEqual(get_game_round(154), 6)

    def test_sharpening_functions(self):
        # Extreme high confidence
        self.assertAlmostEqual(apply_sharpening_scalar(0.97, d_seed=5), 0.992)
        # Extreme low confidence
        self.assertAlmostEqual(apply_sharpening_scalar(0.03, d_seed=-5), 0.008)
        # Mismatch boost
        self.assertAlmostEqual(apply_sharpening_scalar(0.91, d_seed=12), 0.965)
        self.assertAlmostEqual(apply_sharpening_scalar(0.09, d_seed=-12), 0.035)
        # Normal moderate confidence unchanged
        self.assertAlmostEqual(apply_sharpening_scalar(0.65, d_seed=2), 0.65)

        # Vectorized test
        p_vec = np.array([0.97, 0.03, 0.91, 0.65])
        d_seed_vec = np.array([5, -5, 12, 2])
        sharp_vec = apply_sharpening_vector(p_vec, d_seed_vec)
        self.assertAlmostEqual(sharp_vec[0], 0.992)
        self.assertAlmostEqual(sharp_vec[1], 0.008)
        self.assertAlmostEqual(sharp_vec[2], 0.965)
        self.assertAlmostEqual(sharp_vec[3], 0.65)

    def test_compute_calibration_metrics(self):
        # Perfect calibration synthetic data
        y_true = np.array([0, 0, 1, 1, 0, 1, 1, 1, 1, 1])
        y_prob = np.array([0.1, 0.1, 0.9, 0.9, 0.4, 0.6, 0.8, 0.8, 0.9, 1.0])
        metrics = compute_calibration_metrics(y_true, y_prob, n_bins=5)
        self.assertIn("ece", metrics)
        self.assertIn("mce", metrics)
        self.assertIn("bins", metrics)
        self.assertGreaterEqual(metrics["ece"], 0.0)
        self.assertLessEqual(metrics["ece"], 1.0)

    def test_evaluate_kaggle_submission_file(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            # Create mock submission CSV: ID format {Season}_{Team1}_{Team2} (Team1 < Team2)
            sub_csv = tmp_path / "test_sub.csv"
            pd.DataFrame([
                {"ID": "2024_1001_1002", "Pred": 0.80},
                {"ID": "2024_1001_1003", "Pred": 0.40},
                {"ID": "2024_1002_1003", "Pred": 0.60},
            ]).to_csv(sub_csv, index=False)

            # Mock actual compact tournament results:
            # Game 1: 1001 beat 1002 (W=1001, L=1002) -> actual = 1, pred = 0.80, sq_err = 0.04
            # Game 2: 1003 beat 1001 (W=1003, L=1001) -> 1001 < 1003, so actual = 0, pred = 0.40, sq_err = 0.16
            tourney_df = pd.DataFrame([
                {"Season": 2024, "DayNum": 136, "WTeamID": 1001, "LTeamID": 1002},
                {"Season": 2024, "DayNum": 138, "WTeamID": 1003, "LTeamID": 1001},
            ])

            res = evaluate_kaggle_submission_file(sub_csv, tourney_df, season=2024)
            self.assertEqual(res["status"], "SUCCESS")
            self.assertEqual(res["n_games_evaluated"], 2)
            # Mean of (0.04 + 0.16) / 2 = 0.10000
            self.assertAlmostEqual(res["brier_score"], 0.1000, places=4)
            # Both predictions on correct side of 0.5 (0.80 >= 0.5 for actual 1, 0.40 < 0.5 for actual 0)
            self.assertEqual(res["accuracy"], 1.0)
            self.assertEqual(res["baseline_5050_brier"], 0.25)
            # Improvement = (0.25 - 0.10) / 0.25 = 60%
            self.assertAlmostEqual(res["improvement_vs_5050_pct"], 60.0, places=1)


if __name__ == "__main__":
    unittest.main()
