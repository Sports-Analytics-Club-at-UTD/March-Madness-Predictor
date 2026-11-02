import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from combo_eval import simulate_blind_bracket
from run_pipeline import discover_top_ensembles
from shap_monte_carlo import load_shap_recipe


class ConfidenceAndTop25Tests(unittest.TestCase):
    def test_game_confidence_properties(self):
        # 0 diff should give 0.5 confidence
        diff_zero = 0.0
        conf_zero = 1.0 / (1.0 + np.exp(-0.85 * diff_zero))
        self.assertAlmostEqual(conf_zero, 0.5)

        # Monotonically increasing
        diffs = [0.0, 0.5, 1.0, 2.0, 3.0]
        confs = [1.0 / (1.0 + np.exp(-0.85 * d)) for d in diffs]
        self.assertTrue(all(confs[i] <= confs[i + 1] for i in range(len(confs) - 1)))
        self.assertTrue(all(0.5 <= c <= 1.0 for c in confs))

    def test_simulate_blind_bracket_returns_valid_confidence(self):
        # Build minimal mock 63 games bracket
        teams = [f"Team_{i}" for i in range(64)]
        games = []
        # Round 1: 32 games
        for i in range(32):
            h, a = teams[2 * i], teams[2 * i + 1]
            games.append({
                'homeTeam': h,
                'awayTeam': a,
                'Actual_Winner': h,
                'homeWinner': True
            })
        # Round 2: 16 games
        for i in range(16):
            h, a = teams[4 * i], teams[4 * i + 2]
            games.append({
                'homeTeam': h,
                'awayTeam': a,
                'Actual_Winner': h,
                'homeWinner': True
            })
        # Round 3: 8 games
        for i in range(8):
            h, a = teams[8 * i], teams[8 * i + 4]
            games.append({
                'homeTeam': h,
                'awayTeam': a,
                'Actual_Winner': h,
                'homeWinner': True
            })
        # Round 4: 4 games
        for i in range(4):
            h, a = teams[16 * i], teams[16 * i + 8]
            games.append({
                'homeTeam': h,
                'awayTeam': a,
                'Actual_Winner': h,
                'homeWinner': True
            })
        # Round 5: 2 games
        for i in range(2):
            h, a = teams[32 * i], teams[32 * i + 16]
            games.append({
                'homeTeam': h,
                'awayTeam': a,
                'Actual_Winner': h,
                'homeWinner': True
            })
        # Round 6: 1 game
        games.append({
            'homeTeam': teams[0],
            'awayTeam': teams[32],
            'Actual_Winner': teams[0],
            'homeWinner': True
        })

        df_games = pd.DataFrame(games)
        scores = {t: float(64 - idx) for idx, t in enumerate(teams)}

        blind_error, bracket_conf, mean_game_conf, expected_pts = simulate_blind_bracket(df_games, scores)
        self.assertGreaterEqual(blind_error, 0.0)
        self.assertGreaterEqual(bracket_conf, 0.5)
        self.assertLessEqual(bracket_conf, 1.0)
        self.assertGreaterEqual(mean_game_conf, 0.5)
        self.assertLessEqual(mean_game_conf, 1.0)
        self.assertGreater(expected_pts, 0.0)
        self.assertLessEqual(expected_pts, 1920.0)

    def test_discover_top_ensembles_prioritizes_top_25(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ensemble_dir = Path(tmpdir)
            top20_dir = ensemble_dir / "top_20_shap_weights"
            top25_dir = ensemble_dir / "top_25_shap_weights"
            top20_dir.mkdir(parents=True)
            top25_dir.mkdir(parents=True)

            # Create Rank_1 in top_20 and Rank_1..Rank_25 in top_25
            for r in range(1, 21):
                (top20_dir / f"Rank_{r}_Test_SHAP.csv").write_text("feature,shap_importance\nx,1.0\n")
            for r in range(1, 26):
                (top25_dir / f"Rank_{r}_Test_SHAP.csv").write_text("feature,shap_importance\nx,1.0\n")

            discovered = discover_top_ensembles(ensemble_dir)
            self.assertEqual(len(discovered), 25)
            self.assertEqual(discovered[0][0], 1)
            self.assertEqual(discovered[-1][0], 25)
            self.assertEqual(discovered[0][1].parent.name, "top_25_shap_weights")

    def test_load_shap_recipe_supports_top_25(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ensemble_dir = Path(tmpdir)
            top25_dir = ensemble_dir / "top_25_shap_weights"
            top25_dir.mkdir(parents=True)
            (top25_dir / "Rank_1_Ridge_SHAP.csv").write_text("feature,shap_importance\nx,0.5\ny,0.3\n")
            
            top25_csv = ensemble_dir / "TOP_25_BEST_PERFORMING_METRICS.csv"
            df_metrics = pd.DataFrame([{
                'Rank': 1,
                'Ensemble_Name': 'Ridge',
                'Model_Count': 1,
                'Global_Accuracy': 0.72,
                'Global_Per_Game_Confidence': 0.71,
                'Global_Per_Bracket_Confidence': 0.69,
                'Global_Expected_Points': 1250.0,
                'Global_Blind_Error': 0.65,
                'Global_NDCG': 0.86,
                'Global_Spearman': 0.52,
                'Global_MARE': 14.1,
                'Composite_Rank_Score': 10.5
            }])
            df_metrics.to_csv(top25_csv, index=False)

            df, info = load_shap_recipe(ensemble_dir)
            self.assertEqual(len(df), 2)
            self.assertEqual(info['ensemble_rank'], 1)
            self.assertIn('Global_Per_Bracket_Confidence', info['ensemble_metrics'])
            self.assertAlmostEqual(info['ensemble_metrics']['Global_Per_Bracket_Confidence'], 0.69)

    def test_feature_polarity_classification(self):
        from combo_eval import get_feature_polarity
        # Detrimental stats should return -1.0
        self.assertEqual(get_feature_polarity("game_mean_opponentStats.points.total"), -1.0)
        self.assertEqual(get_feature_polarity("game_mean_opponentStats.twoPointFieldGoals.pct"), -1.0)
        self.assertEqual(get_feature_polarity("starters_mean_turnovers"), -1.0)
        self.assertEqual(get_feature_polarity("game_mean_calc_turnover_ratio"), -1.0)
        self.assertEqual(get_feature_polarity("starters_mean_fouls"), -1.0)

        # Opponent mistakes that benefit the team should return +1.0
        self.assertEqual(get_feature_polarity("game_mean_opponentStats.turnovers"), 1.0)
        self.assertEqual(get_feature_polarity("game_mean_opponentStats.fouls.total"), 1.0)

        # Standard offensive/efficiency stats should return +1.0
        self.assertEqual(get_feature_polarity("game_mean_homeTeamEloStart"), 1.0)
        self.assertEqual(get_feature_polarity("starters_mean_offensiveRating"), 1.0)
        self.assertEqual(get_feature_polarity("starters_mean_points"), 1.0)
        self.assertEqual(get_feature_polarity("starters_mean_winShares.total"), 1.0)


if __name__ == "__main__":
    unittest.main()
