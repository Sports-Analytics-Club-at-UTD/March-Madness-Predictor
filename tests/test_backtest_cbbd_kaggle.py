#!/usr/bin/env python3
"""Unit tests for backtest_cbbd_kaggle.py."""

import unittest
from pathlib import Path
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, ".")

from backtest_cbbd_kaggle import (
    build_cbbd_to_kaggle_team_map,
    parse_tournament_round_from_notes,
    apply_sharpening_scalar,
    apply_sharpening_vector,
    compute_calibration_metrics,
    extract_game_matchup_vector,
    CBBDMatchupModel,
    FEATURE_COLS,
)


class TestCBBDKaggleBacktest(unittest.TestCase):
    """Test suite covering CBBD Kaggle data recreation and backtesting functions."""

    def test_team_mapping_aliases(self):
        """Verify that known colloquial team names map to valid Kaggle TeamIDs."""
        kaggle_dir = Path("data/kaggle/march-machine-learning-mania-2026")
        team_map = build_cbbd_to_kaggle_team_map(kaggle_dir)
        self.assertEqual(team_map.get("app state"), 1111)
        self.assertEqual(team_map.get("miami"), 1274)
        self.assertEqual(team_map.get("ualbany"), 1107)
        self.assertEqual(team_map.get("connecticut"), 1163)
        self.assertEqual(team_map.get("uconn"), 1163)
        self.assertEqual(team_map.get("purdue"), 1345)
        self.assertEqual(team_map.get("houston"), 1222)

    def test_round_notes_parser(self):
        """Verify notes text correctly translates to tournament round integers."""
        self.assertEqual(parse_tournament_round_from_notes("First Four"), 0)
        self.assertEqual(parse_tournament_round_from_notes("1st Round - Midwest Region"), 1)
        self.assertEqual(parse_tournament_round_from_notes("2nd Round - West Region"), 2)
        self.assertEqual(parse_tournament_round_from_notes("Sweet 16"), 3)
        self.assertEqual(parse_tournament_round_from_notes("Elite 8"), 4)
        self.assertEqual(parse_tournament_round_from_notes("Final Four"), 5)
        self.assertEqual(parse_tournament_round_from_notes("National Championship"), 6)
        self.assertEqual(parse_tournament_round_from_notes(None), 1)

    def test_sharpening_properties(self):
        """Verify Horan sharpening preserves 0.5 anchor and pushes extreme probabilities."""
        # 0.5 anchor
        self.assertAlmostEqual(apply_sharpening_scalar(0.5, gamma=1.20), 0.5, places=5)

        # Favorite push: 0.85 should become higher
        p_fav = apply_sharpening_scalar(0.85, gamma=1.20)
        self.assertGreater(p_fav, 0.85)

        # Underdog push: 0.15 should become lower
        p_dog = apply_sharpening_scalar(0.15, gamma=1.20)
        self.assertLess(p_dog, 0.15)

        # Vectorized consistency
        vec_in = np.array([0.15, 0.50, 0.85])
        vec_out = apply_sharpening_vector(vec_in, gamma=1.20)
        self.assertAlmostEqual(vec_out[0], p_dog, places=5)
        self.assertAlmostEqual(vec_out[1], 0.5, places=5)
        self.assertAlmostEqual(vec_out[2], p_fav, places=5)

    def test_extract_game_matchup_vector(self):
        """Verify feature differentials are correctly computed."""
        t1 = pd.Series({
            "Seed": 1, "AdjNet": 30.0, "SRS": 22.0, "Elo": 2200.0,
            "NetRtg": 25.0, "WinPct": 0.90, "eFG": 0.58, "TOV_Rate": 0.12,
            "ORB_Rate": 0.35, "FT_Rate": 0.38,
        })
        t2 = pd.Series({
            "Seed": 16, "AdjNet": -2.0, "SRS": -5.0, "Elo": 1600.0,
            "NetRtg": -1.0, "WinPct": 0.55, "eFG": 0.49, "TOV_Rate": 0.18,
            "ORB_Rate": 0.25, "FT_Rate": 0.28,
        })
        diffs = extract_game_matchup_vector(t1, t2)
        # Seed diff = 16 - 1 = +15 in favor of t1
        self.assertEqual(diffs["dSeed"], 15.0)
        self.assertEqual(diffs["dAdjNet"], 32.0)
        self.assertEqual(diffs["dSRS"], 27.0)
        self.assertEqual(diffs["dElo"], 600.0)
        self.assertGreater(diffs["dTOV"], 0.0)  # T1 committed fewer turnovers
        self.assertIn("dSeed_x_dAdjNet", diffs)
        self.assertEqual(diffs["dSeed_x_dAdjNet"], 15.0 * 32.0)

    def test_calibration_metrics(self):
        """Verify ECE and MCE computation on known probabilities."""
        y_true = np.array([1, 1, 1, 0, 0, 0])
        y_prob = np.array([0.9, 0.8, 0.85, 0.1, 0.2, 0.15])
        metrics = compute_calibration_metrics(y_true, y_prob, n_bins=5)
        self.assertIn("ece", metrics)
        self.assertIn("mce", metrics)
        self.assertLessEqual(metrics["ece"], 0.25)
        self.assertLessEqual(metrics["mce"], 0.50)

    def test_model_fit_and_predict(self):
        """Verify model trains and outputs valid probabilities in [0, 1]."""
        np.random.seed(42)
        n = 100
        X = np.random.randn(n, len(FEATURE_COLS))
        d_seed = np.random.randint(-15, 16, size=n)
        y = (X[:, 0] * 0.5 + d_seed * 0.1 + np.random.randn(n) * 0.2 > 0).astype(int)

        model = CBBDMatchupModel(sharpen_gamma=1.20)
        model.fit(X, y, d_seed)

        preds = model.predict_proba(X, d_seed)
        for key in ["p_5050", "p_seed", "p_lr", "p_raw", "p_sharp"]:
            self.assertIn(key, preds)
            self.assertEqual(len(preds[key]), n)
            self.assertTrue(np.all(preds[key] >= 0.0))
            self.assertTrue(np.all(preds[key] <= 1.0))


if __name__ == "__main__":
    unittest.main()
