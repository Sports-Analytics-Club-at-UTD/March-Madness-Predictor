"""Calibration symmetry and the target-season training boundary."""
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from kaggle_tuning import women as W


class WomenTuningTests(unittest.TestCase):
    def test_calibration_is_symmetric_and_bounded(self):
        oof = np.array([.01, .1, .2, .4, .6, .8, .9, .99])
        y = np.array([0, 0, 1, 0, 1, 0, 1, 1])
        p = np.array([.001, .2, .5, .8, .999])
        for method in ('temperature', 'isotonic'):
            calibrated = W.calibrate(method, oof, y, p)
            np.testing.assert_allclose(calibrated, 1 - W.calibrate(method, oof, y, 1-p), atol=1e-10)
            self.assertTrue(((calibrated >= 0) & (calibrated <= 1)).all())
            if method == 'isotonic':
                self.assertTrue(((calibrated >= .001 - 1e-12) & (calibrated <= .999 + 1e-12)).all())

    def test_original_features_exclude_late_regular_season_rows(self):
        reg = pd.DataFrame({'DayNum': [132, 133]})
        stats = pd.DataFrame({'Season':[2015], 'TeamID':[3001]})
        seeds = pd.DataFrame({'Season':[2015], 'TeamID':[3001], 'seed':[1]})
        W.original_table.cache_clear()
        with patch.object(W.F, 'read', return_value=reg), patch.object(W.F, 'seeds', return_value=seeds), patch('kaggle_mania_brackets.compute_team_season_stats', return_value=stats) as compute:
            W.original_table()
            self.assertEqual(compute.call_args.args[0].DayNum.tolist(), [132])
        W.original_table.cache_clear()

    def test_target_season_never_enters_training_or_calibration(self):
        games = pd.DataFrame({'Season': [2013,2013,2014,2014,2015,2015], 'margin': [1,-1]*3})
        calls = []
        def frame(pairs):
            return pairs.Season.to_numpy().reshape(-1,1), pairs
        def classifier(cfg, X, y, test):
            calls.append(X.copy())
            return np.full(len(test), .5)
        with patch.object(W.F, 'tourney_games', return_value=games), patch.object(W, 'original_frame', side_effect=frame), patch.object(W, 'classifier', side_effect=classifier), patch('kaggle_tuning.backtest.test_pairs', return_value=pd.DataFrame({'Season':[2015]})):
            W.predict({'model':'women_original', 'women_calibration':'temperature'}, 2015, None)
        self.assertEqual(len(calls), 3)
        self.assertTrue(all(np.all(X < 2015) for X in calls))
        self.assertTrue(all(len(np.unique(X)) == 1 for X in calls[1:]))


if __name__ == '__main__':
    unittest.main()
