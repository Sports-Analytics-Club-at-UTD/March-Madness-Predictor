"""Kaggle submission validator and the as-of-Selection-Sunday data cutoff."""
import unittest

import numpy as np
import pandas as pd

from kaggle_tuning.submit import as_of, validate

SAMPLE = pd.DataFrame({"ID": ["2026_1101_1102", "2026_1101_1103", "2026_3101_3102"], "Pred": 0.5})


class ValidateTests(unittest.TestCase):
    def good(self):
        return SAMPLE.assign(Pred=[0.0, 0.7, 1.0])

    def assertInvalid(self, sub, msg):
        with self.assertRaisesRegex(ValueError, msg):
            validate(sub, SAMPLE)

    def test_accepts_exact_sample_rows(self):
        validate(self.good(), SAMPLE)

    def test_rejects_bad_rows_and_values(self):
        g = self.good()
        self.assertInvalid(g.iloc[:2], "(?s)2 rows.*1 sample IDs missing")
        self.assertInvalid(g.iloc[[1, 0, 2]], "not in sample order")
        self.assertInvalid(pd.concat([g, g.iloc[:1]]), "duplicate")
        self.assertInvalid(g.assign(ID=g.ID.replace("2026_1101_1103", "2026_1101_1104")), "missing, 1 extra")
        self.assertInvalid(g.assign(Pred=[np.nan, 0.5, 0.5]), "NaN")
        self.assertInvalid(g.assign(Pred=[np.inf, 0.5, 0.5]), "NaN/inf")
        self.assertInvalid(g.assign(Pred=["x", 0.5, 0.5]), "non-numeric")
        self.assertInvalid(g.assign(Pred=[-0.1, 0.5, 1.2]), "2 Pred values outside")
        self.assertInvalid(g.assign(extra=1), "columns")

    def test_rejects_higher_id_first(self):
        sample = pd.DataFrame({"ID": ["2026_1102_1101"], "Pred": 0.5})
        with self.assertRaisesRegex(ValueError, "XXXX < YYYY"):
            validate(sample, sample)


class AsOfTests(unittest.TestCase):
    def test_drops_target_tournament_and_later_rows(self):
        games = pd.DataFrame({"Season": [2025, 2026, 2026, 2027], "DayNum": [140, 132, 133, 10]})
        self.assertEqual(as_of(games, "MRegularSeasonCompactResults", 2026).values.tolist(), [[2025, 140], [2026, 132]])
        self.assertEqual(as_of(games, "MNCAATourneyCompactResults", 2026).Season.tolist(), [2025])
        ranks = pd.DataFrame({"Season": [2026, 2026], "RankingDayNum": [133, 140]})
        self.assertEqual(as_of(ranks, "MMasseyOrdinals", 2026).RankingDayNum.tolist(), [133])
        seeds = pd.DataFrame({"Season": [2026, 2027], "Seed": ["W01", "W01"]})
        self.assertEqual(as_of(seeds, "MNCAATourneySeeds", 2026).Season.tolist(), [2026])


if __name__ == "__main__":
    unittest.main()
