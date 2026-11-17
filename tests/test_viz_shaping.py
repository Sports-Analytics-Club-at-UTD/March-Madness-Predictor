import unittest

import numpy as np
import pandas as pd

from viz.shaping import (actual_wins, advancement_matrix, bracket_tree, brier_by_season, exact_win_probs,
                         prob_col, reliability_table)


class ShapingTests(unittest.TestCase):
    def test_exact_win_probs_differences_thresholds(self):
        adv = np.array([[1.0, 0.5, 0.25, 0.25, 0.1, 0.0]])
        out = exact_win_probs(adv)
        np.testing.assert_allclose(out, [[0.0, 0.5, 0.25, 0.0, 0.15, 0.1, 0.0]])
        self.assertAlmostEqual(out.sum(), 1.0)
        with self.assertRaises(ValueError):
            exact_win_probs(np.array([[0.5, 0.6, 0, 0, 0, 0]]))

    def test_advancement_matrix_prefers_tree_columns(self):
        raw = {prob_col("probability_ge_", k): [0.1] for k in range(1, 7)}
        tree = {prob_col("tree_prob_ge_", k): [0.9 - 0.1 * k] for k in range(1, 7)}
        adv, label = advancement_matrix(pd.DataFrame({**raw, **tree}))
        self.assertEqual(label, "bracket-tree simulation")
        np.testing.assert_allclose(adv[0], [0.8, 0.7, 0.6, 0.5, 0.4, 0.3])
        _, label = advancement_matrix(pd.DataFrame({**raw, **{c: [np.nan] for c in tree}}))
        self.assertEqual(label, "simulated frequency")

    def test_reliability_table_counts_both_sides(self):
        r = reliability_table([0.9, 0.9, 0.2], [1, 0, 0], n_bins=10)
        self.assertEqual(r.n.sum(), 6)
        top = r[r.lo == 0.9].iloc[0]
        self.assertAlmostEqual(top.mean_p, 0.9)
        self.assertAlmostEqual(top.frac, 0.5)
        # mirror symmetry: the bin at 1-p has the complementary observed rate
        self.assertAlmostEqual(r[r.lo == 0.0].iloc[0].frac, 0.5)

    def test_brier_by_season_pools_genders_for_combined(self):
        games = pd.DataFrame({"Season": [2025] * 3, "g": ["M", "M", "W"], "y": [1, 0, 1],
                              "a": [1.0, 1.0, 0.5], "b": [np.nan, np.nan, 0.0]})
        t = brier_by_season(games, ["a", "b"]).set_index(["scope", "series"])
        self.assertAlmostEqual(t.loc[("Combined", "a"), "brier"], (0 + 1 + 0.25) / 3)
        self.assertAlmostEqual(t.loc[("Men", "a"), "brier"], 0.5)
        self.assertEqual(t.loc[("Combined", "b"), "n"], 1)
        self.assertNotIn(("Men", "b"), t.index)

    def test_bracket_tree_resolves_play_in_and_orders_rounds(self):
        seeds = pd.DataFrame({"Seed": ["W01", "W02", "W03", "W04a", "W04b"], "TeamID": [1, 2, 3, 4, 5]})
        slots = pd.DataFrame({"Slot": ["W04", "R1W1", "R1W2", "R2W1"],
                              "StrongSeed": ["W04a", "W01", "W02", "R1W1"],
                              "WeakSeed": ["W04b", "W04", "W03", "R1W2"]})
        teams, pairs, edges = bracket_tree(slots, seeds, field={1, 2, 3, 5})
        self.assertEqual(teams, [1, 5, 2, 3])
        self.assertEqual(pairs, [(0, 1), (2, 3)])
        self.assertEqual(edges, [(0, 1)])
        with self.assertRaises(ValueError):
            bracket_tree(slots, seeds, field={1, 2, 3, 4, 5})

    def test_actual_wins_counts_losers_as_zero(self):
        played = pd.DataFrame({"T1": [1, 3, 1], "T2": [2, 4, 3], "y": [1, 0, 0]})
        self.assertEqual(actual_wins(played).to_dict(), {1: 1, 2: 0, 3: 1, 4: 1})


if __name__ == "__main__":
    unittest.main()
