"""Synthetic checks for the issue #67 combiners (no Kaggle data needed)."""
import unittest

import numpy as np

from kaggle_tuning.strat_reeval import fit_sigma, hedge_weights, history_seasons, stack_weights


class HedgeTest(unittest.TestCase):
    def test_equal_losses_keep_prior(self):
        np.testing.assert_allclose(hedge_weights([1, 1, 2], [[.2, .2, .2]] * 3, 8), [.25, .25, .5])

    def test_lower_loss_gains_weight(self):
        w = hedge_weights([1, 1], [[.15, .20], [.16, .21]], 8)
        np.testing.assert_allclose(w, np.array([1, np.exp(-8 * .1)]) / (1 + np.exp(-8 * .1)))

    def test_no_history_is_prior(self):
        np.testing.assert_allclose(hedge_weights([3, 1], np.empty((0, 2)), 8), [.75, .25])


class HistoryTest(unittest.TestCase):
    def test_strictly_before_target_and_frozen_at_2024(self):
        self.assertEqual(history_seasons(2015), [])
        self.assertEqual(history_seasons(2021, (2012, 2013)), [2012, 2013, 2015, 2016, 2017, 2018, 2019])
        self.assertEqual(history_seasons(2026), history_seasons(2025))
        self.assertEqual(max(history_seasons(2026)), 2024)


class StackTest(unittest.TestCase):
    def test_recovers_coefficients_and_ridge_pulls_to_ref(self):
        rng = np.random.default_rng(0)
        Z = rng.normal(size=(20000, 2))
        y = (rng.random(len(Z)) < 1 / (1 + np.exp(-Z @ [1.2, .3]))).astype(float)
        Z2, y2 = np.vstack([Z, -Z]), np.r_[y, 1 - y]
        np.testing.assert_allclose(stack_weights(Z2, y2, [1, 1], 0), [1.2, .3], atol=.06)
        np.testing.assert_allclose(stack_weights(Z2, y2, [1, 1], 1e9), [.5, .5], atol=1e-3)

    def test_symmetric(self):
        rng = np.random.default_rng(1)
        Z = rng.normal(size=(500, 3))
        y = (rng.random(500) < .5).astype(float)
        b = stack_weights(np.vstack([Z, -Z]), np.r_[y, 1 - y], [1, 1, 1], 5)
        z = rng.normal(size=3)
        self.assertAlmostEqual(1 / (1 + np.exp(-z @ b)) + 1 / (1 + np.exp(z @ b)), 1.0)


class LeakageTest(unittest.TestCase):
    def test_combiners_only_read_outcomes_before_target(self):
        from unittest import mock
        import pandas as pd
        from kaggle_tuning import backtest, strat_reeval as S
        outcomes, preds = [], []
        rng = np.random.default_rng(3)
        pairs = lambda g, h: outcomes.append(h) or pd.DataFrame({"y": rng.integers(0, 2, 10)})
        member = lambda cfg, sub, g, h, table: preds.append(h) or rng.random(10)
        cfg = {"members": {"W": [({"a": 1}, 1), ({"b": 1}, 1)]}}
        with mock.patch.object(backtest, "test_pairs", pairs), mock.patch.object(S, "member_pred", member):
            for model, opts in [("strat_hedge", {}), ("strat_stack", {"warmup": (2012,)})]:
                for year in (2015, 2019, 2026):
                    outcomes.clear(); preds.clear()
                    S.combine({**cfg, "model": model, "combiner": opts}, "W", year, None)
                    self.assertTrue(all(h < min(year, 2025) for h in outcomes), (model, year, outcomes))
                    self.assertTrue(all(h < min(year, 2025) or h == year for h in preds))


class SigmaTest(unittest.TestCase):
    def test_recovers_sigma(self):
        from scipy.stats import norm
        rng = np.random.default_rng(2)
        m = rng.normal(0, 10, 50000)
        y = (rng.random(len(m)) < norm.cdf(m / 11)).astype(float)
        self.assertAlmostEqual(fit_sigma(m, y), 11, delta=.5)


if __name__ == "__main__":
    unittest.main()
