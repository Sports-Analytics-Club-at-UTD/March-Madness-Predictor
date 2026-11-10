"""Synthetic checks for season isolation, scaling, and matchup symmetry."""
import unittest
import numpy as np
from kaggle_tuning.penalized import fold_arrays, estimator


class PenalizedTests(unittest.TestCase):
    def test_scaler_excludes_held_out_season(self):
        x = np.array([[1.], [3.], [100.], [200.]])
        seasons = np.array([2010, 2010, 2011, 2011])
        folds = list(fold_arrays(x, seasons))
        mask, train, valid = folds[1]
        np.testing.assert_array_equal(mask, [False, False, True, True])
        np.testing.assert_allclose(train.ravel(), [-1, 1])
        np.testing.assert_allclose(valid.ravel(), [98, 198])

    def test_all_penalties_respect_matchup_reversal(self):
        rng = np.random.default_rng(19)
        x = rng.normal(size=(80, 4))
        x = np.r_[x, -x]
        margin = x[:, 0]*4 + x[:, 1]
        for kind in ['l1','l2','en25','en50','en75','lasso_margin','ridge_margin']:
            m = estimator(kind,.1).fit(x, margin if kind.endswith('margin') else margin>0)
            if kind.endswith('margin'):
                np.testing.assert_allclose(m.predict(x), -m.predict(-x), atol=1e-10)
            else:
                np.testing.assert_allclose(m.predict_proba(x)[:,1],1-m.predict_proba(-x)[:,1],atol=1e-10)


if __name__ == '__main__':
    unittest.main()
