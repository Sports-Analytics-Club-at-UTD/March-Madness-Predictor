"""Women's regular-season consensus ratings and walk-forward tournament classifier."""
from functools import lru_cache

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler

from kaggle_tuning import features as F


@lru_cache(None)
def ratings():
    games = F.team_games('W').query('DayNum <= 132 and T1 < T2')
    rows = []
    for season, s in games.groupby('Season'):
        teams = np.unique(np.r_[s.T1, s.T2])
        n, k = len(s), len(teams)
        x = sparse.csr_matrix((np.r_[np.ones(n), -np.ones(n)],
            (np.r_[np.arange(n), np.arange(n)],
             np.r_[np.searchsorted(teams, s.T1), np.searchsorted(teams, s.T2)])), shape=(n, k))
        x = sparse.hstack([x, s['loc'].to_numpy()[:, None]], format='csr')
        poss = (F.possessions(s, 'T1_') + F.possessions(s, 'T2_')) / 2
        efficiency = Ridge(alpha=1, fit_intercept=False).fit(x, np.clip(100 * s.margin / poss, -60, 60)).coef_[:k]
        target = expit(s.margin.to_numpy() / 7)
        def objective(beta):
            z = x @ beta
            return (np.logaddexp(0, z).sum() - target @ z + .05 * (beta @ beta),
                    x.T @ (expit(z) - target) + .1 * beta)
        fit = minimize(objective, np.zeros(k + 1), jac=True, method='L-BFGS-B', options={'maxiter': 300})
        assert fit.success, fit.message
        rows.append(pd.DataFrame({'Season': season, 'TeamID': teams,
                                  'bt': fit.x[:k], 'adj_eff': efficiency}))
    return pd.concat(rows, ignore_index=True)


def predict(cfg, year, table):
    from kaggle_tuning.backtest import test_pairs
    feats = cfg.get('consensus_features', ['seed', 'quality', 'elo', 'bt', 'adj_eff'])
    table = table.merge(ratings(), on=['Season', 'TeamID'], how='left')
    tr = F.tourney_games('W').query('2010 <= Season < @year')
    assert tr.Season.max() < year
    te = test_pairs('W', year)
    cols = ['diff_' + f for f in feats]
    x = F.matchup_frame(tr, table, feats)[cols].fillna(0).to_numpy()
    z = F.matchup_frame(te, table, feats)[cols].fillna(0).to_numpy()
    if cfg.get('consensus_original'):
        from kaggle_tuning.women import original_frame
        x = np.c_[original_frame(tr)[0], x]
        z = np.c_[original_frame(te)[0], z]
    scale = StandardScaler().fit(x)
    model = LogisticRegression(C=cfg.get('C', .1), fit_intercept=False, max_iter=2000).fit(scale.transform(x), (tr.margin > 0).astype(int))
    return model.predict_proba(scale.transform(z))[:, 1]
