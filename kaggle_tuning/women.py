"""Women's original-pipeline features and historical-season calibration experiments."""
from functools import lru_cache

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import expit, logit
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from . import features as F


@lru_cache(None)
def original_table():
    from kaggle_mania_brackets import compute_team_season_stats
    reg = F.read('WRegularSeasonDetailedResults')
    return compute_team_season_stats(reg[reg.DayNum <= 132]).merge(F.seeds('W'), on=['Season', 'TeamID'])


def original_frame(pairs):
    names = ['seed', 'NetRtg', 'OffRtg', 'DefRtg', 'eFG', 'TOV', 'ORB', 'FTR', 'Win', 'QualityWins', 'SoS']
    frame = F.matchup_frame(pairs, original_table(), names)
    values = [-frame.diff_seed, np.zeros(len(frame)), frame.diff_NetRtg,
              frame.diff_OffRtg, -frame.diff_DefRtg, 100 * frame.diff_eFG,
              -100 * frame.diff_TOV, 100 * frame.diff_ORB, 100 * frame.diff_FTR,
              100 * frame.diff_Win, frame.diff_QualityWins, frame.diff_SoS]
    return np.column_stack(values), frame


def classifier(cfg, X, y, test):
    sc = StandardScaler().fit(X)
    X, test = sc.transform(X), sc.transform(test)
    lr = LogisticRegression(C=cfg.get('C_W', cfg.get('C', .25)), max_iter=2000, random_state=42).fit(X, y)
    p = lr.predict_proba(test)[:, 1]
    weight = cfg.get('original_lgb_weight', .35)
    if weight:
        from lightgbm import LGBMClassifier
        model = LGBMClassifier(n_estimators=85, learning_rate=.03, num_leaves=15,
                               min_child_samples=20, random_state=42, verbose=-1, n_jobs=1)
        model.fit(X, y)
        p = (1 - weight) * p + weight * model.predict_proba(test)[:, 1]
    return p


def calibrate(method, oof, y, p):
    """Symmetric calibration, fitted exclusively on training seasons' LOSO predictions."""
    if method == 'isotonic':
        iso = IsotonicRegression(y_min=.001, y_max=.999, out_of_bounds='clip').fit(oof, y)
        return (iso.predict(p) + 1 - iso.predict(1 - p)) / 2
    if method == 'temperature':
        logits = logit(np.clip(oof, .001, .999))
        result = minimize_scalar(lambda a: np.mean((expit(a * logits) - y) ** 2), bounds=(.5, 2), method='bounded')
        return expit(result.x * logit(np.clip(p, .001, .999)))
    raise ValueError(method)


def predict(cfg, year, table):
    from .backtest import test_pairs
    tr = F.tourney_games('W')
    tr = tr[(tr.Season < year) & (tr.Season >= cfg.get('start_W', 2010))]
    te = test_pairs('W', year)
    original = cfg['model'] == 'women_original'
    if original:
        X, _ = original_frame(tr)
        Z, _ = original_frame(te)
        if cfg.get('women_extra'):
            cols = ['diff_' + f for f in cfg['women_extra']]
            X = np.c_[X, F.matchup_frame(tr, table, cfg['women_extra'])[cols].fillna(0).values]
            Z = np.c_[Z, F.matchup_frame(te, table, cfg['women_extra'])[cols].fillna(0).values]
    else:
        cols = ['diff_' + f for f in cfg['W']]
        X = F.matchup_frame(tr, table, cfg['W'])[cols].fillna(0).values
        Z = F.matchup_frame(te, table, cfg['W'])[cols].fillna(0).values
    y = (tr.margin > 0).astype(int).values
    local = cfg if original else {**cfg, 'original_lgb_weight': 0}
    both = classifier(local, X, y, np.r_[Z, -Z])
    p = (both[:len(Z)] + 1 - both[len(Z):]) / 2
    method = cfg.get('women_calibration')
    if method:
        oof = np.empty(len(tr))
        for season in sorted(tr.Season.unique()):
            mask = tr.Season.values == season
            oof[mask] = classifier(local, X[~mask], y[~mask], X[mask])
        p = calibrate(method, oof, y, p)
    if cfg.get('horan_W'):
        from kaggle_mania_brackets import apply_sharpening_vector
        seed = F.matchup_frame(te, F.seeds('W'), ['seed']).diff_seed.values
        p = apply_sharpening_vector(p, -seed)
    return p
