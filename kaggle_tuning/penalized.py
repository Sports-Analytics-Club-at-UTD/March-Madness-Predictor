"""Wide matchup penalties: season-held-out tuning, always restricted to years < target.

Run `python -m kaggle_tuning.penalized` to cache fits and score the research grid.
"""
from functools import lru_cache
import hashlib
import json
from pathlib import Path

from joblib import Memory, Parallel, delayed
import numpy as np
from scipy.special import ndtr, expit
from scipy.optimize import minimize_scalar
from sklearn.linear_model import LogisticRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler

from . import features as F

# Version caches by implementation and all input-file metadata, never predictions' row order.
STAMP = hashlib.sha256(Path(__file__).read_bytes() + repr([
    (str(p), p.stat().st_size, p.stat().st_mtime_ns)
    for p in sorted(F.DATA.glob('*.csv'))]).encode()).hexdigest()[:16]
MEM = Memory('/private/tmp/mania_penalized_' + STAMP, verbose=0)
KINDS = ['l1', 'l2', 'en25', 'en50', 'en75', 'lasso_margin', 'ridge_margin']
YEARS = [2015, 2016, 2017, 2018, 2019, 2021, 2022, 2023, 2024, 2025, 2026]


@lru_cache(None)
def wide_table(g):
    # Existing helpers assume their inputs stop at the deadline; verify that explicitly.
    for suffix in ['RegularSeasonDetailedResults', 'RegularSeasonCompactResults']:
        assert F.read(g + suffix).DayNum.max() <= 132
    t = F.team_table(g).copy()
    d = F.team_games(g).query('DayNum <= 132')
    box = [f'T{i}_{c}' for i in (1, 2) for c in ['Score'] + F.BOX]
    extra = [c for c in box if c not in t]
    a = d.groupby(['Season', 'T1'])[extra].mean().reset_index().rename(columns={'T1': 'TeamID'})
    t = t.merge(a, on=['Season', 'TeamID'], how='left')
    feats = [c for c in t if c not in ['Season', 'TeamID', 'massey_med'] and not c.startswith('rank_')]
    assert all(c in feats for c in box + ['quality', 'quality_rec', 'elo', 'seed', 'net_eff'])
    return t, feats


@lru_cache(None)
def training(g):
    t, feats = wide_table(g)
    tr = F.tourney_games(g)
    tr = tr[tr.Season >= (2003 if g == 'M' else 2010)].reset_index(drop=True)
    tr = F.matchup_frame(tr, t, feats)
    return tr, feats


def estimator(kind, a):
    if kind == 'ridge_margin':
        return Ridge(alpha=a, fit_intercept=False)
    if kind == 'lasso_margin':
        return Lasso(alpha=a, fit_intercept=False, max_iter=100000, tol=1e-4, selection="random", random_state=19)
    ratio = {'l1': 1., 'l2': 0., 'en25': .25, 'en50': .5, 'en75': .75}[kind]
    return LogisticRegression(C=a, l1_ratio=ratio, solver='liblinear' if kind in ('l1','l2') else 'saga', fit_intercept=False,
                              random_state=19, max_iter=10000, tol=1e-3)


def fold_arrays(X, seasons):
    for s in np.unique(seasons):
        mask = seasons == s
        scaler = StandardScaler().fit(X[~mask])
        yield mask, scaler.transform(X[~mask]), scaler.transform(X[mask])


@MEM.cache
def fit(g, year, kind):
    tr, feats = training(g)
    tr = tr[tr.Season < year]
    assert tr.Season.max() < year
    cols = ['diff_' + f for f in feats]
    X = tr[cols].fillna(0).to_numpy()
    margin = tr.margin.to_numpy()
    y = (margin > 0).astype(int)
    regression = kind.endswith('_margin')
    target = margin if regression else y
    grid = ([.01, .1, 1., 10.] if kind == 'lasso_margin' else
            [1., 10., 100., 1000., 10000.] if regression else [.001, .01, .1, 1., 10.])
    folds = list(fold_arrays(X, tr.Season.to_numpy()))
    scores, predictions = [], []
    for a in grid:
        oof = np.zeros(len(X))
        losses = []
        for mask, train, valid in folds:
            m = estimator(kind, a).fit(train, target[~mask])
            pred = m.predict(valid) if regression else m.predict_proba(valid)[:, 1]
            oof[mask] = pred
            # Equal weight per season. Regression selects on margin MSE; classification on log loss.
            loss = (pred - target[mask]) ** 2 if regression else -(y[mask]*np.log(np.clip(pred, 1e-12, 1)) + (1-y[mask])*np.log(np.clip(1-pred, 1e-12, 1)))
            losses.append(float(loss.mean()))
        scores.append(float(np.mean(losses)))
        predictions.append(oof)
    idx = int(np.argmin(scores))
    a = grid[idx]
    sigma = None
    if regression:
        # Calibration sees only held-out predictions of pre-target seasons.
        oof = predictions[idx]
        def loss(sigma):
            p = np.clip(ndtr(oof / sigma), 1e-12, 1-1e-12)
            return -np.mean(y*np.log(p) + (1-y)*np.log(1-p))
        sigma = float(minimize_scalar(loss, bounds=(3, 30), method='bounded').x)
    sc = StandardScaler().fit(X)
    m = estimator(kind, a).fit(sc.transform(X), target)
    coef = m.coef_.ravel()
    meta = dict(g=g, year=year, kind=kind, chosen=a, grid=grid, cv=scores, sigma=sigma,
                train_seasons=sorted(map(int, tr.Season.unique())),
                coefficients=dict(zip(feats, map(float, coef))),
                kept=[f for f, v in zip(feats, coef) if abs(v) > 1e-6])
    return sc, m, meta


def predict(cfg, g, year):
    from . import backtest as B
    t, feats = wide_table(g)
    sc, m, meta = fit(g, year, cfg['penalized'])
    te = F.matchup_frame(B.test_pairs(g, year)[['Season', 'T1', 'T2']], t, feats)
    X = te[['diff_' + f for f in feats]].fillna(0).to_numpy()
    if cfg['penalized'].endswith('_margin'):
        return ndtr(m.predict(sc.transform(X)) / meta['sigma'])
    return m.predict_proba(sc.transform(X))[:, 1]


def mix_config(kind, weight=1., replace='all', external=False):
    from .configs.allyears import BEST
    from copy import deepcopy
    cfg = deepcopy(BEST)
    sub = {'model': 'penalized', 'penalized': kind}
    for g in 'MW':
        parts = cfg['men_models'] if g == 'M' else cfg['by_gender']['W']['models']
        if replace == 'all':
            total = sum(w for _, w in parts)
            parts = [(c, (1-weight)*w/total) for c, w in parts] + [(sub, weight)]
        else:
            # Replace the requested fraction of the original logit/tree member's weight.
            changed = []
            for c, w in parts:
                if (replace == 'tree' and c['model'] == 'xgb_spline') or (replace == 'logit' and c['model'] in ('logit', 'women_original')):
                    changed.extend([(c, w*(1-weight)), (sub, w*weight)])
                else:
                    changed.append((c, w))
            parts = changed
        parts = [(c, w) for c, w in parts if w > 0]
        if g == 'M':
            cfg['men_models'] = parts
        else:
            cfg['by_gender']['W']['models'] = parts
    if external:
        cfg['ext_blend'] = {'M': ['538', 'torvik_adjem', 'futures'], 'W': ['538']}
    return cfg


def fit_task(g, y, k):
    return fit(g, y, k)


if __name__ == '__main__':
    import os
    from kaggle_tuning.penalized import fit_task
    jobs = int(os.environ.get('GRID_JOBS', 4))
    tasks = [(g, y, k) for g in 'MW' for y in YEARS for k in KINDS]
    fitted = Parallel(n_jobs=jobs, verbose=10)(delayed(fit_task)(*t) for t in tasks)
    Path('kaggle_tuning/penalized_fits.json').write_text(json.dumps([f[2] for f in fitted], indent=2))
