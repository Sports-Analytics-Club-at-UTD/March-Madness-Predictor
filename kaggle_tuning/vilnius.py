"""Walk-forward port of raddar/vilnius-ncaa (March 17, 2025 notebook).

Gaussian identity GLM is solved as sparse least squares with the same categorical
coding as PointDiff ~ -1 + T1_TeamID + T2_TeamID. No ridge or home adjustment.
"""
from functools import lru_cache
import json

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import lsqr
from scipy.interpolate import UnivariateSpline

from . import features as F

STATS = ['Score', 'FGA', 'Blk', 'PF', 'opponent_FGA', 'opponent_Blk', 'opponent_PF', 'PointDiff']


def feature_names(modeh=False):
    stats = STATS.copy()
    if modeh:
        stats[2:2] = ['OR', 'DR']
    return (['men_women', 'T1_seed', 'T2_seed', 'Seed_diff']
            + [f'T{i}_avg_{s}' for i in (1, 2) for s in stats]
            + ['T1_elo', 'T2_elo', 'elo_diff', 'T1_quality', 'T2_quality'])


def prepare(df):
    df = df.copy()
    for side in ('W', 'L'):
        for stat in ['Score'] + F.BOX:
            df[side + stat] = df[side + stat] / ((40 + 5 * df.NumOT) / 40)
    rows = []
    for a, b in [('W', 'L'), ('L', 'W')]:
        mapping = {side + stat: team + stat for side, team in [(a, 'T1_'), (b, 'T2_')]
                   for stat in ['TeamID', 'Score'] + F.BOX}
        rows.append(df.rename(columns=mapping))
    out = pd.concat(rows, ignore_index=True).rename(columns={'T1_TeamID': 'T1', 'T2_TeamID': 'T2'})
    out['margin'] = out.T1_Score - out.T2_Score
    return out


def quality(s, seeded):
    kept = set(seeded) | set(s.loc[(s.margin > 0) & s.T2.isin(seeded), 'T1'])
    s = s[s.T1.isin(kept) | s.T2.isin(kept)]
    a = s.T1.where(s.T1.isin(kept), 0).to_numpy()
    b = s.T2.where(s.T2.isin(kept), 0).to_numpy()
    teams = np.unique(np.r_[a, b])
    i = np.searchsorted(teams, a)
    j = np.searchsorted(teams, b)
    n, k = len(s), len(teams)
    # Patsy keeps all T1 levels and drops the first T2 level.
    use = j > 0
    design = sparse.csr_matrix((np.ones(n + use.sum()),
                               (np.r_[np.arange(n), np.arange(n)[use]],
                                np.r_[i, k + j[use] - 1])), shape=(n, 2 * k - 1))
    coef = lsqr(design, s.margin.to_numpy(), atol=1e-11, btol=1e-11)[0][:k]
    return pd.DataFrame({'TeamID': teams, 'quality': coef})


@lru_cache(None)
def team_table():
    tables = []
    for g in ('M', 'W'):
        raw = F.read(g + 'RegularSeasonDetailedResults')
        raw = raw[(raw.Season >= 2003) & (raw.DayNum <= 132)]
        regular = prepare(raw)
        seeds = F.seeds(g)
        for season, s in regular.groupby('Season', sort=True):
            if season not in set(seeds.Season):
                continue
            cols = ['T1_' + c for c in ['Score'] + F.BOX] + ['T2_' + c for c in ['Score'] + F.BOX]
            avg = s.groupby('T1')[cols + ['margin']].mean().reset_index()
            avg = avg.rename(columns={'T1': 'TeamID', 'margin': 'avg_PointDiff', **{
                c: 'avg_' + c.replace('T1_', '').replace('T2_', 'opponent_') for c in cols}})
            ratings = dict.fromkeys(avg.TeamID, 1000.)
            # Source order is chronological; stable sorting keeps same-day order.
            wins = s[s.margin > 0].sort_values('DayNum', kind='stable')
            for w, l in zip(wins.T1, wins.T2):
                delta = 100 * (1 - 1 / (1 + 10 ** ((ratings[l] - ratings[w]) / 400)))
                ratings[w] += delta
                ratings[l] -= delta
            avg['elo'] = avg.TeamID.map(ratings)
            sd = seeds[seeds.Season == season]
            avg = avg.merge(quality(s, set(sd.TeamID)), on='TeamID', how='left')
            avg = avg.merge(sd[['TeamID', 'seed']], on='TeamID', how='left')
            avg['Season'] = season
            tables.append(avg)
    return pd.concat(tables, ignore_index=True)


def matchup(pairs):
    table = team_table()
    out = pairs.copy()
    for i in (1, 2):
        out = out.merge(table.add_prefix(f'T{i}_').rename(columns={f'T{i}_Season': 'Season', f'T{i}_TeamID': f'T{i}'}),
                        on=['Season', f'T{i}'], how='left')
    out['men_women'] = (out.T1 < 2000).astype(int)
    out['Seed_diff'] = out.T2_seed - out.T1_seed
    out['elo_diff'] = out.T1_elo - out.T2_elo
    return out


def cauchy(pred, data):
    x = pred - data.get_label()
    c = 5000.  # Exact paris-madness scale; nearly quadratic at basketball margins.
    return x / (1 + x*x/(c*c)), c*c*(c*c-x*x)/(x*x+c*c)**2


@lru_cache(None)
def predict_year(year, options_json):
    import xgboost as xgb
    from .backtest import test_pairs
    cfg = json.loads(options_json)
    train = pd.concat([prepare(F.read(g + 'NCAATourneyDetailedResults')) for g in ('M', 'W')], ignore_index=True)
    train = train[(train.Season >= 2003) & (train.Season < year)].reset_index(drop=True)
    assert train.Season.max() < year
    tr = matchup(train[['Season', 'T1', 'T2', 'margin']])
    test = pd.concat([test_pairs(g, year)[['Season', 'T1', 'T2']] for g in ('M', 'W')], ignore_index=True)
    reverse = test.rename(columns={'T1': 'T2', 'T2': 'T1'})
    cols = feature_names(cfg.get('modeh', False))
    x = tr[cols].to_numpy()
    xt = matchup(pd.concat([test, reverse], ignore_index=True))[cols].to_numpy()
    params = dict(objective='reg:squarederror', eta=.01, subsample=.6, colsample_bynode=.8,
                  num_parallel_tree=2, min_child_weight=4, max_depth=4, tree_method='hist',
                  grow_policy='lossguide', max_bin=32, nthread=1, base_score=0.5)
    if cfg.get('modeh'):
        params.update(eta=.0093, max_bin=38)
    params.update(cfg.get('params', {}))
    results = []
    for seed in cfg.get('seeds', [0]):
        params['seed'] = seed
        oof = np.empty(len(tr))
        margins = []
        for season in sorted(tr.Season.unique()):
            val = (tr.Season == season).to_numpy()
            model = xgb.train(params, xgb.DMatrix(x[~val], label=tr.margin.to_numpy()[~val]),
                              num_boost_round=cfg.get('rounds', 704 if cfg.get('modeh') else 700),
                              obj=cauchy if cfg.get('cauchy') else None)
            oof[val] = model.predict(xgb.DMatrix(x[val]))
            margins.append(model.predict(xgb.DMatrix(xt)))
        order = np.argsort(oof, kind='stable')
        spline = UnivariateSpline(np.clip(oof[order], -25, 25), (tr.margin.to_numpy()[order] > 0).astype(float), k=5)
        probs = np.mean([np.clip(spline(np.clip(m, -25, 25)), .01, .99) for m in margins], axis=0)
        results.append((probs[:len(test)] + 1 - probs[len(test):]) / 2)
    p = np.mean(results, axis=0)
    assert np.isfinite(p).all()
    n = len(test_pairs('M', year))
    return {'M': p[:n], 'W': p[n:]}


def predict(cfg, g, year):
    return predict_year(year, json.dumps(cfg.get('vilnius', {}), sort_keys=True))[g]
