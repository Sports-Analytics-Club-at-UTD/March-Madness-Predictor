"""Cache expensive member predictions, fit blend weights on preceding history only.

MANIA_DATA=... GRID_JOBS=5 uv run python -m kaggle_tuning.vilnius_experiment
"""
import json
import os
from functools import lru_cache
from pathlib import Path

import numpy as np
from joblib import Memory, Parallel, delayed
from scipy.optimize import minimize
from scipy.special import expit

from . import features as F
from .backtest import HIST, VERIFY, test_pairs, fit_predict, _logit, _post
from .configs.final import L, X, BASE

CACHE = Path(__file__).resolve().parent.parent / 'model_outputs' / 'vilnius'
memory = Memory(CACHE, verbose=0)
MEMBERS = {'L': L, 'X': X, 'V': {'model': 'vilnius'},
           'modeh': {'model': 'vilnius', 'vilnius': {'modeh': True}},
           'cauchy': {'model': 'vilnius', 'vilnius': {'cauchy': True}},
           'seeds': {'model': 'vilnius', 'vilnius': {'seeds': [0, 1, 2]}}}


@memory.cache
def member(name, year, data_path):
    cfg = MEMBERS[name]
    out = {}
    for g in ('M', 'W'):
        table = None if cfg.get('model') == 'vilnius' else F.team_table(g, q_clip=60)
        out[g] = fit_predict(cfg, g, year, table)
    print(f'finished {name} {year}', flush=True)
    return out


def get(name, year):
    return member(name, year, str(F.DATA))


def weights(names, year, g):
    # Freeze at 2024 for both target years; historical evaluation is itself rolling.
    years = [s for s in HIST if s < min(year, 2025)]
    if not years:
        return np.ones(len(names)) / len(names)
    x = np.concatenate([np.column_stack([_logit(get(n, s)[g]) for n in names]) for s in years])
    y = np.concatenate([test_pairs(g, s).y.to_numpy() for s in years])
    result = minimize(lambda w: np.mean((expit(x @ w) - y)**2), np.ones(len(names))/len(names),
                      bounds=[(0, 1)]*len(names), constraints=[{'type': 'eq', 'fun': lambda w: w.sum()-1}],
                      method='SLSQP', options={'ftol': 1e-12})
    if not result.success:
        raise RuntimeError(result.message)
    return result.x


@lru_cache(None)
def ensemble(variant, g, year):
    names = ('L', 'X', variant)
    w = weights(names, year, g)
    return expit(sum(a * _logit(get(n, year)[g]) for n, a in zip(names, w)))


def summary(preds):
    rows = {}
    for year, parts in preds.items():
        scores = {g: float(np.mean((parts[g]-test_pairs(g, year).y.to_numpy())**2)) for g in ('M', 'W')}
        counts = [len(test_pairs(g, year)) for g in ('M', 'W')]
        rows[year] = {**scores, 'brier': float(np.average([scores['M'], scores['W']], weights=counts))}
    return {'hist': np.mean([rows[s]['brier'] for s in HIST]),
            'histM': np.mean([rows[s]['M'] for s in HIST]), 'histW': np.mean([rows[s]['W'] for s in HIST]),
            'verify': np.mean([rows[s]['brier'] for s in VERIFY]), 'years': rows}


def main():
    years = HIST + [2025, 2026]
    Parallel(n_jobs=int(os.environ.get('GRID_JOBS', '5')))(
        delayed(member)(name, year, str(F.DATA)) for name in MEMBERS for year in years)
    rows = {}
    rows['final_model_only'] = summary({y: {g: expit((_logit(get('L', y)[g])+_logit(get('X', y)[g]))/2) for g in ('M','W')} for y in years})
    rows['final'] = summary({y: {g: _post(BASE,g,y,expit((_logit(get('L',y)[g])+_logit(get('X',y)[g]))/2)) for g in ('M','W')} for y in years})
    for variant in ('V', 'modeh', 'cauchy', 'seeds'):
        rows[variant] = summary({y: get(variant, y) for y in years})
        preds = {y: {g: ensemble(variant, g, y) for g in ('M','W')} for y in years}
        rows[variant+'_ensemble'] = summary(preds)
        rows[variant+'_market'] = summary({y: {g: _post(BASE,g,y,preds[y][g]) for g in ('M','W')} for y in years})
    result = {'scores': rows, 'weights': {v: {g: weights(('L','X',v),2025,g).tolist() for g in ('M','W')} for v in ('V','modeh','cauchy','seeds')}}
    CACHE.mkdir(parents=True, exist_ok=True)
    (CACHE/'results.json').write_text(json.dumps(result,indent=2))
    print('config hist hM hW v21-24 2025 M25 W25 2026 M26 W26')
    for name,r in rows.items():
        a,b = r['years'][2025], r['years'][2026]
        print(name, ' '.join(f'{v:.5f}' for v in [r['hist'],r['histM'],r['histW'],r['verify'],a['brier'],a['M'],a['W'],b['brier'],b['M'],b['W']]))
    print(json.dumps(result['weights'],indent=2))


if __name__ == '__main__':
    # Keep joblib cache identity identical for CLI and harness imports.
    from kaggle_tuning.vilnius_experiment import main as run_main
    run_main()
