"""Walk-forward temperature for model-only women's pairwise scoring metrics.

A market blend has a different calibration distribution. The bracket metric also
needs a separate fit, so both retain the existing sharpening.
"""
import hashlib
import json
from pathlib import Path

from joblib import Memory
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import expit, logit

from kaggle_tuning import features as F


def _prior_predictions(cfg_json, year):
    from kaggle_tuning import backtest as B
    from kaggle_tuning.vilnius import predict_year
    cfg = json.loads(cfg_json)
    original = B.test_pairs
    # Always use all seeded pairs: Vilnius caches a year's joint M/W predictions.
    def pairs(g, y):
        if y != year:
            return original(g, y)
        teams = sorted(F.seeds(g).query('Season == @year').TeamID)
        out = pd.DataFrame([(a,b) for i,a in enumerate(teams) for b in teams[i+1:]], columns=['T1','T2'])
        out.insert(0, 'Season', year)
        return out
    try:
        B.test_pairs = pairs
        predict_year.cache_clear()
        table = F.team_table('W', cfg.get('q_alpha',1),cfg.get('q_recency',0),cfg.get('q_clip_W',30))
        p = B.fit_predict(cfg, 'W', year, table)
        out = pairs('W',year).assign(p=p)
        truth = F.tourney_games('W').query('Season == @year and T1 < T2')
        return out.merge(truth[['T1','T2','margin']],on=['T1','T2']).assign(y=lambda d:(d.margin>0).astype(int))[['p','y']]
    finally:
        B.test_pairs = original
        predict_year.cache_clear()


def fit_temperature(logits, labels, metric, prior=1.1, penalty=.01):
    def objective(a):
        z = a * logits
        loss = np.mean(np.logaddexp(0,z)-labels*z) if metric == 'log_loss' else np.mean((expit(z)-labels)**2)
        return loss + penalty*(a-prior)**2
    return float(minimize_scalar(objective,bounds=(.7,1.5),method='bounded').x)


def temperature(cfg, year):
    raw = {k:v for k,v in cfg.items() if k not in ('women_model_temperature','sharpen','market_w','men_models','clip_p')}
    if 'by_gender' in raw:
        raw['by_gender'] = {'W':raw['by_gender']['W']}
    opts = cfg['women_model_temperature']
    seasons = [s for s in range(opts.get('start',2014),year) if s != 2020]
    if len(seasons)<3:
        return cfg.get('sharpen',{}).get('W',1.1)
    digest = hashlib.sha256()
    for path in sorted(Path(__file__).parents[1].rglob('*.py')):
        digest.update(path.read_bytes())
    digest.update((Path(__file__).parents[2]/'kaggle_mania_brackets.py').read_bytes())
    for path in sorted(F.DATA.glob('*.csv')):
        stat=path.stat()
        digest.update(f'{path.resolve()}:{stat.st_size}:{stat.st_mtime_ns}'.encode())
    cache = Memory('/private/tmp/mania_women_temperature_'+digest.hexdigest()[:16],verbose=0).cache(_prior_predictions)
    cfg_json = json.dumps(raw,sort_keys=True)
    history = pd.concat([cache(cfg_json,s) for s in seasons],ignore_index=True)
    metric = 'log_loss' if year <= 2022 else 'brier'
    return fit_temperature(logit(np.clip(history.p.to_numpy(),1e-4,1-1e-4)),history.y.to_numpy(),metric,
                           prior=cfg.get('sharpen',{}).get('W',1.1),penalty=opts.get('penalty',.01))
