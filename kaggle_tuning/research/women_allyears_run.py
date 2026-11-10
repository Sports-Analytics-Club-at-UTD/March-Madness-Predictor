"""Cache all seeded pairs for reproducible women-only blend research."""
import os
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from kaggle_tuning import backtest as B, features as F
from kaggle_tuning.configs.combined import BEST, LW, X, ORIGINAL_LOGIT, V

CACHE = Path('/private/tmp/mania_women_allyears_v1')
MEMBERS = {'H': {'model': 'women_consensus', 'women_q_alpha': .1, 'consensus_original': True},
           'L': LW, 'X': X, 'O': ORIGINAL_LOGIT, 'V': V,
           'C': {'model': 'women_consensus', 'women_q_alpha': .1},
           'B': {'model': 'women_consensus', 'women_q_alpha': .1, 'consensus_features': ['seed', 'bt', 'elo']},
           'E': {'model': 'women_consensus', 'women_q_alpha': .1, 'consensus_features': ['seed', 'adj_eff', 'elo']}}
YEARS = [2014, 2015, 2016, 2017, 2018, 2019, 2021, 2022, 2023, 2024, 2025, 2026]


def one(year):
    CACHE.mkdir(exist_ok=True)
    path = CACHE / f'{year}.pkl'
    out = pickle.loads(path.read_bytes()) if path.exists() else {}
    orig = B.test_pairs
    def pairs(g, y):
        if y != year:
            return orig(g, y)
        teams = sorted(F.seeds(g).query('Season == @year').TeamID)
        df = pd.DataFrame([(a,b) for i,a in enumerate(teams) for b in teams[i+1:]], columns=['T1','T2'])
        df.insert(0, 'Season', year)
        return df.merge(orig(g,y)[['T1','T2','y']], how='left')
    B.test_pairs = pairs
    out['pairs'] = pairs('W', year)
    table = F.team_table('W', q_clip=60)
    for name, cfg in MEMBERS.items():
        if name not in out:
            out[name] = B.fit_predict({**cfg, 'market_w': {}, 'sharpen': {}}, 'W', year, table)
            path.write_bytes(pickle.dumps(out))
            print(year, name, flush=True)
    if year >= 2015 and 'M' not in out:
        mt = F.team_table('M', q_clip=60)
        out['Mpairs'] = pairs('M', year)
        out['M'] = B.fit_predict(BEST, 'M', year, mt)
        path.write_bytes(pickle.dumps(out))
    B.test_pairs = orig
    return year

if __name__ == '__main__':
    print(Parallel(n_jobs=3)(delayed(one)(y) for y in YEARS))
