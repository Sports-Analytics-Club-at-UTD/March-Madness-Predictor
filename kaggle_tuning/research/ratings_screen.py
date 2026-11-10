"""Fast logit-only screen. Select candidates on historical seasons, then test full ensemble."""
import json
import numpy as np
from joblib import Parallel, delayed
from kaggle_tuning import backtest as B, features as F
from kaggle_tuning.research.team_ratings import rating_table
from kaggle_tuning.configs.final import L
YEARS = B.HIST+[2025,2026]


def screen(g):
    table = F.team_table(g,q_clip=60).merge(rating_table(g),on=['Season','TeamID'])
    sets = {'baseline':[]}
    for col in table:
        if col.startswith('rt_') and not col.startswith(('rt_off','rt_def')):
            sets[col] = [col]
    for a in (1,10,100):
        sets[f'offdef{a}'] = [f'rt_off{a}',f'rt_def{a}']
    results = {}
    for name,cols in sets.items():
        for mode in ('add','replace') if cols else ('add',):
            feats = [f for f in L[g] if mode!='replace' or f!='quality'] + cols
            cfg={**L,g:feats}
            scores={y:float(np.mean((B.fit_predict(cfg,g,y,table)-B.test_pairs(g,y).y.to_numpy())**2)) for y in YEARS}
            results[name+'_'+mode]={'hist':np.mean([scores[y] for y in B.HIST]),'early':np.mean([scores[y] for y in range(2015,2020)]),'scores':scores}
    return g,results
if __name__=='__main__':
    results=dict(Parallel(n_jobs=2)(delayed(screen)(g) for g in 'MW'))
    with open('kaggle_tuning/research/ratings_results/screen.json','w') as f: json.dump(results,f,indent=2)
    for g,r in results.items():
        print(g,flush=True)
        for n,v in sorted(r.items(),key=lambda kv:kv[1]['hist']):
            print(n,round(v['hist'],6),round(v['early'],6),round(v['scores'][2025],6),round(v['scores'][2026],6),flush=True)
