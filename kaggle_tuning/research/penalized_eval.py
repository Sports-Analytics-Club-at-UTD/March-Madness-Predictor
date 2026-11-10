"""Cache shared components, then evaluate fixed blends with the standard real-year scorer."""
import json
from pathlib import Path
from functools import lru_cache
import numpy as np
from scipy.special import expit, logit
from joblib import Memory, Parallel, delayed
from kaggle_tuning import features as F, backtest as B, all_years as A
from kaggle_tuning.penalized import STAMP, KINDS, YEARS, mix_config
from kaggle_tuning.configs.allyears import BEST, EXT
from kaggle_tuning.configs.combined import BEST as COMBINED
from kaggle_tuning.research.women_temperature import fit_temperature

MEM = Memory('/private/tmp/mania_penalized_eval_' + STAMP, verbose=0)
EVAL_YEARS = [2014] + YEARS


@lru_cache(None)
def pairs(g, year):
    teams = sorted(F.seeds(g).query('Season == @year').TeamID)
    import pandas as pd
    d = pd.DataFrame([(a,b) for i,a in enumerate(teams) for b in teams[i+1:]], columns=['T1','T2'])
    d.insert(0, 'Season', year)
    return d.merge(B._orig_test_pairs(g, year)[['T1','T2','y']], on=['T1','T2'], how='left')


@MEM.cache
def component(g, year, cfg_json):
    cfg = json.loads(cfg_json)
    original = B.test_pairs
    B._orig_test_pairs = getattr(B, '_orig_test_pairs', original)
    B.test_pairs = lambda sex, y: pairs(sex,y)
    try:
        from kaggle_tuning.vilnius import predict_year
        predict_year.cache_clear()
        table = F.team_table(g, cfg.get('q_alpha',1), cfg.get('q_recency',0), cfg.get('q_clip_'+g,30))
        p = B.fit_predict(cfg,g,year,table)
        return pairs(g,year).assign(p=p)
    finally:
        B.test_pairs = original
        predict_year.cache_clear()


def parts(cfg,g):
    return cfg['men_models'] if g == 'M' else cfg['by_gender']['W']['models']


def component_cfg(cfg,g,sub):
    base = {k:v for k,v in cfg.items() if k not in ('models','men_models','by_gender','ext_blend')}
    return {**base, **sub, 'market_w':{},'sharpen':{},'women_model_temperature':None,
            'clip_p':(0,1),'models_inner':True}


@lru_cache(None)
def raw(cfg_json,g,year):
    cfg = json.loads(cfg_json)
    ps = [(component(g,year,json.dumps(component_cfg(cfg,g,c),sort_keys=True)),w) for c,w in parts(cfg,g)]
    z = sum(w * B._logit(d.p.to_numpy()) for d,w in ps)/sum(w for _,w in ps)
    return ps[0][0].assign(p=expit(z))


def post(cfg_json,g,year,market):
    cfg = json.loads(cfg_json)
    d = raw(cfg_json,g,year).copy()
    cfg.pop('ext_blend',None)
    opts = cfg.pop('women_model_temperature',None)
    if not market:
        cfg['market_w'] = {}
    if g == 'W' and opts and not (market and year in (2025,2026)) and year != 2024:
        seasons = [s for s in EVAL_YEARS if opts['start'] <= s < year]
        if len(seasons) >= 3:
            prior = [raw(cfg_json,g,s).dropna(subset=['y']) for s in seasons]
            p = np.concatenate([h.p.to_numpy() for h in prior]); y = np.concatenate([h.y.to_numpy() for h in prior])
            a = fit_temperature(logit(np.clip(p,1e-4,1-1e-4)),y,'log_loss' if year <= 2022 else 'brier',prior=cfg['sharpen']['W'],penalty=opts['penalty'])
            cfg['sharpen'] = {**cfg['sharpen'],'W':a}
    original = B.test_pairs
    B.test_pairs = lambda sex,y: d if sex == g and y == year else original(sex,y)
    try:
        return d.assign(p=B._post(cfg,g,year,d.p.to_numpy()))
    finally:
        B.test_pairs = original


def evaluate(name,cfg):
    j = json.dumps(cfg,sort_keys=True)
    history = {y:{g:post(j,g,y,False) for g in 'MW'} for y in YEARS if y < 2025}
    real = {}
    for y in YEARS:
        if y >= 2021:
            p = {g:post(j,g,y,True) for g in 'MW'}
            real[y] = A.score_year(cfg,y,p,{s:h for s,h in history.items() if s<y})
    brier = {y:{g:float(np.mean((d.dropna(subset=['y']).p-d.dropna(subset=['y']).y)**2)) for g,d in h.items()} for y,h in history.items()}
    return name, {'real':real,'brier':brier,'hist':{g:float(np.mean([brier[y][g] for y in brier])) for g in 'MW'},
                  'early':{g:float(np.mean([brier[y][g] for y in range(2015,2020)])) for g in 'MW'}}


def warm_task(g,y,j):
    return component(g,y,j).shape


def main():
    import os
    configs = {'BEST':BEST,'EXT':EXT,'COMBINED':COMBINED}
    for k in KINDS:
        for replacement in ['all','logit','tree']:
            for w in [.25,.5,1.]:
                configs[f'{k}_{replacement}_{w}'] = mix_config(k,w,replacement)
    # Materialize shared fits once before scoring combinations.
    tasks = set()
    for cfg in configs.values():
        for g in 'MW':
            for sub,w in parts(cfg,g):
                j = json.dumps(component_cfg(cfg,g,sub),sort_keys=True)
                tasks.update((g,y,j) for y in EVAL_YEARS)
    Parallel(n_jobs=int(os.environ.get('GRID_JOBS',4)),verbose=10)(delayed(warm_task)(*t) for t in sorted(tasks))
    results = {}
    for name,cfg in configs.items():
        _,r = evaluate(name,cfg)
        results[name] = r
        print(name, json.dumps(r),flush=True)
        Path('kaggle_tuning/penalized_results.json').write_text(json.dumps(results,indent=2))


if __name__ == '__main__':
    # Import workers by module name so loky can serialize their cached dependencies.
    from kaggle_tuning.research.penalized_eval import main
    main()
