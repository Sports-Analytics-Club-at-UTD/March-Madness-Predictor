"""Evaluate cached all-pair predictions; selection never changes the fitted inputs."""
import json
import pickle
from pathlib import Path
import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.special import expit, logit
from kaggle_tuning import all_years as A, market as MK
from .women_allyears_run import CACHE

BASE = np.array([.4,.1,.3,.2])

def load():
    return {int(p.stem): pickle.loads(p.read_bytes()) for p in CACHE.glob('*.pkl')}

def blend(d, weights, names='LXOV'):
    return sum(w * logit(np.clip(d[n],1e-4,1-1e-4)) for n,w in zip(names,weights))

def fit_prior(data, year, weights, names, metric, penalty=.01, stacking=False):
    prior = [d for y,d in sorted(data.items()) if y < year and all(n in d for n in names)]
    if len(prior) < 3:
        return weights, 1.1
    mask = np.concatenate([d['pairs'].y.notna() for d in prior])
    x = np.concatenate([np.column_stack([logit(np.clip(d[n],1e-4,1-1e-4)) for n in names]) for d in prior])[mask]
    y = np.concatenate([d['pairs'].y for d in prior])[mask]
    def loss(z):
        if metric == 'log':
            return np.mean(np.logaddexp(0,z)-y*z)
        return np.mean((expit(z)-y)**2)
    if stacking:
        result = minimize(lambda w: loss(x @ w * 1.1) + penalty*np.sum((w-weights)**2), weights,
                          bounds=[(0,1)]*len(weights), constraints=[{'type':'eq','fun':lambda w:w.sum()-1}], method='SLSQP')
        return result.x, 1.1
    result = minimize_scalar(lambda a: loss(a*(x @ weights)) + penalty*(a-1.1)**2, bounds=(.7,1.5), method='bounded')
    return weights, result.x

def evaluate(data, weights=BASE, names='LXOV', sharp=1.1, calibration=None, penalty=.01):
    out = {}
    for year,d in sorted(data.items()):
        if not all(n in d for n in names):
            continue
        w,a = weights,sharp
        if calibration:
            w,a = fit_prior(data,year,weights,names,'log' if year in (2021,2022) else 'brier',penalty,calibration=='stack')
        p = expit(a*blend(d,w,names))
        if year in MK.PATHS:
            mk=d['pairs'][['T1','T2']].merge(MK.market(year),on=['T1','T2'],how='left').mkt.to_numpy()
            p=expit(a*(.5*blend(d,w,names)+.5*1.2*logit(np.clip(mk,1e-4,1-1e-4))))
        mask=d['pairs'].y.notna().to_numpy()
        y=d['pairs'].y.to_numpy()[mask]
        q=p[mask]
        out[year]={'brier':float(np.mean((y-q)**2)), 'll':float(np.mean(-y*np.log(q)-(1-y)*np.log1p(-q))), 'a':float(a),'w':np.asarray(w).tolist()}
        if year==2024:
            pairs=d['pairs'].assign(p=p)
            out[year]['bracket']=A.bracket_score('W',year,pairs)
    ys=[2015,2016,2017,2018,2019,2021,2022,2023,2024]
    out['histW']=float(np.mean([out[y]['brier'] for y in ys]))
    out['earlyW']=float(np.mean([out[y]['brier'] for y in ys[:5]]))
    return out

if __name__=='__main__':
    data=load()
    results={'baseline':evaluate(data)}
    for n in 'CBEH':
        results[n]=evaluate(data,[1],n)
        for w in [.1,.2,.3,.5]:
            results[f'{n}{w}']=evaluate(data,np.r_[BASE*(1-w),w],'LXOV'+n)
    for a in [.9,1,1.05,1.15,1.2]:
        results[f'sharp{a}']=evaluate(data,sharp=a)
    for method in ['temp','stack']:
        for penalty in [.001,.01,.1]:
            results[f'{method}{penalty}']=evaluate(data,calibration=method,penalty=penalty)
    print('name earlyW histW LL21 LL22 B23 BR24 W25 W26')
    for n,r in results.items():
        print(n, ' '.join(f'{v:.6f}' for v in [r['earlyW'],r['histW'],r[2021]['ll'],r[2022]['ll'],r[2023]['brier'],r[2024]['bracket'],r[2025]['brier'],r[2026]['brier']]))
    Path('kaggle_tuning/research/women_allyears_results.json').write_text(json.dumps(results,indent=2))
