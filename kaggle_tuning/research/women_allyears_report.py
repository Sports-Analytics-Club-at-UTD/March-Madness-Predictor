"""Reproduce comparison and guards from cached model predictions."""
import json
from pathlib import Path
import numpy as np
from scipy.special import expit, logit
from kaggle_tuning import all_years as A
from .women_allyears_eval import load, blend, BASE, evaluate
from .women_temperature import fit_temperature


def main():
    data=load()
    results={}
    for name in ['baseline','recommended']:
        result={}
        for year,d in sorted(data.items()):
            a=1.1
            if name=='recommended' and year<=2023:
                prior=[v for y,v in sorted(data.items()) if y<year]
                if len(prior)>=3:
                    x=np.concatenate([blend(v,BASE)[v['pairs'].y.notna()] for v in prior])
                    y=np.concatenate([v['pairs'].y.dropna() for v in prior])
                    a=fit_temperature(x,y,'log_loss' if year<=2022 else 'brier')
            p=expit(a*blend(d,BASE))
            if year>=2025:
                # These are the unchanged production market blends.
                b=evaluate({**data})[year]
                w=b['brier']
            else:
                mask=d['pairs'].y.notna()
                yy=d['pairs'].y[mask]
                pp=p[mask]
                w=float(np.mean((pp-yy)**2))
            r={'W_brier':w,'temperature':a}
            if year in [2021,2022]:
                r['W_ll']=float(np.mean(-yy*np.log(pp)-(1-yy)*np.log1p(-pp)))
                r['W_rank']=A.rank(A.LB[year]['W'],r['W_ll'])[0]
                mask=d['Mpairs'].y.notna();yy=d['Mpairs'].y[mask];pp=d['M'][mask]
                r['M_ll']=float(np.mean(-yy*np.log(pp)-(1-yy)*np.log1p(-pp)))
                r['M_rank']=A.rank(A.LB[year]['M'],r['M_ll'])[0]
            if year>=2015:
                mask=d['Mpairs'].y.notna()
                r['M_brier']=float(np.mean((d['M'][mask]-d['Mpairs'].y[mask])**2))
                r['combined_brier']=(r['M_brier']+w)/2
            if year==2024:
                r['W_bracket']=A.bracket_score('W',year,d['pairs'].assign(p=p))
                r['M_bracket']=A.bracket_score('M',year,d['Mpairs'].assign(p=d['M']))
                r['score']=(r['W_bracket']+r['M_bracket'])/2
                r['rank']=A.rank(A.LB[year],r['score'])[0]
            elif year in [2023,2025,2026]:
                r['score']=r['combined_brier'];r['rank']=A.rank(A.LB[year],r['score'])[0]
            result[year]=r
        for key,years in [('early',list(range(2015,2020))),('hist',[2015,2016,2017,2018,2019,2021,2022,2023,2024])]:
            result[key]={k:float(np.mean([result[y][k] for y in years])) for k in ['W_brier','M_brier','combined_brier']}
        results[name]=result
        print(name)
        for y in [2021,2022,2023,2024,2025,2026,'early','hist']:
            print(y,result[y],flush=True)
    Path('kaggle_tuning/research/women_allyears_comparison.json').write_text(json.dumps(results,indent=2))

if __name__=='__main__':
    main()
