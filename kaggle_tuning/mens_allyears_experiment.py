"""Reproducible research grid; writes men's probabilities and guard metrics."""
import json
from pathlib import Path
import numpy as np
from joblib import Memory
from . import backtest as B
from . import mens_allyears as M
from .configs.final import BASE

CACHE=Memory('/tmp/mania_mens_allyears_v1',verbose=0)
@CACHE.cache
def pred(opts,year):
    return M.predict_margin(opts,year) if opts.get('margin') else M.predict({'mens_allyears':opts},year)

def main(mode="margin"):
    if mode not in ("margin", "regular"):
        raise ValueError(mode)
    years=B.HIST+[2025,2026]
    ys={s:B.test_pairs('M',s).y.values for s in years}
    baseline=json.loads(Path('kaggle_tuning/results/mens/pool_worst_conf.json').read_text())
    bp={s:np.array(baseline['p'][str(s)]) for s in years}
    def metrics(ps):
        b={s:float(np.mean((ps[s]-ys[s])**2)) for s in years}
        ll=lambda s:float(-np.mean(ys[s]*np.log(ps[s])+(1-ys[s])*np.log1p(-ps[s])))
        return {'oldM':float(np.mean([b[s] for s in years[:5]])), 'histM':float(np.mean([b[s] for s in B.HIST])),
                'll21':ll(2021),'ll22':ll(2022),'M23':b[2023],
                **{f'M{s}':float(np.mean((B._post(BASE,'M',s,ps[s])-ys[s])**2)) for s in [2025,2026]}}
    results={'baseline':{'metrics':metrics(bp)}}
    jobs = []
    if mode == 'margin':
        for feats in [['seed','q_pool','elo','mstr_med','mstr_worst','net_eff','conf_quality'],['seed','q_pool','mstr_med'],['seed','q_pool','mstr_med','normal_POM','normal_SAG','normal_MOR','normal_WLK','normal_DOL','normal_COL','normal_MAS','normal_RTH','normal_WOL','normal_BIH']]:
          for alpha in [1,100,1000]:
            jobs.append((f'margin_f{len(feats)}_a{alpha}',{'features':feats,'alpha':alpha,'margin':True}))
    else:
        for feats in [['q','qrec','off','def','consensus','POM'],['q','consensus'],['q','qrec','off','def','consensus','POM','SAG','MOR','WLK','DOL','COL','MAS','RTH','WOL','BIH']]:
          for rw,neutral in [(0,False),(.5,False),(1,False),(3,False),(1,True),(3,True)]:
            jobs.append((f'f{len(feats)}_w{rw}_n{neutral}',{'features':feats,'regular_weight':rw,'neutral':neutral,'C':.1}))
    for name, opts in jobs:
        ps={s:pred(opts,s) for s in years}
        results[name]={'opts':opts,'metrics':metrics(ps),'p':{s:p.tolist() for s,p in ps.items()}}
        print(name,results[name]['metrics'],flush=True)
        Path(f'kaggle_tuning/results/mens_allyears/{mode}_grid.json').write_text(json.dumps(results,indent=2))
if __name__=='__main__':
    import sys
    main(sys.argv[1] if len(sys.argv)>1 else 'margin')
