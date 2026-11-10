"""Full model-only historical guards (same fits and years as grid)."""
import json
import sys
import numpy as np
from joblib import Parallel, delayed
from kaggle_tuning import backtest as B

def one(name,cfg):
    cfg={**cfg,'market_w':{}}
    scores=B.run(cfg, years=B.HIST, verbose=False,hist=False)
    return name,{'hist':float(np.mean([scores[y]['brier'] for y in B.HIST])),
                 'early':float(np.mean([scores[y]['brier'] for y in range(2015,2020)])),
                 'earlyM':float(np.mean([scores[y]['M'] for y in range(2015,2020)])),
                 'earlyW':float(np.mean([scores[y]['W'] for y in range(2015,2020)])),
                 'scores':scores}
if __name__=='__main__':
    from kaggle_tuning.configs.allyears import BEST
    from kaggle_tuning.configs.combined import BEST as COMBINED
    from kaggle_tuning.configs.ratings import CONFIGS
    configs={'baseline':BEST,'combined':COMBINED,**CONFIGS}
    names=sys.argv[1:] or list(configs)
    results=dict(Parallel(n_jobs=min(4,len(names)))(delayed(one)(n,configs[n]) for n in names))
    for n,r in results.items():
        print(n,json.dumps(r),flush=True)
        with open(f'kaggle_tuning/research/ratings_results/{n}_hist.json','w') as f:json.dump(r,f,indent=2)
