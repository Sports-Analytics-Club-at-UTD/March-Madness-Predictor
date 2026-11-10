"""Small deterministic checks for the research's scoring and temporal boundaries."""
import numpy as np
import pandas as pd
from scipy.special import expit
from kaggle_tuning.research.bust_years import coefficient, walkforward


def test_scoring_and_temporal_boundaries():
    x=np.tile(np.array([-12,-8,-4,4,8,12]),10)
    y=np.tile(np.array([0,0,1,0,1,1]),10)
    beta=coefficient(pd.DataFrame({'x':x,'y':y}))
    p=expit(beta*x)
    actual=-(y*np.log(p)+(1-y)*np.log1p(-p))
    entropy=-(p*np.log(p)+(1-p)*np.log1p(-p))
    np.testing.assert_allclose(actual-entropy,(p-y)*beta*x,atol=1e-12)
    rng=np.random.default_rng(1)
    d=pd.DataFrame({'Season':np.arange(1985,2026),'f':rng.normal(size=41),
                    'surprise_past':rng.normal(size=41)})
    _,_,before=walkforward(d,['f'])
    changed=d.copy(); changed.loc[changed.Season>=2020,'surprise_past']+=100
    changed.loc[changed.Season>=2020,'f']-=100
    _,_,after=walkforward(changed,['f'])
    pd.testing.assert_frame_equal(before[before.Season<2020],after[after.Season<2020])
    assert before.Season.min()==1995

if __name__=='__main__':
    test_scoring_and_temporal_boundaries()
    print('Scoring identity and past-only forecast checks passed.')
