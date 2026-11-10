"""Optional team-rating features; all ensemble weights and calibration stay fixed."""
from copy import deepcopy
from kaggle_tuning.configs.allyears import BEST, EXT


def variant(men=(), women=(), replace=False):
    cfg = deepcopy(EXT)
    def update(c, g, cols):
        if c.get('model') not in ('logit','xgb_spline') or not cols:
            return c
        c['team_ratings'] = True
        c[g] = [f for f in c[g] if not (replace and f == 'quality')] + list(cols)
        if c['model'] == 'xgb_spline':
            c['xgb'] = {**c.get('xgb',{}),'n_jobs':1}
        return c
    cfg['men_models'] = [(update(c,'M',men),w) for c,w in cfg['men_models']]
    cfg['by_gender']['W']['models'] = [(update(c,'W',women),w) for c,w in cfg['by_gender']['W']['models']]
    return cfg

CARRY = variant(men=['rt_carry10'])
EFFICIENCY = variant(men=['rt_off1','rt_def1'],women=['rt_off1','rt_def1'])
WF = variant(women=['rt_wf'],replace=True)
CONFIGS = {'carry':CARRY,'efficiency':EFFICIENCY,'wf':WF}
