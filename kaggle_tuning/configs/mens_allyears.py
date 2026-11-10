"""Fixed men's margin/tree blends; women's and market settings from BEST."""
from kaggle_tuning.configs.combined import BEST as BASELINE

LOGIT, TREE = [c for c, _ in BASELINE['men_models']]
MARGIN = {'model':'mens_allyears','mens_allyears':{'margin':True,'alpha':1,
          'features':['seed','q_pool','elo','mstr_med','mstr_worst','net_eff','conf_quality']}}
TREE75 = {**BASELINE,'men_models':[(TREE,3),(MARGIN,1)]}
BALANCED = {**BASELINE,'men_models':[(LOGIT,3),(TREE,9),(MARGIN,4)]}
CONFIGS = {'tree75':TREE75,'balanced':BALANCED}

SHRUNK = {**BASELINE,"men_models":[(TREE,1),(MARGIN,1)],"model_sharpen":{"M":.9}}
CONFIGS["shrunk"] = SHRUNK

RECOMMENDED = SHRUNK
