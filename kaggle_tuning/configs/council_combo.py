"""Men's offense/defense ridge ratings (configs/ratings.py) + women's margin-lasso member (configs/penalized.py)."""
from copy import deepcopy
from kaggle_tuning.configs.penalized import CANDIDATE_EXT
from kaggle_tuning.configs.ratings import variant

EFF_M = variant(men=['rt_off1', 'rt_def1'])
COMBO = deepcopy(CANDIDATE_EXT)
COMBO['men_models'] = EFF_M['men_models']
CONFIGS = {'combo': COMBO, 'eff_men': EFF_M}
