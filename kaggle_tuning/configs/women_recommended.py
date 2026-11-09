"""Women's ensembles; prefer the conservative blend with improved history."""
from kaggle_tuning.configs.final import BASE, L, X

ORIGINAL_LOGIT = {'model': 'women_original', 'original_lgb_weight': 0}
WOMEN = {'models': [({**L, 'women_q_alpha': .1}, .5), (X, .1), (ORIGINAL_LOGIT, .4)]}
RECOMMENDED = {**BASE, 'by_gender': {'W': WOMEN}, 'sharpen': {'W': 1.1}}
AGGRESSIVE = {**RECOMMENDED, 'by_gender': {'W': {
    'models': [({**L, 'women_q_alpha': .1}, .4), (X, .1), (ORIGINAL_LOGIT, .5)]}}}
CONFIGS = {'women_recommended': RECOMMENDED, 'women_aggressive': AGGRESSIVE}
