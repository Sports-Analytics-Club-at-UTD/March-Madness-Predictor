from kaggle_tuning.configs.final import BASE, L, ENS
from kaggle_tuning.configs.women import women, ORIGINAL

OG = {**ORIGINAL, 'original_lgb_weight': 0, 'horan_W': False}
CONFIGS = {}
for weight in (.1, .2, .3, .5):
    CONFIGS[f'original_ensemble{weight}'] = women({'models': [(ENS, 1-weight), (OG, weight)]})
for extra in (['quality'], ['quality','elo'], ['quality','elo','quality_rec']):
    for weight in (0, .35):
        CONFIGS[f'original_extra_{"_".join(extra)}_lgb{weight}'] = women({**OG, 'women_extra': extra, 'original_lgb_weight': weight})
for a in (.8, 1.0, 1.2, 1.4):
    CONFIGS[f'baseline_sharp{a}'] = {**BASE, 'sharpen': {'W': a}}
