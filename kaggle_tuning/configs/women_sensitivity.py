from kaggle_tuning.configs.women_recommended import AGGRESSIVE as RECOMMENDED, ORIGINAL_LOGIT
from kaggle_tuning.configs.final import BASE, L, X

CONFIGS = {}
for market in (0, .4, .5, .6):
    CONFIGS[f'recommended_market{market}'] = {**RECOMMENDED, 'market_w': {'M': .85, 'W': market}}
for original in (.4, .6):
    CONFIGS[f'recommended_original{original}'] = {**RECOMMENDED, 'by_gender': {'W': {'models': [({**L,'women_q_alpha':.1}, .9-original), (X,.1), (ORIGINAL_LOGIT,original)]}}}
