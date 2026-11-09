"""Women's research; men's final configuration stays fixed."""
from kaggle_tuning.configs.final import BASE, L


def women(override, market=.5):
    return {**BASE, 'market_w': {'M': .85, 'W': market}, 'by_gender': {'W': override}}


ORIGINAL = {'model': 'women_original'}
CONFIGS = {'baseline': BASE}
for weight in (0, .35):
    for sharp in (False, True):
        name = f'original_lgb{weight}_horan{int(sharp)}'
        CONFIGS[name] = women({**ORIGINAL, 'original_lgb_weight': weight, 'horan_W': sharp})
for method in ('temperature', 'isotonic'):
    CONFIGS['original_' + method] = women({**ORIGINAL, 'women_calibration': method})
    CONFIGS['logit_' + method] = women({**L, 'women_calibration': method})
for c in (.01, .1, 1, 10):
    CONFIGS[f'logit_C{c}'] = women({**L, 'C_W': c})
for alpha, clip, recency in ((.1, 60, 0), (10, 60, 0), (1, 30, 0), (1, 100, 0), (1, 60, 1.5)):
    CONFIGS[f'logit_q{alpha}_clip{clip}_rec{recency}'] = women({**L, 'women_q_alpha': alpha, 'women_q_clip': clip, 'women_q_recency': recency})
for k in (10, 40):
    CONFIGS[f'logit_elo{k}'] = women({**L, 'women_elo_k': k})
