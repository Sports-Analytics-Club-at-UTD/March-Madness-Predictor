from kaggle_tuning.configs.final import L, X
from kaggle_tuning.configs.women import women
from kaggle_tuning.configs.women_round2 import OG

CONFIGS = {}
for ow, xw in ((.3,.35), (.4,.3), (.5,.25), (.3,.2), (.4,.2), (.5,.1)):
    for sharp in (1, 1.1):
        CONFIGS[f'q01_og{ow}_x{xw}_s{sharp}'] = {**women({'models': [({**L, 'women_q_alpha': .1},1-ow-xw), (X,xw), (OG,ow)]}), 'sharpen': {'W':sharp}}
for ow in (.2,.3,.4):
    for sharp in (1.05,1.1):
        CONFIGS[f'og{ow}_s{sharp}'] = {**women({'models': [(L,(1-ow)/2), (X,(1-ow)/2), (OG,ow)]}), 'sharpen': {'W':sharp}}
