"""Faithful Vilnius variants and historical-weight L/X ensembles.

The ensemble fits nonnegative, sum-to-one log-odds weights separately by gender,
using only HIST seasons before each target (capped at 2024 for 2025/2026).
"""
from kaggle_tuning.configs.final import BASE

V = {'model': 'vilnius'}
CONFIGS = {}
for name, options in [('V', {}), ('modeh', {'modeh': True}),
                      ('cauchy', {'cauchy': True}), ('seeds', {'seeds': [0, 1, 2]})]:
    CONFIGS[name] = {**V, 'vilnius': options}
    ens = {'model': 'vilnius_ensemble', 'variant': name}
    CONFIGS[name + '_ensemble'] = ens
    CONFIGS[name + '_market'] = {**ens, 'market_w': BASE['market_w'], 'mkt_sharpen': BASE['mkt_sharpen']}
