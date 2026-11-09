"""Recommended men's feature improvement; retain the established market weight.

The 80% weight is a descriptive target-year optimum, not independently validated.
"""

from kaggle_tuning.configs.mens_research import CONFIGS as RESEARCH

RECOMMENDED = RESEARCH["pool_worst_conf"]
CONFIGS = {
    "mens_recommended": RECOMMENDED,
    "mens_market80_diagnostic": {**RECOMMENDED, "market_w": {"M": 0.8, "W": 0.5}},
}
