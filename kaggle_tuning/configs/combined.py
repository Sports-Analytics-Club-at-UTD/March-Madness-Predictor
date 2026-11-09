"""Women's blend (women_recommended) + men's features (mens_best), optionally with Vilnius in the women's blend."""
from kaggle_tuning.configs.final import L, X
from kaggle_tuning.configs.mens_best import RECOMMENDED as MEN
from kaggle_tuning.configs.women_recommended import RECOMMENDED as WREC, ORIGINAL_LOGIT

V = {"model": "vilnius", "vilnius": {}}
LW = {**L, "women_q_alpha": .1}
COMBINED = {**WREC, "men_models": MEN["men_models"]}
CONFIGS = {
    "combined": COMBINED,
    "combined_WV.2": {**COMBINED, "by_gender": {"W": {"models": [(LW, .4), (X, .1), (ORIGINAL_LOGIT, .3), (V, .2)]}}},
    "combined_WV.3": {**COMBINED, "by_gender": {"W": {"models": [(LW, .35), (ORIGINAL_LOGIT, .35), (V, .3)]}}},
    "combined_MV.3": {**COMBINED, "men_models": MEN["men_models"] + [(V, .9)]},
}

# Current best: better than "combined" in every 2021-2024 year and in 2025/2026.
BEST = CONFIGS["combined_WV.2"]
