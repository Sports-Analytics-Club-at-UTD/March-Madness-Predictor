"""Men's ridge/xgb blend (mens_allyears) + women's walk-forward temperature (women_allyears)."""
from kaggle_tuning.configs.mens_allyears import RECOMMENDED as MEN
from kaggle_tuning.configs.women_allyears import RECOMMENDED as WOMEN

BEST = {**MEN, "women_model_temperature": WOMEN["women_model_temperature"]}
CONFIGS = {"allyears": BEST}
