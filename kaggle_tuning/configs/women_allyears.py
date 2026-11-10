"""Model-only women's calibration, fitted on earlier walk-forward predictions."""
from kaggle_tuning.configs.combined import BEST

RECOMMENDED = {**BEST, 'women_model_temperature': {'start': 2014, 'penalty': .01}}
CONFIGS = {'women_allyears': RECOMMENDED}
