"""Market-weight insurance variants of the current best (men's weight lowered)."""
from kaggle_tuning.configs.combined import BEST
CONFIGS = {f"best_mkM{w}": {**BEST, "market_w": {"M": w, "W": 0.5}} for w in (0.85, 0.65, 0.5, 0.25)}
CONFIGS["best_mkM0.5_noclipfix"] = {**BEST, "market_w": {"M": 0.5, "W": 0.5}, "mkt_sharpen": 1.0}
