"""Pre-tournament betting-odds probabilities (goto_conversion), published before each deadline.

2025: kaggle dataset kaito510/538data (2025-03-18). 2026: PyPI goto-conversion==4.0.1 (2026-03-17).
"""
from functools import lru_cache
import pandas as pd
from .features import DATA

PATHS = {2025: DATA.parent / "market25" / "submission.csv", 2026: DATA.parent / "market26" / "submission.csv"}


@lru_cache(maxsize=None)
def market(year: int) -> pd.DataFrame:
    d = pd.read_csv(PATHS[year])
    return pd.DataFrame({"T1": d.ID.str[5:9].astype(int), "T2": d.ID.str[10:14].astype(int), "mkt": d.Pred})
