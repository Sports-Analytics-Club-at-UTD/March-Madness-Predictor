"""Sensitivity around the retained 50% market rule, reusing women's fitted models."""
import json
from pathlib import Path
import numpy as np
from . import backtest as B
from .configs.women_recommended import RECOMMENDED


def main():
    results = {str(w): {} for w in (0, .4, .5, .6)}
    table = B.F.team_table('W', q_clip=60)
    for year in (2025, 2026):
        p = B.fit_predict({**RECOMMENDED, 'market_w': {}, 'sharpen': {}}, 'W', year, table)
        y = B.test_pairs('W', year).y.values
        for w, row in results.items():
            cfg = {**RECOMMENDED, 'market_w': {'W':float(w)}}
            row[year] = float(np.mean((B._post(cfg, 'W', year, p) - y) ** 2))
    print(json.dumps(results, indent=2))
    Path('kaggle_tuning/women_market_results.json').write_text(json.dumps(results, indent=2) + '\n')


if __name__ == '__main__':
    main()
