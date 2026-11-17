"""Data-free checks for the issue #59 EDA helpers."""
from collections import Counter
from itertools import combinations

import numpy as np
import pandas as pd

from kaggle_tuning.research.kaggle_eda import round_of, walk_forward


def test_round_of_matches_bracket_structure():
    seeds = [f"{r}{s:02d}" for r in "WXYZ" for s in range(1, 17)]
    # each team can meet 1, 2, 4, 8, 16, 32 opponents in rounds 1..6
    per_team = Counter(round_of("W01", b) for b in seeds if b != "W01")
    assert per_team == {1: 1, 2: 2, 3: 4, 4: 8, 5: 16, 6: 32}
    assert round_of("W01", "W16") == 1 and round_of("W08", "W01") == 2 and round_of("X02", "X03") == 3
    assert round_of("Y01", "Y02") == 4 and round_of("Y11a", "Z01") == 5 and round_of("W01", "Y01") == 6
    assert all(round_of(a, b) == round_of(b, a) for a, b in combinations(seeds, 2))


def test_walk_forward_uses_only_earlier_seasons():
    rng = np.random.default_rng(0)
    d = pd.DataFrame({"Season": np.repeat(np.arange(2000, 2010), 40), "x": rng.integers(-15, 16, 400)})
    d["y"] = (rng.random(400) < 1 / (1 + np.exp(-0.15 * d.x))).astype(int)
    before = walk_forward(d, [2005, 2006])
    changed = d.copy()
    changed.loc[changed.Season >= 2006, "y"] = 1 - changed.y  # rewrite the future
    after = walk_forward(changed, [2005, 2006])
    # targets 2005 and 2006 are fitted on <2005 and <2006, so neither sees the rewritten seasons
    pd.testing.assert_series_equal(before.p, after.p)
    assert not np.allclose(before[before.Season == 2006].p, 0.5)
