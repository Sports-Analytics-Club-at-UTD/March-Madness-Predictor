"""Score a config on 2021-2026 with each year's real Kaggle metric and rank it on that year's leaderboard.

2021, 2022: log loss, separate men's / women's competitions.
2023, 2025, 2026: Brier on men's + women's games together.
2024: "Average Brier Bracket Score" (per-round team advancement probabilities, 6 rounds averaged,
      men's and women's averaged). Our pairwise probabilities give the exact round probabilities a
      100k-bracket portfolio sampled from them would approximate.
2021-2024 are model-only: no pre-deadline market file exists for those years.

    uv run python -m kaggle_tuning.all_years [configs_file:NAME]
"""
from __future__ import annotations

import glob
import runpy
import sys

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from .features import DATA

LB = {
    2021: {"M": "ncaam-march-mania-2021", "W": "ncaaw-march-mania-2021"},
    2022: {"M": "mens-march-mania-2022", "W": "womens-march-mania-2022"},
    2023: "march-machine-learning-mania-2023",
    2024: "2024",
    2025: "2025",
    2026: "2026",
}


def lb_scores(key: str) -> np.ndarray:
    return np.sort(pd.read_csv(glob.glob(str(DATA.parent / "lb" / key / "*.csv"))[0]).Score.values)


def rank(key: str, score: float) -> tuple[int, int]:
    s = lb_scores(key)
    return int((s < score).sum() + 1), len(s)


def all_pair_predictions(cfg: dict, year: int) -> dict[str, pd.DataFrame]:
    """Predict every pair of seeded teams; y filled for games actually played (play-ins excluded)."""
    from . import backtest as B
    from . import features as F

    played = {g: B.test_pairs(g, year) for g in "MW"}

    def all_pairs(g, y):
        assert y == year
        teams = sorted(F.read(f"{g}NCAATourneySeeds").query("Season == @year").TeamID)
        pairs = pd.DataFrame([(a, b) for i, a in enumerate(teams) for b in teams[i + 1:]], columns=["T1", "T2"])
        pairs.insert(0, "Season", year)
        return pairs.merge(played[g][["T1", "T2", "y"]], on=["T1", "T2"], how="left")

    B.test_pairs = all_pairs  # every module looks this up at call time
    out = {}
    for g in "MW":
        table = F.team_table(g, cfg.get("q_alpha", 1.0), cfg.get("q_recency", 0.0), cfg.get("q_clip_" + g, 30.0))
        df = all_pairs(g, year)
        df["p"] = B.fit_predict(cfg, g, year, table)
        out[g] = df
    return out


def bracket_score(g: str, year: int, pairs: pd.DataFrame) -> float:
    """Kaggle 2024 Average Brier Bracket Score for one gender from pairwise probabilities."""
    from . import features as F

    seeds = F.read(f"{g}NCAATourneySeeds").query("Season == @year")
    slots = F.read(f"{g}NCAATourneySlots").query("Season == @year")
    games = F.read(f"{g}NCAATourneyCompactResults").query("Season == @year")
    P = {}
    for t1, t2, p in zip(pairs.T1, pairs.T2, pairs.p):
        P[(t1, t2)], P[(t2, t1)] = p, 1 - p
    winner_of = {frozenset((w, l)): w for w, l in zip(games.WTeamID, games.LTeamID)}
    dist = {s: {t: 1.0} for s, t in zip(seeds.Seed, seeds.TeamID)}
    actual = dict(zip(seeds.Seed, seeds.TeamID))
    pos = {t: s[:3] for s, t in zip(seeds.Seed, seeds.TeamID)}
    pending = slots.copy()
    while len(pending):
        ready = pending[pending.StrongSeed.isin(dist) & pending.WeakSeed.isin(dist)]
        for slot, a, b in zip(ready.Slot, ready.StrongSeed, ready.WeakSeed):
            d = {}
            for ta, pa in dist[a].items():
                for tb, pb in dist[b].items():
                    d[ta] = d.get(ta, 0) + pa * pb * P[(ta, tb)]
                    d[tb] = d.get(tb, 0) + pa * pb * P[(tb, ta)]
            dist[slot] = d
            actual[slot] = winner_of[frozenset((actual[a], actual[b]))]
        pending = pending[~pending.Slot.isin(ready.Slot)]
    briers = []
    for r in range(1, 7):
        rs = slots[slots.Slot.str.startswith(f"R{r}")].Slot
        pred = {}
        for s in rs:
            for t, p in dist[s].items():
                pred[pos[t]] = pred.get(pos[t], 0) + p
        won = {pos[actual[s]] for s in rs}
        positions = sorted({pos[t] for t in seeds.TeamID})
        briers.append(np.mean([(pred.get(q, 0) - (q in won)) ** 2 for q in positions]))
    return float(np.mean(briers))


def score_year(cfg: dict, year: int) -> dict:
    preds = all_pair_predictions(cfg, year)
    played = {g: d.dropna(subset=["y"]) for g, d in preds.items()}
    res = {"year": year}
    if year in (2021, 2022):
        for g in "MW":
            d = played[g]
            p = np.clip(d.p.values, 1e-15, 1 - 1e-15)
            ll = float(-np.mean(d.y * np.log(p) + (1 - d.y) * np.log(1 - p)))
            res[g] = (ll, *rank(LB[year][g], ll))
        res["metric"] = "log loss"
    elif year == 2024:
        sc = float(np.mean([bracket_score(g, year, preds[g]) for g in "MW"]))
        res["all"] = (sc, *rank(LB[year], sc))
        res["metric"] = "bracket Brier"
    else:
        y = np.concatenate([played[g].y.values for g in "MW"])
        p = np.concatenate([played[g].p.values for g in "MW"])
        b = float(np.mean((p - y) ** 2))
        res["all"] = (b, *rank(LB[year], b))
        res["metric"] = "Brier"
    return res


def main():
    spec = sys.argv[1] if len(sys.argv) > 1 else "kaggle_tuning/configs/combined.py:BEST"
    path, name = spec.split(":")
    ns = runpy.run_path(path)
    cfg = ns[name] if name in ns else ns["CONFIGS"][name]
    out = Parallel(n_jobs=6)(delayed(score_year)(cfg, y) for y in [2021, 2022, 2023, 2024, 2025, 2026])
    print("* 2020: no tournament (COVID)")
    for r in out:
        if "all" in r:
            s, k, n = r["all"]
            print(f"* {r['year']}: rank #{k} of {n}, {s:.5f} ({r['metric']})")
        else:
            print(f"* {r['year']}: men's rank #{r['M'][1]} of {r['M'][2]}, {r['M'][0]:.5f}; "
                  f"women's rank #{r['W'][1]} of {r['W'][2]}, {r['W'][0]:.5f} ({r['metric']})")


if __name__ == "__main__":
    main()
