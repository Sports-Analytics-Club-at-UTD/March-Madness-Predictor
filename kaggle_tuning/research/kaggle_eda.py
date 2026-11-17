"""Reproducible Kaggle MMLM data audit for issue #59 (read-only on data/kaggle).

    MANIA_DATA=/abs/path/to/data/kaggle/mirror26 uv run --extra training python -m kaggle_tuning.research.kaggle_eda

Writes kaggle_tuning/research/kaggle_eda/summary.json. Every probability below comes from a
seed-only logistic fitted on earlier seasons only; nothing here tunes a submission.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit

from kaggle_tuning import features as F
from kaggle_tuning.all_years import lb_scores
from kaggle_tuning.research.bust_years import coefficient

OUT = Path(__file__).with_name("kaggle_eda")
ORDER = [1, 16, 8, 9, 5, 12, 4, 13, 6, 11, 3, 14, 7, 10, 2, 15]  # standard 16-team region order
SEMIS = [{"W", "X"}, {"Y", "Z"}]  # Kaggle region letters: W plays X, Y plays Z in the Final Four
EVAL = [2015, 2016, 2017, 2018, 2019, 2021, 2022, 2023, 2024, 2025, 2026]
ROUNDS = ["R64", "R32", "S16", "E8", "F4", "NC"]


def round_of(a: str, b: str) -> int:
    """Main-bracket round (1-6) in which seed strings like 'W01' and 'X16a' can meet."""
    if a[0] != b[0]:
        return 5 if {a[0], b[0]} in SEMIS else 6
    i, j = ORDER.index(int(a[1:3])), ORDER.index(int(b[1:3]))
    return next(r for r, w in ((1, 2), (2, 4), (3, 8), (4, 16)) if i // w == j // w)


def games(g: str) -> pd.DataFrame:
    """Main-bracket games, one row per game with T1 < T2 (Kaggle ID orientation), 1985/1998-2026."""
    sd = F.read(f"{g}NCAATourneySeeds").set_index(["Season", "TeamID"]).Seed
    t = F.read(f"{g}NCAATourneyCompactResults")
    rows = pd.DataFrame({"Season": t.Season, "T1": t[["WTeamID", "LTeamID"]].min(axis=1),
                         "T2": t[["WTeamID", "LTeamID"]].max(axis=1), "WLoc": t.WLoc})
    rows["y"] = (t.WTeamID == rows.T1).astype(int)
    rows["home"] = np.where(t.WLoc == "N", 0, np.where((t.WLoc == "H") == (rows.y == 1), 1, -1))  # +1 = T1 at home
    s = F.solution_2026()
    sol = pd.DataFrame({"Season": 2026, "T1": s.ID.str[5:9].astype(int), "T2": s.ID.str[10:14].astype(int),
                        "WLoc": "?", "y": s.y, "home": 0})  # solution file has no venue: home unknown in 2026
    lo = 3000 if g == "W" else 1000
    rows = pd.concat([rows, sol[(sol.T1 >= lo) & (sol.T1 < lo + 1000)]], ignore_index=True)
    s1 = sd.reindex(pd.MultiIndex.from_arrays([rows.Season, rows.T1])).to_numpy()
    s2 = sd.reindex(pd.MultiIndex.from_arrays([rows.Season, rows.T2])).to_numpy()
    rows = rows[np.array([a[:3] != b[:3] for a, b in zip(s1, s2)])].copy()  # drop First Four (same seed line)
    s1, s2 = s1[rows.index], s2[rows.index]
    rows["x"] = [int(b[1:3]) - int(a[1:3]) for a, b in zip(s1, s2)]
    rows["round"] = [round_of(a, b) for a, b in zip(s1, s2)]
    return rows.reset_index(drop=True)


def walk_forward(d: pd.DataFrame, years=EVAL) -> pd.DataFrame:
    """Seed-only probability for each target season, fitted on strictly earlier seasons."""
    out = []
    for y in years:
        past = d[d.Season < y]
        both = pd.concat([past[["x", "y"]], past[["x", "y"]].assign(x=-past.x, y=1 - past.y)])  # symmetric fit
        cur = d[d.Season == y].copy()
        cur["p"] = expit(coefficient(both) * cur.x)
        out.append(cur)
    return pd.concat(out, ignore_index=True)


def brier(p, y) -> float:
    return float(np.mean((np.asarray(p) - np.asarray(y)) ** 2))


def inventory() -> list[dict]:
    rows = []
    for g in "MW":
        for f in ["RegularSeasonCompactResults", "RegularSeasonDetailedResults", "NCAATourneyCompactResults",
                  "NCAATourneyDetailedResults", "NCAATourneySeeds", "ConferenceTourneyGames",
                  "SecondaryTourneyCompactResults", "MasseyOrdinals", "TeamCoaches"]:
            if not (F.DATA / f"{g}{f}.csv").exists():
                rows.append({"file": g + f, "rows": 0})
                continue
            d = F.read(g + f)
            day = next((c for c in ("DayNum", "RankingDayNum", "LastDayNum") if c in d), None)
            rows.append({"file": g + f, "rows": len(d), "seasons": [int(d.Season.min()), int(d.Season.max())],
                         "max_day": int(d[day].max()) if day else None})
    return rows


def scoring_population(d: pd.DataFrame, g: str) -> dict:
    t = F.read(f"{g}NCAATourneyCompactResults")
    raw = t[t.Season >= 2015].groupby("Season").size()
    main = d[d.Season >= 2015].groupby("Season").size()
    per_round = d[d.Season >= 2015].groupby(["Season", "round"]).size().unstack()
    return {"compact_games": {int(k): int(v) for k, v in raw.items()},
            "scored_main_bracket": {int(k): int(v) for k, v in main.items()},
            "round_counts_ok": bool((per_round.drop(2021, errors="ignore") == [32, 16, 8, 4, 2, 1]).all().all()),
            "lower_id_win_rate": float(d.y.mean())}


def baselines(wf: dict[str, pd.DataFrame]) -> dict:
    rng = np.random.default_rng(20261116)
    out = {}
    for y in EVAL:
        cur = {g: wf[g][wf[g].Season == y] for g in "MW"}
        both = pd.concat(cur.values())
        se = (both.p - both.y) ** 2
        boot = [se.sample(frac=1, replace=True, random_state=rng.integers(1 << 31)).mean() for _ in range(2000)]
        out[y] = {"n": len(both), "seed_M": brier(cur["M"].p, cur["M"].y), "seed_W": brier(cur["W"].p, cur["W"].y),
                  "seed_all": float(se.mean()), "seed_all_boot95": [float(np.quantile(boot, .025)), float(np.quantile(boot, .975))],
                  "constant_half": 0.25}
    return out


def by_round(wf: dict[str, pd.DataFrame]) -> dict:
    out = {}
    for g in "MW":
        d = wf[g]
        out[g] = {ROUNDS[r - 1]: {"n": int(len(x)), "seed_brier": brier(x.p, x.y), "favorite_win": float(((x.p > .5) == (x.y == 1))[x.p != .5].mean()),
                                  "share_of_total_sq_error": float(((x.p - x.y) ** 2).sum() / ((d.p - d.y) ** 2).sum())}
                  for r, x in d.groupby("round")}
    return out


def upsets_r64(d: pd.DataFrame) -> dict:
    r = d[d["round"] == 1]
    better_won = np.where(r.x > 0, r.y, 1 - r.y)
    worse = (8.5 + r.x.abs() / 2).astype(int)  # seeds sum to 17, so worse seed = (17 + |diff|) / 2
    return {f"{17 - s}v{s}": {"n": int((worse == s).sum()), "lower_seed_win": float(1 - better_won[worse == s].mean())}
            for s in range(9, 17)}


def women_host(wf: pd.DataFrame) -> dict:
    """Non-neutral women's tournament games: did the home team beat its seed-only expectation?"""
    h = wf[wf.home != 0]
    p_home = np.where(h.home == 1, h.p, 1 - h.p)
    y_home = np.where(h.home == 1, h.y, 1 - h.y)
    by = {int(s): int(n) for s, n in h.groupby("Season").size().items()}
    resid = pd.Series(y_home - p_home).groupby(h.Season.to_numpy()).agg(["sum", "count"])
    rng = np.random.default_rng(20261116)
    boot = [resid.iloc[i].sum().pipe(lambda t: t["sum"] / t["count"]) for i in rng.integers(0, len(resid), (4000, len(resid)))]
    return {"n": int(len(h)), "seasons": by, "home_win": float(y_home.mean()), "seed_expected": float(p_home.mean()),
            "excess_season_boot95": [float(np.quantile(boot, .025)), float(np.quantile(boot, .975))],
            "home_was_better_seed": float((np.where(h.home == 1, h.x, -h.x) > 0).mean()),
            "rounds": {ROUNDS[r - 1]: int(n) for r, n in h.groupby("round").size().items()}}


def leakage_audit() -> dict:
    out = {}
    for g in "MW":
        reg = F.read(f"{g}RegularSeasonCompactResults")
        conf = F.read(f"{g}ConferenceTourneyGames")
        key = ["Season", "DayNum", "WTeamID", "LTeamID"]
        out[g] = {"regular_max_day": int(reg.DayNum.max()),
                  "conference_tourney_in_regular": float(conf.merge(reg[key], on=key).shape[0] / len(conf)),
                  "tourney_min_day": int(F.read(f"{g}NCAATourneyCompactResults").DayNum.min()),
                  "secondary_tourney_min_day": int(F.read(f"{g}SecondaryTourneyCompactResults").DayNum.min())}
    m = F.read("MMasseyOrdinals")
    out["M"]["massey_max_ranking_day"] = int(m.RankingDayNum.max())
    out["M"]["massey_rows_day_133"] = int((m.RankingDayNum == 133).sum())
    c = F.read("MTeamCoaches")
    out["M"]["coach_rows_ending_after_132"] = int(((c.LastDayNum > 132) & (c.LastDayNum < c.groupby("Season").LastDayNum.transform("max"))).sum())
    out["W"]["massey_available"] = (F.DATA / "WMasseyOrdinals.csv").exists()
    return out


def leaderboard() -> dict:
    out = {}
    for y in ("2025", "2026"):
        s = lb_scores(y)
        out[y] = {"teams": len(s), "best": float(s[0]), "top20": float(s[19]), "top100": float(s[99]),
                  "median": float(np.median(s)), "teams_within_0.001_of_top100": int((abs(s - s[99]) <= .001).sum())}
    return out


def main():
    OUT.mkdir(exist_ok=True)
    d = {g: games(g) for g in "MW"}
    wf = {g: walk_forward(d[g]) for g in "MW"}
    sub = F.read("SampleSubmissionStage2")
    res = {"inventory": inventory(),
           "stage2_rows": len(sub), "scored_2026_rows": int(len(F.solution_2026())),
           "scoring": {g: scoring_population(d[g], g) for g in "MW"},
           "leakage": leakage_audit(), "seed_walk_forward": baselines(wf), "by_round": by_round(wf),
           "r64_upsets": {g: upsets_r64(d[g][d[g].Season <= 2025]) for g in "MW"},
           "women_host": women_host(walk_forward(d["W"], sorted(d["W"].Season.unique())[1:])), "leaderboard": leaderboard()}
    (OUT / "summary.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
