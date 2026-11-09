"""Leak-free team-season features for March Machine Learning Mania backtests.

Everything here is computed from regular-season data (DayNum <= 132) plus seeds and
pre-tournament Massey ordinals, so features for season Y never see Y's tournament.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import Ridge

import os
DATA = Path(os.environ.get("MANIA_DATA", Path(__file__).resolve().parent.parent / "data" / "kaggle" / "mirror26"))

BOX = ["FGM", "FGA", "FGM3", "FGA3", "FTM", "FTA", "OR", "DR", "Ast", "TO", "Stl", "Blk", "PF"]


@lru_cache(maxsize=None)
def read(name: str) -> pd.DataFrame:
    return pd.read_csv(DATA / f"{name}.csv")


def team_games(g: str) -> pd.DataFrame:
    """Regular-season detailed results, one row per team per game (T1 = the team)."""
    df = read(f"{g}RegularSeasonDetailedResults")
    w = {"W": "T1_", "L": "T2_"}
    l = {"W": "T2_", "L": "T1_"}

    def ren(m):
        cols = {}
        for c in df.columns:
            if c[0] in "WL" and c != "WLoc":
                cols[c] = m[c[0]] + c[1:]
        return df.rename(columns=cols)

    a = ren(w)
    a["loc"] = a.WLoc.map({"H": 1, "A": -1, "N": 0})
    b = ren(l)
    b["loc"] = -b.WLoc.map({"H": 1, "A": -1, "N": 0})
    out = pd.concat([a, b], ignore_index=True).drop(columns="WLoc")
    out = out.rename(columns={"T1_TeamID": "T1", "T2_TeamID": "T2"})
    out["margin"] = out.T1_Score - out.T2_Score
    return out


def possessions(d: pd.DataFrame, p: str) -> pd.Series:
    return d[p + "FGA"] - d[p + "OR"] + d[p + "TO"] + 0.475 * d[p + "FTA"]


def box_features(g: str) -> pd.DataFrame:
    d = team_games(g)
    d["poss"] = (possessions(d, "T1_") + possessions(d, "T2_")) / 2
    d["off_eff"] = 100 * d.T1_Score / d.poss
    d["def_eff"] = 100 * d.T2_Score / d.poss
    d["efg"] = (d.T1_FGM + 0.5 * d.T1_FGM3) / d.T1_FGA
    d["opp_efg"] = (d.T2_FGM + 0.5 * d.T2_FGM3) / d.T2_FGA
    d["tov"] = d.T1_TO / d.poss
    d["opp_tov"] = d.T2_TO / d.poss
    d["orb"] = d.T1_OR / (d.T1_OR + d.T2_DR)
    d["drb"] = d.T1_DR / (d.T1_DR + d.T2_OR)
    d["ftr"] = d.T1_FTA / d.T1_FGA
    d["opp_ftr"] = d.T2_FTA / d.T2_FGA
    d["win"] = (d.margin > 0).astype(float)
    late = d[d.DayNum >= 115].groupby(["Season", "T1"]).agg(late_margin=("margin", "mean"), late_win=("win", "mean"))
    cols = ["margin", "poss", "off_eff", "def_eff", "efg", "opp_efg", "tov", "opp_tov", "orb", "drb", "ftr", "opp_ftr", "win",
            "T1_Score", "T2_Score", "T1_FGA3", "T1_Ast", "T1_Stl", "T1_Blk", "T2_Stl", "T2_Blk"]
    agg = d.groupby(["Season", "T1"])[cols].mean()
    agg = agg.join(late).reset_index().rename(columns={"T1": "TeamID"})
    agg["net_eff"] = agg.off_eff - agg.def_eff
    return agg


def ridge_quality(g: str, alpha: float = 1.0, recency: float = 0.0, clip: float = 30.0) -> pd.DataFrame:
    """Per-season adjusted margin rating: margin ~ q[T1] - q[T2] + h*loc (ridge)."""
    d = team_games(g)
    d = d[d.T1 < d.T2]  # one row per game
    rows = []
    for season, s in d.groupby("Season"):
        teams = np.unique(np.r_[s.T1.values, s.T2.values])
        idx = {t: i for i, t in enumerate(teams)}
        n = len(s)
        i1 = s.T1.map(idx).values
        i2 = s.T2.map(idx).values
        r = np.arange(n)
        X = sparse.csr_matrix(
            (np.r_[np.ones(n), -np.ones(n)], (np.r_[r, r], np.r_[i1, i2])), shape=(n, len(teams))
        )
        X = sparse.hstack([X, sparse.csr_matrix(s["loc"].values.reshape(-1, 1).astype(float))]).tocsr()
        y = np.clip(s.margin.values, -clip, clip).astype(float)
        w = np.ones(n)
        if recency:
            w = np.exp(recency * (s.DayNum.values - 132) / 132)
        m = Ridge(alpha=alpha, fit_intercept=False).fit(X, y, sample_weight=w)
        q = m.coef_[:-1]
        rows.append(pd.DataFrame({"Season": season, "TeamID": teams, "quality": q - q.mean()}))
    return pd.concat(rows, ignore_index=True)


def elo(g: str, k: float = 20.0, home: float = 75.0, carry: float = 0.75) -> pd.DataFrame:
    """Margin-aware Elo across regular-season compact results; snapshot at end of each season."""
    df = read(f"{g}RegularSeasonCompactResults").sort_values(["Season", "DayNum"])
    ratings: dict[int, float] = {}
    out = []
    for season, s in df.groupby("Season", sort=True):
        ratings = {t: 1500 + carry * (r - 1500) for t, r in ratings.items()}
        for w, l, ws, ls, loc in zip(s.WTeamID.values, s.LTeamID.values, s.WScore.values, s.LScore.values, s.WLoc.values):
            rw, rl = ratings.get(w, 1500.0), ratings.get(l, 1500.0)
            hadj = home if loc == "H" else (-home if loc == "A" else 0.0)
            exp_w = 1 / (1 + 10 ** ((rl - rw - hadj) / 400))
            mov = np.log1p(abs(ws - ls)) * 2.2 / ((rw - rl + hadj) * 0.001 + 2.2)
            delta = k * mov * (1 - exp_w)
            ratings[w] = rw + delta
            ratings[l] = rl - delta
        teams = set(s.WTeamID) | set(s.LTeamID)
        out.append(pd.DataFrame({"Season": season, "TeamID": list(teams), "elo": [ratings[t] for t in teams]}))
    return pd.concat(out, ignore_index=True)


def seeds(g: str) -> pd.DataFrame:
    s = read(f"{g}NCAATourneySeeds").copy()
    s["seed"] = s.Seed.str[1:3].astype(int)
    return s[["Season", "TeamID", "seed"]]


MASSEY_SYSTEMS = ["POM", "SAG", "MOR", "WLK", "DOL", "COL", "MAS", "RTH", "WOL", "BIH"]


def massey() -> pd.DataFrame:
    m = read("MMasseyOrdinals")
    m = m[m.RankingDayNum <= 133]
    last = m.sort_values("RankingDayNum").groupby(["Season", "SystemName", "TeamID"]).OrdinalRank.last().reset_index()
    piv = last[last.SystemName.isin(MASSEY_SYSTEMS)].pivot_table(index=["Season", "TeamID"], columns="SystemName", values="OrdinalRank")
    allsys = last.groupby(["Season", "TeamID"]).OrdinalRank.median().rename("massey_med")
    piv = piv.join(allsys).reset_index()
    piv.columns = [c if c in ("Season", "TeamID", "massey_med") else f"rank_{c}" for c in piv.columns]
    # rank -> roughly Gaussian strength (lower rank better)
    for c in [c for c in piv.columns if c.startswith("rank_") or c == "massey_med"]:
        piv[c.replace("rank_", "mstr_") if c != "massey_med" else "mstr_med"] = -np.log(piv[c])
    piv["mstr_worst"] = -np.log(piv[["rank_POM", "rank_WLK", "rank_MOR"]].max(axis=1))
    piv["mstr_mean3"] = piv[["mstr_POM", "mstr_WLK", "mstr_MOR"]].mean(axis=1)
    return piv


@lru_cache(maxsize=None)
def team_table(g: str, q_alpha: float = 1.0, q_recency: float = 0.0, q_clip: float = 30.0) -> pd.DataFrame:
    t = box_features(g)
    t = t.merge(ridge_quality(g, q_alpha, q_recency, q_clip), on=["Season", "TeamID"], how="left")
    t = t.merge(ridge_quality(g, q_alpha, 1.5, q_clip).rename(columns={"quality": "quality_rec"}), on=["Season", "TeamID"], how="left")
    t = t.merge(elo(g), on=["Season", "TeamID"], how="left")
    t = t.merge(seeds(g), on=["Season", "TeamID"], how="left")
    if g == "M":
        t = t.merge(massey(), on=["Season", "TeamID"], how="left")
    t["seed"] = t.seed.fillna(17)
    return t


def tourney_games(g: str, include_first_four: bool = False) -> pd.DataFrame:
    df = read(f"{g}NCAATourneyCompactResults")
    if not include_first_four:
        # play-in = both teams share a seed line (e.g. W16a vs W16b); women's First Four days differ from men's
        sd = read(f"{g}NCAATourneySeeds").set_index(["Season", "TeamID"]).Seed.str[:3]
        ws = sd.reindex(pd.MultiIndex.from_arrays([df.Season, df.WTeamID])).values
        ls = sd.reindex(pd.MultiIndex.from_arrays([df.Season, df.LTeamID])).values
        df = df[ws != ls]
    a = pd.DataFrame({"Season": df.Season, "T1": df.WTeamID, "T2": df.LTeamID, "margin": df.WScore - df.LScore})
    b = pd.DataFrame({"Season": df.Season, "T1": df.LTeamID, "T2": df.WTeamID, "margin": df.LScore - df.WScore})
    return pd.concat([a, b], ignore_index=True)


def solution_2026() -> pd.DataFrame:
    s = pd.read_csv(DATA.parent / "sol26" / "2026 MM Solution File.csv")
    return s[["ID", "Pred"]].rename(columns={"Pred": "y"})


def matchup_frame(pairs: pd.DataFrame, table: pd.DataFrame, feats: list[str]) -> pd.DataFrame:
    """pairs: Season,T1,T2 (+ optional cols). Adds T1_f, T2_f and diff_f for each feature."""
    t = table[["Season", "TeamID"] + feats]
    out = pairs.merge(t.add_prefix("T1_").rename(columns={"T1_Season": "Season", "T1_TeamID": "T1"}), on=["Season", "T1"], how="left")
    out = out.merge(t.add_prefix("T2_").rename(columns={"T2_Season": "Season", "T2_TeamID": "T2"}), on=["Season", "T2"], how="left")
    for f in feats:
        out[f"diff_{f}"] = out[f"T1_{f}"] - out[f"T2_{f}"]
    return out
