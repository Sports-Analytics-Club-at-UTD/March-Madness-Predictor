"""Optional men's research features; all inputs stop before the tournament."""

from functools import lru_cache

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.special import ndtri
from sklearn.linear_model import Ridge

from . import features as F


@lru_cache(maxsize=1)
def ratings():
    d = F.team_games("M")
    d = d[d.DayNum <= 132].copy()
    sd = F.seeds("M")
    conf = F.read("MTeamConferences").set_index(["Season", "TeamID"]).ConfAbbrev
    out = []
    for year, s in d.groupby("Season"):
        teams = np.unique(np.r_[s.T1, s.T2])
        idx = {t: i for i, t in enumerate(teams)}
        n, nt = len(s), len(teams)
        a, b, r = s.T1.map(idx).values, s.T2.map(idx).values, np.arange(n)
        loc = sparse.csr_matrix(s["loc"].values[:, None])
        X = sparse.csr_matrix(
            (np.r_[np.ones(n), -np.ones(n)], (np.r_[r, r], np.r_[a, b])), shape=(n, nt)
        )
        X = sparse.hstack([X, loc]).tocsr()
        y = s.margin.values.astype(float)
        rec = np.exp(1.5 * (s.DayNum.values - 132) / 132)
        seeded = sd[sd.Season == year].set_index("TeamID").seed
        seed_weight = 1 + (17 - s.T1.map(seeded).fillna(17).values) / 16
        seed_weight += (17 - s.T2.map(seeded).fillna(17).values) / 16
        ft_y = y + 0.5 * ((s.T1_FTA - s.T1_FTM) - (s.T2_FTA - s.T2_FTM)).values
        fit = lambda x, target, w: (
            Ridge(alpha=2, fit_intercept=False).fit(x, target, sample_weight=w).coef_
        )
        z = pd.DataFrame({"Season": year, "TeamID": teams})
        z["q_dayseed"] = fit(X, np.clip(y, -60, 60), rec * seed_weight)[:nt]
        z["q_ft"] = fit(X, np.clip(ft_y, -60, 60), rec * seed_weight)[:nt]
        good = set(seeded.index) if len(seeded) else set(teams)  # No 2020 tournament.
        good |= set(s.loc[(s.margin > 0) & s.T2.isin(good), "T1"])
        pooled = {t: (idx[t] if t in good else nt) for t in teams}
        pa, pb = s.T1.map(pooled).values, s.T2.map(pooled).values
        XP = sparse.csr_matrix(
            (np.r_[np.ones(n), -np.ones(n)], (np.r_[r, r], np.r_[pa, pb])),
            shape=(n, nt + 1),
        )
        XP = sparse.hstack([XP, loc]).tocsr()
        keep = s.T1.isin(good).values | s.T2.isin(good).values
        z["q_pool"] = fit(XP[keep], np.clip(y[keep], -60, 60), np.ones(keep.sum()))[
            [pooled[t] for t in teams]
        ]
        z["q_raddar"] = fit(
            XP[keep], np.clip(ft_y[keep], -60, 60), (rec * seed_weight)[keep]
        )[[pooled[t] for t in teams]]
        poss = (F.possessions(s, "T1_") + F.possessions(s, "T2_")).values / 2
        # Each scoring observation has its own offense and the opponent's defense.
        XO = sparse.csr_matrix(
            (np.ones(2 * n), (np.r_[r, r], np.r_[a, nt + b])), shape=(n, 2 * nt)
        )
        XO = sparse.hstack([XO, loc]).tocsr()
        target = 100 * s.T1_Score.values / poss
        for alpha in (10, 100):
            m = Ridge(alpha=alpha).fit(XO, target, sample_weight=np.ones(n))
            z[f"adj_o{alpha}"], z[f"adj_d{alpha}"] = m.coef_[:nt], -m.coef_[nt : 2 * nt]
            z[f"adj_net{alpha}"] = z[f"adj_o{alpha}"] - z[f"adj_d{alpha}"]
        oppq = dict(zip(teams, fit(X, np.clip(y, -60, 60), np.ones(n))[:nt]))
        ss = s.copy()
        ss["performance"] = y + s.T2.map(oppq).values - 3 * s["loc"].values
        ss["late_var"] = ss["performance"]
        late = ss[ss.DayNum >= 100].groupby("T1").performance.mean()
        z["late_adj"] = z.TeamID.map(late)
        z["form_delta"] = z.late_adj - z.TeamID.map(oppq)
        z["late_stability"] = z.TeamID.map(
            ss[ss.DayNum >= 100].groupby("T1").late_var.std()
        )
        c1 = conf.reindex(pd.MultiIndex.from_arrays([s.Season, s.T1])).values
        c2 = conf.reindex(pd.MultiIndex.from_arrays([s.Season, s.T2])).values
        cs = sorted(set(c1) | set(c2))
        ci = {c: i for i, c in enumerate(cs)}
        ca, cb = np.array([ci[c] for c in c1]), np.array([ci[c] for c in c2])
        XC = sparse.csr_matrix(
            (np.r_[np.ones(n), -np.ones(n)], (np.r_[r, r], np.r_[ca, cb])),
            shape=(n, len(cs)),
        )
        XC = sparse.hstack([XC, loc]).tocsr()
        mask = c1 != c2
        cq = fit(XC[mask], np.clip(ft_y[mask], -60, 60), rec[mask])[: len(cs)]
        team_conf = conf.reindex(
            pd.MultiIndex.from_arrays([np.full(nt, year), teams])
        ).values
        z["conf_quality"] = [cq[ci[c]] for c in team_conf]
        out.append(z)
    return pd.concat(out, ignore_index=True)


@lru_cache(maxsize=1)
def elo_replay():
    d = F.read("MRegularSeasonCompactResults")
    d = d[d.DayNum <= 132].sort_values(["Season", "DayNum"])
    out = []
    for year, s in d.groupby("Season"):
        ts = sorted(set(s.WTeamID) | set(s.LTeamID))
        rating = {t: 0.0 for t in ts}
        for _ in range(5):
            for w, l, ws, ls, loc in zip(
                s.WTeamID, s.LTeamID, s.WScore, s.LScore, s.WLoc
            ):
                home = 3 if loc == "H" else -3 if loc == "A" else 0
                delta = 0.08 * ((ws - ls) - (rating[w] - rating[l] + home))
                rating[w] += delta
                rating[l] -= delta
        out.append(
            pd.DataFrame(
                {"Season": year, "TeamID": ts, "elo5": [rating[t] for t in ts]}
            )
        )
    return pd.concat(out, ignore_index=True)


@lru_cache(maxsize=1)
def ordinals():
    m = F.read("MMasseyOrdinals")
    m = m[m.RankingDayNum <= 133]
    last = (
        m.sort_values("RankingDayNum")
        .groupby(["Season", "SystemName", "TeamID"])
        .tail(1)
    )
    # Stable, broad coverage is selected from input availability, never outcomes.
    systems = [
        "POM",
        "SAG",
        "MOR",
        "WLK",
        "DOL",
        "COL",
        "MAS",
        "RTH",
        "WOL",
        "BIH",
        "TRK",
        "KPK",
        "DOK",
        "BOB",
        "RPI",
        "NET",
    ]
    p = last[last.SystemName.isin(systems)].pivot(
        index=["Season", "TeamID"], columns="SystemName", values="OrdinalRank"
    )
    count = last.groupby("Season").TeamID.nunique()
    n = p.index.get_level_values("Season").map(count).to_numpy()
    z = pd.DataFrame(index=p.index)
    for c in p:
        z[f"normal_{c}"] = ndtri(np.clip(1 - (p[c].values - 0.5) / n, 0.001, 0.999))
        z[f"log_{c}"] = -np.log(p[c])
    z["normal_consensus"] = z[[c for c in z if c.startswith("normal_")]].mean(axis=1)
    z["normal_worst"] = z[[f"normal_{c}" for c in ["POM", "WLK", "MOR"]]].min(axis=1)
    recent = m[m.RankingDayNum >= 119]
    for agg in ["mean", "median", "min"]:
        a = (
            recent.groupby(["Season", "SystemName", "TeamID"])
            .OrdinalRank.agg(agg)
            .reset_index()
        )
        a = a[a.SystemName.isin(["POM", "WLK", "MOR"])].pivot(
            index=["Season", "TeamID"], columns="SystemName", values="OrdinalRank"
        )
        z[f"recent_{agg}"] = -np.log(a).mean(axis=1)
        z[f"recent_POM_{agg}"] = -np.log(a.POM)
    return z.reset_index()


@lru_cache(maxsize=1)
def extra_table():
    t = ratings().merge(elo_replay(), on=["Season", "TeamID"], how="left")
    return t.merge(ordinals(), on=["Season", "TeamID"], how="left")


def self_check():
    t = extra_table()
    assert not t.duplicated(["Season", "TeamID"]).any()
    assert np.isfinite(t[["q_pool", "q_raddar", "adj_net10", "elo5"]]).all().all()
    assert t.Season.max() == 2026
    # Inject impossible post-deadline observations: every optional feature must ignore them.
    original = F.read

    def contaminated(name):
        frame = original(name)
        day = "RankingDayNum" if name == "MMasseyOrdinals" else "DayNum"
        if name not in (
            "MMasseyOrdinals",
            "MRegularSeasonCompactResults",
            "MRegularSeasonDetailedResults",
        ):
            return frame
        extra = frame.tail(10).copy()
        extra[day] = 200
        if name == "MMasseyOrdinals":
            extra["OrdinalRank"] = 9999
        else:
            extra["WScore"] = 9999
        return pd.concat([frame, extra], ignore_index=True)

    def clear():
        for fn in (ratings, elo_replay, ordinals, extra_table):
            fn.cache_clear()

    try:
        F.read = contaminated
        clear()
        pd.testing.assert_frame_equal(t, extra_table())
    finally:
        F.read = original
        clear()


if __name__ == "__main__":
    self_check()
