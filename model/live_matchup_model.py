# /// script
# requires-python = ">=3.13,<3.14"
# dependencies = ["numpy==2.5.3", "pandas==3.0.6", "scikit-learn==1.9.1", "scipy==1.18.1"]
# ///
"""Live matchup model: export, backtest and parity check.

This is the `logit` member of kaggle_tuning's `final` men's ensemble
(origin/kaggle-score-tuning, kaggle_tuning/backtest.py + features.py at 07db721),
restricted to the three features that exist during the regular season:

    quality  ridge-adjusted scoring margin (alpha 1, margins clipped to +/-60, home term)
    elo      margin-aware Elo (k 20, home 75, 75% carried between seasons)
    net_eff  mean per-game offensive minus defensive efficiency

`seed` and the Massey ordinals from the full model are dropped: neither exists
before Selection Sunday. The XGBoost member and the betting-market blend are
also dropped. Training: StandardScaler + LogisticRegression(C=0.1, no intercept)
on men's NCAA tournament games 2003-2025 (First Four excluded), both team
orders, with features from the same season's regular season.

    uv run model/live_matchup_model.py export --kaggle-dir /path/to/mirror26
    pnpm live:sync --dump /tmp/live.json
    uv run model/live_matchup_model.py parity --dump /tmp/live.json --team-a Duke --team-b Houston
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
FEATURES = ["quality", "elo", "net_eff"]
C = 0.1
Q_ALPHA, Q_CLIP = 1.0, 60.0
ELO_K, ELO_HOME, ELO_CARRY = 20.0, 75.0, 0.75
TRAIN_START, TRAIN_END = 2003, 2025


# --- features (same formulas as kaggle_tuning/features.py) -------------------

def ridge_quality(games: pd.DataFrame, solver: str = "auto") -> tuple[pd.DataFrame, float]:
    """games: one row per game with T1, T2, margin (T1 - T2), loc (+1 T1 home). Returns q and home edge."""
    teams = np.unique(np.r_[games.T1.values, games.T2.values])
    idx = {t: i for i, t in enumerate(teams)}
    n = len(games)
    r = np.arange(n)
    X = sparse.csr_matrix(
        (np.r_[np.ones(n), -np.ones(n)], (np.r_[r, r], np.r_[games.T1.map(idx).values, games.T2.map(idx).values])),
        shape=(n, len(teams)),
    )
    X = sparse.hstack([X, sparse.csr_matrix(games["loc"].values.reshape(-1, 1).astype(float))]).tocsr()
    y = np.clip(games.margin.values, -Q_CLIP, Q_CLIP).astype(float)
    m = Ridge(alpha=Q_ALPHA, fit_intercept=False, solver=solver).fit(X, y)
    q = m.coef_[:-1]
    return pd.DataFrame({"TeamID": teams, "quality": q - q.mean()}), float(m.coef_[-1])


def elo_update(ratings: dict, w, l, ws, ls, loc_w: int) -> None:
    rw, rl = ratings.get(w, 1500.0), ratings.get(l, 1500.0)
    hadj = ELO_HOME * loc_w
    exp_w = 1 / (1 + 10 ** ((rl - rw - hadj) / 400))
    mov = np.log1p(abs(ws - ls)) * 2.2 / ((rw - rl + hadj) * 0.001 + 2.2)
    delta = ELO_K * mov * (1 - exp_w)
    ratings[w] = rw + delta
    ratings[l] = rl - delta


def kaggle_tables(data: Path) -> tuple[pd.DataFrame, dict[int, float], dict[int, float]]:
    """Per-season team table, end-of-2026-regular-season Elo, and each season's ridge home edge."""
    compact = pd.read_csv(data / "MRegularSeasonCompactResults.csv").sort_values(["Season", "DayNum"], kind="stable")
    ratings: dict[int, float] = {}
    elo_rows = []
    for season, s in compact.groupby("Season", sort=True):
        ratings = {t: 1500 + ELO_CARRY * (r - 1500) for t, r in ratings.items()}
        for w, l, ws, ls, loc in zip(s.WTeamID.values, s.LTeamID.values, s.WScore.values, s.LScore.values, s.WLoc.values):
            elo_update(ratings, w, l, ws, ls, 1 if loc == "H" else (-1 if loc == "A" else 0))
        elo_rows += [{"Season": season, "TeamID": t, "elo": ratings[t]} for t in set(s.WTeamID) | set(s.LTeamID)]
    elo_end = {int(t): float(r) for t, r in ratings.items()}

    d = pd.read_csv(data / "MRegularSeasonDetailedResults.csv")
    loc = d.WLoc.map({"H": 1, "A": -1, "N": 0})
    rows, home_edges = [], {}
    for season, s in d.groupby("Season"):
        sl = loc.loc[s.index]
        one = pd.DataFrame({"T1": s.WTeamID.values, "T2": s.LTeamID.values, "margin": (s.WScore - s.LScore).values, "loc": sl.values})
        flip = one.T1 > one.T2  # features.py keeps T1 < T2; ridge is order-invariant, kept for parity
        one.loc[flip, ["T1", "T2"]] = one.loc[flip, ["T2", "T1"]].values
        one.loc[flip, ["margin", "loc"]] *= -1
        q, home_edges[int(season)] = ridge_quality(one)
        eff = []
        for p, o, pts, opp in (("W", "L", s.WScore, s.LScore), ("L", "W", s.LScore, s.WScore)):
            poss = ((s[p + "FGA"] - s[p + "OR"] + s[p + "TO"] + 0.475 * s[p + "FTA"])
                    + (s[o + "FGA"] - s[o + "OR"] + s[o + "TO"] + 0.475 * s[o + "FTA"])) / 2
            eff.append(pd.DataFrame({"TeamID": s[p + "TeamID"].values, "net": (100 * pts / poss - 100 * opp / poss).values}))
        net = pd.concat(eff).groupby("TeamID").net.mean().rename("net_eff").reset_index()
        rows.append(q.merge(net, on="TeamID").assign(Season=season))
    table = pd.concat(rows, ignore_index=True).merge(pd.DataFrame(elo_rows), on=["Season", "TeamID"], how="left")
    return table, elo_end, home_edges


def tourney_pairs(data: Path) -> pd.DataFrame:
    df = pd.read_csv(data / "MNCAATourneyCompactResults.csv")
    sd = pd.read_csv(data / "MNCAATourneySeeds.csv").set_index(["Season", "TeamID"]).Seed.str[:3]
    ws = sd.reindex(pd.MultiIndex.from_arrays([df.Season, df.WTeamID])).values
    ls = sd.reindex(pd.MultiIndex.from_arrays([df.Season, df.LTeamID])).values
    df = df[ws != ls]  # drop First Four
    a = pd.DataFrame({"Season": df.Season, "T1": df.WTeamID, "T2": df.LTeamID, "y": 1})
    b = pd.DataFrame({"Season": df.Season, "T1": df.LTeamID, "T2": df.WTeamID, "y": 0})
    return pd.concat([a, b], ignore_index=True)


def diffs(pairs: pd.DataFrame, table: pd.DataFrame) -> np.ndarray:
    t = table[["Season", "TeamID"] + FEATURES]
    out = pairs.merge(t.rename(columns={"TeamID": "T1"}), on=["Season", "T1"], how="left")
    out = out.merge(t.rename(columns={"TeamID": "T2"}), on=["Season", "T2"], how="left", suffixes=("_1", "_2"))
    return np.column_stack([(out[f + "_1"] - out[f + "_2"]).fillna(0).values for f in FEATURES])


def fit(X: np.ndarray, y: np.ndarray) -> tuple[StandardScaler, LogisticRegression]:
    sc = StandardScaler().fit(X)
    return sc, LogisticRegression(C=C, fit_intercept=False, max_iter=2000).fit(sc.transform(X), y)


def symmetric(sc, m, X: np.ndarray) -> np.ndarray:
    """backtest.py averages both team orders: p = (p(A,B) + 1 - p(B,A)) / 2."""
    pf = m.predict_proba(sc.transform(X))[:, 1]
    pr = m.predict_proba(sc.transform(-X))[:, 1]
    return (pf + 1 - pr) / 2


# --- commands -----------------------------------------------------------------

def export(args) -> None:
    data = Path(args.kaggle_dir)
    table, elo_end, home_edges = kaggle_tables(data)
    pairs = tourney_pairs(data)
    train = pairs[(pairs.Season >= TRAIN_START) & (pairs.Season <= TRAIN_END)]
    sc, m = fit(diffs(train, table), train.y.values)

    backtest = {}
    for year in range(2015, TRAIN_END + 1):
        if year == 2020:
            continue
        tr, te = pairs[(pairs.Season >= TRAIN_START) & (pairs.Season < year)], pairs[pairs.Season == year]
        te = te[te.T1 < te.T2]
        s2, m2 = fit(diffs(tr, table), tr.y.values)
        p = symmetric(s2, m2, diffs(te, table))
        backtest[year] = {"games": int(len(te)), "brier": round(float(np.mean((p - te.y.values) ** 2)), 5),
                          "accuracy": round(float(np.mean((p > 0.5) == (te.y.values == 1))), 4)}
    print("walk-forward (train on earlier tournaments only):")
    for y, r in backtest.items():
        print(f"  {y}: {r['games']} games  brier={r['brier']:.4f}  accuracy={r['accuracy']:.3f}")

    model = {
        "id": "kaggle-logit-live-v1",
        "source": "kaggle_tuning `final` logit member (origin/kaggle-score-tuning @ 07db721), seed and Massey features removed",
        "features": FEATURES,
        "scalerMean": [float(v) for v in sc.mean_],
        "scalerScale": [float(v) for v in sc.scale_],
        "coefficients": [float(v) for v in m.coef_[0]],
        "regularizationC": C,
        "trainingSeasons": [TRAIN_START, TRAIN_END],
        "trainingRows": int(len(train)),
        "quality": {"alpha": Q_ALPHA, "marginClip": Q_CLIP},
        "elo": {"k": ELO_K, "home": ELO_HOME, "carry": ELO_CARRY},
        # Early-season fits confound home court with scheduling (strong teams host weak ones; 6-8 pts
        # in the first weeks), so home games use the mean full-regular-season fit instead.
        "homeEdgePoints": float(np.mean(list(home_edges.values()))),
        "homeEdgeSeasons": [min(home_edges), max(home_edges)],
        "backtest": backtest,
    }
    (ROOT / "convex/data/liveMatchupModel.json").write_text(json.dumps(model, indent=2) + "\n")

    teams = pd.read_csv(data / "MTeams.csv")
    spell = pd.read_csv(data / "MTeamSpellings.csv", encoding="cp1252")
    active = teams[teams.LastD1Season >= 2026]
    prior = [{"kaggleTeamId": int(r.TeamID), "teamName": r.TeamName,
              "spellings": sorted(set(spell[spell.TeamID == r.TeamID].TeamNameSpelling.str.lower())),
              "elo": round(elo_end[int(r.TeamID)], 4)} for r in active.itertuples() if int(r.TeamID) in elo_end]
    (ROOT / "model/elo-prior-2026.json").write_text(json.dumps(
        {"season": 2026, "note": "End of 2026 regular season Elo from the Kaggle mirror (before carry-over).", "teams": prior},
        indent=1) + "\n")
    print(f"coefficients {dict(zip(FEATURES, model['coefficients']))}; {len(prior)} team priors written")


def parity(args) -> None:
    """Recompute live features in Python from the sync script's CBBD dump, compare, and score one matchup."""
    dump = json.loads(Path(args.dump).read_text())
    prior = {int(k): v for k, v in dump["eloPrior"].items()}
    games = pd.DataFrame(dump["games"])  # already sorted by start time then id
    one = pd.DataFrame({"T1": games.homeTeamId, "T2": games.awayTeamId,
                        "margin": games.homePoints - games.awayPoints, "loc": np.where(games.neutralSite, 0, 1)})
    # Training used sklearn's default (iterative sparse_cg, tol 1e-4); the site solves exactly.
    q, home = ridge_quality(one, solver=args.solver)
    ratings = {t: 1500 + ELO_CARRY * (r - 1500) for t, r in prior.items()}
    for g in games.itertuples():
        hw = g.homePoints > g.awayPoints
        w, l = (g.homeTeamId, g.awayTeamId) if hw else (g.awayTeamId, g.homeTeamId)
        ws, ls = max(g.homePoints, g.awayPoints), min(g.homePoints, g.awayPoints)
        elo_update(ratings, w, l, ws, ls, 0 if g.neutralSite else (1 if hw else -1))
    box = pd.DataFrame(dump["box"])
    poss = lambda p: box[p + "FGA"] - box[p + "OR"] + box[p + "TO"] + 0.475 * box[p + "FTA"]
    box["net"] = 100 * (box.pts - box.oppPts) / ((poss("") + poss("opp")) / 2)
    net = box.groupby("teamId").net.mean()
    py = q.set_index("TeamID").assign(elo=lambda f: [ratings.get(t, 1500.0) for t in f.index], net_eff=net)

    ts = {int(t["teamId"]): t for t in dump["teams"]}
    worst = {f: max(abs(py.loc[t, f] - ts[t][f.replace("net_eff", "netEff")]) for t in py.index if t in ts) for f in FEATURES}
    print(f"python home edge {home:.6f} vs ts {dump.get('homeEdge')}; max |python - ts| per feature: {worst}")

    model = json.loads((ROOT / "convex/data/liveMatchupModel.json").read_text())
    sc = StandardScaler().fit(np.zeros((2, 3)))
    sc.mean_, sc.scale_ = np.array(model["scalerMean"]), np.array(model["scalerScale"])
    m = LogisticRegression(fit_intercept=False)
    m.classes_, m.coef_, m.intercept_ = np.array([0, 1]), np.array([model["coefficients"]]), np.array([0.0])
    name = {int(t["teamId"]): t["teamName"] for t in ts.values()}
    a = next(t for t, n in name.items() if n == args.team_a)
    b = next(t for t, n in name.items() if n == args.team_b)
    x = np.array([[py.loc[a, f] - py.loc[b, f] for f in FEATURES]])
    print(f"python P({args.team_a} beats {args.team_b}, neutral) = {symmetric(sc, m, x)[0]:.6f}")
    x[0, 0] += model["homeEdgePoints"]
    x[0, 1] += model["elo"]["home"]
    print(f"python P({args.team_a} beats {args.team_b}, {args.team_a} at home) = {symmetric(sc, m, x)[0]:.6f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(required=True)
    e = sub.add_parser("export")
    e.add_argument("--kaggle-dir", required=True)
    e.set_defaults(run=export)
    p = sub.add_parser("parity")
    p.add_argument("--dump", required=True)
    p.add_argument("--team-a", required=True)
    p.add_argument("--team-b", required=True)
    p.add_argument("--solver", default="cholesky", help="ridge solver; 'auto' reproduces training exactly")
    p.set_defaults(run=parity)
    a = ap.parse_args()
    a.run(a)
