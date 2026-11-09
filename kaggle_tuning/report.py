"""Score each step from the original pipeline to the final config; write JSON for charts.

    uv run python -m kaggle_tuning.report --og-dir <dir with {year}_{M,W}/kaggle_mania_{year}/kaggle_submission_{year}.csv>
"""
import argparse, glob, json
from pathlib import Path
import numpy as np, pandas as pd
from . import backtest as B, market as MK
from .features import DATA

BOXF = ["margin","off_eff","def_eff","efg","opp_efg","tov","opp_tov","orb","drb","ftr","opp_ftr","win","poss","late_margin"]
M0 = ["seed","quality","elo","mstr_med","mstr_POM","net_eff"]
W0 = ["seed","quality","elo","net_eff"]
Q = {"q_clip_M": 60, "q_clip_W": 60}
L = {"M": M0, "W": W0, "model": "logit", "C": 0.1, **Q}
X = {"M": M0 + BOXF, "W": W0 + BOXF, "model": "xgb_spline", **Q, "xgb": {"max_depth": 2}}
ENS = {"models": [(L, 1), (X, 1)], **Q}
FINAL = {**ENS, "market_w": {"M": 0.85, "W": 0.5}, "mkt_sharpen": 1.2}

STEPS = [
    ("Seed-only baseline", {"M": ["seed"], "W": ["seed"], "model": "logit"}),
    ("Original pipeline (week-5-51)", "OG"),
    ("New ratings + logistic", L),
    ("+ XGBoost ensemble", ENS),
    ("+ 50/50 market blend", {**ENS, "market_w": {"M": 0.5, "W": 0.5}}),
    ("+ per-gender weights, de-clip market (final)", FINAL),
]


def og_preds(og_dir, g, year):
    sub = pd.read_csv(f"{og_dir}/{year}_{g}/kaggle_mania_{year}/kaggle_submission_{year}.csv")
    sub = pd.DataFrame({"T1": sub.ID.str[5:9].astype(int), "T2": sub.ID.str[10:14].astype(int), "p": sub.Pred})
    return B.test_pairs(g, year).merge(sub, on=["T1", "T2"], how="left").p.values


def lb(year):
    f = glob.glob(str(DATA.parent / "lb" / str(year) / "*.csv"))[0]
    return np.sort(pd.read_csv(f).Score.values)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--og-dir", required=True); ap.add_argument("--out", default="kaggle_tuning/report.json")
    a = ap.parse_args()
    out = {"steps": [], "lb": {}}
    for y in B.YEARS:
        s = lb(y); out["lb"][y] = {"n": int(len(s)), "top100": float(s[99]), "top10": float(s[9]), "first": float(s[0]), "median": float(np.median(s))}
    for name, cfg in STEPS:
        row = {"name": name}
        for y in B.YEARS:
            per = {}
            allp, ally = [], []
            for g in "MW":
                p = og_preds(a.og_dir, g, y) if cfg == "OG" else B.fit_predict(cfg, g, y, B.F.team_table(g, cfg.get("q_alpha", 1.0), cfg.get("q_recency", 0.0), cfg.get("q_clip_" + g, 30.0)))
                t = B.test_pairs(g, y).y.values
                per[g] = float(np.mean((p - t) ** 2)); allp.append(p); ally.append(t)
            br = float(np.mean((np.concatenate(allp) - np.concatenate(ally)) ** 2))
            rank = int((lb(y) < br).sum() + 1)
            row[y] = {"brier": br, "M": per["M"], "W": per["W"], "rank": rank}
        print(f"{name:48s} " + "  ".join(f"{y}: {row[y]['brier']:.5f} (#{row[y]['rank']})" for y in B.YEARS))
        out["steps"].append(row)
    # market alone, for reference
    row = {"name": "Market odds alone (reference)"}
    for y in B.YEARS:
        per, ap_, ay = {}, [], []
        for g in "MW":
            t = B.test_pairs(g, y).merge(MK.market(y), on=["T1", "T2"])
            per[g] = float(np.mean((t.mkt - t.y) ** 2)); ap_.append(t.mkt.values); ay.append(t.y.values)
        br = float(np.mean((np.concatenate(ap_) - np.concatenate(ay)) ** 2))
        row[y] = {"brier": br, **per, "rank": int((lb(y) < br).sum() + 1)}
    print(f"{row['name']:48s} " + "  ".join(f"{y}: {row[y]['brier']:.5f} (#{row[y]['rank']})" for y in B.YEARS))
    out["market"] = row
    Path(a.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
