"""Walk-forward backtest: for each target year, train only on earlier seasons, score Brier
on that year's men's + women's tournament games (Kaggle 2025/2026 metric, First Four excluded).

    uv run python -m kaggle_tuning.backtest --config base
"""

from __future__ import annotations

import argparse
import json
import warnings

import numpy as np
import pandas as pd
from scipy.interpolate import UnivariateSpline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from . import features as F
from . import market as MK

warnings.filterwarnings("ignore")

LB_TOP100 = {2025: 0.11078, 2026: 0.12430}  # 2024 used a bracket metric, so it is not scored here
YEARS = [2025, 2026]
HIST = [2015, 2016, 2017, 2018, 2019, 2021, 2022, 2023, 2024]  # overfitting guard: same walk-forward, older tourneys
VERIFY = [2021, 2022, 2023, 2024]  # re-checked after every ~50-rank gain

CONFIGS: dict[str, dict] = {
    "seed_only": {"M": ["seed"], "W": ["seed"], "model": "logit"},
    "base": {
        "M": ["seed", "quality", "elo", "mstr_med", "mstr_POM", "net_eff"],
        "W": ["seed", "quality", "elo", "net_eff"],
        "model": "logit",
    },
}


def test_pairs(g: str, year: int) -> pd.DataFrame:
    if year == 2026:
        s = F.solution_2026()
        p = pd.DataFrame({"Season": 2026, "T1": s.ID.str[5:9].astype(int), "T2": s.ID.str[10:14].astype(int), "y": s.y})
        lo = 3000 if g == "W" else 1000
        return p[(p.T1 >= lo) & (p.T1 < lo + 1000)].reset_index(drop=True)
    t = F.tourney_games(g)
    t = t[(t.Season == year) & (t.T1 < t.T2)]
    return pd.DataFrame({"Season": year, "T1": t.T1.values, "T2": t.T2.values, "y": (t.margin > 0).astype(int).values})


def _logit(x):
    x = np.clip(x, 1e-4, 1 - 1e-4)
    return np.log(x / (1 - x))


def fit_predict(cfg: dict, g: str, year: int, table: pd.DataFrame) -> np.ndarray:
    """cfg may list sub-models in cfg["models"] = [(subcfg, weight), ...]; blended in log-odds."""
    if g == "M" and "men_models" in cfg:
        cfg = {**cfg, "models": cfg["men_models"]}
    cfg = {k: v for k, v in cfg.items() if k != "men_models"}
    if g in cfg.get("by_gender", {}):
        override = cfg["by_gender"][g]
        cfg = {**cfg, **override}
        cfg.pop("by_gender")
        if "model" in override:
            cfg.pop("models", None)
    if g == "W" and any(k in cfg for k in ("women_q_alpha", "women_q_recency", "women_q_clip", "women_elo_k")):
        table = F.team_table(g, cfg.get("women_q_alpha", 1.0), cfg.get("women_q_recency", 0.0),
                             cfg.get("women_q_clip", 60), cfg.get("women_elo_k", 20.0))
    if g == "W" and cfg.get("model") == "women_consensus":
        from .research.women_allyears import predict
        return _post(cfg, g, year, predict(cfg, year, table))
    if g == "W" and (cfg.get("model") == "women_original" or cfg.get("women_calibration")):
        from .women import predict
        return _post(cfg, g, year, predict(cfg, year, table))
    if "models" in cfg:
        base = {k: v for k, v in cfg.items() if k != "models"}
        parts = [(fit_predict({**base, **sub, "market_w": {}, "sharpen": {}, "women_model_temperature": None, "clip_p": (0, 1), "models_inner": True}, g, year, table), w)
                 for sub, w in cfg["models"]]
        tot = sum(w for _, w in parts)
        p = 1 / (1 + np.exp(-sum(w * _logit(pp) for pp, w in parts) / tot))
        return _post(cfg, g, year, p)
    if cfg.get("model") == "vilnius_ensemble":
        from .vilnius_experiment import ensemble
        p = ensemble(cfg.get("variant", "V"), g, year)
        return p if cfg.get("models_inner") else _post(cfg, g, year, p)
    if cfg.get("model") == "vilnius":
        from .vilnius import predict
        p = predict(cfg, g, year)
        return p if cfg.get("models_inner") else _post(cfg, g, year, p)
    if g == "M" and cfg.get("model") == "mens_allyears":
        from .mens_allyears import predict, predict_margin
        opts = cfg.get("mens_allyears", {})
        p = predict_margin(opts, year) if opts.get("margin") else predict(cfg, year)
        return p if cfg.get("models_inner") else _post(cfg, g, year, p)
    feats = cfg[g]
    if g == "M" and cfg.get("mens_features"):
        from .mens_features import extra_table
        table = table.merge(extra_table(), on=["Season", "TeamID"], how="left")
    tr = F.tourney_games(g)
    start = 2003 if g == "M" else 2010
    tr = tr[(tr.Season < year) & (tr.Season >= cfg.get("start_" + g, start))]
    tr = F.matchup_frame(tr, table, feats)
    te = test_pairs(g, year)
    te_f = F.matchup_frame(te[["Season", "T1", "T2"]], table, feats)
    te_r = F.matchup_frame(te[["Season", "T1", "T2"]].rename(columns={"T1": "T2", "T2": "T1"}), table, feats)

    cols = [f"diff_{f}" for f in feats]
    model = cfg["model"]
    Xtr = tr[cols].fillna(0).values
    if model == "logit":
        y = (tr.margin > 0).astype(int).values
        sc = StandardScaler().fit(Xtr)
        C = cfg.get("C_" + g, cfg.get("C", 1.0))
        m = LogisticRegression(C=C, fit_intercept=False, max_iter=2000).fit(sc.transform(Xtr), y)
        pf = m.predict_proba(sc.transform(te_f[cols].fillna(0).values))[:, 1]
        pr = m.predict_proba(sc.transform(te_r[cols].fillna(0).values))[:, 1]
    elif model == "xgb_spline":
        import xgboost as xgb

        params = dict(objective="reg:squarederror", learning_rate=0.02, max_depth=3, subsample=0.6,
                      colsample_bytree=0.8, min_child_weight=40, reg_lambda=1.0, n_estimators=cfg.get("n_est", 600))
        params.update(cfg.get("xgb", {}))
        # out-of-fold margins by season for spline calibration
        seasons = tr.Season.unique()
        oof = np.zeros(len(tr))
        for s in seasons:
            mask = tr.Season.values == s
            mdl = xgb.XGBRegressor(**params).fit(Xtr[~mask], tr.margin.values[~mask])
            oof[mask] = mdl.predict(Xtr[mask])
        y = (tr.margin > 0).astype(float).values
        order = np.argsort(oof)
        clip = cfg.get("clip", 25)
        spl = UnivariateSpline(np.clip(oof[order], -clip, clip) + np.arange(len(order)) * 1e-7, y[order], k=5)
        mdl = xgb.XGBRegressor(**params).fit(Xtr, tr.margin.values)
        pf = np.clip(spl(np.clip(mdl.predict(te_f[cols].fillna(0).values), -clip, clip)), 0.01, 0.99)
        pr = np.clip(spl(np.clip(mdl.predict(te_r[cols].fillna(0).values), -clip, clip)), 0.01, 0.99)
    else:
        raise ValueError(model)
    p = (pf + 1 - pr) / 2
    if cfg.get("models_inner"):
        return p
    return _post(cfg, g, year, p)


def _post(cfg, g, year, p):
    # Optional calibration of the model component before blending market odds.
    model_a = cfg.get("model_sharpen", {}).get(g, 1.0)
    if model_a != 1.0:
        p = 1 / (1 + np.exp(-model_a * _logit(p)))
    w = cfg.get("market_w", {}).get(g, 0.0)
    if w and year in MK.PATHS:
        mk = test_pairs(g, year)[["T1", "T2"]].merge(MK.market(year), on=["T1", "T2"], how="left").mkt.values
        lg = lambda x: np.log(np.clip(x, 1e-4, 1 - 1e-4) / np.clip(1 - x, 1e-4, 1))
        # market file is clipped to ~[0.025, 0.975]; mkt_sharpen undoes some of that
        p = 1 / (1 + np.exp(-((1 - w) * lg(p) + w * cfg.get("mkt_sharpen", 1.0) * lg(mk))))
    a = cfg.get("sharpen", {}).get(g, 1.0)
    if g == "W" and cfg.get("women_model_temperature") and not (w and year in MK.PATHS) and year != 2024:
        from .research.women_temperature import temperature
        a = temperature(cfg, year)
    if a != 1.0:
        lp = np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, 1))
        p = 1 / (1 + np.exp(-a * lp))
    lo, hi = cfg.get("clip_p", (0.0, 1.0))
    return np.clip(p, lo, hi)


def run(cfg: dict, years=YEARS, verbose=True, hist=True) -> dict:
    res = {}
    if hist:
        h = run(cfg, HIST, verbose=False, hist=False)
        res["hist"] = float(np.mean([h[y]["brier"] for y in HIST]))
        res["histM"] = float(np.mean([h[y]["M"] for y in HIST]))
        res["histW"] = float(np.mean([h[y]["W"] for y in HIST]))
        res["verify"] = float(np.mean([h[y]["brier"] for y in VERIFY]))
        res["verify_by_year"] = {y: h[y]["brier"] for y in VERIFY}
        if verbose:
            print(f"hist 2015-23 mean: {res['hist']:.5f}  " + " ".join(f"{y % 100}:{h[y]['brier']:.3f}" for y in HIST))
    for year in years:
        ys, ps, per = [], [], {}
        for g in ["M", "W"]:
            table = F.team_table(g, cfg.get("q_alpha", 1.0), cfg.get("q_recency", 0.0), cfg.get("q_clip_" + g, 30.0))
            p = fit_predict(cfg, g, year, table)
            y = test_pairs(g, year).y.values
            per[g] = float(np.mean((p - y) ** 2))
            ys.append(y)
            ps.append(p)
        y, p = np.concatenate(ys), np.concatenate(ps)
        res[year] = {"brier": float(np.mean((p - y) ** 2)), **per}
        if verbose:
            tgt = LB_TOP100.get(year)
            flag = "" if tgt is None else ("  TOP100" if res[year]["brier"] <= tgt else f"  (need {tgt})")
            print(f"{year}: brier={res[year]['brier']:.5f}  M={per['M']:.5f}  W={per['W']:.5f}{flag}")
    if verbose and len(years) > 1:
        print(f"mean 25/26: {np.mean([res[y]['brier'] for y in years]):.5f}")
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="base")
    ap.add_argument("--json", help="inline JSON config overriding --config")
    a = ap.parse_args()
    cfg = json.loads(a.json) if a.json else CONFIGS[a.config]
    run(cfg)
