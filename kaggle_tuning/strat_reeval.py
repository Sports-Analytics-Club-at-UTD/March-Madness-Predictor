"""Issue #67: alternative combiners and GBM probability mappings, judged by the existing walk-forward.

Model types (dispatched from backtest.fit_predict):

- ``strat_hedge``: Hedge / exponential weights (Freund & Schapire 1997) over fixed members, per the
  protocol pre-registered in PR #66 (``kaggle_tuning/research/RL_METHODS.md`` on ``t3/kaggle-eda-rl-59``).
  Member weights for season Y use only walk-forward member losses from seasons < Y.
- ``strat_stack``: logistic-regression stacking (no intercept) on member log odds, fitted on the
  pooled walk-forward member predictions of seasons < Y, optionally ridge-penalised toward the
  reference weights.
- ``xgb_normal`` / ``xgb_binary``: the existing ``xgb_spline`` member with the spline replaced by a
  one-parameter normal-CDF map of the margin, or with a logistic objective that predicts the
  probability directly.

Combiner members are the same (subconfig, weight) pairs ``backtest`` blends; weights are the prior
(Hedge) or the ridge target (stacking). Post-processing (market blend, sharpening) is unchanged.

    MANIA_DATA=... GRID_JOBS=5 uv run --extra training python -m kaggle_tuning.strat_reeval \
        kaggle_tuning/configs/strat_reeval.py
"""
from __future__ import annotations

import json
import os
import runpy
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.stats import norm

from . import features as F

MODELS = ("strat_hedge", "strat_stack", "xgb_normal", "xgb_binary")
HIST = [2015, 2016, 2017, 2018, 2019, 2021, 2022, 2023, 2024]
_MEMO: dict[str, np.ndarray] = {}


def _logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def history_seasons(year: int, warmup=()) -> list[int]:
    """Seasons whose walk-forward member predictions may inform weights for ``year`` (all < year)."""
    return [s for s in sorted(set(warmup) | set(HIST)) if s < min(year, 2025)]


def hedge_weights(prior, losses, eta: float) -> np.ndarray:
    """prior: (k,), losses: (n_seasons, k) season-mean Brier per member. Returns normalised weights."""
    prior = np.asarray(prior, float) / np.sum(prior)
    cum = np.asarray(losses, float).reshape(-1, len(prior)).sum(axis=0)
    w = prior * np.exp(-eta * (cum - cum.min()))
    return w / w.sum()


def stack_weights(Z, y, ref, lam: float) -> np.ndarray:
    """Minimise summed log loss of sigmoid(Z @ b) + lam * ||b - ref||^2 (no intercept: symmetric)."""
    ref = np.asarray(ref, float) / np.sum(ref)

    def f(b):
        s = Z @ b
        ll = np.sum(np.logaddexp(0, s) - y * s)
        return ll + lam * np.sum((b - ref) ** 2)

    def grad(b):
        s = Z @ b
        return Z.T @ (1 / (1 + np.exp(-s)) - y) + 2 * lam * (b - ref)

    return minimize(f, ref, jac=grad, method="L-BFGS-B").x


def _member_cfg(cfg: dict, sub: dict) -> dict:
    base = {k: v for k, v in cfg.items() if k not in ("members", "model", "combiner")}
    return {**base, **sub, "market_w": {}, "sharpen": {}, "women_model_temperature": None,
            "clip_p": (0, 1), "models_inner": True}


def member_pred(cfg: dict, sub: dict, g: str, year: int, table) -> np.ndarray:
    from .backtest import fit_predict
    mc = _member_cfg(cfg, sub)
    key = json.dumps([mc, g, year], sort_keys=True, default=str)
    if key not in _MEMO:
        _MEMO[key] = fit_predict(mc, g, year, table)
    return _MEMO[key]


def combine(cfg: dict, g: str, year: int, table) -> np.ndarray:
    from .backtest import test_pairs
    members = cfg["members"][g]
    subs, prior = [m for m, _ in members], np.array([w for _, w in members], float)
    opts = cfg.get("combiner", {})
    hist = history_seasons(year, opts.get("warmup", ()))
    target = np.array([_logit(member_pred(cfg, s, g, year, table)) for s in subs]).T
    if cfg["model"] == "strat_hedge":
        losses = [[np.mean((member_pred(cfg, s, g, h, table) - test_pairs(g, h).y.values) ** 2) for s in subs]
                  for h in hist]
        start = np.ones_like(prior) if opts.get("uniform") else prior
        w = hedge_weights(start, losses, opts.get("eta", 8.0)) if losses else start / start.sum()
        return 1 / (1 + np.exp(-target @ w))
    if not hist:
        return 1 / (1 + np.exp(-target @ (prior / prior.sum())))
    Z = np.vstack([np.array([_logit(member_pred(cfg, s, g, h, table)) for s in subs]).T for h in hist])
    y = np.concatenate([test_pairs(g, h).y.values for h in hist]).astype(float)
    # Both orientations: logits are odd, so this keeps the fit exactly symmetric.
    b = stack_weights(np.vstack([Z, -Z]), np.r_[y, 1 - y], prior, opts.get("lam", 0.0))
    return 1 / (1 + np.exp(-target @ b))


def xgb_member(cfg: dict, g: str, year: int, table) -> np.ndarray:
    import xgboost as xgb
    from .backtest import test_pairs

    feats = cfg[g]
    tr = F.tourney_games(g)
    tr = tr[(tr.Season < year) & (tr.Season >= cfg.get("start_" + g, 2003 if g == "M" else 2010))]
    tr = F.matchup_frame(tr, table, feats)
    te = test_pairs(g, year)[["Season", "T1", "T2"]]
    cols = [f"diff_{f}" for f in feats]
    Xtr = tr[cols].fillna(0).values
    Xf = F.matchup_frame(te, table, feats)[cols].fillna(0).values
    Xr = F.matchup_frame(te.rename(columns={"T1": "T2", "T2": "T1"}), table, feats)[cols].fillna(0).values
    y = (tr.margin > 0).astype(float).values
    params = dict(learning_rate=0.02, max_depth=3, subsample=0.6, colsample_bytree=0.8, min_child_weight=40,
                  reg_lambda=1.0, n_estimators=cfg.get("n_est", 600))
    params.update(cfg.get("xgb", {}))
    if cfg["model"] == "xgb_binary":
        # Hessians are p(1-p) <= .25, so the regression min_child_weight is scaled by .25.
        params["min_child_weight"] = params["min_child_weight"] * 0.25
        mdl = xgb.XGBClassifier(objective="binary:logistic", **params).fit(Xtr, y)
        pf, pr = mdl.predict_proba(Xf)[:, 1], mdl.predict_proba(Xr)[:, 1]
    else:
        params["objective"] = "reg:squarederror"
        oof = np.zeros(len(tr))
        for s in tr.Season.unique():
            mask = tr.Season.values == s
            oof[mask] = xgb.XGBRegressor(**params).fit(Xtr[~mask], tr.margin.values[~mask]).predict(Xtr[mask])
        sigma = fit_sigma(oof, y)
        mdl = xgb.XGBRegressor(**params).fit(Xtr, tr.margin.values)
        pf, pr = norm.cdf(mdl.predict(Xf) / sigma), norm.cdf(mdl.predict(Xr) / sigma)
    return np.clip((pf + 1 - pr) / 2, 0.01, 0.99)


def fit_sigma(margin, y) -> float:
    """Max-likelihood sigma for P(win) = Phi(margin / sigma), bounded like penalized.py (3..30)."""
    def nll(s):
        p = np.clip(norm.cdf(margin / s), 1e-6, 1 - 1e-6)
        return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
    return float(minimize_scalar(nll, bounds=(3, 30), method="bounded").x)


def predict(cfg: dict, g: str, year: int, table) -> np.ndarray:
    if cfg["model"] in ("strat_hedge", "strat_stack"):
        return combine(cfg, g, year, table)
    return xgb_member(cfg, g, year, table)


def _one(name, cfg):
    """Same aggregates as backtest.run, plus per-season scores and market-free 2025/2026."""
    os.environ["OMP_NUM_THREADS"] = "1"
    from .backtest import HIST as H, VERIFY, YEARS, run
    t = time.time()
    per = run(cfg, H + YEARS, verbose=False, hist=False)
    res = {y: per[y] for y in YEARS}
    res["hist"] = float(np.mean([per[y]["brier"] for y in H]))
    res["histM"] = float(np.mean([per[y]["M"] for y in H]))
    res["histW"] = float(np.mean([per[y]["W"] for y in H]))
    res["verify"] = float(np.mean([per[y]["brier"] for y in VERIFY]))
    res["by_season"] = {y: per[y] for y in H}
    mo = run({**cfg, "market_w": {}}, YEARS, verbose=False, hist=False)
    res["model_only"] = {y: mo[y] for y in YEARS}
    return name, res, time.time() - t


def main():
    from joblib import Parallel, delayed
    path = Path(sys.argv[1])
    confs = runpy.run_path(str(path))["CONFIGS"]
    out = Parallel(n_jobs=min(int(os.environ.get("GRID_JOBS", 5)), len(confs)))(delayed(_one)(n, c) for n, c in confs.items())
    res = {n: r for n, r, _ in out}
    ref = res.get("ref")
    cols = ["hist", "histM", "histW", "verify"]
    print(f"{'config':22s} " + " ".join(f"{c:>8s}" for c in cols + ["2025", "M25", "W25", "2026", "M26", "W26", "mean", "mo25", "mo26"])
          + "   d_hist  d_v2124  d_mean guard seasons<ref")
    for n, r, dt in sorted(out, key=lambda x: x[1][2025]["brier"] + x[1][2026]["brier"]):
        vals = [r[c] for c in cols] + [r[y][k] for y in (2025, 2026) for k in ("brier", "M", "W")]
        mean = (r[2025]["brier"] + r[2026]["brier"]) / 2
        vals += [mean, r["model_only"][2025]["brier"], r["model_only"][2026]["brier"]]
        line = f"{n:22s} " + " ".join(f"{v:8.5f}" for v in vals)
        if ref:
            dh, dv = r["hist"] - ref["hist"], r["verify"] - ref["verify"]
            dm = mean - (ref[2025]["brier"] + ref[2026]["brier"]) / 2
            wins = sum(r["by_season"][y]["brier"] < ref["by_season"][y]["brier"] for y in r["by_season"])
            line += f"  {dh:+.5f} {dv:+.5f} {dm:+.5f} {'ok' if max(dh, dv) <= 0.0005 else 'FAIL'}  {wins}/{len(r['by_season'])}"
        print(line + f"  {dt:.0f}s")
    path.with_suffix(".results.json").write_text(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    main()
