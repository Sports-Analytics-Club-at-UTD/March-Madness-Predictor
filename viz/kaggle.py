"""Plots for the Kaggle walk-forward Brier backtest (kaggle_tuning).

    MANIA_DATA=/abs/path/mirror26 uv run --extra training python -m viz.kaggle

Each (season, gender) is fitted once on earlier seasons only, exactly as kaggle_tuning.backtest does,
and scored on every pair of tournament teams. The played games are a subset of those pairs, so the same
cache feeds both these plots and viz.bracket. Cached under <out>/cache/; pass --refresh after code changes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import runpy
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from .shaping import brier_by_season, reliability_table

DEFAULT_CONFIG = "kaggle_tuning/configs/final.py:final"
DEFAULT_OUT = Path("model_outputs/plots")
LB_TOP20 = {2025: 0.10921, 2026: 0.12016}  # kaggle_tuning/README.md; top-100 lives in backtest.LB_TOP100
SERIES = {"full": "Pipeline (submitted)", "model_only": "Model only", "market": "Market only", "seed": "Seed-only logit"}


def load_config(spec: str) -> tuple[str, dict]:
    path, _, name = spec.partition(":")
    confs = runpy.run_path(path)["CONFIGS"]
    name = name or next(iter(confs))
    return name, confs[name]


def field_pairs(g: str, year: int) -> pd.DataFrame:
    from kaggle_tuning import features as F

    s = F.read(f"{g}NCAATourneySeeds")
    teams = sorted(s.loc[s.Season == year, "TeamID"].astype(int))
    t1, t2 = zip(*combinations(teams, 2))
    return pd.DataFrame({"Season": year, "T1": t1, "T2": t2, "y": 0})


def _predict(cfg: dict, g: str, year: int, pairs: pd.DataFrame) -> np.ndarray:
    """backtest.fit_predict, but scored on `pairs` instead of the played games. fit_predict and _post read
    the test set through backtest.test_pairs, so swap that for the duration of the call."""
    from kaggle_tuning import backtest as B
    from kaggle_tuning import features as F

    played = B.test_pairs
    B.test_pairs = lambda *_: pairs
    try:
        table = F.team_table(g, cfg.get("q_alpha", 1.0), cfg.get("q_recency", 0.0), cfg.get("q_clip_" + g, 30.0))
        p = np.asarray(B.fit_predict(cfg, g, year, table), float)
    finally:
        B.test_pairs = played
    if p.shape != (len(pairs),) or not np.isfinite(p).all():
        raise ValueError(f"{g} {year}: config does not score arbitrary pairs through backtest.test_pairs "
                         f"(got {p.shape}, expected {len(pairs)} finite values).")
    return p


def _task(cfg: dict, g: str, year: int, variant: str) -> pd.DataFrame:
    pairs = field_pairs(g, year)
    return pairs.drop(columns="y").assign(g=g, variant=variant, p=_predict(cfg, g, year, pairs))


def field_predictions(spec: str = DEFAULT_CONFIG, seasons=None, out: Path = DEFAULT_OUT,
                      refresh: bool = False, jobs: int | None = None) -> pd.DataFrame:
    """Season, g, T1, T2 (T1 < T2), variant, p for every pair of tournament teams; cached."""
    from joblib import Parallel, delayed

    from kaggle_tuning import backtest as B
    from kaggle_tuning import features as F
    from kaggle_tuning import market as MK

    name, cfg = load_config(spec)
    seasons = sorted(seasons or B.HIST + B.YEARS)
    key = hashlib.sha1(json.dumps([cfg, seasons, str(F.DATA)], sort_keys=True, default=str).encode()).hexdigest()[:10]
    cache = out / "cache" / f"field_{name}_{key}.csv"
    if cache.exists() and not refresh:
        return pd.read_csv(cache)
    cfg = {**cfg, "xgb": {"n_jobs": 1, **cfg.get("xgb", {})}}
    tasks = []
    for year in seasons:
        for g in "MW":
            tasks.append((cfg, g, year, "full"))
            tasks.append((B.CONFIGS["seed_only"], g, year, "seed"))
            if cfg.get("market_w", {}).get(g) and year in MK.PATHS:
                tasks.append(({**cfg, "market_w": {}}, g, year, "model_only"))
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    jobs = jobs or min(8, os.cpu_count() or 1)
    print(f"fitting {len(tasks)} walk-forward models for {name} ({jobs} jobs)...")
    preds = pd.concat(Parallel(n_jobs=jobs)(delayed(_task)(*t) for t in tasks), ignore_index=True)
    cache.parent.mkdir(parents=True, exist_ok=True)
    preds.to_csv(cache, index=False)
    return preds


def scored_games(preds: pd.DataFrame) -> pd.DataFrame:
    """One row per played (non-First-Four) game: Season, g, T1, T2, y and one column per series."""
    from kaggle_tuning import backtest as B
    from kaggle_tuning import market as MK

    wide = preds.pivot_table(index=["Season", "g", "T1", "T2"], columns="variant", values="p").reset_index()
    rows = []
    for (year, g), _ in wide.groupby(["Season", "g"]):
        played = B.test_pairs(g, int(year)).assign(g=g)
        if year in MK.PATHS:
            played = played.merge(MK.market(int(year)).rename(columns={"mkt": "market"}), on=["T1", "T2"], how="left")
        rows.append(played)
    games = pd.concat(rows, ignore_index=True).merge(wide, on=["Season", "g", "T1", "T2"], how="left")
    if games["full"].isna().any():
        raise ValueError("Some played games are missing from the field predictions.")
    for col in SERIES:
        if col not in games:
            games[col] = np.nan
    return games


def plot_brier_by_season(table: pd.DataFrame, path: Path, name: str) -> Path:
    import matplotlib.pyplot as plt

    from kaggle_tuning.backtest import LB_TOP100

    from . import style as S

    seasons = np.arange(table.Season.min(), table.Season.max() + 1)  # missing 2020 breaks the line
    look = {"full": (S.BLUE, "-", "o"), "seed": (S.GRAY, "--", "o"), "model_only": (S.AQUA, "", "s"), "market": (S.ORANGE, "", "D")}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharey=True)
    for ax, scope in zip(axes, ["Combined", "Men", "Women"]):
        d = table[table.scope == scope]
        for series, (color, ls, marker) in look.items():
            s = d[d.series == series].set_index("Season").brier.reindex(seasons)
            if s.notna().any():
                ax.plot(seasons, s, color=color, ls=ls or "none", marker=marker, ms=7 if ls else 8,
                        label=SERIES[series], zorder=3 if series == "full" else 2)
        if scope == "Combined":
            for year in LB_TOP100:
                for cut, ls, lab in ((LB_TOP100, "-", "LB top-100 cutoff"), (LB_TOP20, ":", "LB top-20 cutoff")):
                    ax.hlines(cut[year], year - 0.4, year + 0.4, color=S.INK, ls=ls, lw=1.4,
                              label=lab if year == min(LB_TOP100) else None)
        ax.set_title({"Combined": "Men + women (Kaggle metric)", "Men": "Men", "Women": "Women"}[scope], loc="left")
        ax.set_xticks(seasons[seasons != 2020])
        ax.set_xticklabels([f"'{y % 100:02d}" for y in seasons if y != 2020])
        ax.set_xlabel("Tournament season (no 2020 tournament)")
    axes[0].set_ylabel("Brier score (lower is better)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=6, bbox_to_anchor=(0.5, -0.06))
    fig.suptitle(f"Walk-forward Brier by season: {name}  (2025–26 blend in betting-market odds; earlier seasons are model only)",
                 x=0.01, ha="left", fontsize=13, fontweight="bold")
    return S.save(fig, path)


def plot_calibration(games: pd.DataFrame, path: Path, name: str, n_bins: int = 10) -> Path:
    import matplotlib.pyplot as plt

    from . import style as S

    fig, axes = plt.subplots(2, 2, figsize=(12, 7.2), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    years = f"{games.Season.min()}–{games.Season.max()}"
    for col, (g, label) in enumerate((("M", "Men"), ("W", "Women"))):
        d = games[games.g == g]
        top, bottom = axes[0, col], axes[1, col]
        top.plot([0, 1], [0, 1], color=S.GRAY, lw=1, ls=":", label="Perfect calibration")
        for series, color in (("full", S.BLUE), ("seed", S.GRAY)):
            r = reliability_table(d[series], d.y, n_bins)
            brier = np.mean((d[series] - d.y) ** 2)
            top.plot(r.mean_p, r.frac, color=color, marker="o", ms=7, ls="-" if series == "full" else "--",
                     label=f"{SERIES[series]} (Brier {brier:.4f})")
            if series == "full":
                bottom.bar((r.lo + r.hi) / 2, r.n, width=(r.hi - r.lo) * 0.9, color=S.BLUE)
        top.set_title(f"{label}: {len(d)} games, {years}", loc="left")
        top.set_ylabel("Observed win rate")
        top.set_xlim(0, 1)
        top.set_ylim(0, 1)
        top.legend(loc="upper left")
        bottom.set_ylabel("Predictions")
        bottom.set_xlabel("Predicted win probability (each game counted from both sides)")
    fig.suptitle(f"Calibration of walk-forward predictions: {name}", x=0.01, ha="left", fontsize=13, fontweight="bold")
    return S.save(fig, path)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=DEFAULT_CONFIG, help="path/to/configs.py[:NAME] (default %(default)s)")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--refresh", action="store_true", help="refit instead of reading the prediction cache")
    ap.add_argument("--jobs", type=int)
    a = ap.parse_args(argv)
    name = load_config(a.config)[0]
    games = scored_games(field_predictions(a.config, out=a.out, refresh=a.refresh, jobs=a.jobs))
    table = brier_by_season(games, list(SERIES))
    out = a.out / "kaggle"
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "brier_by_season.csv", index=False)
    print(table[table.scope == "Combined"].pivot(index="Season", columns="series", values="brier").round(5).to_string())
    plot_brier_by_season(table, out / "brier_by_season.png", name)
    plot_calibration(games, out / "calibration.png", name)


if __name__ == "__main__":
    main()
