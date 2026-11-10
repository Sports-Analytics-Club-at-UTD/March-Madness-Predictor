"""Blend model probabilities with pre-deadline external forecasts for years that have no betting-odds file (2016-2024).

Sources (log-odds averaged into one consensus per pair; missing pairs fall back to the model):
  538      FiveThirtyEight Selection-Sunday ratings, M+W 2016-19, 2021-23 (research/ext.py)
  torvik   Torvik T-Rank as of Selection Sunday, M 2015-24 (ext_sources.py -> data/kaggle/ext_pairs)
  futures  vig-free championship futures -> team strengths, M 2021-24, W 2023 (ext_sources.py)
The blend weight w (model vs consensus, in log-odds) is chosen walk-forward: for target year Y it minimises the
target year's metric over earlier seasons that had every listed source family available at all (seasons >= 2016).
"""
from __future__ import annotations

import os
from functools import lru_cache

import numpy as np
import pandas as pd

from .features import DATA

EP = DATA.parent / "ext_pairs"
GRID = np.round(np.arange(0, 1.01, 0.1), 1)
SEASONS = [2016, 2017, 2018, 2019, 2021, 2022, 2023, 2024]


def _lg(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


@lru_cache(maxsize=None)
def source(name: str, year: int, g: str) -> pd.DataFrame | None:
    if name == "538":
        if not (2016 <= year <= 2023) or year == 2020:
            return None
        from .research.ext import fte
        return fte(year, g).rename(columns={"lo": "z"})[["T1", "T2", "z"]]
    f = EP / (f"futures_{g}_{g}{year}.csv" if name == "futures" else f"{name}_{g}{year}.csv")
    if not os.path.exists(f):
        return None
    d = pd.read_csv(f)
    return pd.DataFrame({"T1": d.T1, "T2": d.T2, "z": _lg(d.p.values)})


def consensus(pairs: pd.DataFrame, names: list[str], year: int, g: str) -> tuple[np.ndarray, bool]:
    zm = _lg(pairs.p.values)
    zs = []
    for n in names:
        s = source(n, year, g)
        if s is not None:
            z = pairs[["T1", "T2"]].merge(s, on=["T1", "T2"], how="left").z.values
            zs.append(np.where(np.isnan(z), zm, z))
    return (np.mean(zs, axis=0) if zs else zm), bool(zs)


def blend(pairs: pd.DataFrame, w: float, ze: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-((1 - w) * _lg(pairs.p.values) + w * ze)))


def _ll(p, y):
    p = np.clip(p, 1e-15, 1 - 1e-15)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def walk_forward_weight(names, g, year, metric, history: dict) -> float:
    """history: {season: pairs DataFrame with p (model-only) and y} for seasons < year."""
    f = _ll if metric == "log loss" else (lambda p, y: float(np.mean((p - y) ** 2)))
    prev = []
    for s in SEASONS:
        if s >= year or s not in history:
            continue
        ze, ok = consensus(history[s], names, s, g)
        if ok:
            d = history[s].assign(ze=ze).dropna(subset=["y"])
            prev.append(d)
    if not prev:
        return 0.0
    sc = {w: np.mean([f(blend(d, w, d.ze.values), d.y.values) for d in prev]) for w in GRID}
    return float(min(sc, key=sc.get))
