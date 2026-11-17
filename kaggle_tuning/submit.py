"""Build, validate and (optionally) score a Kaggle March Machine Learning Mania submission.

    MANIA_DATA=/abs/mirror26 uv run --extra training python -m kaggle_tuning.submit \
        --season 2026 --out submission.csv [--market odds.csv | --model-only] [--score solution.csv]

Fits `configs/women_recommended.RECOMMENDED` (men = configs/final, women = recommended ensemble,
market blend M 0.85 / W 0.5). Competition CSVs are cut at Selection Sunday of --season; market
odds are taken as given and must predate the submission deadline. It predicts every pair of seeded teams, writes 0.5 for all other rows of the Stage 2 sample (never scored), validates the
file and writes `<out>.provenance.json`.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from datetime import datetime, timezone
from functools import lru_cache
from importlib.metadata import version
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

from . import backtest as B
from . import features as F
from . import market as MK
from .configs.women_recommended import RECOMMENDED

CONFIGS = {"recommended": RECOMMENDED, "model_only": {**RECOMMENDED, "market_w": {}}}
ID_RE = re.compile(r"^\d{4}_\d{4}_\d{4}$")


def validate(sub: pd.DataFrame, sample: pd.DataFrame) -> None:
    """Raise ValueError listing every way `sub` is not a valid upload for `sample`."""
    errs = []
    if list(sub.columns) != ["ID", "Pred"]:
        errs.append(f"columns must be exactly ['ID', 'Pred'], got {list(sub.columns)}")
    if "ID" in sub:
        ids = sub.ID.astype(str)
        if len(sub) != len(sample):
            errs.append(f"{len(sub)} rows, sample has {len(sample)}")
        if ids.duplicated().any():
            errs.append(f"{int(ids.duplicated().sum())} duplicate IDs")
        if set(ids) != set(sample.ID):
            errs.append(f"{len(set(sample.ID) - set(ids))} sample IDs missing, {len(set(ids) - set(sample.ID))} extra")
        elif ids.tolist() != sample.ID.tolist():
            errs.append("rows are not in sample order")
        bad = ~ids.str.match(ID_RE) | (ids.str[5:9] >= ids.str[10:14])
        if bad.any():
            errs.append(f"{int(bad.sum())} IDs not SSSS_XXXX_YYYY with XXXX < YYYY, e.g. {ids[bad].iloc[0]}")
    if "Pred" in sub:
        p = pd.to_numeric(sub.Pred, errors="coerce").to_numpy(float)
        if (~np.isfinite(p)).any():
            errs.append(f"{int((~np.isfinite(p)).sum())} Pred values are NaN/inf/non-numeric")
        if ((p < 0) | (p > 1)).any():
            errs.append(f"{int(((p < 0) | (p > 1)).sum())} Pred values outside [0, 1]")
    if errs:
        raise ValueError("invalid submission:\n  " + "\n  ".join(errs))


def as_of(df: pd.DataFrame, name: str, season: int) -> pd.DataFrame:
    """Drop rows not known before `season`'s tournament (Selection Sunday, DayNum 132)."""
    if "Season" not in df:
        return df
    df = df[df.Season <= (season - 1 if "NCAATourney" in name and "Results" in name else season)]
    for col, last in (("DayNum", 132), ("RankingDayNum", 133)):
        if col in df:
            df = df[(df.Season < season) | (df[col] <= last)]
    return df


def parse_pairs(ids: pd.Series) -> pd.DataFrame:
    return pd.DataFrame({"Season": ids.str[:4].astype(int), "T1": ids.str[5:9].astype(int), "T2": ids.str[10:14].astype(int)})


def predict(cfg: dict, season: int, sample: pd.DataFrame) -> tuple[np.ndarray, dict]:
    pairs = parse_pairs(sample.ID)
    pred = np.full(len(pairs), 0.5)
    counts = {}
    # shortcut: team_table/original_table stay lru-cached per process, so run one season per process.
    raw = F.read.__wrapped__
    with mock.patch.object(F, "read", lru_cache(None)(lambda name: as_of(raw(name), name, season))):
        for g, lo in (("M", 1000), ("W", 3000)):
            seeded = set(F.seeds(g).query("Season == @season").TeamID)
            if len(seeded) < 64:
                raise SystemExit(f"{g}: only {len(seeded)} seeded teams for {season}; seeds come out on Selection Sunday")
            mask = (pairs.T1 >= lo) & (pairs.T1 < lo + 1000) & pairs.T1.isin(seeded) & pairs.T2.isin(seeded)
            te = pairs[mask].assign(y=0).reset_index(drop=True)
            if cfg.get("market_w", {}).get(g):
                mk = te.merge(MK.market(season), on=["T1", "T2"], how="left").mkt
                if mk.isna().any() or not mk.between(0, 1).all():
                    raise SystemExit(f"{g}: market file lacks valid odds for {int(mk.isna().sum())} seeded pairs")
            table = F.team_table(g, cfg.get("q_alpha", 1.0), cfg.get("q_recency", 0.0), cfg.get("q_clip_" + g, 30.0))
            with mock.patch.object(B, "test_pairs", lambda gg, yy: te):
                pred[mask.to_numpy()] = B.fit_predict(cfg, g, season, table)
            counts[g] = {"seeded_teams": len(seeded), "modelled_rows": int(mask.sum())}
    return pred, counts


def score(sub: pd.DataFrame, solution: Path) -> dict:
    """Brier on the solution file's games (the 63 + 63 scored games; validated against the 2025/2026 LB)."""
    s = pd.read_csv(solution)[["ID", "Pred"]].rename(columns={"Pred": "y"}).merge(sub, on="ID", how="left")
    if s.Pred.isna().any():
        raise ValueError(f"{int(s.Pred.isna().sum())} solution games missing from submission")
    err = (s.Pred - s.y) ** 2
    men = s.ID.str[5:9].astype(int) < 3000
    return {"brier": float(err.mean()), "M": float(err[men].mean()), "W": float(err[~men].mean()), "games": len(s)}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git(*args: str) -> str:
    return subprocess.run(["git", *args], capture_output=True, text=True, cwd=Path(__file__).parent).stdout.strip()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--sample", type=Path, help="default: $MANIA_DATA/SampleSubmissionStage2.csv")
    ap.add_argument("--market", type=Path, help="pre-deadline market odds in submission format (default: market.PATHS)")
    ap.add_argument("--model-only", action="store_true", help="no market odds (use if odds are ruled out)")
    ap.add_argument("--score", type=Path, help="solution CSV (ID, Pred = outcome) to score against afterwards")
    a = ap.parse_args(argv)

    name = "model_only" if a.model_only else "recommended"
    cfg = CONFIGS[name]
    sample_path = a.sample or F.DATA / "SampleSubmissionStage2.csv"
    sample = pd.read_csv(sample_path)
    if not sample.ID.str.startswith(f"{a.season}_").all():
        raise SystemExit(f"{sample_path} has rows outside season {a.season}; use the Stage 2 sample for that season")
    if a.market:
        MK.PATHS[a.season] = a.market
        MK.market.cache_clear()
    if not a.model_only and a.season not in MK.PATHS:
        raise SystemExit(f"no market odds for {a.season}: pass --market path/to/odds.csv or --model-only")
    market_path = MK.PATHS.get(a.season) if not a.model_only else None

    pred, counts = predict(cfg, a.season, sample)
    sub = pd.DataFrame({"ID": sample.ID, "Pred": pred})
    validate(sub, sample)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(a.out, index=False)
    written = pd.read_csv(a.out)
    validate(written, sample)
    print(f"wrote {a.out}: {len(written)} rows, validation passed; config={name}, " +
          ", ".join(f"{g}: {c['modelled_rows']} modelled pairs of {c['seeded_teams']} seeded teams" for g, c in counts.items()))

    used = sorted(F.DATA.glob("*.csv")) + [Path(sample_path)] + ([Path(market_path)] if market_path else [])
    prov = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "season": a.season, "config_name": name, "config": cfg,
        "config_source": "kaggle_tuning/configs/women_recommended.py (men: configs/final.py)",
        "data_cutoff": f"regular season DayNum <= 132, Massey RankingDayNum <= 133, NCAA results seasons < {a.season}"
                       + ("; market odds as given (must predate the submission deadline)" if market_path else ""),
        "rows": len(written), "counts": counts,
        "pred_summary": {k: float(v) for k, v in written.Pred.describe()[["min", "mean", "max"]].items()},
        "git": {"sha": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain"))},
        "packages": {p: version(p) for p in ("numpy", "pandas", "scikit-learn", "xgboost", "scipy")},
        "files": {str(p): sha256(Path(p)) for p in dict.fromkeys(map(str, used))},
        "submission_sha256": sha256(a.out),
    }
    if a.score:
        prov["score"] = score(written, a.score)
        print("score vs {}: brier={brier:.5f}  M={M:.5f}  W={W:.5f}  ({games} games)".format(a.score.name, **prov["score"]))
    prov_path = a.out.with_suffix(".provenance.json")
    prov_path.write_text(json.dumps(prov, indent=2, default=list) + "\n")
    print(f"wrote {prov_path}")


if __name__ == "__main__":
    main()
