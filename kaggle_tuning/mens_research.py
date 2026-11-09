"""Cached men's walk-forward ablations and descriptive market-weight sensitivity.

MANIA_DATA=... GRID_JOBS=5 uv run python -m kaggle_tuning.mens_research
"""

import json
import os
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
import numpy as np
from joblib import Memory, Parallel, delayed
from scipy.special import expit

from . import backtest as B
from . import features as F
from . import market as MK
from .configs.final import BASE
from .configs.mens_research import CONFIGS

OUT = Path("kaggle_tuning/results/mens")
CACHE = Memory("/tmp/mania_mens_research_cache", verbose=0)
YEARS = B.HIST + [2025, 2026]


@CACHE.cache
def predict(cfg, g, year):
    # Cache leaves because most ablations retain the other ensemble component.
    if g == "M" and "men_models" in cfg:
        cfg = {**cfg, "models": cfg["men_models"]}
    cfg = {k: v for k, v in cfg.items() if k != "men_models"}
    if "models" in cfg:
        base = {k: v for k, v in cfg.items() if k != "models"}
        parts = [
            (
                predict(
                    {
                        **base,
                        **sub,
                        "market_w": {},
                        "sharpen": {},
                        "clip_p": (0, 1),
                        "models_inner": True,
                    },
                    g,
                    year,
                ),
                w,
            )
            for sub, w in cfg["models"]
        ]
        return expit(sum(w * B._logit(p) for p, w in parts) / sum(w for _, w in parts))
    cfg = {**cfg, "market_w": {}, "xgb": {"n_jobs": 1, **cfg.get("xgb", {})}}
    table = F.team_table(
        g,
        cfg.get("q_alpha", 1.0),
        cfg.get("q_recency", 0.0),
        cfg.get("q_clip_" + g, 30.0),
    )
    return B.fit_predict(cfg, g, year, table)


def evaluate(name, cfg):
    records = {}
    predictions = {}
    for year in YEARS:
        p = predict(cfg, "M", year)
        y = B.test_pairs("M", year).y.values
        records[year] = float(np.mean((p - y) ** 2))
        predictions[str(year)] = p.tolist()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}.json").write_text(
        json.dumps({"cfg": cfg, "scores": records, "p": predictions}, indent=2)
    )
    return name, records


def summarize():
    women = {
        y: float(
            np.mean(
                (
                    B._post(BASE, "W", y, predict(BASE, "W", y))
                    - B.test_pairs("W", y).y.values
                )
                ** 2
            )
        )
        for y in YEARS
    }
    report = []
    for file in sorted(OUT.glob("*.json")):
        r = json.loads(file.read_text())
        scores = {int(k): v for k, v in r["scores"].items()}
        name = file.stem
        row = {
            "name": name,
            "hM": np.mean([scores[y] for y in B.HIST]),
            "vM": np.mean([scores[y] for y in B.VERIFY]),
        }
        # The cancelled VCU/Oregon game leaves 62 men's games in 2021.
        combined = {
            y: np.average(
                [scores[y], women[y]],
                weights=[len(B.test_pairs(g, y)) for g in ("M", "W")],
            )
            for y in B.HIST
        }
        row["hist"] = np.mean([combined[y] for y in B.HIST])
        row["verify"] = np.mean([combined[y] for y in B.VERIFY])
        weights = []
        for w in [0, 0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]:
            z = []
            for year in [2025, 2026]:
                p = np.array(r["p"][str(year)])
                y = B.test_pairs("M", year).y.values
                pp = B._post({**BASE, "market_w": {"M": w}}, "M", year, p)
                z.append(float(np.mean((pp - y) ** 2)))
            weights.append({"w": w, "M25": z[0], "M26": z[1], "mean": np.mean(z)})
        fixed = next(z for z in weights if z["w"] == 0.85)
        row.update(
            {
                "M25_only": scores[2025],
                "M26_only": scores[2026],
                "M25": fixed["M25"],
                "M26": fixed["M26"],
                "W25": women[2025],
                "W26": women[2026],
            }
        )
        row["2025"] = (row["M25"] + row["W25"]) / 2
        row["2026"] = (row["M26"] + row["W26"]) / 2
        row["weight_sweep"] = weights
        row["best_weight"] = min(weights, key=lambda z: z["mean"])["w"]
        # Descriptive diagnostics, never training inputs.
        for year in [2025, 2026]:
            p = np.array(r["p"][str(year)])
            y = B.test_pairs("M", year).y.values
            mk = (
                B.test_pairs("M", year)[["T1", "T2"]]
                .merge(MK.market(year), on=["T1", "T2"], how="left")
                .mkt.values
            )
            row[f"corr{year}"] = float(np.corrcoef(p, mk)[0, 1])
            row[f"error_corr{year}"] = float(np.corrcoef(p - y, mk - y)[0, 1])
        report.append(row)
    baseline = next(r for r in report if r["name"] == "baseline")
    for r in report:
        r["accepted"] = bool(
            r["hist"] <= baseline["hist"] + 0.0005
            and r["verify"] <= baseline["verify"] + 0.0005
            and r["2025"] + r["2026"] < baseline["2025"] + baseline["2026"]
        )
    (OUT / "summary.txt").write_text(json.dumps(report, indent=2))
    print(
        "name hist hM v21-24 vM M25only M26only 2025 M25 W25 2026 M26 W26 best_w",
        flush=True,
    )
    for r in sorted(report, key=lambda r: r["2025"] + r["2026"]):
        print(
            r["name"],
            *[
                f"{r[k]:.5f}"
                for k in [
                    "hist",
                    "hM",
                    "verify",
                    "vM",
                    "M25_only",
                    "M26_only",
                    "2025",
                    "M25",
                    "W25",
                    "2026",
                    "M26",
                    "W26",
                    "best_weight",
                ]
            ],
            flush=True,
        )


if __name__ == "__main__":
    import sys

    names = sys.argv[1:]
    if names == ["summary"]:
        summarize()
    else:
        from kaggle_tuning.mens_research import evaluate

        configs = {k: v for k, v in CONFIGS.items() if not names or k in names}
        result = Parallel(n_jobs=int(os.environ.get("GRID_JOBS", "5")), verbose=10)(
            delayed(evaluate)(n, c) for n, c in configs.items()
        )
        summarize()
