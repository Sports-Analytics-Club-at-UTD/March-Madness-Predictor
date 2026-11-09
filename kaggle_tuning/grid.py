"""Run many configs in parallel; print one summary row each.

    uv run python -m kaggle_tuning.grid path/to/configs.py   (file defines CONFIGS = {name: cfg})
"""
import runpy, sys, time
import numpy as np
from joblib import Parallel, delayed


def one(name, cfg):
    import os
    os.environ["OMP_NUM_THREADS"] = "1"
    from .backtest import run
    t = time.time()
    cfg = {**cfg, "xgb": {"n_jobs": 1, **cfg.get("xgb", {})}}
    r = run(cfg, verbose=False)
    return name, r, time.time() - t


def main():
    confs = runpy.run_path(sys.argv[1])["CONFIGS"]
    out = Parallel(n_jobs=min(11, len(confs)), verbose=0)(delayed(one)(n, c) for n, c in confs.items())
    print(f"{'config':34s} {'hist':>7s} {'hM':>7s} {'hW':>7s} {'v21-24':>7s} {'2025':>7s} {'M25':>7s} {'W25':>7s} {'2026':>7s} {'M26':>7s} {'W26':>7s} {'mean':>7s}")
    for n, r, dt in sorted(out, key=lambda x: (x[1][2025]['brier'] + x[1][2026]['brier'])):
        a, b = r[2025], r[2026]
        flag = ("*" if a['brier'] <= 0.11078 else " ") + ("*" if b['brier'] <= 0.1243 else " ")
        print(f"{n:34s} {r.get('hist', np.nan):7.4f} {r['histM']:7.4f} {r['histW']:7.4f} {r['verify']:7.4f} {a['brier']:7.4f} {a['M']:7.4f} {a['W']:7.4f} {b['brier']:7.4f} {b['M']:7.4f} {b['W']:7.4f} {(a['brier']+b['brier'])/2:7.4f} {flag} {dt:4.0f}s")


if __name__ == "__main__":
    main()
