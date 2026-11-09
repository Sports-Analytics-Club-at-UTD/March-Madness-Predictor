"""Women's grid, caching reusable fits outside the repository (set MANIA_DATA)."""
import os
import runpy
import sys
import time
import json
import hashlib
from pathlib import Path
from joblib import Memory, Parallel, delayed


def one(name, cfg):
    os.environ['OMP_NUM_THREADS'] = '1'
    from . import backtest as B
    original = B.fit_predict
    # Include implementation and input snapshots: joblib cannot track imported helpers.
    digest = hashlib.sha256()
    for path in (Path(B.__file__), Path(B.F.__file__), Path(__file__).with_name('women.py'), Path(B.__file__).parent.parent / 'kaggle_mania_brackets.py'):
        digest.update(path.read_bytes())
    digest.update(str(B.F.DATA.resolve()).encode())
    for path in sorted(B.F.DATA.parent.rglob('*.csv')):
        stat = path.stat()
        digest.update(f'{path}:{stat.st_size}:{stat.st_mtime_ns}'.encode())
    cached = Memory('/private/tmp/mania_women_' + digest.hexdigest()[:16], verbose=0).cache(original, ignore=['table'])
    def predict(config, gender, year, table):
        if gender == 'M':
            config = {k: v for k, v in config.items() if k != 'by_gender'}
        return cached(config, gender, year, table)
    B.fit_predict = predict
    try:
        t = time.time()
        r = B.run(cfg, verbose=False)
        print(name, json.dumps(r), flush=True)
        return name, r, time.time() - t
    finally:
        B.fit_predict = original


def main():
    configs = runpy.run_path(sys.argv[1])['CONFIGS']
    out = Parallel(n_jobs=int(os.environ.get('GRID_JOBS', 5)))(delayed(one)(n,c) for n,c in configs.items())
    print('config hist hM hW v21-24 2025 M25 W25 2026 M26 W26 mean')
    for n,r,dt in sorted(out, key=lambda v: v[1][2025]['brier'] + v[1][2026]['brier']):
        vals = [r['hist'], r['histM'], r['histW'], r['verify']] + [r[y][k] for y in (2025,2026) for k in ('brier','M','W')]
        vals.append((r[2025]['brier']+r[2026]['brier'])/2)
        print(n, ' '.join(f'{v:.5f}' for v in vals), f'{dt:.0f}s')
    with open(Path(sys.argv[1]).with_suffix('.results.json'), 'w') as f:
        json.dump({n:r for n,r,_ in out}, f, indent=2)


if __name__ == '__main__':
    main()
