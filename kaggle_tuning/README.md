# kaggle_tuning

Leak-free walk-forward backtests for Kaggle March Machine Learning Mania (2025/2026 metric:
Brier on the 63 men's + 63 women's tournament games, First Four excluded).

Data (gitignored) lives in `data/kaggle/` (override with `MANIA_DATA=/abs/path/to/mirror26`):
- `mirror26/` – competition CSVs as of 2026-04-05 (Kaggle dataset muhammadtojiddinov/march-machine-learning-mania-2026)
- `sol26/` – 2026 solution file (brisamarina/2026-march-machine-learning-mania-solution-file)
- `market25/submission.csv`, `market26/submission.csv` – pre-deadline betting-odds probabilities (goto_conversion)
- `lb/{2024,2025,2026}/` – final leaderboards

Scorer is validated: the goto_conversion file scores exactly 0.12584 (2026) and 0.10921 (2025, with its Drake gamble), matching the leaderboard.

- `python -m kaggle_tuning.grid kaggle_tuning/configs/r6.py` – run configs in parallel. Columns: hist (2015–2024 model-only mean, no market), v21-24 (overfitting check), 2025, 2026 (+ men/women).
- `python -m kaggle_tuning.report --og-dir ...` – step-by-step comparison vs the original pipeline.

Leaderboard cutoffs: top-100 = 0.11078 / 0.12430, top-20 = 0.10921 / 0.12016 (2025 / 2026).

Women's research (`configs/women_recommended.py`):

- Leaves men on `final`; women combine 50% logistic (ridge alpha 0.1), 10% XGBoost,
  and 40% logistic using the original pipeline's trimmed efficiency, four factors,
  quality wins and schedule strength. Ensembles combine log odds. Women's final
  probabilities receive 1.1 log-odds sharpening; the market weight remains 0.5.
- Overall 2025/2026: **0.10838 / 0.12142**; women: **0.09061 / 0.09826**.
  Historical mean improves 0.000077, hW improves 0.000147, and v21–24 improves
  0.000031. The aggressive variant increases the original member to 50% and
  scores 0.10814 / 0.12110, but historical mean rises 0.000292 and v21–24 rises
  0.000447, close to the shared 0.0005 guard.
- Women's model alone scores 0.09415 / 0.09661. Market weights 0.4, 0.5, 0.6
  give overall two-year means 0.11484, 0.11490, 0.11503. Keep 0.5 by the existing
  equal-weight rule: there is no historical market evidence to select 0.4.
  `women_market_sensitivity.py` reproduces these checks; results are stored in
  `women_market_results.json`.
- `women_results.json` records all 58 configurations and per-season checks.
  `configs/women.py`, `women_round2.py`, `women_round3.py`, and
  `women_sensitivity.py` reproduce the research grids. Full original-model
  replacement, isotonic calibration and stronger sharpening failed the guard.
  No external ratings were used; joint-gender training was not tested.

Run the recommendation with the ordinary grid runner. `python -m
kaggle_tuning.women_grid <config.py>` additionally caches fits under `/private/tmp`,
keyed by implementation, full configuration, and data-file metadata, and saves a
`.results.json` beside the config. Set `MANIA_DATA` and `GRID_JOBS=5` as above.
The women's tests check calibration symmetry, target-season exclusion, and the
DayNum 132 cutoff. The original `configs/final.py` still reproduces 0.1099 / 0.1227.
