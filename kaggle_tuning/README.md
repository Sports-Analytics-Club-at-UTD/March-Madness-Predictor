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
