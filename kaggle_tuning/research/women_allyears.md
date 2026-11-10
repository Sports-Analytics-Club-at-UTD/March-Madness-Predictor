# Women's 2021–2024 model-only research

Recommendation: retain the four model members and fit the log-odds multiplier
walk-forward for model-only pairwise metrics. Keep the bracket metric and market
blends on the established multiplier 1.1. This is a modest gain, not a top-20
solution. No team overrides or external data were used.

Exact configuration:

```python
from kaggle_tuning.configs.combined import BEST
RECOMMENDED = {**BEST, 'women_model_temperature': {'start': 2014, 'penalty': .01}}
```

The women’s members remain 0.4 logistic / 0.1 XGBoost / 0.3 original logistic /
0.2 Vilnius, blended in log odds. Fit the multiplier in [0.7, 1.5], minimizing
mean historical loss plus `0.01 * (a - 1.1)**2`. Require three prior seasons,
otherwise use 1.1. Every historical prediction is itself trained only on older
tournament seasons. Use log loss through 2022 and pairwise Brier thereafter.
The multiplier is fitted on seasons from 2014 through Y−1, excluding 2020.
Do not apply this model-only calibration to market blends or the bracket metric.

| Scored year / metric | combined BEST | Recommendation | Old → new rank |
| --- | ---: | ---: | --- |
| 2021 W log loss | .43140011 | .42941061 | 35 → 32 |
| 2022 W log loss | .43655803 | .43636623 | 103 → 99 |
| 2023 combined Brier | .18649365 | .18541900 | 237 → 192 |
| 2024 combined bracket | .05773548 | .05773548 | 25 → 25 |
| 2025 combined Brier | .10796536 | .10796536 | 10 → 10 |
| 2026 combined Brier | .12121942 | .12121942 | 31 → 31 |

Women's 2023 Brier: .16907112 → .16692183. Women's 2024 bracket Brier stays
.03759447. Women's 2025/2026 stay .09013734 / .09790535.
The fitted 2021/2022/2023 multipliers are .99296487 / .99114145 / .97574977.
Men's predictions are unchanged (.59639/#81 and .64912/#418 in 2021/2022).

| Model-only Brier guard | BEST | Recommendation |
| --- | ---: | ---: |
| 2015–2019 women | .13813283 | .13774786 |
| 2015–2019 combined | .15520000 | .15500751 |
| Historical women | .13957949 | .13898792 |
| Historical combined | .16329733 | .16300155 |

Historical is the grid's nine tournament seasons 2015–2019 and 2021–2024.
No tournament existed in 2020. The old-period women's guard improves .000385,
comfortably satisfying the maximum allowed deterioration .0005.

Sensitivity to penalty .001 / .01 / .1, holding market/bracket processing fixed:
2021 log loss .429375 / .429411 / .429912; 2022 .436459 / .436366 / .436124;
2023 women's Brier .166314 / .166922 / .168493. All improve these target losses.

Experiments rejected:

- Nonnegative walk-forward stacking with penalty .001 brought 2021 to .417327,
  but worsened 2022 to .447916 and 2015–2019 women's Brier by .000676,
  **failing the .0005 guard**. Softer stacking still hurt 2022 and market years.
- Fixed sharpening 1.2 worsened old women's Brier by about .000900: **guard fail**.
- A new consensus logistic member combined margin-aware Bradley–Terry,
  possession-adjusted ridge efficiency, ridge margin, Elo and seeds. It improved
  old-period Brier but worsened 2022 and 2025/2026. Adding these ratings to the
  original feature model also hurt 2022. The experimental route is opt-in.
- Applying pairwise calibration to the market blends worsened 2025/2026.
- Fitting temperature to historical bracket loss reduced sharpening and worsened
  2024 women's bracket Brier (.038043 at penalty .01 versus .037594 baseline).
- Regular-season games train the new ratings, but were not added as classifier
  targets. No external ratings were evaluated in this track.

Reproduction (`MANIA_DATA` as specified by the shared brief, one BLAS thread):

```sh
uv run python -m kaggle_tuning.research.women_allyears_run
uv run python -m kaggle_tuning.research.women_allyears_eval
uv run python -m kaggle_tuning.research.women_allyears_report
GRID_JOBS=3 uv run python -m kaggle_tuning.grid kaggle_tuning/configs/women_allyears.py
uv run python -m kaggle_tuning.all_years kaggle_tuning/configs/women_allyears.py:RECOMMENDED
uv run python -m unittest discover -s tests -p 'test_women*.py'
```

The all-years runner currently defaults to six jobs; validation called its `main`
with `Parallel(n_jobs=3)` instead to respect the shared core budget.
Raw-member caches are outside the repository in `/private/tmp` and the actual
calibrator fingerprints source and input files. JSON files record the experiments
and exact per-year comparison. No existing configuration was changed.
