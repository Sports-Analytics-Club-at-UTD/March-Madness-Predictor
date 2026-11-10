# Men's model-only research

Recommendation: `configs/mens_allyears.py:RECOMMENDED` (`SHRUNK`). This is one fixed method: 50% existing men's XGBoost spline and 50% new ridge-margin model, blended in log odds, then multiplied by 0.9 before the existing market blend. Women, market weights, and market sharpening remain BEST's. No year-specific parameters or external data.

| Real-year metric | combined.py:BEST | SHRUNK | TREE75 sensitivity |
|---|---:|---:|---:|
| 2021 men's log loss | .596388 (#81) | .588919 (#48) | .588587 (#48) |
| 2022 men's log loss | .649115 (#418) | .631267 (#270) | .648847 (#417) |
| 2023 combined Brier | .186494 (#237) | .186422 (#231) | .185919 (#209) |
| 2024 combined bracket Brier | .057735 (#25) | .057725 (#25) | .057659 (#24) |
| 2025 combined Brier | .107965 (#10) | .107959 (#10) | .107910 (#10) |
| 2026 combined Brier | .121219 (#31) | .121218 (#31) | .121189 (#31) |

2021/2022 women's log loss is unchanged: .431400 (#35) / .436558 (#103). The requested top-20 objective is **not achieved**. Improvements in 2025/2026 are extremely small; do not interpret them as robust gains in expected performance.

| Guard (model-only) | BEST | SHRUNK | TREE75 |
|---|---:|---:|---:|
| 2015–2019 men's mean Brier | .172267 | .172275 | .172520 |
| 2015–2019 combined mean Brier | .155200 | .155204 | .155326 |
| hist 2015–2024 men's mean Brier | .187015 | .186031 | .186349 |
| hist 2015–2024 combined mean Brier | .163268 | .162777 | .162937 |

2020 is absent. 2023 men's Brier: BEST .203916, SHRUNK .203773, TREE75 .202766. Blended men's 2025/2026: BEST .125793/.144533, SHRUNK .125781/.144530. The cancelled VCU/Oregon game receives no fabricated label; combined scores weight actual observations.

## Exact configuration

Fully expanded JSON: `results/mens_allyears/recommended_config.json`.

```python
from kaggle_tuning.configs.combined import BEST
TREE = BEST['men_models'][1][0]
MARGIN = {
    'model': 'mens_allyears',
    'mens_allyears': {
        'margin': True, 'alpha': 1,
        'features': ['seed', 'q_pool', 'elo', 'mstr_med', 'mstr_worst',
                     'net_eff', 'conf_quality'],
    },
}
RECOMMENDED = {
    **BEST, 'men_models': [(TREE, 1), (MARGIN, 1)],
    'model_sharpen': {'M': .9},
}
```

Ridge predicts margins from standardized feature differences, using tournament seasons 2003 through Y−1. Logistic calibration has no intercept, C=100, and is fitted on season-held-out ridge predictions from those earlier seasons. The final ridge refits all those seasons. Its calibration is LOSO within past training seasons, never fitted to target outcomes. Features use regular-season results through day 132, ordinal ranks through day 133, and seeds. Source compact and detailed regular-season files both have maximum DayNum 132. No Torvik tables were used.

## Ablations, sensitivity, and guard failures

18 logistic variants trained on day-101–132 regular-season games with features frozen at day 100, alongside past tournament games. Tested regular/tournament total-weight ratios 0, .5, 1, 3, with neutral-only alternatives; 2, 6, or 15 features. Neither neutral-only nor pooled regular training beat the baseline across the requested targets. Broad ordinal neutral-only weight 3 failed the old-men guard: .176815 versus .172267.

Nine margin variants tested 3/7/13 features and ridge alpha 1/100/1000. Pure seven-feature alpha 1 improves 2022 to .631944 and old-men Brier to .171396, but worsens 2021 to .615764, 2023 men to .210396, and market-blended 2026 men to .144735. The 13-feature alpha-1 margin model fails the guard at .174939. The seven-feature model is useful as a blend member, not as a replacement.

An initial temperature diagnostic on BEST at .8 improves 2021/2022 log loss to .584230/.627754 but fails the guard: old-men Brier .174123 (+.001856). Rolling earlier-season temperature fits over the saved 2015+ predictions gave smaller gains; these diagnostics have a short calibration history and were not promoted. The recommendation uses a fixed .9 across all years and metrics, not target-fitted annual temperatures.

Coarse log-odds blend sensitivity tested ridge shares .25/.5, tree shares .5/.75/1 in the non-ridge component, and multipliers .9/1/1.1. TREE75 is 75% tree / 25% ridge without temperature; it passes all requested checks. BALANCED (18.75% logit / 56.25% tree / 25% ridge) passes history, but slightly worsens 2024 bracket Brier (.057738991). SHRUNK improves all reported targets with the better historical mean. These target seasons were inspected during selection; the guard is a retrospective robustness check, not an untouched holdout.

## Reproduction and files

Set `MANIA_DATA=/Users/spro/code/March-Madness-Predictor/data/kaggle/mirror26`, `GRID_JOBS=3`, `OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1`.

- `uv run python -m kaggle_tuning.all_years kaggle_tuning/configs/mens_allyears.py:RECOMMENDED` reproduces real metrics. Research called the same `score_year` functions with joblib n_jobs=3; the stock CLI uses six workers.
- `uv run python -m kaggle_tuning.grid kaggle_tuning/configs/mens_allyears.py` reproduces model-only hist and market years.
- `uv run python -m kaggle_tuning.mens_allyears_experiment regular` and `... margin` reproduce component grids. Delete `/tmp/mania_mens_allyears_v1` after feature/input changes.
- `uv run python -m unittest discover -s tests -p test_mens_allyears.py`: three synthetic checks for frozen snapshots, target-season exclusion/symmetry, and calibration before the market blend.

Added model/research modules, own configs, synthetic tests, this report, and raw JSON results. `backtest.py` has two opt-in hooks (new men's predictor; model calibration before market blending). Existing configs were not edited.

Final validation: all three synthetic tests passed; `git diff --check` passed. An exact six-year regression rerun after the implementation matched the saved `combined.py:BEST` scores and ranks byte-for-byte after JSON normalization.
