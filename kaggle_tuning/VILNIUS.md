# Vilnius port

Run with the shared competition mirror:

```sh
export MANIA_DATA=/Users/spro/code/March-Madness-Predictor/data/kaggle/mirror26
export GRID_JOBS=5 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
uv sync --extra training
uv run python -m kaggle_tuning.vilnius_experiment
uv run python -m kaggle_tuning.grid kaggle_tuning/configs/vilnius.py
uv run python -m unittest discover -s tests -p test_vilnius.py
```

`vilnius_experiment` caches member probabilities under ignored `model_outputs/vilnius`
and writes `results.json` there. Clear this directory after changing recipe code or
replacing input CSVs. Its cache is keyed by member, target year, and input directory;
it does not fingerprint the CSV contents.

## Source and fidelity

Sources are the supplied Kaggle JSON dumps for `raddar/vilnius-ncaa`, `modeh`, and
`raddar/paris-madness`, read as data only. The Vilnius notebook contains execution
metadata dated 2025-03-17. No external ratings or new downloads are used.

- Joint men's/women's training begins in 2003 (women's detailed games begin in
  2010). Men's flag is always 1; women's is always 0.
- Regular-season features use DayNum <= 132 and seeds. Scores and all box counts,
  including training tournament margins, are divided by `(40+5*NumOT)/40`.
- Elo starts at 1000 each season, uses width 400 and K=100 without a home or margin
  adjustment, and processes games chronologically, retaining source order within days.
- Quality uses the source's Gaussian identity GLM design, with all T1 categorical
  levels and the first T2 level dropped. Sparse least squares replaces dense GLM
  fitting; a test checks equality against statsmodels. Each season/gender retains
  seeded teams and teams that beat a seeded team, pools other teams as 0000, and
  drops games between two pooled teams.
- The base has the exact 25-feature list and 700-round XGBoost settings. `modeh`
  adds T1/T2 offensive and defensive rebounds (29 total), eta=.0093,
  max_bin=38, and 704 rounds. Its manual submission gambles are excluded.
- `cauchy` changes the Vilnius loss to the Paris Cauchy objective (c=5000), keeping
  Vilnius features/LOSO/hyperparameters to isolate the objective. It is not a full
  recreation of Paris's older features and random-KFold tuning. At basketball
  margin scales this loss is nearly quadratic.
- `seeds` averages three independently fitted/calibrated recipes with seeds 0,1,2.
- LOSO training and spline fitting for target Y use only seasons < Y. Each LOSO
  model's margin is clipped at +/-25, calibrated by a degree-5 UnivariateSpline,
  clipped to [.01,.99], and then averaged across LOSO models. There is no full-data
  refit. First Four games are retained in training, as in the source, but excluded
  from scoring. The harness's forward/reverse probability averaging is retained.
- XGBoost's base_score=.5 preserves the fixed intercept used by the older source
  version, instead of newer XGBoost's automatically estimated intercept.

The ensemble fits nonnegative weights summing to one for L/X/V log odds, separately
by gender, minimizing Brier on earlier HIST walk-forward predictions. Its historical
scores use rolling weight fitting; 2015 uses equal weights because no earlier HIST
prediction is available. Both target years freeze weights using 2015–2024 only.
Historical scoring weights individual games (including the 2021 men’s no-contest),
matching the harness rather than averaging genders equally.
The market blend uses `final`'s existing weights (M=.85, W=.5) and sharpening=1.2.

## Results

See `vilnius_results.json` for all per-year scores and fitted weights, and
`vilnius_results.md` for the grid and acceptance assessment.
