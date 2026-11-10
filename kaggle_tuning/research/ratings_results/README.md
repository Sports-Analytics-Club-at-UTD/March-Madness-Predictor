Regularized team-rating experiment

All new feature fits use regular-season DayNum <=132. Rating alpha selection uses only tournaments strictly before the feature season. Tournament prediction models retain the established earlier-season training cutoff. No external data was added.

`team_ratings.py` builds centered per-team ratings for each gender:
- Adjusted offense and defense: 100 * points / possessions = centered league average + offense[team] + defense_allowed[opponent] + home * location. Possessions average both teams' FGA - OR + TO + .475 FTA. Ridge alpha 1, 10, 100; higher `rt_def` means better defense.
- Margin ridge alpha .1, 1, 10, 100, margins clipped at 60. `rt_wf` minimizes earlier tournament Brier using a fixed expit(rating_difference / 10) probability link; season means get equal weight, default alpha 1 without history. It does not tune the probability scale. `rt_gcv` minimizes regular-season generalized cross-validation over the same grid.
- Lasso alpha .01 and .1, clip 60. Huber via 12 IRLS ridge fits, residual threshold 10 points and ridge alpha 1, using raw margins.
- Carry-over minimizes squared clipped-margin residuals plus alpha * ||rating - .7 * previous_season_rating||², alpha 1 or 10. Previous ratings are the previous season's ordinary alpha-1 ridge; new teams have prior zero. Home coefficient prior is zero.
- Recency weights exp(k * (day - 132)/132), k=1.5 or 3. Opponent-quality weighting uses 1 + the mean percentile rank of the two teams' within-season initial ridge strength, so game weights lie in [1,2]. Both sides get the same weight.
- Bradley–Terry is logistic win/loss on team-indicator differences and location, C=1, no intercept (L2 penalty).

`screen.json` is a fast logistic screen against alpha-1 ordinary quality for both genders, NOT the full baseline ensemble. `women_actual_logit_screen.json` separately checks the actual alpha-.1 women's logistic member. It confirms much of the women's low-alpha gain is already present. Do not interpret the first screen's women's gains as full-ensemble gains.

Candidate selection was informed by historical averages, with target-year results visible; this is exploratory research, not a new untouched holdout. Alpha/feature/weight grids are round-number sensitivity checks, not nested selection of the complete final model.

Full candidates inherit `allyears.py:EXT`, with original ensemble weights, market blend, external sources, model sharpening, and walk-forward women's temperature unchanged. Only logit/xgb members receive features; the men's margin member and women's original/Vilnius members retain their inputs. `carry` adds rt_carry10 to the men's tree. `efficiency` adds rt_off1 and rt_def1 to men's tree and women's logit/tree. `wf` replaces women's quality by rt_wf in logit/tree. There is no men's logit member in allyears BEST.

Exact expanded configs: `carry_config.json`, `efficiency_config.json`, `wf_config.json`. Full historical evaluations: `*_hist.json`. HIST is 2015–2019 and 2021–2024, equal-season average, with market disabled. 2020 has no tournament. The ordinary all_years CLI output is saved in `*_ext.txt`; baseline_ext reproduces every number in the shared brief.

Reproduce (from repository root):

    export MANIA_DATA=/Users/spro/code/March-Madness-Predictor/data/kaggle/mirror26
    export GRID_JOBS=4 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
    uv run python -m unittest tests.test_team_ratings_research
    uv run python -m kaggle_tuning.research.ratings_screen
    uv run python -m kaggle_tuning.research.ratings_evaluate
    uv run python -m kaggle_tuning.all_years kaggle_tuning/configs/allyears.py:EXT
    uv run python -m kaggle_tuning.all_years kaggle_tuning/configs/ratings.py:CARRY
    uv run python -m kaggle_tuning.all_years kaggle_tuning/configs/ratings.py:EFFICIENCY

Run the expensive commands sequentially to keep this research track within four cores. all_years now honors GRID_JOBS; default behavior stays at six workers.
