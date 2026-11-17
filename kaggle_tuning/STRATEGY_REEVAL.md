# Re-evaluating the current strategy (issue #67)

**Result: negative.** Seven alternatives were run against the current model under the repo's protocol, and none
beats it on 2025 *and* 2026 *and* history. The production configs are unchanged. The one configuration that
improves the 2025/2026 mean (`xgb_binary`, −0.00007) fails the 0.0005 guard on v21–24 (+0.00149).

## What was already settled, and not redone

- **RL with a policy.** PR #66 (`kaggle_tuning/research/RL_METHODS.md` on `t3/kaggle-eda-rl-59`) argues that the
  MDP, bandit and policy-gradient settings don't fit the task. There is one pre-tournament submission, it does
  not affect outcomes, every outcome is revealed, and the metric is a strictly proper score. The one RL-adjacent
  method it found plausible is Hedge (online learning with expert advice). PR #66 wrote a protocol for it but did
  not run it. **Candidate A below runs that protocol as written.**
- **Season noise.** PR #66's `KAGGLE_EDA.md` also measured season-to-season Brier variation of about ±0.03, which
  is far larger than any difference here. Only paired, same-games comparisons are meaningful.
- **Rating features.** These are covered in existing notes, so no new rating features were added:
  - Elo variants: `MENS_RESEARCH.md` elo5 and the Vilnius Elo.
  - Massey consensus, pooled and "worst" ordinals: `MENS_RESEARCH.md`.
  - Ridge, offence/defence, carry-over, lasso and Bradley–Terry ratings: `research/ratings_results/REPORT.md`.
- **Other covered approaches.**
  - The raddar/Vilnius XGBoost-margin recipe and history-fitted L/X/V weights: `VILNIUS.md`.
  - Penalized wide-feature models with normal-CDF linear margins: `penalized_report.txt`.
  - Market weights and sharpening: `MENS_RESEARCH.md` and `README.md`.
  - Women's temperature and isotonic calibration: `README.md` and `research/women_allyears.md`.

## Protocol

- **Reference** (`ref`): `configs/women_recommended.py:RECOMMENDED`. Men use `final` (logit + xgb_spline, 1:1 in
  log odds). Women use 0.5 narrow logit, 0.1 xgb_spline and 0.4 original-pipeline logit. Post-processing is
  market 0.85 for men and 0.5 for women, market sharpening 1.2, and women's sharpening 1.1. `ref` reproduces the
  README's 0.10838 / 0.12142.
- **Change scope.** Each candidate changes one part of the model. Everything else, including all
  post-processing, is inherited from `ref`.
- **Columns.**
  - `hist` is the model-only mean over 2015–2019 and 2021–2024.
  - `v21–24` is the overfitting check.
  - 2025 and 2026 include the market blend.
  - `mo25` and `mo26` are the same targets with the market weight set to 0.
- **Guard.** `hist` and `v21–24` may each rise by at most 0.0005 against `ref`.
- **Leakage.** Walk-forward is strict: member fits and combiner weights for season Y use only seasons before Y.
  Combiner history is frozen at 2024 for both target years, the same as `vilnius_ensemble`.
- **Fixed settings.** Every setting was fixed before any run, and none was tuned on 2025/2026.

## Candidates

| Config | Family | What changes |
|---|---|---|
| `hedge` | **A. Hedge** (PR #66 protocol) | Uniform prior over the existing members. η = 8 per season-mean Brier. Losses come from walk-forward member predictions on HIST seasons < Y. |
| `hedge_refprior` | A, sensitivity | Same, but the prior is the current hand-set weights. |
| `stack` | **B. Stacking** | Logistic regression without intercept on member log odds. Fitted on pooled walk-forward member predictions from the 2012–2014 warm-up seasons plus HIST < Y. The sum of the coefficients acts as a learned temperature. |
| `stack_ridge` | B | Same, with an L2 penalty toward the current weights (λ = 63, about one season of games). |
| `stack_nosharpen` | B | `stack` without the women's post-market 1.1 sharpening, which would otherwise double-count the learned temperature. |
| `xgb_normal` | **C. GBM margin → probability** | The xgb_spline member's degree-5 spline is replaced by P = Φ(margin/σ). σ is fitted by maximum likelihood on season-held-out margins (bounds 3–30). |
| `xgb_binary` | C | Same trees, but trained on `binary:logistic` so they predict the probability directly (min_child_weight scaled by 0.25 for logistic hessians). No spline. |

## Results

Lower is better. The Δ columns are against `ref`. "Seasons better" counts the HIST seasons in which the candidate
beats `ref`.

| Config | hist | hM | hW | v21–24 | 2025 | M25 | W25 | 2026 | M26 | W26 | mean | mo25 | mo26 | Δhist | Δv21–24 | Δmean | guard | seasons better |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:--:|:--:|
| **ref** | .16422 | .18810 | .14041 | .17516 | .10838 | .12614 | .09061 | .12142 | .14458 | .09826 | .11490 | .11419 | .12253 | — | — | — | — | — |
| hedge | .16455 | .18813 | .14104 | .17528 | .10854 | .12614 | .09094 | .12155 | .14455 | .09855 | .11504 | .11489 | .12274 | +.00033 | +.00012 | +.00014 | ok | 3/9 |
| hedge_refprior | .16411 | .18813 | .14016 | .17489 | .10865 | .12614 | .09115 | .12176 | .14455 | .09898 | .11521 | .11477 | .12321 | −.00011 | −.00027 | +.00031 | ok | 5/9 |
| stack | .16385 | .18827 | .13949 | .17278 | .10982 | .12663 | .09300 | .12300 | .14446 | .10154 | .11641 | .12085 | .12699 | −.00037 | −.00238 | +.00151 | ok | 5/9 |
| stack_ridge | .16407 | .18770 | .14049 | .17400 | .10875 | .12655 | .09094 | .12164 | .14454 | .09874 | .11519 | .11785 | .12377 | −.00016 | −.00116 | +.00029 | ok | 5/9 |
| stack_nosharpen | .16347 | .18827 | .13873 | .17258 | .11052 | .12663 | .09442 | .12339 | .14446 | .10232 | .11695 | .12153 | .12721 | −.00076 | −.00258 | +.00205 | ok | 6/9 |
| xgb_normal | .16415 | .18798 | .14038 | .17498 | .10838 | .12618 | .09057 | .12143 | .14463 | .09823 | .11490 | .11440 | .12266 | −.00007 | −.00018 | +.00000 | ok | 4/9 |
| xgb_binary | .16500 | .18899 | .14108 | .17666 | .10826 | .12595 | .09057 | .12140 | .14497 | .09783 | .11483 | .11337 | .12342 | +.00078 | +.00149 | −.00007 | **FAIL** | 2/9 |

Raw per-season and per-gender scores are in [`configs/strat_reeval.results.json`](configs/strat_reeval.results.json).

### Verdicts

- **A. Hedge does not help.**
  - With η = 8, the season-mean Brier gaps between members (≈0.005–0.02) move the weights only slightly. For
    2025/2026 they reach 0.46/0.54 for men and 0.45/0.28/0.27 for women.
  - The pre-registered uniform-prior run is worse on every column of the guard and the targets.
  - Starting from the current weights helps history a little (−0.00011) but hurts the targets (+0.00031).
  - This is the measured answer to PR #66's open question: the RL-adjacent method does not beat the current blend.
- **B. Stacking wins history and loses the targets.** It gives the largest history gain of any run, with v21–24
  improving by up to −0.00258 and 6 of 9 seasons better. But the unpenalized stacks worsen the 2025/2026 mean by
  0.0015–0.0021, even without the market (`mo25` .12085 vs .11419). `stack_ridge` worsens it by 0.00029. Two effects push in opposite directions:
  - **Men.** The stack learns a temperature of 0.81–0.86, so the model shrinks toward 0.5. That matches the fixed
    0.9 multiplier in `MENS_ALLYEARS.md` and helps the upset-heavy 2021–2024. It hurts the chalky 2025 (men's
    model-only .14315 vs .13424).
  - **Women.** History moves the original-pipeline logit's weight to about 0. Its walk-forward Brier is worse than
    the narrow logit in 7 of 9 HIST seasons. In 2025 and 2026, though, it is the best member by a wide margin:
    .0931 / .0923 vs .1006 / .1039 for the narrow logit.

  So the women's 0.4 weight on the original logit is supported by the target years and not by history.
  `women_recommended` was selected with the targets in view (README: "hist improves 0.000077"), and this stack is
  the first independent check of that choice. It is a risk to know about, not a reason to switch: neither side has
  enough seasons to settle it.
- **C. GBM probability mappings tie or fail.**
  - `xgb_normal` matches `ref` on the 2025/2026 mean to 5 decimals and is slightly better on history (−0.00007).
    That is not a clear win, since 2026 is +0.00001 worse, so the spline stays.
  - `xgb_binary` is the only config that improves the targets (−0.00007). It comes from women's 2026 and men's
    2025, and it fails the guard (+0.00149 v21–24, worse in 7 of 9 seasons).
  - Fitting the tree directly to the binary outcome throws away the margin information that the regression +
    calibration recipe uses.

### Scope and caveats

- All differences except stacking's are below 0.0005. PR #66 measured season noise of about ±0.03, so these
  differences are indistinguishable from ties.
- 2025 and 2026 were used to *judge* the candidates, not to fit them. They are still not untouched holdouts,
  because `ref` itself was selected on them.
- Not tested:
  - other learning rates η and other ridge strengths (deliberately not tuned);
  - stacking with extra members (Vilnius, ridge-margin), which `VILNIUS.md` already partly covers;
  - joint-gender stacking;
  - LightGBM or CatBoost as replacements for xgb_spline.

## Recommendation

Keep `configs/final.py` / `configs/women_recommended.py` as they are. Hedge and stacking show that the
history-optimal blend differs from the target-optimal one. The disagreement is in the women's original-logit
weight and the men's temperature, and more tournaments are needed before switching. If 2027 again rewards
shrinkage and the narrow women's logit, `stack_ridge` is the conservative candidate to revisit: it passes the
guard and improves v21–24 by 0.00116.

## Reproduce

```sh
export MANIA_DATA=/Users/spro/code/March-Madness-Predictor/data/kaggle/mirror26 GRID_JOBS=5 OMP_NUM_THREADS=1
uv run --extra training python -m kaggle_tuning.strat_reeval kaggle_tuning/configs/strat_reeval.py   # ~45 s, writes the results JSON
uv run --extra training python -m unittest tests.test_strat_reeval
```

Code: `strat_reeval.py`, with a 4-line opt-in hook in `backtest.fit_predict`. Configs: `configs/strat_reeval.py`.
Tests: `tests/test_strat_reeval.py`, which covers the Hedge and stacking maths, symmetry, σ recovery, and a
leakage check that the combiners read outcomes only from seasons before the target. A rerun reproduced every
number in the table exactly.
