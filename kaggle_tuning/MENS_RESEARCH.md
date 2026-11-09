# Men’s model research

Recommendation: `kaggle_tuning/configs/mens_best.py:RECOMMENDED`, exact expanded dictionary in [mens_recommendation.json](results/mens_recommendation.json). Keep 85% market, 50% women, sharpening 1.2.

All variants use only regular-season DayNum ≤132, ordinal RankingDayNum ≤133, and seeds; walk-forward training uses tournament years strictly before the scored year. Women remain exactly BASE. `v21–24` below is the combined-gender score used by the shared acceptance rule; `vM` is men only. Targets in the ablation table are model only.

| Ablation | hM | v21–24 | vM | M25 model | M26 model | Accepted* |
|---|---:|---:|---:|---:|---:|:---:|
| baseline | 0.18810 | 0.17519 | 0.20794 | 0.13424 | 0.14844 | — |
| conference | 0.18797 | 0.17515 | 0.20786 | 0.13296 | 0.14833 | yes |
| dayseed | 0.18860 | 0.17487 | 0.20729 | 0.13523 | 0.15140 | — |
| depth1 | 0.18835 | 0.17575 | 0.20906 | 0.13350 | 0.14881 | — |
| depth3 | 0.18875 | 0.17520 | 0.20795 | 0.13396 | 0.14782 | yes |
| elo5 | 0.18867 | 0.17498 | 0.20752 | 0.13656 | 0.14964 | — |
| form | 0.18750 | 0.17510 | 0.20776 | 0.13719 | 0.14787 | — |
| ft | 0.18812 | 0.17455 | 0.20665 | 0.13561 | 0.15248 | — |
| learned_consensus | 0.18782 | 0.17448 | 0.20650 | 0.13926 | 0.14813 | — |
| logit_C0.01 | 0.18977 | 0.17585 | 0.20928 | 0.13786 | 0.15234 | — |
| logit_C0.1 | 0.19015 | 0.17688 | 0.21134 | 0.13260 | 0.15070 | — |
| logit_C1.0 | 0.19051 | 0.17747 | 0.21253 | 0.13100 | 0.15028 | — |
| more_systems | 0.18743 | 0.17422 | 0.20598 | 0.13819 | 0.14879 | — |
| normal | 0.18823 | 0.17555 | 0.20866 | 0.13603 | 0.14834 | — |
| normal_worst | 0.18763 | 0.17484 | 0.20724 | 0.13564 | 0.14852 | — |
| offdef10 | 0.18800 | 0.17523 | 0.20802 | 0.13612 | 0.14839 | — |
| offdef100 | 0.18822 | 0.17510 | 0.20776 | 0.13492 | 0.14859 | — |
| pool | 0.18781 | 0.17447 | 0.20649 | 0.13315 | 0.14821 | yes |
| pool_worst | 0.18715 | 0.17397 | 0.20548 | 0.13243 | 0.14783 | yes |
| pool_worst_conf | 0.18702 | 0.17395 | 0.20545 | 0.13153 | 0.14820 | yes |
| pool_worst_form | 0.18692 | 0.17408 | 0.20570 | 0.13526 | 0.14692 | yes |
| quality_recency | 0.18869 | 0.17502 | 0.20760 | 0.13550 | 0.15100 | — |
| raddar | 0.18848 | 0.17428 | 0.20610 | 0.13335 | 0.15156 | — |
| raddar_bundle | 0.18794 | 0.17412 | 0.20578 | 0.13483 | 0.15114 | — |
| recent_mean | 0.18799 | 0.17490 | 0.20736 | 0.13592 | 0.14770 | — |
| recent_median | 0.18787 | 0.17479 | 0.20714 | 0.13553 | 0.14792 | — |
| recent_min | 0.18783 | 0.17468 | 0.20693 | 0.13492 | 0.14768 | yes |
| regularized | 0.18744 | 0.17481 | 0.20718 | 0.13502 | 0.14853 | — |
| worst | 0.18746 | 0.17466 | 0.20688 | 0.13351 | 0.14816 | yes |

*Acceptance compares full blended target-year mean at fixed 85% market against BASE, with hist and v21–24 deterioration ≤0.0005. Baseline is a reference, not a candidate. Eight candidates pass. No promising target-improving candidate failed the history guard; pure logistic models failed history and did not improve blended mean. More Massey systems improved history but failed the target test.

## Final grid (85% market; standard backtest)

| Config | hist | hM | hW | v21–24 | 2025 | M25 | W25 | 2026 | M26 | W26 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 0.16430 | 0.18810 | 0.14056 | 0.17519 | 0.10992 | 0.12614 | 0.09370 | 0.12267 | 0.14458 | 0.10076 |
| pool_worst_conf | 0.16376 | 0.18702 | 0.14056 | 0.17395 | 0.10975 | 0.12579 | 0.09370 | 0.12265 | 0.14453 | 0.10076 |
| pool_worst | 0.16383 | 0.18715 | 0.14056 | 0.17397 | 0.10982 | 0.12593 | 0.09370 | 0.12265 | 0.14454 | 0.10076 |
| conference | 0.16423 | 0.18797 | 0.14056 | 0.17515 | 0.10982 | 0.12593 | 0.09370 | 0.12263 | 0.14450 | 0.10076 |

## Market sensitivity

The observed two-target-year optimum is 80% (grid resolution 5 points), conditional on market sharpening 1.2. This is descriptive tuning on the evaluation years. No pre-2025 market data was reconstructed, so retain the established 85% setting. The difference is tiny; it does not justify a separately accepted market-only change.

| Market weight | M25 | M26 | Men mean |
|---|---:|---:|---:|
| 70% | 0.125578 | 0.144849 | 0.135213 |
| 75% | 0.125607 | 0.144728 | 0.135168 |
| 80% | 0.125679 | 0.144623 | 0.135151 |
| 85% | 0.125793 | 0.144533 | 0.135163 |
| 90% | 0.125946 | 0.144458 | 0.135202 |
| 95% | 0.126136 | 0.144398 | 0.135267 |
| 100% | 0.126360 | 0.144353 | 0.135357 |

The recommendation improves model accuracy, not independence: model/market probability correlations rise from 0.9720/0.9815 to 0.9724/0.9838. Error correlations likewise rise from 0.9804/0.9915 to 0.9809/0.9924. The feature gain is small, and the target years were examined across 29 configurations; the historical guard is not an untouched holdout.

## Interpretation and scope

- `pool`: tournament teams plus direct regular-season victors get separate quality parameters; others share one pool, retaining games touching the kept teams. Symmetric game rows use ridge alpha 2, equivalent to alpha 1 on one row per game.
- `dayseed` uses exp(1.5 × (DayNum−132)/132), multiplied by 1 plus both seeded-team strengths (17−seed)/16. `ft` adds half the difference in missed free throws. These are transparent implementations inspired by the write-up, not exact reproductions of unavailable 2026 code.
- `elo5` replays each season five times with point-margin residual updates (gain .08, home advantage 3); no cross-season carry. Conference quality fits inter-conference games only.
- `offdef10/100`: separate offensive and opposing defensive effects in a ridge regression of points per 100 possessions, plus home advantage; alpha 10 or 100.
- `recent_*`: ordinal mean/median/min in days 119–133, then log transform. Normal scores use the inverse Gaussian CDF of each season’s team-count-scaled rank. The learned consensus adds nine normal-system features with logistic C=.01, alongside the tree component; missing values use the existing zero imputation.
- Form uses opponent-adjusted performance since day 100, its change from whole-season quality, and its variability. It is only an availability/form proxy, not verified injury information.
- Torvik was excluded before fitting: historical files contain postseason records. See [source audit](TORVIK_PROVENANCE.md). No score is reported for a leaky feature.

## Reproduction and verification

Set MANIA_DATA to `/Users/spro/code/March-Madness-Predictor/data/kaggle/mirror26` and GRID_JOBS=5. Run `uv run python -m kaggle_tuning.mens_research` for all ablations, `uv run python -m kaggle_tuning.grid kaggle_tuning/configs/mens_best.py` for the recommendation, and `uv run python -m kaggle_tuning.mens_features` for the cutoff-invariance self-check. The research runner caches predictions in `/tmp/mania_mens_research_cache`; clear this cache when inputs or feature code change. Raw per-year predictions/configs are in `results/mens/`.

Cutoff-invariance self-check passed after injecting extreme day-200 observations. Original `configs/final.py` was not edited; its final regression rerun reproduced 0.1099/0.1227. The cancelled VCU/Oregon game means 2021 has 62 men’s and 63 women’s observations; combined history uses observation-weighted scores as the standard backtest does.
