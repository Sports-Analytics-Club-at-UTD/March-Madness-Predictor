# Expanded Bayesian evaluation: 12 seasons, 801 games

**The current Bayesian model trails both available baselines.** Across the same
801 games, Bayesian Brier score is 0.2088, versus 0.1998 for seeds and 0.1918
for the Ridge SHAP matchup baseline. Bayesian has lower Brier score in 2 of 12
seasons versus seeds and 1 of 12 versus the matchup baseline. These results
support keeping the current Bayesian model experimental while improving data
coverage and the model before changing the pipeline default.

The matchup comparison uses the existing score-plus-seed logistic specification
with **one fixed member, Ridge SHAP**. Only Ridge artifacts are available in this
checkout. This does not evaluate an absent full selected ensemble recipe, and
these SHAP scores are not Ridge's raw predicted tournament-win totals.

The baseline history starts in 2009; evaluated seasons are 2010–2016, 2019 and
2022–2025. All three models predicted all 801 games, with no missing predictions
or game exclusions inside this cohort. Cutoffs are the first NCAA tipoff, with
training games strictly before that instant. The 2010 tournament has 64 games;
each other evaluated tournament has 67, including First Four.

Bayesian priors, 4,000 draws, RNG seed 42 and 10 calibration bins are unchanged
from the pilot. The comparison was expanded after reviewing the pilot, so it
is retrospective and not an independent confirmatory test. Seed baseline values
for 2024–2025 differ from the pilot because the earlier training history expanded.

Player snapshot availability relies on each cached manifest's declared
pre-tournament scope; no independent point-in-time archive was verified.
Raw feature files, fitted artifact metadata and their hashes are recorded,
but historical corrections and all upstream tuning decisions cannot be
independently reconstructed from those declarations.

Full artifacts are saved locally in
`model_outputs/bayesian_evaluation/expanded-ridge/`; score exports and source
provenance are in `model_outputs/bayesian_evaluation/historical-ridge-scores/`.
Both directories are ignored by Git. Use the [reproduction instructions](bayesian-evaluation.md#export-historical-scores-and-run-the-expanded-comparison)
with new output directories to repeat the workflow.

This evaluates individual held-out NCAA game probabilities, including First Four games. It does not evaluate championship or advancement probabilities.

## Results

| Season | Model | Games | Missing | Brier ↓ | Log loss ↓ | Accuracy ↑ | ECE ↓ |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2010 | bayesian | 64 | 0 | 0.1911 | 0.5659 | 0.7031 | 0.0996 |
| 2010 | seed | 64 | 0 | 0.2122 | 0.6114 | 0.6719 | 0.1518 |
| 2010 | matchup | 64 | 0 | 0.2017 | 0.6012 | 0.6875 | 0.1495 |
| 2011 | bayesian | 67 | 0 | 0.2256 | 0.6353 | 0.6418 | 0.1333 |
| 2011 | seed | 67 | 0 | 0.2140 | 0.6091 | 0.6567 | 0.1469 |
| 2011 | matchup | 67 | 0 | 0.2026 | 0.5798 | 0.6418 | 0.1500 |
| 2012 | bayesian | 67 | 0 | 0.2085 | 0.5995 | 0.6418 | 0.1135 |
| 2012 | seed | 67 | 0 | 0.1967 | 0.5798 | 0.7015 | 0.0910 |
| 2012 | matchup | 67 | 0 | 0.1886 | 0.5563 | 0.7313 | 0.1127 |
| 2013 | bayesian | 67 | 0 | 0.2083 | 0.6033 | 0.6866 | 0.1342 |
| 2013 | seed | 67 | 0 | 0.2130 | 0.6143 | 0.6716 | 0.1142 |
| 2013 | matchup | 67 | 0 | 0.1899 | 0.5528 | 0.6866 | 0.0607 |
| 2014 | bayesian | 67 | 0 | 0.2188 | 0.6221 | 0.6119 | 0.1233 |
| 2014 | seed | 67 | 0 | 0.2188 | 0.6193 | 0.6269 | 0.1281 |
| 2014 | matchup | 67 | 0 | 0.2141 | 0.6104 | 0.6567 | 0.1030 |
| 2015 | bayesian | 67 | 0 | 0.1941 | 0.5684 | 0.6418 | 0.1860 |
| 2015 | seed | 67 | 0 | 0.1780 | 0.5354 | 0.7463 | 0.1145 |
| 2015 | matchup | 67 | 0 | 0.1625 | 0.4966 | 0.8060 | 0.1526 |
| 2016 | bayesian | 67 | 0 | 0.2111 | 0.6073 | 0.6866 | 0.1077 |
| 2016 | seed | 67 | 0 | 0.2043 | 0.5942 | 0.6567 | 0.1189 |
| 2016 | matchup | 67 | 0 | 0.1963 | 0.5773 | 0.7313 | 0.1082 |
| 2019 | bayesian | 67 | 0 | 0.1921 | 0.5573 | 0.6418 | 0.1472 |
| 2019 | seed | 67 | 0 | 0.1791 | 0.5306 | 0.6716 | 0.0888 |
| 2019 | matchup | 67 | 0 | 0.1876 | 0.5502 | 0.7015 | 0.1037 |
| 2022 | bayesian | 67 | 0 | 0.2288 | 0.6479 | 0.6269 | 0.0990 |
| 2022 | seed | 67 | 0 | 0.2189 | 0.6258 | 0.6567 | 0.1341 |
| 2022 | matchup | 67 | 0 | 0.2024 | 0.5906 | 0.6716 | 0.0948 |
| 2023 | bayesian | 67 | 0 | 0.2394 | 0.6749 | 0.5821 | 0.1169 |
| 2023 | seed | 67 | 0 | 0.2069 | 0.6039 | 0.6716 | 0.0978 |
| 2023 | matchup | 67 | 0 | 0.2111 | 0.6225 | 0.7015 | 0.1052 |
| 2024 | bayesian | 67 | 0 | 0.2139 | 0.6088 | 0.6119 | 0.1152 |
| 2024 | seed | 67 | 0 | 0.1946 | 0.5683 | 0.6567 | 0.0718 |
| 2024 | matchup | 67 | 0 | 0.1869 | 0.5498 | 0.7164 | 0.1226 |
| 2025 | bayesian | 67 | 0 | 0.1733 | 0.5209 | 0.7164 | 0.1429 |
| 2025 | seed | 67 | 0 | 0.1612 | 0.4940 | 0.7761 | 0.0953 |
| 2025 | matchup | 67 | 0 | 0.1583 | 0.4848 | 0.7612 | 0.1238 |
| pooled | bayesian | 801 | 0 | 0.2088 | 0.6011 | 0.6492 | 0.0411 |
| pooled | seed | 801 | 0 | 0.1998 | 0.5821 | 0.6804 | 0.0387 |
| pooled | matchup | 801 | 0 | 0.1918 | 0.5642 | 0.7079 | 0.0268 |

## Comparisons on identical games

- seed: 801 shared games; 0 excluded. Bayesian minus baseline Brier = +0.0090; log loss = +0.0190. Negative differences favor Bayesian.
  Season bootstrap 95% intervals: Brier [+0.0017, +0.0157], log loss [+0.0034, +0.0331]. Bayesian has lower Brier in 2/12 seasons and lower log loss in 2/12 seasons.
- matchup: 801 shared games; 0 excluded. Bayesian minus baseline Brier = +0.0170; log loss = +0.0369. Negative differences favor Bayesian.
  Season bootstrap 95% intervals: Brier [+0.0100, +0.0229], log loss [+0.0193, +0.0506]. Bayesian has lower Brier in 1/12 seasons and lower log loss in 1/12 seasons.

Bayesian probabilities scored worse than the seed baseline on both pooled Brier score and log loss in this run.

These are descriptive comparisons, not significance tests. They do not establish that the Bayesian model improves future forecasts or is ready to become the default.

## Coverage and fitting boundaries

- 2010: 1853 unique training games, 380 teams, 64 tournament games, 65 entrants. Entrant training-game counts: 29–36. 277 fitted teams have fewer than 10 observed games.
- 2011: 1889 unique training games, 375 teams, 67 tournament games, 68 entrants. Entrant training-game counts: 30–35. 281 fitted teams have fewer than 10 observed games.
- 2012: 1906 unique training games, 387 teams, 67 tournament games, 68 entrants. Entrant training-game counts: 30–36. 286 fitted teams have fewer than 10 observed games.
- 2013: 1943 unique training games, 389 teams, 67 tournament games, 68 entrants. Entrant training-game counts: 28–36. 287 fitted teams have fewer than 10 observed games.
- 2014: 1926 unique training games, 385 teams, 67 tournament games, 68 entrants. Entrant training-game counts: 30–35. 281 fitted teams have fewer than 10 observed games.
- 2015: 1929 unique training games, 396 teams, 67 tournament games, 68 entrants. Entrant training-game counts: 29–35. 294 fitted teams have fewer than 10 observed games.
- 2016: 1915 unique training games, 405 teams, 67 tournament games, 68 entrants. Entrant training-game counts: 28–35. 305 fitted teams have fewer than 10 observed games.
- 2019: 1954 unique training games, 391 teams, 67 tournament games, 68 entrants. Entrant training-game counts: 29–35. 287 fitted teams have fewer than 10 observed games.
- 2022: 1911 unique training games, 406 teams, 67 tournament games, 68 entrants. Entrant training-game counts: 28–35. 304 fitted teams have fewer than 10 observed games.
- 2023: 1939 unique training games, 413 teams, 67 tournament games, 68 entrants. Entrant training-game counts: 29–36. 307 fitted teams have fewer than 10 observed games.
- 2024: 1964 unique training games, 422 teams, 67 tournament games, 68 entrants. Entrant training-game counts: 31–36. 311 fitted teams have fewer than 10 observed games.
- 2025: 1902 unique training games, 409 teams, 67 tournament games, 68 entrants. Entrant training-game counts: 29–35. 306 fitted teams have fewer than 10 observed games.

## Baseline provenance

- 2010: earlier tournament seasons [2009]; seed: fitted on earlier tournaments; matchup: fitted on earlier tournaments.
- 2011: earlier tournament seasons [2009, 2010]; seed: fitted on earlier tournaments; matchup: fitted on earlier tournaments.
- 2012: earlier tournament seasons [2009, 2010, 2011]; seed: fitted on earlier tournaments; matchup: fitted on earlier tournaments.
- 2013: earlier tournament seasons [2009, 2010, 2011, 2012]; seed: fitted on earlier tournaments; matchup: fitted on earlier tournaments.
- 2014: earlier tournament seasons [2009, 2010, 2011, 2012, 2013]; seed: fitted on earlier tournaments; matchup: fitted on earlier tournaments.
- 2015: earlier tournament seasons [2009, 2010, 2011, 2012, 2013, 2014]; seed: fitted on earlier tournaments; matchup: fitted on earlier tournaments.
- 2016: earlier tournament seasons [2009, 2010, 2011, 2012, 2013, 2014, 2015]; seed: fitted on earlier tournaments; matchup: fitted on earlier tournaments.
- 2019: earlier tournament seasons [2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016]; seed: fitted on earlier tournaments; matchup: fitted on earlier tournaments.
- 2022: earlier tournament seasons [2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2019]; seed: fitted on earlier tournaments; matchup: fitted on earlier tournaments.
- 2023: earlier tournament seasons [2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2019, 2022]; seed: fitted on earlier tournaments; matchup: fitted on earlier tournaments.
- 2024: earlier tournament seasons [2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2019, 2022, 2023]; seed: fitted on earlier tournaments; matchup: fitted on earlier tournaments.
- 2025: earlier tournament seasons [2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2019, 2022, 2023, 2024]; seed: fitted on earlier tournaments; matchup: fitted on earlier tournaments.

## Matchup score provenance

Comparison: **single-member Ridge SHAP + seed**. Members: ridge. This is the existing score-plus-seed matchup specification with those members; it is not evidence about an unavailable full ensemble recipe.

Every member uses a SHAP validation fold from an earlier season; model training seasons are earlier still. Feature construction filters regular-season game starts at the configured cutoff. Player snapshot scope relies on the saved statistics manifest.

- Player snapshot availability relies on the existing statistics manifest; no independent point-in-time archive was verified.
- This uses strictly earlier SHAP validation folds, not a full-history Rank 1 recipe or a same-season fold.
- Saved artifact training metadata is checked; historical data corrections and upstream tuning provenance are not independently established.

## Calibration

`calibration.csv` contains equal-width probability bins, counts, mean predicted probability and observed win rate for every season and pooled model. Empty bins are retained. ECE weights absolute bin gaps by game count; its value depends on binning and sample size and is not a calibration guarantee. Probabilities and outcomes refer to team A, ordered by provider ID independently of the outcome.

## Limitations and interpretation

- Regular-season caches focus on tournament entrants. Opponent schedules may be incomplete; Division I membership and full-league coverage are not established.
- Training uses regular-season games strictly before each explicit cutoff. Start times do not establish completion or historical data availability; later corrections remain possible.
- Bayesian uncertainty uses a Laplace approximation and fixed priors. Probability credible intervals are saved but interval coverage is not evaluated against binary game outcomes.
- The models use different inputs: Bayesian ratings use game win/loss outcomes, while the matchup baseline also uses seeds and SHAP-weighted statistics. Performance differences cannot be attributed solely to Bayesian versus non-Bayesian inference.
- Seed coefficients and the existing score-plus-seed logistic model are trained only on earlier configured tournaments. Future-season results are never baseline training data.
- Existing matchup comparison requires actual ensemble scores with declared pre-cutoff, earlier-trained provenance. Declarations cannot independently prove upstream feature or ensemble-selection provenance. Missing scores are reported; no scores are invented.
- Missing baseline predictions change the comparison sample. Use the saved identical-game comparisons rather than comparing model-wide metrics with different coverage.
- Games within a tournament share teams and are dependent. Multiple seasons, uncertainty analysis and settings fixed before evaluation are needed for stronger conclusions.
- Bootstrap intervals resample entire seasons (2,000 draws, seed 42). They are descriptive and conditional on these seasons; expanding training histories share information, so this is not an independent-season significance test. Two-season intervals are especially unstable.
- Accuracy chooses team A at exactly 0.5. Log loss clips probabilities to [1e-12, 1-1e-12].

## Artifacts

`predictions.csv`, `metrics.json`, `calibration.csv`, `coverage.json`, `baseline_training.json`, `run.json`, and per-season posterior NPZ artifacts preserve the inputs, fitted state, predictions and diagnostics. `run.json` records file hashes and configuration.

## Cohort exclusions

| Season | Reason |
| --- | --- |
| 2008 | Incomplete cached tournament and team-game coverage. |
| 2017 | Manifest marks incomplete tournament coverage (65 cached games). |
| 2018 | Statistics manifest is incomplete; player snapshot cannot be accepted by the strict exporter. |
| 2020 | Tournament cancelled. |
| 2021 | Manifest incomplete; includes a no-contest game requiring a separate outcome policy. |

These exclusions apply to the scored/evaluated cohort. Earlier saved artifacts
retain their declared training and SHAP validation histories; the exporter
records those histories rather than claiming their input coverage was complete.

## Exact score sources

Membership was fixed to `ridge` because it is the only available local member.
No held-out outcomes selected weights or ensemble membership. Each scored
season uses an earlier validation fold, whose model training ends earlier still.
All selected SHAP features were present: missing-weight fraction was zero in
every scored season.

| Score season | SHAP validation season | Last model training season | Snapshot |
| --- | ---: | ---: | --- |
| 2009 | 2008 | 2007 | 2009-03-17T23:29:59.999000+00:00 |
| 2010 | 2009 | 2008 | 2010-03-16T23:29:59.999000+00:00 |
| 2011 | 2010 | 2009 | 2011-03-15T22:29:59.999000+00:00 |
| 2012 | 2011 | 2010 | 2012-03-13T22:39:59.999000+00:00 |
| 2013 | 2012 | 2011 | 2013-03-19T22:39:59.999000+00:00 |
| 2014 | 2013 | 2012 | 2014-03-18T22:39:59.999000+00:00 |
| 2015 | 2014 | 2013 | 2015-03-17T22:39:59.999000+00:00 |
| 2016 | 2015 | 2014 | 2016-03-15T22:39:59.999000+00:00 |
| 2019 | 2018 | 2017 | 2019-03-19T22:39:59.999000+00:00 |
| 2022 | 2021 | 2019 | 2022-03-15T22:39:59.999000+00:00 |
| 2023 | 2022 | 2021 | 2023-03-14T22:39:59.999000+00:00 |
| 2024 | 2023 | 2022 | 2024-03-19T22:39:59.999000+00:00 |
| 2025 | 2024 | 2023 | 2025-03-18T22:39:59.999000+00:00 |

## Reproducibility fingerprints

- Configuration SHA-256: `c9b5d88218f7eb2ca415cfec9be160a9f7fcabc4378d07153fe0669a4f2cbefc`.
- Exported score CSV SHA-256: `5e87cde72b34a858ff79a7c129f067aeda46d3821b60848c4a6dc851b51fc501`.
- `run.json` records every tournament and regular-season input file hash.
- `matchup_score_provenance.json` records every feature cache, manifest, SHAP CSV and inference bundle hash.

## Validation

- All 60 repository tests pass, including causal cutoff, earlier artifact selection, snapshot availability, provenance hash and complete-comparison checks.
- All 801 game IDs are unique. Pooled Brier score, log loss and accuracy agree with scikit-learn for all three models.
- Every exported score matches the existing scorer across all 13 input seasons when supplied the same earlier weights.
- Future malformed numeric values, missing future IDs and postseason rows cannot alter historical features. Same-season and future SHAP fallback are rejected.
