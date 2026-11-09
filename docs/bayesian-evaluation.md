# Historical Bayesian evaluation

`run_bayesian_evaluation.py` evaluates held-out **individual NCAA game win
probabilities**. It audits regular-season coverage, fits one Bayesian model per
evaluation season, saves predictions, measures calibration, and compares
probabilities against earlier-trained baselines. It does not run Monte Carlo
or change the pipeline's model selection.

## Run the historical pilot

```bash
uv sync --extra bayesian
uv run --extra bayesian python run_bayesian_evaluation.py \
  --config docs/bayesian-evaluation.example.json \
  --output-dir model_outputs/bayesian_evaluation/pilot-1 \
  --fetch-results
```

The example uses 2023 results to train the initial seed baseline and evaluates
2024 and 2025. Cutoffs are midnight UTC before each tournament's First Four;
the window includes all 67 games, including First Four. The 2025 seed baseline
uses 2023 and 2024 results. Settings must be fixed before using the results to
assess performance; rerunning tuned settings on these seasons is exploratory.

`--fetch-results` explicitly enables requests to the existing CBBD `/games`
endpoint using `CBBD_API_KEY` from the environment or `.env`. It fetches only
missing `data/<season>/tournament_games.csv` files, one request per missing
season. It saves a source timestamp and file hash beside each cache. It does
not fetch or overwrite regular-season team caches. After caching the results,
omit `--fetch-results` to run offline. API errors fail clearly rather than
silently skipping a season. Use a new output directory for every run, including
after a failed run; previous outputs are never replaced.

The first real run's findings are saved in
[the pilot report](bayesian-evaluation-pilot.md). The existing matchup comparison
is explicitly unavailable in that pilot because no eligible ensemble scores
were supplied.

## Inputs and coverage

The configuration has `seasons` and `evaluation_seasons`. Each season needs:

```json
{
  "season": 2025,
  "cutoff": "2025-03-18T00:00:00Z",
  "until": "2025-05-01T00:00:00Z",
  "expected_games": 67
}
```

Only `evaluation_seasons` need regular-season caches at
`data/<season>/*/*_game_stats.csv`. All configured seasons need tournament
results, including the earlier years used to train baselines. Add more seasons
to expand the benchmark. Explicit game counts support historical formats and
exceptional tournaments; do not assume every year has 67 games or an NCAA
tournament. Set the count to the independently verified complete result count,
not the number of rows that happened to download.

Tournament results use the CBBD `/games` CSV schema:

| Columns | Purpose |
| --- | --- |
| `id` (or `gameId`), `season`, `startDate` | Stable game ID, season, timezone-aware start |
| `homeTeamId`, `awayTeamId` (or `homeId`, `awayId`) | Stable provider IDs, matched to training caches |
| `homeTeam`, `awayTeam`, `homeSeed`, `awaySeed` | Display names and seeds from 1 to 16 |
| `homePoints`, `awayPoints`, `neutralSite` | Final score and venue |
| `tournament`, `status` | NCAA selection and final/completed status, when supplied |

The designated tournament CSV is assumed NCAA if the tournament column is
absent; otherwise only rows explicitly marked `NCAA` are selected. A missing
status column is treated as a completed-score historical cache, matching the
existing loader's policy. This is not a live-score feed. Score-derived outcomes
are used; season-end Elo and winner columns are never prediction features.

`--results-dir` accepts a separate root with the same `<season>/tournament_games.csv`
layout. No fuzzy name joins are used. Identical duplicate results are deduplicated;
conflicting IDs, seeds or names, incomplete result counts, invalid scores,
games outside the held-out window, and missing entrant ratings fail clearly.
`coverage.json` is written before fitting and records missing entrants,
overlapping training/result IDs, connected components, and observed game counts
for every fitted team. Counts below `--min-games` (default 10) are warnings in
the audit, not reasons to silently drop games. These counts do not establish
complete league coverage or Division I membership.

## Existing matchup baseline

The seed baseline fits `P(A wins) = sigmoid(beta * (seed_B - seed_A))` using
only earlier configured tournaments. It adds both perspectives and uses
zero-intercept logistic regression with `C=1`. If no earlier tournaments are
configured, seed predictions are explicitly unavailable.

The existing simulator's matchup model uses the ensemble-score difference and
seed difference in the same zero-intercept logistic regression. The evaluator
uses this model specification, sharing its probability calculation with
`shap_monte_carlo.py`, but trains it only on **earlier** tournaments. It requires
at least 10 original training games, matching the existing model's 20 symmetric
row minimum. It does not substitute the simulator's fixed fallback when training
data is missing, and it does not use future seasons as historical training data.

Supply genuine scores from your existing ensemble with:

```bash
uv run --extra bayesian python run_bayesian_evaluation.py \
  --config docs/bayesian-evaluation.example.json \
  --output-dir model_outputs/bayesian_evaluation/with-matchup-1 \
  --matchup-scores path/to/ensemble-scores.csv
```

The score CSV needs these columns, with one row per season/team:

```csv
season,team_id,ensemble_score,as_of,trained_through_season
2023,cbbd:1,0.42,2023-03-13T00:00:00Z,2022
```

This row illustrates the schema only; it is not a real team score. Include
scores for all entrants in both baseline-training and evaluation seasons.
`ensemble_score` must be the existing model's actual score, with its original
normalization. Map names to provider IDs using the caches. `as_of` must describe
the latest feature/score availability and be strictly before that season's
cutoff; `trained_through_season` must be earlier than the scored season. Include
model and ensemble-selection training when making that declaration. If using
`shap_monte_carlo.score_season`, verify feature snapshots and member-weight
selection first: its fallback to the latest saved member weights does not by
itself prove historical availability. Do not relabel a current fit as historical.

Invalid or duplicate score rows fail. Missing training scores and target scores
are recorded explicitly. Without the CSV, Bayesian versus seed still runs,
and the report marks the existing matchup comparison unavailable. This prevents
an unevaluated baseline from appearing to have passed.

## Outputs and interpretation

| Artifact | Contents |
| --- | --- |
| `predictions.csv` | Every held-out game, actual outcome, all available model probabilities, Bayesian 90% probability interval and unavailable-baseline reasons |
| `metrics.json` | Per-season and pooled Brier score, clipped log loss, accuracy, ECE, missing counts, and comparisons on identical game sets |
| `calibration.csv` | Equal-width bins with counts, mean probability and observed win rate; empty bins retained |
| `coverage.json` | Training input exclusions, team counts, schedule connectivity and leakage checks |
| `baseline_training.json` | Earlier seasons, original game counts and fitted baseline coefficients |
| `ratings_<season>.npz` and `.summary.json` | Saved posterior draws, training fingerprint, cutoff and fit diagnostics |
| `run.json` | Configuration, settings, dependency versions, and input file SHA-256 hashes |
| `report.md` | Findings, tables, coverage, baseline availability and limitations |

Lower Brier score and log loss are better. Accuracy uses 0.5 as the decision
threshold and chooses team A on ties. Calibration compares predicted probability
with observed win frequency; ECE is an equal-width-bin summary, not proof of
calibration. `--bins` defaults to 10. Small bins and shared teams make estimates
noisy. The report gives descriptive differences, without claiming statistical
significance. Stronger conclusions require more seasons and uncertainty analysis.

Every comparison scores both models on exactly the same game IDs. Individual
model metrics may cover different samples if inputs are missing, so use the
paired comparison objects for model selection. Inspect per-season results as
well as pooled numbers. Tournament advancement and championship probabilities
still require a separate bracket simulation evaluation.

## Verification

```bash
uv run --extra bayesian python -m unittest discover -s tests
```

Tests exercise a complete synthetic multi-season run, future/target outcome
isolation, cutoff and score-provenance rejection, missing entrants, overlapping
game IDs, partial/conflicting result feeds, unavailable comparisons, calibration
boundary bins, artifact preservation and API caching. Synthetic results validate
the workflow, not the model's predictive quality.
