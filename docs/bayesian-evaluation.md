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

The follow-up [expanded report](bayesian-evaluation-expanded.md) evaluates all
three models on 801 identical games across 12 seasons, using the strict score
export below. The available matchup baseline has one member, Ridge; the full
selected ensemble recipe is not present in this checkout.

## Export historical scores and run the expanded comparison

```bash
uv run --extra bayesian python run_historical_matchup_scores.py \
  --config docs/bayesian-evaluation.expanded.json \
  --members ridge \
  --output-dir model_outputs/bayesian_evaluation/historical-scores-1

uv run --extra bayesian python run_bayesian_evaluation.py \
  --config docs/bayesian-evaluation.expanded.json \
  --matchup-scores model_outputs/bayesian_evaluation/historical-scores-1/ensemble_scores.csv \
  --matchup-provenance model_outputs/bayesian_evaluation/historical-scores-1/provenance.json \
  --output-dir model_outputs/bayesian_evaluation/expanded-1 \
  --fetch-results --require-complete-comparisons
```

Both output directories must be new. Exporting scores requires only the
Bayesian dependencies, not the plotting/SHAP training extras. It reads existing
trusted local SHAP CSVs and inference bundles under `model_outputs/<member>/`;
it does not retrain them. The current checkout contains only Ridge artifacts,
so this run is explicitly a **single-member Ridge SHAP + seed** comparison.
The existing score-plus-seed logistic model is fitted on earlier tournaments.
It is not a test of the absent full ensemble or of Ridge's raw win predictions.

Membership is fixed in `matchup_members` in the configuration and must match
`--members`. There is no selection of the globally best combination using held-out
tournament outcomes. If other members have compatible inference metadata and
SHAP exports, configure a fixed member list before running that comparison.

For each scored season, the exporter selects the latest SHAP validation fold
**strictly earlier** than that season, verifies its bundle's training years,
and rejects same-season/future fallback. This is deliberately stricter than
`shap_monte_carlo.score_season`'s historical fallback. Exported
`trained_through_season` includes the SHAP validation-feature season, not just
the underlying model's last training season. Game/player feature formulas,
population standardization, per-member SHAP dot-product normalization and member
averaging follow the existing scorer; the dot-product implementation is shared.

Game features use only regular-season rows with starts strictly before the
cutoff. Numeric types are inferred after filtering so a future malformed value
cannot change historical feature selection. Season-end Elo is excluded, as in
the existing scorer. Player aggregates lack per-game timestamps, so export
requires a complete same-season manifest declaring regular-season statistics
and a snapshot time before the scoring cutoff. This relies on that declaration;
it does not independently establish point-in-time player-data availability.
Missing player files, invalid team identities, contradictory fold metadata,
unusable weights and constant scores fail rather than silently dropping teams.

The expanded configuration uses the first NCAA tipoff as the exclusive training
cutoff. Declared snapshots are one millisecond earlier. It uses 2009 for initial
baseline history and evaluates 2010–2016, 2019 and 2022–2025. Incomplete cached
manifests exclude 2017, 2018 and 2021; 2020 was cancelled. Selection reasons are
recorded in the configuration. This is a retrospective extension of the pilot,
not an independent confirmatory test.

`provenance.json` records exact member folds, model training years, missing
feature/weight fractions, snapshot declarations, and file hashes for SHAP
weights, bundles, feature caches and manifests. `--matchup-provenance` verifies
the score/config hashes and copies that provenance into the evaluation run.
`--require-complete-comparisons` exits with an error if any model lacks a
prediction; the saved report still identifies the gaps. Use it when claiming
that all three models were compared on the full evaluation set.

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
| `matchup_score_provenance.json` | Checked score-export provenance, when supplied |
| `report.md` | Findings, tables, coverage, baseline availability and limitations |

Lower Brier score and log loss are better. Accuracy uses 0.5 as the decision
threshold and chooses team A on ties. Calibration compares predicted probability
with observed win frequency; ECE is an equal-width-bin summary, not proof of
calibration. `--bins` defaults to 10. Small bins and shared teams make estimates
noisy. The report gives descriptive differences, without claiming statistical
significance. It also reports 95% percentile intervals by resampling whole
seasons (2,000 draws, seed 42) and counts seasons where Bayesian Brier score and
log loss improve. These intervals preserve games within each tournament but
share information through expanding baseline-training histories. They are not
independent-season significance tests; two-season intervals are especially
unstable. Stronger conclusions require better data coverage and independent
future validation.

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

Historical score tests also reject future/same-season SHAP fallback, contradictory
training metadata, post-cutoff or incomplete player snapshots, missing player
files, unusable weights and changes to the fixed member list. Future text and
postseason rows cannot change historical feature values or their numeric types.
