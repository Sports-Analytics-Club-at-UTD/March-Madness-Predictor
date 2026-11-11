# Schedule coverage and isolated Bayesian experiments

The 801-game CBBD/Ridge benchmark could **not** be reproduced in this checkout.
A separate retrospective Kaggle experiment completed; it is not a reproduction
of that benchmark and does not justify changing pipeline defaults.

## Reproduction attempted first

On branch `bayesian-model-improve`, before implementation:

```bash
uv run --extra bayesian python run_historical_matchup_scores.py \
  --config docs/bayesian-evaluation.expanded.json --members ridge \
  --output-dir model_outputs/bayesian_evaluation/reproduction-scores
uv run --extra bayesian python run_bayesian_evaluation.py \
  --config docs/bayesian-evaluation.expanded.json \
  --output-dir model_outputs/bayesian_evaluation/reproduction-attempt --fetch-results
```

Exact blockers:

- No `model_outputs/bayesian_evaluation/` artifacts existed before the attempt.
- No `*_game_stats.csv` caches existed under `data/`, and no Ridge artifacts
  existed under `model_outputs/`.
- Historical exporter: `Error: 2009: need nonempty, unique cached team identities`.
- Result acquisition: `Error: --fetch-results requires CBBD_API_KEY in the environment or .env`.

No credential values were printed. Current credentials could not regenerate
CBBD data. Restoring regular schedules alone would not restore Ridge's saved
historical models, feature snapshots and SHAP provenance.

## Coverage acquisition

```bash
uv run --extra bayesian python run_schedule_coverage.py \
  --config docs/bayesian-evaluation.expanded.json \
  --output-dir model_outputs/bayesian_evaluation/league-schedules
```

This explicitly fetches `/games?season=YEAR&seasonType=regular`, once per
evaluation season, without filtering on tournament entrants. It normalizes
home/away records into the existing team-cache format in a separate root, retains
all opponents, and uses the existing loader for cutoff, status, score, identity,
and duplicate validation. Raw responses and source manifests preserve request
parameters, retrieval time, cache hashes, exclusions and per-team game counts.
A season is published only after validation; existing seasons are never replaced.
If a later season fails, earlier successfully published seasons remain. To retry,
use a configuration containing only the remaining seasons and their evaluation list.

The existing evaluator accepts the result with `--data-dir PATH`; point
`--results-dir` at the separate tournament cache root. This acquisition does not
alter player caches, export SHAP scores, or fetch tournament results.

A successful league-wide response does **not** prove provider completeness or
Division I membership. There is no independently verified expected regular-season
game count. Inspect the recorded game counts and low-coverage teams. API schema,
truncation, live credentials and real download behavior remain unverified here;
request shape, conflicting duplicates, validation before publication and overwrite
protection are covered by mocked tests.

## Experiment protocol

```bash
uv run --extra bayesian python run_bayesian_experiments.py \
  --config docs/bayesian-evaluation.expanded.json \
  --plan docs/bayesian-experiments.plan.json \
  --kaggle-dir data/kaggle/mirror26 \
  --output-dir model_outputs/bayesian_evaluation/new-experiment
```

Alternatively use `--data-dir` with CBBD regular and tournament caches in the
existing layout. The output directory must be new. Defaults remain 4,000 posterior
draws, RNG seed 42, rating prior SD 1.5 and home prior SD 1.0.

The plan partitions evaluation seasons into tuning (2010–2016, 2019; 533 games)
and later held-out evaluation (2022–2025; 268 games). Every candidate changes
exactly one setting relative to the full-schedule default. Candidate settings
were fixed before running this experiment. Selection minimizes pooled tuning
Brier score, preferring the baseline on exact ties. `selection.json` is saved
before loading held-out season datasets. Only the selected candidate and baseline
are scored on the held-out set. The seed comparator is trained on configured
2009–2019 tournaments and frozen: none of the four held-out seasons trains it.

Kaggle IDs stay in a separate `kaggle:` namespace. Compact results have day
resolution; the training cutoff is conservatively moved to midnight at the start
of the configured NCAA tipoff day. Games on that day are excluded from training.
Expected tournament counts are enforced, ties/invalid scores and duplicate keys
are rejected, and orientation follows team ID rather than the winner. No CBBD
identity mapping or point-in-time data provenance is claimed.

`protocol.json` records the plan, config, source hashes, draw settings and (in the
current tool) code hashes and dependency versions. Each candidate saves posterior
NPZ files, predictions, per-season/pooled metrics and coverage audits. Failed
outputs are retained. The completed real run below preceded the addition of code
hash/version fields; its protocol contains input/config/plan hashes and settings.

## Actual measured results

Artifacts: `model_outputs/bayesian_evaluation/kaggle-experiment-2/` (ignored by Git).
These use the existing local Kaggle mirror without modifying it. The baseline
and selected candidate have exactly 268 identical unique held-out game keys;
Brier and log loss were independently checked against scikit-learn.

| Tuning candidate | Games | Brier | Log loss |
| --- | ---: | ---: | ---: |
| Full schedules, original priors | 533 | 0.198378 | 0.575788 |
| Rating prior SD 0.75 | 533 | 0.208342 | 0.602799 |
| Rating prior SD 3.0 (selected) | 533 | 0.197475 | 0.573779 |
| Home prior SD 0.5 | 533 | 0.198360 | 0.575747 |
| Entrant schedules only | 533 | 0.205004 | 0.591991 |

The coverage ablation keeps only regular games involving an eventual NCAA entrant,
while retaining the opponent ratings. Full schedules supplied 5,246–5,463 games
per tuning season and zero observed teams below ten games. Entrant-only schedules
supplied 1,818–1,914 games and 243–251 teams below ten games. These are Kaggle
counts, not the missing CBBD cache counts. The coverage difference on tuning
seasons is isolated within this source; it was not separately tested on the later
holdout after seeing the results.

| Held-out model, 2022–2025 | Games | Brier | Log loss | Accuracy | ECE |
| --- | ---: | ---: | ---: | ---: | ---: |
| Full schedules, original priors | 268 | 0.202219 | 0.587921 | 0.664179 | 0.063448 |
| Selected rating prior SD 3.0 | 268 | 0.198479 | 0.580433 | 0.682836 | 0.082645 |
| Frozen earlier-trained seed baseline | 268 | 0.195187 | 0.572235 | 0.701493 | 0.068291 |

The selected candidate reduced held-out Brier by 0.003740 and log loss by
0.007488 versus the original priors. It still trailed seeds by 0.003293 Brier
and 0.008198 log loss. Its ECE worsened. Brier improved in three of four seasons;
2022 worsened (0.226177 to 0.229451). These descriptive results do not establish
significance or future improvement, and no default prior was changed.

## Numerical fix and validation

The first experiment (`kaggle-experiment-1`) stopped in the tuning phase, before
selection or held-out evaluation, on the 2019 stronger-home-prior fit:

```
Rating fit failed convergence checks: CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH; gradient=0.0002227125152165499
```

Large schedules can meet relative objective tolerance before meeting the existing
stationarity requirement. `fit_ratings` now retries that specific case from the
candidate optimum with relative tolerance disabled. It still requires successful
optimization, finite parameters and maximum absolute gradient at most `1e-4`.
Ordinary successful fits, priors and optional integration defaults are unchanged.
The second run completed with the same experiment plan.

Tests cover chronological split rejection, multi-setting ablation rejection,
selection saved before held-out loading, invariance of selection and predictions
to reversed held-out outcomes, frozen seed training, Kaggle cutoff/identity/count
validation, validated acquisition, artifact preservation, and the optimizer retry.
Run `uv run --extra bayesian python -m unittest discover -s tests`.

The historical outcomes were already summarized in earlier project reports, so
this is a retrospective development holdout, not a pristine independent test.
Kaggle includes a different population of regular-season games than CBBD (and
may omit non-Division-I opponents); historical corrections and source provenance
are not independently established. No Ridge comparison, championship evaluation,
Monte Carlo integration, commits or pushes were performed.
