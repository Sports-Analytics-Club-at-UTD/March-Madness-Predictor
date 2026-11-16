# Bayesian team ratings: first implementation

This is a standalone Bayesian Bradley–Terry baseline. It learns a season-static
strength for each team and a shared home-court effect from game results. It does
not replace Ridge, change SHAP features, or automatically enter the ensemble.

For a game viewed from team A:

```text
logit(P(A wins)) = rating[A] - rating[B] + home_indicator * home_advantage
home_indicator = +1 for A at home, -1 for B at home, 0 for neutral
```

The default priors are independent `Normal(0, 1.5²)` team ratings and
`Normal(0, 1²)` home advantage. Prior scales are fixed configuration, not learned
hyperparameters. The priors shrink weakly observed teams toward average and keep
undefeated teams and disconnected schedules finite. A positive home effect is
learned from evidence, not forced by a constraint.

Ratings use log-odds units, not points, Elo units or expected tournament wins.
All rating draws are centered across the fitted team registry; pairwise
differences and matchup predictions are unchanged by centering.

## Inference and uncertainty

This first version uses a **Laplace posterior approximation**, not MCMC. It finds
the maximum a posteriori estimate with SciPy L-BFGS-B, calculates the analytic
negative-log-posterior Hessian, and samples a joint multivariate Gaussian using
its inverse precision. It checks optimizer success, the gradient, positive
definiteness and finite draws. This is the usual mode-and-Hessian construction
described in the [Stan reference](https://mc-stan.org/docs/reference-manual/laplace.html).

Saved summaries report approximate 90% credible intervals. These are conditional
on the model, fixed priors, and observed schedule. They are not validated coverage
guarantees. Gaussian approximation quality can be poor for sparse or strongly
skewed posteriors. Optimizer diagnostics do not establish posterior calibration;
there are no MCMC chains, R-hat or effective-sample-size diagnostics to report.
Drawing more samples improves Monte Carlo precision, not the approximation.

## Input and leakage boundaries

`load_games` reads `data/<season>/*/*_game_stats.csv`, using existing CBBD columns:

| Input | Purpose |
| --- | --- |
| `gameId`, `season`, `startDate` | Deduplication, season and exclusive date cutoff |
| `seasonType` | Fit command includes only `regular` games |
| `teamId`, `opponentId`, `team`, `opponent` | Stable `cbbd:<numeric ID>` registry and display names |
| `neutralSite`, `isHome` | Venue from the row team's perspective |
| `teamStats.points.total`, `opponentStats.points.total` | Determine the winner; margin is not modeled |

Games present in both teams' files are normalized to the same orientation and
counted once. Conflicting duplicates, inconsistent team names, invalid scores,
and tied nonzero scores fail with file and line information. Missing scores,
zero-zero rows and explicitly non-final statuses are excluded and counted.
Files lacking a status column are treated as historical completed-score caches;
they are not a supported live-score ingestion feed.

Only starts **strictly before** an explicit timezone-aware cutoff are eligible.
Later scores and season-end Elo/statistics are not used. A start-time cutoff
does not prove a game had finished by that instant, or prevent later score
corrections: use a completed historical snapshot and a cutoff clear of games
in progress. The fit command intentionally excludes postseason games, including
conference tournaments if their feed classifies them as postseason.

All opponents in supplied files are kept, not just NCAA tournament entrants.
However, this repository's caches focus on tournament teams. Opponents may have
very incomplete schedules, and some are outside Division I. The model does not
claim full-league coverage or automatically filter divisions. For a serious
benchmark, supply complete, consistent Division I regular-season schedules and
choose an explicit policy for non-Division-I opponents. The loader makes no
network requests and does not repair coverage by guessing.

## Commands

Install dependencies from the project root:

```bash
uv sync --extra bayesian
```

Fit a historical season, using a cutoff before its tournament starts:

```bash
uv run --extra bayesian python run_bayesian.py fit \
  --season 2025 --cutoff 2025-03-18T00:00:00Z \
  --draws 4000 --seed 42 \
  --output model_outputs/bayesian/2025-run-1.npz \
  --summary model_outputs/bayesian/2025-run-1.summary.json
```

Choose a new output filename for each run. Existing artifacts and summaries are
not overwritten. The NPZ stores the team registry, joint rating/home draws, and
metadata without Python pickles. Metadata includes cutoff, prior scales, seed,
data-selection counts, training game IDs and a fingerprint of the canonical
training games. Numerical draws reproduce for a fixed seed and identical input
under the same numerical environment; metadata generation timestamps change.
Artifacts remain ignored under `model_outputs/`.

Predict without training again:

```bash
uv run --extra bayesian python run_bayesian.py predict \
  model_outputs/bayesian/2025-run-1.npz \
  --team-a Duke --team-b Houston --venue neutral

uv run --extra bayesian python run_bayesian.py summary \
  model_outputs/bayesian/2025-run-1.npz
```

Names match exactly ignoring case and surrounding whitespace. For ambiguous
names use the saved provider IDs. Unknown teams fail instead of receiving an
invented average rating. Matchup output averages the win probability across
joint draws; its interval describes uncertainty in that probability, not the
binary outcome of the game.

## Held-out evaluation

For multi-season coverage audits, saved game predictions, seed/matchup baseline
comparisons and calibration, use the [historical evaluation runner](bayesian-evaluation.md).
The command below remains available for a single saved fit's basic metrics.

The `evaluate` command requires a directory containing NCAA tournament games in
the **same team-game CSV format**, with `tournament=NCAA`. Supply it separately
from training caches if appropriate:

```bash
uv run --extra bayesian python run_bayesian.py evaluate \
  model_outputs/bayesian/2025-run-1.npz \
  --data-dir path/to/heldout-team-game-caches \
  --until 2025-07-01T00:00:00Z
```

It scores only NCAA games at or after the fitting cutoff and strictly before
`--until`, reporting Brier score, log loss and accuracy. Training IDs, duplicate
games, different seasons and unknown teams are rejected. Existing regular-only
caches will produce a clear no-held-out-games error; there is no automatic API
fetch. A separate `tournament_games.csv` with a different schema is not consumed.

For credible comparisons, repeat across historical seasons with settings chosen
on earlier seasons. Compare against seed and existing matchup baselines on the
same held-out games and inspect calibration. Successful fitting and synthetic
tests do not establish an improvement over the existing model.

## Connection to the existing simulator

```python
from bayesian_ratings import load_ratings

posterior = load_ratings("model_outputs/bayesian/2025-run-1.npz")
# teams64 must use names/IDs present in the saved registry, in bracket index order.
P = posterior.probability_matrix(teams64)
# P[i, j] is the posterior-averaged neutral-site chance team i beats team j.
```

This matrix has the interface accepted by `monte_carlo.simulate_bracket_tree`.
The existing SHAP pipeline is unchanged. For the optional Bayesian simulator,
pipeline stage and forecast JSON export, see [Bayesian bracket Monte Carlo](bayesian-monte-carlo.md).

To preserve shared strength uncertainty across a tournament, select **one joint
posterior draw per simulated tournament** and use
`posterior.probability_matrix(teams64, draw_index=k)` for all its games. A fixed
posterior-average matrix loses that correlation across games. The existing
simulator accepts one fixed matrix per call; use a loop or grouped calls for
per-tournament draws. A scheduled bracket must also be supplied independently;
the existing historical topology parser uses actual winners and is unsuitable
as a pre-tournament schedule source without further work.

`run_bayesian_monte_carlo.py` implements joint-draw simulation using an explicit
pre-tournament schedule and exports run metadata, rating credible intervals,
advancement probabilities and a consistent maximum expected points bracket.
Its JSON follows main's forecast contract; full contract validation runs when
`backend_data` is importable. Team ratings alone are not a bracket.

## Verification

```bash
uv run --extra bayesian python -m unittest discover -s tests -p test_bayesian_ratings.py
uv run --extra training python -m unittest discover -s tests
```

Tests cover opposite-perspective deduplication, cutoff exclusion, bad inputs,
synthetic strength recovery, prior shrinkage for disconnected schedules,
reproducibility, neutral/home symmetry, uncertainty contraction, persistence,
held-out metrics and the command-line workflow.
