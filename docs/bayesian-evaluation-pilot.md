# Bayesian evaluation pilot: 2024–2025

This records the first real historical evaluation on the local CBBD caches.
The configuration uses 2023 as initial baseline-training history and evaluates
2024 and 2025, including First Four games. No ensemble score file was available,
so the existing matchup comparison remains unmeasured.

Run settings: 4,000 posterior draws, RNG seed 42, rating prior SD 1.5, home prior
SD 1.0, 10 calibration bins. These settings were not tuned on these results.
Full artifacts are saved locally under
`model_outputs/bayesian_evaluation/2024-2025-verified/` and are ignored by Git.
To reproduce with the same regular-season caches and a new output directory:

```bash
uv run --extra bayesian python run_bayesian_evaluation.py \
  --config docs/bayesian-evaluation.example.json \
  --output-dir model_outputs/bayesian_evaluation/reproduced-1 --fetch-results
```

The raw result CSVs are cached under `data/<season>/tournament_games.csv`.
The fetch flag makes no request when those caches already exist.
See [the workflow guide](bayesian-evaluation.md) for input schemas, comparison
requirements and how to expand the seasons.

This evaluates individual held-out NCAA game probabilities, including First Four games. It does not evaluate championship or advancement probabilities.

## Results

| Season | Model | Games | Missing | Brier ↓ | Log loss ↓ | Accuracy ↑ | ECE ↓ |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2024 | bayesian | 67 | 0 | 0.2139 | 0.6088 | 0.6119 | 0.1152 |
| 2024 | seed | 67 | 0 | 0.1949 | 0.5702 | 0.6567 | 0.0656 |
| 2024 | matchup | 0 | 67 | — | — | — | — |
| 2025 | bayesian | 67 | 0 | 0.1733 | 0.5209 | 0.7164 | 0.1429 |
| 2025 | seed | 67 | 0 | 0.1618 | 0.4959 | 0.7761 | 0.0980 |
| 2025 | matchup | 0 | 67 | — | — | — | — |
| pooled | bayesian | 134 | 0 | 0.1936 | 0.5648 | 0.6642 | 0.0860 |
| pooled | seed | 134 | 0 | 0.1784 | 0.5331 | 0.7164 | 0.0481 |
| pooled | matchup | 0 | 134 | — | — | — | — |

## Comparisons on identical games

- seed: 134 shared games; 0 excluded. Bayesian minus baseline Brier = +0.0153; log loss = +0.0318. Negative differences favor Bayesian.
- matchup: unavailable; no comparison can be made.

Bayesian probabilities scored worse than the seed baseline on both pooled Brier score and log loss in this run.

These are descriptive comparisons, not significance tests. They do not establish that the Bayesian model improves future forecasts or is ready to become the default.

## Coverage and fitting boundaries

- 2024: 1964 unique training games, 422 teams, 67 tournament games, 68 entrants. Entrant training-game counts: 31–36. 311 fitted teams have fewer than 10 observed games.
- 2025: 1902 unique training games, 409 teams, 67 tournament games, 68 entrants. Entrant training-game counts: 29–35. 306 fitted teams have fewer than 10 observed games.

## Baseline provenance

- 2024: earlier tournament seasons [2023]; seed: fitted on earlier tournaments; matchup: unavailable: fewer than 10 earlier games with supplied ensemble scores.
- 2025: earlier tournament seasons [2023, 2024]; seed: fitted on earlier tournaments; matchup: unavailable: fewer than 10 earlier games with supplied ensemble scores.

## Calibration

`calibration.csv` contains equal-width probability bins, counts, mean predicted probability and observed win rate for every season and pooled model. Empty bins are retained. ECE weights absolute bin gaps by game count; its value depends on binning and sample size and is not a calibration guarantee. Probabilities and outcomes refer to team A, ordered by provider ID independently of the outcome.

## Limitations and interpretation

- Regular-season caches focus on tournament entrants. Opponent schedules may be incomplete; Division I membership and full-league coverage are not established.
- Training uses regular-season games strictly before each explicit cutoff. Start times do not establish completion or historical data availability; later corrections remain possible.
- Bayesian uncertainty uses a Laplace approximation and fixed priors. Probability credible intervals are saved but interval coverage is not evaluated against binary game outcomes.
- Seed coefficients and the existing score-plus-seed logistic model are trained only on earlier configured tournaments. Future-season results are never baseline training data.
- Existing matchup comparison requires actual ensemble scores with declared pre-cutoff, earlier-trained provenance. Declarations cannot independently prove upstream feature or ensemble-selection provenance. Missing scores are reported; no scores are invented.
- Missing baseline predictions change the comparison sample. Use the saved identical-game comparisons rather than comparing model-wide metrics with different coverage.
- Games within a tournament share teams and are dependent. Multiple seasons, uncertainty analysis and settings fixed before evaluation are needed for stronger conclusions.
- Accuracy chooses team A at exactly 0.5. Log loss clips probabilities to [1e-12, 1-1e-12].

## Artifacts

`predictions.csv`, `metrics.json`, `calibration.csv`, `coverage.json`, `baseline_training.json`, `run.json`, and per-season posterior NPZ artifacts preserve the inputs, fitted state, predictions and diagnostics. `run.json` records file hashes and configuration.

## Recorded pooled calibration bins

| Model | Probability bin | Games | Mean prediction | Observed win rate |
| --- | --- | ---: | ---: | ---: |
| bayesian | 0.0–0.1 | 0 | — | — |
| bayesian | 0.1–0.2 | 10 | 0.1502 | 0.0000 |
| bayesian | 0.2–0.3 | 15 | 0.2583 | 0.1333 |
| bayesian | 0.3–0.4 | 14 | 0.3643 | 0.3571 |
| bayesian | 0.4–0.5 | 28 | 0.4571 | 0.6071 |
| bayesian | 0.5–0.6 | 22 | 0.5433 | 0.5909 |
| bayesian | 0.6–0.7 | 22 | 0.6368 | 0.5909 |
| bayesian | 0.7–0.8 | 13 | 0.7394 | 0.7692 |
| bayesian | 0.8–0.9 | 7 | 0.8324 | 1.0000 |
| bayesian | 0.9–1.0 | 3 | 0.9230 | 1.0000 |
| seed | 0.0–0.1 | 0 | — | — |
| seed | 0.1–0.2 | 9 | 0.1359 | 0.0000 |
| seed | 0.2–0.3 | 21 | 0.2522 | 0.2381 |
| seed | 0.3–0.4 | 13 | 0.3648 | 0.3077 |
| seed | 0.4–0.5 | 13 | 0.4506 | 0.4615 |
| seed | 0.5–0.6 | 29 | 0.5368 | 0.5172 |
| seed | 0.6–0.7 | 16 | 0.6421 | 0.6875 |
| seed | 0.7–0.8 | 18 | 0.7500 | 0.8333 |
| seed | 0.8–0.9 | 15 | 0.8504 | 0.9333 |
| seed | 0.9–1.0 | 0 | — | — |

Bin upper bounds are exclusive except the last bin, which includes 1.0.
Small bin counts limit what can be inferred from individual gaps.

## Tournament input fingerprints

| Season | Result CSV SHA-256 |
| --- | --- |
| 2023 | `cc694741249e9b65a43df95cb2095f6bb0fb0cb207932053363eb3200dd3096d` |
| 2024 | `9ce2325a2b23114b4ce300ac0902709583b4205354532736dbf77a643b7b6ced` |
| 2025 | `10f6eeaf273983d2d78885f3bc3d1caa64c09fbf92790795294fa44646cf8376` |

The run manifest records hashes of every regular-season input as well.

## Validation

All 50 repository tests passed. Pooled historical Brier score, log loss and
accuracy were also independently checked against scikit-learn on the 134
saved predictions. No Monte Carlo simulations or pipeline model changes
were needed for this evaluation.
