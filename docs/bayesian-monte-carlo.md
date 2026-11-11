# Bayesian bracket Monte Carlo and forecast JSON export

`run_bayesian_monte_carlo.py` simulates a pre-tournament bracket from the
[Bayesian team ratings](bayesian-ratings.md) and writes one forecast JSON file
in the contract defined on `main` (`backend_data`, schema `1.0.0`).

This is **optional and experimental**. `run_pipeline.py --all` and every
existing default are unchanged. In the [expanded evaluation](bayesian-evaluation-expanded.md)
the current Bayesian game probabilities trail the seed and Ridge SHAP baselines,
and advancement or championship probabilities have not been evaluated at all.

## Run it

```bash
uv sync --extra bayesian

# Fit ratings before the tournament, then simulate.
uv run --extra bayesian python run_bayesian_monte_carlo.py \
  --season 2025 --cutoff 2025-03-18T00:00:00Z \
  --bracket-schedule data/2025/bracket_schedule.json

# Or reuse an artifact saved by `run_bayesian.py fit`.
uv run --extra bayesian python run_bayesian_monte_carlo.py \
  --season 2025 --ratings model_outputs/bayesian/2025-run-1.npz \
  --bracket-schedule data/2025/bracket_schedule.json

# The same thing as an opt-in pipeline stage.
uv run --extra bayesian python run_pipeline.py --stages bayesian_monte_carlo \
  --season 2025 --bayesian-cutoff 2025-03-18T00:00:00Z
```

Supply exactly one of `--ratings` or `--cutoff`. Other options: `--n-simulations`
(10000), `--seed` (42), `--draws` (4000, fitting only), `--data-dir` (`data`,
fitting only) and `--output-dir` (`model_outputs/bayesian_monte_carlo`).

Outputs, each under a new name so nothing is overwritten:

- `forecast_{season}_{uuid}.json`: the forecast document.
- `bayesian_ratings_{season}_{run_id}.npz`: the fitted posterior, with `--cutoff` only.

The command prints the paths, the most likely champion and which contract
validation ran. Any input error exits with status 1 before writing a forecast.

### Pipeline stage

Select the stage with `--stages bayesian_monte_carlo` or
`--run-bayesian-monte-carlo`. `--all` never includes it. It reuses `--season`,
`--n-simulations`, `--seed` and `--data-dir`, and adds:

| Option | Meaning |
| --- | --- |
| `--bayesian-bracket-schedule` | Schedule path; `{season}` is substituted. Default `data/{season}/bracket_schedule.json` |
| `--bayesian-ratings` | Saved ratings artifact; `{season}` is substituted |
| `--bayesian-cutoff` | Fit new ratings with this cutoff (one season per run) |
| `--bayesian-draws` | Posterior draws when fitting. Default 4000 |
| `--bayesian-output-dir` | Default `model_outputs/bayesian_monte_carlo` |

The stage needs numeric seasons; `--season all` is rejected.

## One joint posterior draw per tournament

For each simulated tournament the simulator picks one saved posterior draw
(uniformly, with replacement) and uses that draw's ratings for **every team in
every game** of that tournament:

```text
P(A beats B | draw k) = sigmoid(rating[k, A] - rating[k, B])
```

Ratings are not drawn independently per team and are not redrawn per game.
A team that is better than its posterior mean in a draw is better in all six
rounds of that tournament, and the posterior correlation between teams is kept.
Averaging the win-probability matrix over draws first would lose this and
understate how often strong or weak versions of a team make deep runs.

All games are neutral-site, so the home-advantage draws are not used.
`BracketSimulation.draw_indices` records the draw each tournament used.

## Pre-tournament bracket schedule

The bracket is an explicit file, written when the field is announced. Later
rounds name the slots that feed them; the file never contains who actually
played or won. This replaces the historical topology parser, which reads
actual winners and cannot be used before a tournament.

The usual form lists four seeded regions. A two-team entry is a play-in
(First Four) game for that seed:

```json
{
  "season": 2025,
  "as_of": "2025-03-16T23:00:00Z",
  "source": "Selection Sunday bracket, transcribed by hand",
  "final_four": [["South", "West"], ["East", "Midwest"]],
  "regions": {
    "South": {"1": "cbbd:2", "2": "Michigan State", "11": ["cbbd:153", "cbbd:21"], "16": ["cbbd:2011", "cbbd:2598"]}
  }
}
```

The example is abbreviated: each of the four regions needs every seed `"1"`
to `"16"`. Teams are provider IDs (`cbbd:<number>`) or exact names from the
ratings artifact. Seeds are paired 1–16, 8–9, 5–12, 4–13, 6–11, 3–14, 7–10,
2–15, and `final_four` gives the two national semifinal pairings.

For any other single-elimination shape, list the games yourself:

```json
{
  "season": 2025,
  "as_of": "2025-03-16T23:00:00Z",
  "games": [
    {"slot": 1, "round_number": 1, "participants": [{"team": "cbbd:1", "seed": 1}, {"team": "cbbd:2", "seed": 4}]},
    {"slot": 2, "round_number": 1, "participants": [{"team": "cbbd:3"}, {"team": "cbbd:4"}],
     "scheduled_start_utc": "2025-03-20T16:15:00Z"},
    {"slot": 3, "round_number": 2, "participants": [{"source_slot": 1}, {"source_slot": 2}]}
  ]
}
```

Optional game fields are `play_in`, `round_points` and `scheduled_start_utc`.
Points default to 10, 20, 40, … by main-bracket round, and 0 for play-in games.

The loader rejects, with a specific message:

- any unknown field, including winners, scores or actual results;
- a team that enters twice, which is how a realized later-round matchup shows up;
- a slot that feeds no game or more than one, or references a later slot or round;
- a scheduled start earlier than `as_of`, or a timestamp without a timezone;
- teams missing from the ratings, two entries for the same rated team, or a
  season that differs from the ratings;
- a ratings cutoff later than any scheduled start.

That last check only runs when the schedule has start times. Without them the
export records `cutoff_check: not checkable`, and keeping the cutoff before the
first game is your responsibility.

## What the forecast JSON contains

The file has the same structure and file naming as the Ridge/SHAP exports on
`main`, so the same consumers can read it.

| Contract field | Bayesian content |
| --- | --- |
| `tournament` | `ncaa-men-{season}`, entrant count, `includes_first_four` when play-ins exist |
| `teams` | Name-derived IDs, identical to `backend_data.export.stable_team_id` |
| `games` | Every scheduled slot. `actual_winner_team_id` is always null |
| `prediction_runs[0]` | Method `bayesian_bradley_terry_joint_draw_monte_carlo`, seed, simulation count, fit metadata, schedule hash, artifacts |
| `team_predictions[].probabilities.bracket_tree` | P(at least k main-bracket wins), k = 1…6 |
| `team_predictions[].statistics` | `tree_expected_wins`, `tree_std_wins`, `champion_probability`, `seed`, `play_in_win_probability` |
| `team_predictions[].posterior` | Rating mean, median, standard deviation and a 90% credible interval, in log-odds |
| `brackets[0]` | `max_ev`: the consistent bracket with the highest expected points |

Play-in wins do not count towards `bracket_tree`, so thresholds mean the same
thing as in the 64-team SHAP export. Pick confidence is the simulated frequency
that the team wins that slot, including reaching it. The `max_ev` bracket is an
exact optimum over those frequencies. There is no public-pick model, so
`public_pick_probability`, pool-win estimates and a leverage bracket are absent.

`source_metadata` keeps the fit metadata except the long list of training game
IDs (their count and SHA-256 remain), plus the simulation design, schedule
provenance and a `team_registry` mapping each team ID to its provider ID.

### Contract validation

This branch does not contain `backend_data`. Where it is importable, every
document is validated with `backend_data.models.ForecastDocument` before it is
written. Otherwise only local checks run (finite JSON, unique well-formed IDs,
schema version) and the command says so. The written file is already in the
form the Pydantic models serialize, including default fields.

## Limitations

- The rating model is experimental and currently trails both baselines on
  game probabilities. Bracket outputs inherit that and are unevaluated.
- Ratings are season-static, ignore margin and use a Laplace approximation.
- Team IDs hash the display name. A merge with a SHAP document from `main`
  needs identical names. `backend_data merge` also requires identical `games`
  when both documents have topology. This has not been tried against a SHAP
  export; expect a conflict wherever slot order, play-ins, start times or
  recorded winners differ. Merging with a Ridge document (no topology) avoids it.
- With play-ins, round numbers run 1–7 (play-in round first), as the contract
  requires sources to be in earlier rounds.
- Nothing checks that the schedule matches the real announced bracket.

## Verification

```bash
uv run --extra bayesian python -m unittest discover -s tests -p test_bayesian_monte_carlo.py
```

Tests cover joint-draw behaviour against per-draw probability matrices,
reproducibility, play-in accounting, schedule rejection cases, optimality of
the `max_ev` bracket against brute force, immutable output files, the command
with fitted and saved ratings, and the pipeline stage. One test validates
against main's Pydantic models and is skipped while `backend_data` is absent.
