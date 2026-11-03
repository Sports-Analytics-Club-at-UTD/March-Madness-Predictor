# Forecast JSON contract

The `backend_data` package is the data boundary between forecasting and future
HTTP consumers. It uses Pydantic v2 and has no FastAPI, database, network, or
training dependency. The pipeline now exports validated JSON in addition to
existing analysis CSVs and metadata. Its output includes all numeric team-score
columns, model metadata, structured bracket topology, both current strategies,
and optional historical Kalshi observations.

## Files and commands

- `backend_data/models.py`: authoritative wire models and relationship checks.
- `schemas/forecast.schema.json`: generated JSON Schema, Draft 2020-12.
- `backend_data/export.py`: current Ridge/SHAP adapters and atomic storage.
- `backend_data/kalshi.py`: historical CSV adapter and reviewed mappings.
- `examples/forecast.example.json`: synthetic, small tournament with two runs,
  two strategies, a future posterior example, and market data. Not a forecast.

Install with `uv sync`. Tests also use the development dependency `jsonschema`.
For the existing training tests, install `uv sync --extra training`.

```bash
# Export an existing Ridge table without rerunning any models.
uv run python -m backend_data export \
  --team-scores model_outputs/monte_carlo/team_success_scores_2025.csv \
  --metadata model_outputs/monte_carlo/team_success_scores_2025.metadata.json \
  --mode ridge

# Include the supplied historical markets in a new 2026 SHAP pipeline run.
# Requires the existing SHAP recipes, historical data, and training dependencies.
uv run --extra training python run_pipeline.py --stages monte_carlo \
  --season 2026 --kalshi-csv kalshi_result.csv --no-plots

# Optional reviewed crosswalk (also accepted by run_monte_carlo.py).
# Add: --kalshi-mapping path/to/reviewed-mapping.json

# Validate shape AND references/progression, not just JSON syntax.
uv run python -m backend_data validate examples/forecast.example.json

# Combine distinct runs for the same tournament.
uv run python -m backend_data merge path/to/run-a.json path/to/run-b.json

# Regenerate after editing Pydantic models.
uv run python -m backend_data schema

uv run python -m unittest discover -s tests
```

New files are named `forecast_{season}_{bundle_uuid}.json`. Each prediction run
also has its own ID; a bundle can contain several runs. Storage writes a complete
temporary file, validates it, and atomically publishes it without replacing an
existing filename. No mutable `latest.json` is maintained. The pipeline's artifact
summary lists newly created JSON files. JSON artifacts under `model_outputs/`
remain ignored; schemas, documentation, and examples are tracked.

The conversion command accepts `--kalshi-csv`, `--kalshi-mapping`, `--bracket-csv`,
and `--topology`. A saved bracket CSV alone does not contain structured sources.
The topology JSON must contain `teams64`, zero-based `round1_pairs`, zero-based
`tree_edges`, `slot_rounds`, and optionally `actual_slot_winners` (team indices;
`-1` for unknown) and `scheduled_starts_utc` (one timestamp/null per game).
Normal pipeline exports receive this object directly, preserve `df_63.startDate`
when available, and do
not parse human-readable matchup labels.

## Document and identity model

One `ForecastDocument` describes one tournament:

| Field | Meaning |
| --- | --- |
| `schema_version` | Contract version, currently `1.0.0` |
| `tournament` | Season, competition, topology availability, bracket size and First Four flag |
| `teams` | Shared identity registry for this document |
| `games` | Slots, rounds, scoring points, participants, optional actual winners and schedule |
| `prediction_runs` | Method/model identity, timestamps, seed, simulation count, original metadata, artifact references |
| `team_predictions` | One record per run/team, numeric statistics, probability families, intervals, optional posterior |
| `brackets` | Run-linked strategies, one pick per game, confidence, public frequencies, expected points, summaries |
| `market_data` | Optional Kalshi import provenance, import report and game-winner snapshots |

IDs are separate from names. Current team IDs hash the exact canonical name and
competition, so input row order does not change them. Changing a canonical name
changes its ID; a future team registry can supply persistent identities. Team IDs
do not encode a season. Tournament IDs are `ncaa-men-{season}`; game IDs append
`-g{slot}`. IDs are scoped to this tournament topology, not provider game IDs.

A participant references exactly one `team_id` or `source_game_id`. The graph
must be a connected single-elimination tree. Teams enter once, games reference
earlier rounds, and every bracket must pick a winner that advances from its own
preceding picks. Bracket champions must match final picks. Teams outside the
64-team main bracket may still have team predictions, such as First Four losers.

The shared model supports arbitrary single-elimination sizes. The CURRENT SHAP
adapter only exports its existing 64-team/63-game tree with
`includes_first_four=false`; it does not manufacture a 67-game bracket. Ridge
outputs have no topology or brackets. SHAP may also have no topology when the
existing parser cannot construct one. Its parser currently depends on historical
actual winners; future pre-tournament scheduling needs a separate topology source.

## Source field mapping and statistical meaning

All probabilities use decimals in `[0,1]`. Missing values are `null`, including
pandas/NumPy NaNs. Infinity is rejected. Empty collections mean no such records,
not zero probability. Unknown object fields are rejected except within explicit
extension/metadata dictionaries. Unavailable actual results remain null.

| Current source | JSON destination |
| --- | --- |
| Team CSV `team` / `season` | Shared team reference / tournament season |
| `probability_ge_{k}_win(s)` | `team_predictions[].probabilities.success_score_threshold[]` |
| `calibrated_prob_ge_{k}_win(s)` | `probabilities.calibrated_success_score_threshold[]` |
| `tree_prob_ge_{k}_win(s)` | `probabilities.bracket_tree[]` |
| Other numeric team columns | `team_predictions[].statistics`, preserving source names |
| `percentile_5`, `percentile_95` | Also a 90% `empirical_prediction` interval for the success score |
| Entire companion metadata | `prediction_runs[].source_metadata` |
| Parsed pairs/edges/rounds | `games[].participants`, slot, round, points |
| Parsed actual winner | `games[].actual_winner_team_id` |
| `max_ev_pick`, `leverage_pick` | Two bracket strategies and their winner team references |
| `{strategy}_win_prob` | `picks[].confidence.value` |
| `{strategy}_public_pct` | `picks[].public_pick_probability` (already 0–1) |
| `{strategy}_expected_pts` | `picks[].expected_points` |
| `leverage_ratio` | Leverage strategy's pick ratio |
| Correctness columns | Derived from picked/actual winner IDs when actual is known |
| Pool optimization summary | Typed bracket summary; full source remains in run metadata |

Threshold probabilities remain distinct. Ridge frequencies are thresholds on
continuous success scores, not calibrated matchup/advancement probabilities.
Isotonic-calibrated score thresholds and bracket-tree advancement probabilities
also remain separately named. Frequencies must decrease as thresholds increase.

Pick confidence means **the probability the team wins that slot**, including
reaching it. It is not conditional on a particular opponent. The public-pick
frequency comes from simulated public behavior, not observed Kalshi pricing.
Pool-win estimates preserve the existing optimizer's semantics (ties count as
wins, opponent count is capped); these are not independent validation results.
The source metadata retains the configuration and calibration assumptions.

## Historical Kalshi import

The supplied CSV contains both competitions and season-wide game markets.
Import selects the tournament's competition and dated games in the explicit
ending-year window `[July 1 of season-1, July 1 of season)`. This is a project
convention, not an inferred provider season label. Records without start times
are excluded and reported, rather than assigning a season from ticker text.
Every excluded row records its CSV row number, event ticker and reason. Included
invalid probabilities or timestamps fail validation rather than silently changing
the quote. The 2025–26 CSV will contribute no dated records to a 2025 forecast.

| CSV fields | JSON destination |
| --- | --- |
| `game`, `game_id`, `event_ticker` | Snapshot title, provider game ID, event ticker |
| Competition, schedule, requested snapshot | Snapshot fields with timezone-aware timestamps |
| `team_1`, `team_2` | Two outcome records with original names |
| `team_{i}_*_probability_pct` | Outcome probabilities, divided by 100 |
| Market ticker, probability basis, data status | Each outcome, unmodified |
| Quote as-of, seconds before start | Each outcome, independently preserved |
| Game status and market count | Snapshot status and market count |
| Timing/price basis and source/exception URLs | Snapshot provenance |
| Additional future CSV columns | Snapshot `source_fields` |

Zero is retained. Probabilities are NOT forced to sum to one. An unavailable
quote is not a loss or a 0% prediction. Older, last-trade, canceled and non-binary
settlement statuses are retained for consumers to filter explicitly. These are
historical pregame snapshots, not live quotes. Timestamps are checked against
the requested cutoff, and the supplied quote age must agree with the timestamps.

Exact team-name matches are labeled `exact_name`; other names remain unmatched
unless an alias is supplied. A reviewed mapping file looks like:

```json
{
  "team_aliases": {"vendor team name": "canonical roster name"},
  "event_games": {"provider event ticker": "ncaa-men-2026-g1"}
}
```

The example is a template, not a verified real mapping. Unmatched teams remain
null and are counted. No fuzzy joins or automatic game joins are performed.
Explicit game joins require two known teams from opposite participant branches
of that slot. The reviewer
must also confirm date/event identity: eligible names alone do not prove identity,
especially for later rounds. Regular-season markets remain unlinked unless
reviewed; they are not automatically tournament games or championship odds.
Canceled/irregular games should not be used for model-market comparisons without
an explicit policy. Quotes are never substituted for model probabilities.

## Future integration and compatibility

FastAPI can import `ForecastDocument` as a response model and serve validated
saved documents. Select/paginate records at the API layer; do not rerun training
on a page request or send all historical markets on every bracket request.
API auth, CORS, databases and live ingestion are separate work.

Bayesian producers can construct the same run/team/bracket objects using a new
`method`, populate optional `posterior` summaries/diagnostics and correctly named
credible intervals, and reference large posterior draws using run `artifacts`.
No Bayesian fitting or market-based updating is performed here. Existing bootstrap
percentiles never become credible intervals. A method using a different event
definition must add a corresponding explicit field rather than reuse slot confidence.

`merge` combines distinct runs, including Ridge documents without topology and
SHAP documents with topology. It rejects duplicate run IDs, conflicting team
identities/topologies, and conflicting market datasets. It does not guess how
to resolve differing quotes or alias policies.

For changes, update Pydantic models, regenerate the schema and update examples.
Use additive optional fields in a minor contract version, fixes that preserve
meaning in a patch, and a major version for renamed fields/type or meaning changes.
Consumers should check `schema_version`; old artifacts are not rewritten. The
current validator deliberately accepts only `1.0.0`, so accepting a new version
is an explicit code change. The JSON Schema checks shape; Python validators add
graph integrity, progression, mapping, season and import-count checks.
