# Import forecasts into the demo

The Python pipeline stays on this backend branch. The `site` branch already has
a Convex database and a frontend that lists bracket cards and renders their
tournaments. This integration converts saved forecasts into those existing
tables; it does not require an HTTP server or frontend changes.

## One-time setup in a site checkout

Use a separate checkout of `site` so you can keep the Python project open. Copy
`integrations/convex/forecastImports.ts` from this branch into that checkout's
`convex/forecastImports.ts`. This is the only new file needed on `site`; do not
replace its schema or generated files.

In the site checkout, install dependencies and sync the module:

```bash
npm install
npm run convex:dev
```

Use your existing development deployment, or configure a new one. Follow the
site README to import logos and publish the season's official results card. The
importer needs the official tournament at `march-madness-men-<season>` and a card
pointing to it with a valid stored logo. The current site includes a 2026
reference; importing a different season requires adding that season's reference.
Simply seeding example cards is insufficient; publish the canonical reference.

The new functions are **internal**, accessed through your authenticated Convex
CLI. They do not expose unauthenticated browser writes. No Convex deploy key is
passed to Python or embedded in forecast JSON.

## Try the offline example

From the Python project root:

```bash
uv run python -m backend_data.convex examples/forecast.example.json \
  --roster examples/convex-roster.example.json \
  --dry-run --output /tmp/convex-preview.json
```

This makes no network requests or database writes. Both example files are
synthetic, including the storage ID; they are not suitable for a live import.

## Preview a real forecast

Generate a forecast with complete games and picks, then run:

```bash
uv run python -m backend_data.convex path/to/forecast.json \
  --site-dir /absolute/path/to/site-checkout \
  --dry-run --output /tmp/convex-preview.json
```

This reads the roster from the configured development deployment and writes
nothing to the database. Review the converted JSON. The importer validates the
forecast's schema, references, advancement, champion, and layout compatibility.
Ridge files with team scores but no topology/picks are rejected. No training
runs during import.

Team names match only when equal after trimming whitespace and ignoring case.
For other spellings, supply a reviewed JSON object with `--aliases aliases.json`:

```json
{
  "Your exact team name": "canonical-site-team-id",
  "team-your-forecast-id": "another-canonical-site-team-id"
}
```

Forecast IDs take precedence over name aliases. Targets must exist in the site's
roster; two entrants cannot map to the same site team. Seeds, display names, and
logos come from the site. Predictions for teams outside the bracket, such as
First Four losers, are excluded from the tournament record.

## Import

Remove `--dry-run` after reviewing:

```bash
uv run python -m backend_data.convex path/to/forecast.json \
  --site-dir /absolute/path/to/site-checkout --aliases aliases.json
```

Omit `--aliases` if names already match. Open `/projects/mmp` in the demo. Each
strategy appears as a sidebar card and uses its own predicted participants in
later rounds. Scores stay hidden. The card initially reuses the official card's
stored logo; team logos remain the real canonical logos.

The entire import is one atomic Convex mutation. Stable slugs use tournament ID,
run ID, and bracket ID. Importing the same IDs updates their cards and tournaments;
new run IDs create new cards. No official-results records are replaced. Up to
100 KB of compact mutation arguments is supported; split unusually large bundles
into smaller validated documents. `--output` saves a preview, not an alternate
unvalidated import input.

Development is the default. `--prod` explicitly selects production for both
roster reads and imports; the Convex module must already be deployed there.
This integration does not deploy or change deployment configuration for you.

## Limits and verification

- Supports balanced men's main brackets with 4–64 teams, consecutive rounds
  starting at 1, and two feeder games per later matchup. Current SHAP exports
  use 64 teams/63 games. First Four import is not yet supported.
- Region labels are generic (`Region 1`, etc.) because forecast JSON does not
  contain official region names. Tree traversal preserves the bracket's order.
- Team IDs are translated to canonical IDs. Game IDs remain model-specific;
  existing odds/result code matches canonical team pairs, season, and round.
  Hypothetical matchups have no matching real market.
- The converter trusts the forecast's bracket topology. Review that its opening
  matchups and regional tree agree with the season's official bracket.
- Confidence, uncertainty, team statistics, and historical market snapshots
  stay in the original forecast file. This first integration imports display
  picks only; the site keeps its existing CBBD/Kalshi ingestion and semantics.

Run Python checks with `uv run python -m unittest discover -s tests`. The
integration's TypeScript checks run from the site checkout after copying
`integrations/convex/forecastImports.test.cjs` into `tests/`:

```bash
node --test tests/forecastImports.test.cjs tests/tournament-bracket.test.cjs
```

The new tests exercise the actual mutation handlers against an in-memory context;
they do not contact Convex. After configuring a deployment, verify two cards,
correct per-strategy advancement/champions, logos, and matching-game odds. Import
the same file again and confirm the returned IDs are reused. A real deployment
smoke test requires your configured site checkout and a real forecast with picks.
