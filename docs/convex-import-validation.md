# Convex import: validation evidence (issue #60)

Checked `forecast-convex-import` (Python converter + `forecastImports.ts`) against
the real `site` branch (`convex/schema.ts`, `tournamentModel.ts`,
`data/marchMadness2026.ts`, `src/lib/tournament-bracket.ts`). Nothing here
touched a Convex deployment.

## Verified locally

- **Python:** `uv run python -m unittest discover -s tests` passes (48 tests).
- **Site modules:** `forecastImports.test.cjs` plus the site's own
  `tournament-bracket.test.cjs` pass (16 tests) against the unmodified `site`
  sources, with `convex` and `typescript` installed. `tsc --noEmit -p convex/tsconfig.json`
  is clean with `forecastImports.ts` copied in.
- **Real roster end to end** (new): `tests/fixtures/site-2026-roster.json` is the
  site's actual 68-team 2026 reference and 32 round-one pairs. A 64-team forecast
  built with the real `build_forecast` exporter is converted against it (one team
  spelled differently, resolved by an alias; the 4 First Four losers are omitted),
  and `tests/fixtures/convex-payload-2026.json` is the committed output.
  The Python test fails if the converter drifts from that file.
- **Payload against the site** (new, `FORECAST_IMPORT_PAYLOAD`): the committed
  payload
  - has no keys the site's `tournamentValidator` doesn't declare, and all required ones;
  - passes the real `validateTournament` and the `importForecast` mutation against the
    real 68-team reference;
  - reuses IDs on re-import and leaves the official record untouched;
  - has round-one matchups identical to the official bracket's, in order;
  - lays out with `buildTournamentBracket` identically to the official 64-team bracket
    (same 63 node positions, same edge count);
  - gives the two strategies different champions;
  - is about 50 KB, under the 100 KB argument cap.
- The previous cross-language hook overwrote the reference roster with the payload's
  own teams, so it could not catch a roster mismatch. It now uses the real reference.

## Not verified (needs a live deployment)

- `convex run` / `convex dev` behaviour, authentication, and `--prod`.
- Convex's runtime argument validation (`v.id("_storage")`, strict objects) and
  transaction rollback. The tests use an in-memory context, and checking the
  payload's field names against the validators stands in for strict-object validation.
- A real stored logo and a published official card. `logoStorageId` in the fixture is synthetic.
- Real model output: team-name spellings from an actual run still need aliases.
- Odds matching and card rendering in the browser.

Site observations (not changed here): `npm ci` fails on `site` because
`package-lock.json` is out of sync with `package.json`; the tests above were run
with `npm install convex typescript`.
