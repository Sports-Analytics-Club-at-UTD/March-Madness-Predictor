This is a [Next.js](https://nextjs.org) project bootstrapped with [`create-next-app`](https://nextjs.org/docs/app/api-reference/cli/create-next-app).

## Getting Started

Install dependencies and configure your Convex development deployment:

```bash
npm install
npm run convex:dev
```

On the first run, Convex prompts you to log in and select or create a project.
It writes deployment settings, including `NEXT_PUBLIC_CONVEX_URL`, to `.env.local`
and generates typed helpers in `convex/_generated/`. Keep this command running
to sync backend changes. Add tables to `convex/schema.ts` and backend functions
under `convex/`.

In a second terminal, run the Next.js development server:

```bash
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) with your browser to see the result.

You can start editing the homepage in `src/app/(Home)/page.tsx`. The page auto-updates as you edit the file.

The root layout provides Convex to client components through
`src/app/ConvexClientProvider.tsx`. Use `useQuery` and `useMutation` from
`convex/react` with the generated API to access your backend.

For production, run `npm run convex:deploy` and set `NEXT_PUBLIC_CONVEX_URL` in
your hosting environment to the production deployment URL before building.

This project uses [`next/font`](https://nextjs.org/docs/app/building-your-application/optimizing/fonts) to automatically optimize and load [Geist](https://vercel.com/font), a new font family for Vercel.

## Tournament bracket

`/projects/mmp` loads the selected prediction bracket from Convex. To populate
a new development deployment, run:

```bash
npx convex dev --once --run tournaments:seed2026
npx convex dev --once
node scripts/seed-bracket-cards.mjs
```

The seed inserts the example only if its slug is absent; rerunning it preserves
existing data. It includes all 68 teams and 67 completed games, with results from
the [official NCAA bracket](https://www.ncaa.com/march-madness-live/bracket).
The card seed uploads `public/next-favicon.ico` to Convex file storage and adds
The UT Upheaval, A&M Avenue, and The Real Deal. The first two are illustrative
predictions that advance Texas or Texas A&M to the championship; they are not
model output. The Real Deal contains historical results. Rerunning the script
reuses the uploaded logo and existing cards.

`brackets` stores each card's title, description, storage logo ID, and tournament
reference. `PredictionBracketCard` accepts `imageUrl`, `title`, `bracketId`, and
`onSelect` props. The sidebar loads the selected card's tournament in the main
canvas and refits it when switching cards.

`TournamentBracket` accepts a `tournament` prop with teams, game results, and
references to the games that feed each matchup. It generates locked cards and
connections in mirrored halves meeting at the championship; pan, zoom,
fit-to-content, and theme controls remain available.
Scores are optional and hidden by default. Model predictions need only a
`winnerTeamId` for each game. Pass `showScores` to `TournamentBracket` or
`TournamentBracketCard` to display scores when supplied.
`convex/tournamentModel.ts` defines the data shape and validates advancement,
scores, and the champion. The seed is internal; the browser has read access only.

Run the bracket checks with:

```bash
node --test tests/tournament-bracket.test.cjs
```

## Team logos

The official [`cbbd` TypeScript SDK](https://github.com/CFBD/cbbd-typescript)
provides basketball data using the `CFBD_API_KEY` in `.env.local`. Keep that key
server-side. To import the available standard 500px team logos:

```bash
pnpm exec convex dev --once
pnpm logos:sync
```

The importer requires Node 22.18+ and uses the configured development Convex
deployment. It skips unchanged logos and reports teams without a large logo
and failed downloads; rerun it to retry failures. Images live in Convex file
storage, with team name, display name, CBBD team ID, and original URL in
`teamLogos`. Import functions are internal and cannot be called from the browser.

Retrieve a stored logo with
`api.teamLogos.getByTeamName({ teamName: "Texas" })` (or pass those arguments to
`useQuery`). Name matching ignores case and surrounding whitespace; the result
includes a `url` for rendering. For duplicate school names, name lookup returns
the first match; use `api.teamLogos.getByTeamId` with the CBBD `teamId` to select
an exact API record.

After importing logos, publish the official 2026 tournament with:

```bash
pnpm exec convex run brackets:publish2026
```

This updates or creates the `2026-results` bracket and its tournament, links all
68 teams to stored logos, and makes the official results the default sidebar
selection with scores visible. The card uses Michigan's logo. Existing example
predictions also receive real team logos without changing their picks. Reruns
reuse the bracket and tournament records.

```bash
node --test tests/team-logos.test.mjs
```

## Kalshi odds and graphs

Set the server-side `KALSHI_API_ID` and `KALSHI_PRIVATE_KEY` in `.env.local`.
The private key value can be a PEM file path or a quoted PEM string with escaped
newlines. Keep the key file outside version control. The official
`kalshi-typescript` SDK makes read-only calls from the importer and Convex's
Node action. Browser code never receives these credentials.

```bash
pnpm exec convex dev --once
pnpm odds:sync
```

The hover chart also needs these credentials in the Convex server environment
(already configured for the current development deployment):

```bash
pnpm exec convex env set KALSHI_API_ID
pnpm exec convex env set KALSHI_PRIVATE_KEY --from-file /path/to/private-key.pem
```

The first command accepts the ID through a prompt or stdin; the second reads
the PEM contents, not a path on the Convex server.

The importer matches the official 2026 games against the men's game-winner
series using both team names and the round's scheduled dates. It rejects
ambiguous matches, uses the historical endpoints for archived markets, and
stores each team's first and last available hourly trade prices in Convex's
`gameOdds` table. The championship uses the `KXMARMAD-26` tournament-winner
markets, with history limited to after both finalists were known; the card
labels this distinction.

Hovering for one second opens the graph. The server resolves the game to its
stored market tickers, reads current prices and the last six hours of one-minute
candlesticks, and saves the graph in `gameOddsCharts`. Older markets use the
historical endpoints. Graphs refresh every 15 seconds only while the card is
open and the tab is visible, with a shared 15-second cache for active markets.
Settled graphs are retained without polling or further Kalshi calls. Closing
the preview stops its refresh timer.

The card shows two price curves, time and percentage axes, and a crosshair for
mouse or keyboard inspection. Trade prices represent implied probabilities;
100%/0% final outcomes are shown only for finalized yes/no results, using their
actual settlement timestamp. Empty trade history stays empty rather than
generating a simulated curve. Failed updates offer a retry and preserve any
previously saved graph.

Rerunning the importer skips matched games. Prediction brackets resolve to the
official tournament for the same season (`march-madness-men-<season>`). Matchups
are compared by both canonical team IDs and round, regardless of their game ID
or top/bottom ordering. Matching games share the official market and graph cache,
even when the model predicts a different winner. An unrelated branch continues
to receive odds, and a later matchup that rejoins the real bracket can receive
odds again.

A red border and `Bracket breaks here` mark a pick that differs from a confirmed
CBBD final result. Downstream matchups containing a team CBBD confirmed was
eliminated in an earlier round get a dashed amber border and
`Hypothetical matchup`, with an explanation instead of an unrelated market.
Live leads, scheduled games, missing scores, final ties, postponed games, and
cancelled games do not adjudicate a pick. Pending matchups retain any known
markets without being marked broken. Games without imported markets still show
`Couldn't find game`. A season with no official reference is not marked as broken. Future model imports should
preserve the canonical team IDs and season; model-specific game IDs are fine.
Use `pnpm odds:sync --refresh` only when intentionally replacing cached snapshots.

```bash
node --test tests/kalshi-matching.test.mjs tests/kalshi-chart.test.cjs tests/tournament-bracket.test.cjs
```

## CBBD game results

`cbbdSeasons` stores each season's NCAA schedule and scores separately from model
picks. The comparison reads this cached feed; reference-bracket winner fields
and Kalshi prices do not decide whether a prediction failed. CBBD game IDs,
team IDs, school aliases and round descriptions bind the feed to canonical teams.
Unresolved participants or unrecognized rounds are skipped instead of guessed.

The current development deployment has the 67 official 2026 CBBD finals loaded
from one bulk request. Set the same server-side key for a new deployment and
initialize a season after its official tournament reference is imported:

```bash
pnpm exec convex env set CFBD_API_KEY
pnpm exec convex run cbbd:syncSeason '{"season":2026}'
```

While the bracket page is visible during March–April, a shared refresh runs at
most once per 15 minutes across all viewers. An atomic claim prevents concurrent
viewers from duplicating requests. The schedule refreshes daily and when the
scoreboard reveals a newly scheduled matchup between known tournament teams.
Scoreboard updates preserve games absent from the live feed and do not downgrade
confirmed finals to an older live status. Valid final-score corrections update
the result and automatically recompute the borders through Convex queries.

The live scoreboard requires CBBD Tier 1 access. If denied, the app remembers
that for the season and polls regular game results instead. Completed seasons
stop requesting data; no per-game or hover-driven CBBD requests occur. To
intentionally refresh a completed season or retry upgraded scoreboard access:

```bash
pnpm exec convex run cbbd:syncSeason '{"season":2026,"force":true}'
```

## Live matchups and weekly predictions

`/projects/mmp/matchup` gives the model's win probability for any two Division I
men's teams this season, on a neutral court or at either team's arena.
`/projects/mmp/weekly` lists the predictions for the coming week's games by week
and day, then marks each pick right or wrong and reports accuracy and Brier score
once games are final. Both pages are linked from the mmp sidebar.

**Model.** `model/live_matchup_model.py` reuses the logistic member of the
`final` men's configuration in `kaggle_tuning` (`origin/kaggle-score-tuning`).
That model is a StandardScaler plus an L2 logistic regression with C = 0.1, no
intercept, and both team orders averaged. Here it uses only the three features
that exist during the season:
ridge-adjusted scoring margin (alpha 1, margins clipped at 60, with a home
term), margin-aware Elo (k 20, home 75, 75% carried between seasons), and mean
net efficiency per 100 possessions. Seeds and Massey rankings do not exist before
Selection Sunday, so they are left out, along with the XGBoost member and the
betting-market blend. The model trains on the 2003–2025 men's tournaments (First
Four excluded) from the Kaggle mirror. Walk-forward tournament Brier scores for
2015–2025 range from 0.132 to 0.228, with 62–81% of winners picked; the
`backtest` field of the artifact has each year.

Regenerate the artifact with a pinned environment (Python 3.13; dependency
versions live in the script header):

```bash
uv run model/live_matchup_model.py export --kaggle-dir /path/to/data/kaggle/mirror26
```

This writes the scaler and coefficients to `convex/data/liveMatchupModel.json`
and each team's end-of-2026 regular-season Elo, with its Kaggle spellings, to
`model/elo-prior-2026.json`.

**Sync.** Run the importer against your configured Convex deployment:

```bash
pnpm exec convex dev --once
pnpm live:sync            # optional: --season 2027 --dump /tmp/live.json
```

The importer reads the season's games and team box scores from CBBD in 10-day
windows, staying under the API's 3,000-row limit. It uses final regular-season
games between Division I teams, which includes conference tournaments but not
the NCAA tournament or NIT, the same games the Kaggle regular season covers. From
these it computes the three features in `scripts/lib/live-ratings.mjs` and stores
them in `liveTeamRatings`, with run details in `liveRatingRuns`. It then stores
predictions in `liveGamePredictions` for Division I games in the next seven days,
including NCAA tournament and NIT games,
and refreshes results for stored games that are not final. A prediction is made
only before tip-off, from ratings that exclude the game, and is refreshed on each
sync until tip-off. After that, only the status and score change. Games first
seen after they start are skipped rather than predicted after the fact. Rerun
the sync to update ratings and grade results; it is safe to repeat.

CBBD schools are matched to Kaggle Elo priors by spelling, plus a few aliases
listed in `live-ratings.mjs`. A spelling that maps to more than one team leaves
both unmatched. Teams without a prior, such as new Division I members, start at
1500. Teams without a completed game and box score get no features. Both pages
list these teams.

To compare against Python on the same CBBD data:

```bash
pnpm live:sync --dump /tmp/live.json
uv run model/live_matchup_model.py parity --dump /tmp/live.json --team-a Duke --team-b Houston
```

On November 17, 2026, the Python and TypeScript features agreed within 1e-13.
Both gave Duke a 0.705702 chance against Houston on a neutral court and 0.816507
at Duke. Training used sklearn's default iterative sparse ridge solver, while the
site solves exactly; with `--solver auto` Python reproduces training instead,
with quality differences up to 0.07 points.

**Limits.** Early in the season each team has only a few games, so ratings rest
on little data and lean on last season's Elo. The model was trained on
neutral-court tournament games, so regular-season games, especially mismatches,
fall outside its training data. Home court is an adjustment the model did not
learn: the home team's adjusted margin gains the mean full-season ridge home
edge (3.56 points, 2003–2026) and its Elo gains the 75-point Elo home bonus.
This season's own fitted home edge, 7.4 points in mid-November, is stored but
not used, because early schedules mix home court with strong teams hosting weak
ones. Injuries and lineups enter only through results. The weekly page tracks how
these probabilities hold up on regular-season games.

```bash
node --test tests/live-matchups.test.cjs tests/live-ratings.test.mjs
```

## App theme

The root `ThemeProvider` uses `next-themes` with a system default and applies
the resolved light/dark class before first paint. OS appearance changes are
followed until the user chooses a theme; the canvas toggle saves that explicit
choice in localStorage under `ui-theme` across reloads and navigation. Theme-dependent icons use CSS so server
and client markup stay consistent during hydration.
The canvas toggle uses the sweep from the sibling `karppa.dev` project: a 720ms
radial reveal from the click position with `cubic-bezier(.76, 0, .24, 1)` easing.
Keyboard activation starts at the button center. Reduced motion and browsers
without view transitions change instantly. The shared theme hook applies the
root class synchronously before the new-theme snapshot, and the Next.js provider
persists the choice under `ui-theme` while retaining the pre-paint system default.

Kalshi cards, graph labels, grid lines, chart colors, loading states and errors
use the same shared CSS theme tokens and font as the rest of the app.

## Learn More

To learn more about Next.js, take a look at the following resources:

- [Next.js Documentation](https://nextjs.org/docs) - learn about Next.js features and API.
- [Learn Next.js](https://nextjs.org/learn) - an interactive Next.js tutorial.

You can check out [the Next.js GitHub repository](https://github.com/vercel/next.js) - your feedback and contributions are welcome!

## Deploy on Vercel

The easiest way to deploy your Next.js app is to use the [Vercel Platform](https://vercel.com/new?utm_medium=default-template&filter=next.js&utm_source=create-next-app&utm_campaign=create-next-app-readme) from the creators of Next.js.

Check out our [Next.js deployment documentation](https://nextjs.org/docs/app/building-your-application/deploying) for more details.
