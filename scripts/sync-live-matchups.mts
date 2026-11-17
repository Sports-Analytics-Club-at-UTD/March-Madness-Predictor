import { execFileSync } from "node:child_process";
import { readFileSync, writeFileSync } from "node:fs";
import { getGames, getGameTeams, getTeams } from "cbbd";
import { boxRows, fitLiveRatings, matchKaggleTeams, ratingGames, seasonForDate } from "./lib/live-ratings.mjs";

// Usage: pnpm live:sync [--season 2027] [--dump path.json]
function run(name: string, args: object) {
  return JSON.parse(execFileSync("pnpm", ["exec", "convex", "run", name, JSON.stringify(args)], {
    encoding: "utf8", stdio: ["ignore", "pipe", "inherit"], maxBuffer: 64 * 1024 * 1024,
  }).trim());
}
const flag = (name: string) => {
  const index = process.argv.indexOf(name);
  return index === -1 ? undefined : process.argv[index + 1];
};

const apiKey = process.env.CFBD_API_KEY;
if (!apiKey) throw new Error("Set CFBD_API_KEY in .env.local before syncing live matchups.");
const request = { headers: { Authorization: `Bearer ${apiKey}` }, throwOnError: true } as const;
const now = new Date();
const season = Number(flag("--season") ?? seasonForDate(now));
const DAY = 86_400_000;

// The API returns at most 3000 rows per call, so read the season in 10-day windows.
async function readWindows<T>(fetchWindow: (start: string, end: string) => Promise<T[]>, until: Date) {
  const rows: T[] = [];
  for (let start = Date.UTC(season - 1, 9, 1); start <= until.getTime(); start += 10 * DAY) {
    const page = await fetchWindow(new Date(start).toISOString(), new Date(start + 10 * DAY - 1).toISOString());
    if (page.length >= 3000) throw new Error("A 10-day window hit the 3000-row API limit; shorten the window.");
    rows.push(...page);
  }
  return rows;
}

const { data: teams } = await getTeams({ ...request, query: { season } });
if (!teams?.length) throw new Error(`The basketball API returned no teams for ${season}.`);
const divisionOne = new Map(teams.map(team => [team.id, team]));
const until = new Date(now.getTime() + 8 * DAY);
const games = await readWindows(async (startDateRange, endDateRange) =>
  (await getGames({ ...request, query: { season, startDateRange, endDateRange } })).data ?? [], until);
const box = await readWindows(async (startDateRange, endDateRange) =>
  (await getGameTeams({ ...request, query: { season, startDateRange, endDateRange } })).data ?? [], now);

const readJson = (path: string) => JSON.parse(readFileSync(new URL(path, import.meta.url), "utf8"));
const liveMatchupModel = readJson("../convex/data/liveMatchupModel.json");
const prior = readJson("../model/elo-prior-2026.json");
if (prior.season !== season - 1) console.warn(`Elo prior is from ${prior.season}, not ${season - 1}; Elo carry-over is stale.`);
const { matched, unmatched } = matchKaggleTeams(teams, prior.teams);
const priorElo = new Map([...matched].map(([teamId, entry]) => [teamId, entry.elo]));

const rated = ratingGames(games, new Set(divisionOne.keys()));
const rows = boxRows(box, new Set(rated.map((game: { id: number }) => game.id)));
const { teams: features, homeEdge } = fitLiveRatings({ games: rated, box: rows, priorElo, model: liveMatchupModel });
const without = teams.filter(team => !features.has(team.id)).map(team => team.school);
console.log(`${season}: ${games.length} games listed, ${rated.length} final D1 regular-season games rated, ${rows.length / 2} with box scores.`);
console.log(`${features.size}/${teams.length} teams have live features; this season's fitted home edge ${homeEdge.toFixed(2)} pts (predictions use the model's ${liveMatchupModel.homeEdgePoints.toFixed(2)}).`);
console.log(`No last-season Elo (start at 1500): ${unmatched.join(", ") || "none"}.`);
if (without.length) console.log(`No live features yet: ${without.join(", ")}.`);

const ratings = [...features].map(([teamId, team]) => ({
  teamId, teamName: divisionOne.get(teamId)!.school, conference: divisionOne.get(teamId)!.conference ?? undefined,
  ...team, priorElo: priorElo.get(teamId),
}));
const runRecord = {
  season, modelId: liveMatchupModel.id, updatedAt: now.getTime(), gamesUsed: rated.length,
  lastGameDate: rated.at(-1)?.startDate, fittedHomeEdge: homeEdge, teamsWithoutPrior: unmatched, teamsWithoutFeatures: without,
};
run("liveMatchups:importRatings", { run: runRecord, teams: ratings });
console.log(`Stored ${ratings.length} team ratings.`);

// Predict D1 games (including the NCAA tournament and NIT) in the next seven days and refresh
// results for stored games. Ratings above still use regular-season games only.
const pending = new Set<number>(run("liveMatchups:unfinishedGameIds", { season }));
const horizon = now.getTime() + 7 * DAY;
const toSync = games.filter(game => divisionOne.has(game.homeTeamId) && divisionOne.has(game.awayTeamId)
  && (pending.has(game.id) || (Date.parse(game.startDate) > now.getTime() && Date.parse(game.startDate) <= horizon)))
  .map(game => ({
    gameId: game.id, startDate: game.startDate, status: game.status, neutralSite: Boolean(game.neutralSite),
    homeTeamId: game.homeTeamId, homeTeam: game.homeTeam, awayTeamId: game.awayTeamId, awayTeam: game.awayTeam,
    ...(game.status === "scheduled" ? {} : { homePoints: game.homePoints ?? undefined, awayPoints: game.awayPoints ?? undefined }),
  }));
const totals: Record<string, number> = {};
for (let offset = 0; offset < toSync.length; offset += 200) {
  const result = run("liveMatchups:upsertGames", { season, games: toSync.slice(offset, offset + 200) });
  for (const [key, value] of Object.entries(result)) totals[key] = (totals[key] ?? 0) + Number(value);
}
console.log(`Games: ${toSync.length} sent;`, totals);

const dump = flag("--dump");
if (dump) {
  writeFileSync(dump, JSON.stringify({
    season, homeEdge, games: rated, box: rows, eloPrior: Object.fromEntries(priorElo),
    teams: ratings.map(({ teamId, teamName, quality, elo, netEff }) => ({ teamId, teamName, quality, elo, netEff })),
  }));
  console.log(`Wrote parity dump to ${dump}.`);
}
