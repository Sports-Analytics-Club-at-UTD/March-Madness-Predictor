import { execFileSync } from "node:child_process";
import { Configuration, EventsApi, HistoricalApi, MarketApi, type EventData, type Market } from "kalshi-typescript";
import type { Tournament } from "../convex/tournamentModel.js";
import { matchGameEvent, normalizeTeam, summarizePrices } from "./lib/kalshi-matching.mjs";

function run(name: string, args: object = {}) {
  return JSON.parse(execFileSync("pnpm", ["exec", "convex", "run", name, JSON.stringify(args)], {
    encoding: "utf8", stdio: ["ignore", "pipe", "inherit"],
  }).trim() || "null");
}

async function main() {
  const key = process.env.KALSHI_PRIVATE_KEY;
  if (!process.env.KALSHI_API_ID || !key) throw new Error("Set KALSHI_API_ID and KALSHI_PRIVATE_KEY in .env.local.");
  const configuration = new Configuration({
    apiKey: process.env.KALSHI_API_ID,
    ...(key.includes("-----BEGIN") ? { privateKeyPem: key.replace(/\\n/g, "\n") } : { privateKeyPath: key }),
    baseOptions: { timeout: 20_000 },
  });
  const eventsApi = new EventsApi(configuration);
  const historicalApi = new HistoricalApi(configuration);
  const marketApi = new MarketApi(configuration);
  const slug = "march-madness-men-2026";
  const tournament: Tournament | null = run("tournaments:getBySlug", { slug });
  if (!tournament) throw new Error("Publish the official 2026 tournament before importing odds.");
  const stored: string[] = run("gameOdds:listStored", { tournamentSlug: slug });
  const pending = tournament.games.filter(game => process.argv.includes("--refresh") || !stored.includes(game.id));
  if (!pending.length) { console.log("All 67 games already cached; no Kalshi requests needed."); return; }
  const events: EventData[] = [];
  let cursor: string | undefined;
  do {
    const { data } = await eventsApi.getEvents(200, cursor, false, false, "settled", "KXNCAAMBGAME", undefined, Date.parse("2026-03-17") / 1000);
    events.push(...data.events);
    cursor = data.cursor || undefined;
  } while (cursor);

  const teamById = new Map(tournament.teams.map(team => [team.id, team]));
  let matched = 0;
  let missing = 0;
  let failed = 0;
  for (const game of pending) {
    const top = teamById.get(game.topTeamId)!;
    const bottom = teamById.get(game.bottomTeamId)!;
    try {
      const match = matchGameEvent(events, top.name, bottom.name, game.round);
      // Kalshi's championship is the tournament-winner event, not a head-to-head game event.
      const championship = game.id === "championship" && game.round === 6;
      const eventTicker = match?.event_ticker ?? (championship ? "KXMARMAD-26" : undefined);
      if (!eventTicker) {
        run("gameOdds:save", { records: [{ tournamentSlug: slug, gameId: game.id, outcomes: [], importedAt: Date.now() }] });
        console.log(`Couldn't find game: ${top.name} vs ${bottom.name}`); missing++; continue;
      }
      let markets: Market[];
      const archived = (await historicalApi.getHistoricalMarkets(1000, undefined, undefined, eventTicker)).data;
      markets = archived.markets;
      if (!markets.length) markets = (await eventsApi.getEvent(eventTicker, false, true)).data.markets ?? [];
      const outcomes = [];
      for (const team of [top, bottom]) {
        const candidates = markets.filter(market => normalizeTeam(market.yes_sub_title ?? "") === normalizeTeam(team.name));
        if (candidates.length !== 1) throw new Error("No unique winner market for both teams.");
        const market = candidates[0];
        // Championship history starts after both finalists were known, excluding earlier futures prices.
        const start = championship ? Date.parse("2026-04-05") / 1000 : Date.parse(market.open_time) / 1000;
        // Include the final, partially traded hour; its endpoint may be after market close.
        const end = Math.ceil(Date.parse(market.close_time) / 3_600_000) * 3600;
        let summary;
        if (archived.markets.length) {
          const { data } = await historicalApi.getMarketCandlesticksHistorical(market.ticker, Math.floor(start), Math.ceil(end), 60);
          summary = summarizePrices(data.candlesticks);
        } else {
          const { data } = await marketApi.getMarketCandlesticks(eventTicker.split("-")[0], market.ticker, Math.floor(start), Math.ceil(end), 60);
          summary = summarizePrices(data.candlesticks);
        }
        outcomes.push({ teamId: team.id, teamName: team.name, ticker: market.ticker, ...summary });
      }
      run("gameOdds:save", { records: [{ tournamentSlug: slug, gameId: game.id, eventTicker,
        marketKind: championship ? "championship" : "game", outcomes, importedAt: Date.now() }] });
      matched++;
      console.log(`Cached ${top.name} vs ${bottom.name} (${matched + missing}/${pending.length}).`);
    } catch (error) {
      // Axios errors contain signed request headers; only report a status or a safe local message.
      const status = (error as { response?: { status: number } }).response?.status;
      console.error(`Could not import ${top.name} vs ${bottom.name}: ${status ? `Kalshi HTTP ${status}` : "market matching or history unavailable"}.`);
      failed++;
    }
  }
  console.log(`${matched} games cached; ${missing} unmatched; ${failed} failed.`);
  if (failed) process.exitCode = 1;
}

main().catch(error => {
  const status = (error as { response?: { status: number } }).response?.status;
  console.error(status ? `Kalshi HTTP ${status}. Check the API key and read permissions.` : "Odds import failed. Check local configuration and Convex deployment.");
  process.exitCode = 1;
});
