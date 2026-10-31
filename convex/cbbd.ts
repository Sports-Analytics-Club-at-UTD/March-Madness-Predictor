"use node";

import { ConvexError, v } from "convex/values";
import { getGames, getScoreboard, type ScoreboardGame } from "cbbd";
import { action, internalAction, type ActionCtx } from "./_generated/server";
import { internal } from "./_generated/api";
import { applyScoreboard, confirmedWinner, hasNewTournamentGame, normalizeCbbdGames } from "./cbbdResults";

type SyncStatus = "complete" | "cached" | "updated" | "inactive";

async function sync(ctx: ActionCtx, season: number, manual: boolean, force = false): Promise<SyncStatus> {
  const { reference, saved } = await ctx.runQuery(internal.cbbdSeasons.context, { season });
  if (!reference) return "inactive";
  if (!force && saved && saved.games.length >= reference.games.length && saved.games.every(game => confirmedWinner(game))) return "complete";
  const now = Date.now();
  if (!manual && (now < Date.UTC(season, 2, 1) || now > Date.UTC(season, 3, 16))) return "inactive";
  if (!await ctx.runMutation(internal.cbbdSeasons.claimRefresh, { season, force })) return "cached";
  const key = process.env.CFBD_API_KEY;
  if (!key) throw new ConvexError("CBBD server credentials are not configured.");
  try {
    const headers = { Authorization: `Bearer ${key}` };
    let scoreboardAvailable = force ? true : saved?.scoreboardAvailable;
    let scheduleDue = manual || !saved?.scheduleFetchedAt || now - saved.scheduleFetchedAt >= 24 * 3600_000 || scoreboardAvailable === false;
    let scoreboard: ScoreboardGame[] | undefined;
    let games = saved?.games ?? [];
    if (!scheduleDue) {
      const { data, response } = await getScoreboard({ headers, signal: AbortSignal.timeout(20_000) });
      if (response.status === 403) {
        // Scoreboard requires a paid tier; remember the fallback rather than retrying denied calls.
        scoreboardAvailable = false;
        scheduleDue = true;
      } else {
        if (!response.ok || !data) throw new Error("Scoreboard unavailable.");
        scoreboardAvailable = true;
        scoreboard = data;
        scheduleDue = hasNewTournamentGame(games, data);
        games = applyScoreboard(games, data);
      }
    }
    if (scheduleDue) {
      const { data } = await getGames({ headers, query: { season, seasonType: "postseason", tournament: "NCAA" },
        signal: AbortSignal.timeout(20_000), throwOnError: true });
      if (!data || data.length >= 3000) throw new Error("Incomplete tournament response.");
      games = normalizeCbbdGames(data, reference);
      if (scoreboard) games = applyScoreboard(games, scoreboard);
    }
    await ctx.runMutation(internal.cbbdSeasons.save, { season, games,
      scheduleFetchedAt: scheduleDue ? now : saved!.scheduleFetchedAt,
      ...(scoreboardAvailable === undefined ? {} : { scoreboardAvailable }) });
    return "updated";
  } catch {
    // Never forward SDK errors containing the API Authorization header.
    throw new ConvexError("Could not refresh CBBD results. Saved results are still available.");
  }
}

export const refreshSeason = action({
  args: { season: v.number() },
  handler: (ctx, { season }): Promise<SyncStatus> => sync(ctx, season, false),
});

export const syncSeason = internalAction({
  args: { season: v.number(), force: v.optional(v.boolean()) },
  handler: (ctx, { season, force }): Promise<SyncStatus> => sync(ctx, season, true, force),
});
