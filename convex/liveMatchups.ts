import { v } from "convex/values";
import { internalMutation, internalQuery, query, type QueryCtx } from "./_generated/server";
import liveMatchupModel from "./data/liveMatchupModel.json";
import {
  liveGameInput, liveModel, liveRatingRunValidator, liveTeamRatingFields,
  summarizePredictions, venueValidator, weekStart, winProbability, type TeamFeatures,
} from "./liveMatchupModel";

async function latestRun(ctx: QueryCtx, season?: number) {
  if (season !== undefined) return ctx.db.query("liveRatingRuns").withIndex("by_season", q => q.eq("season", season)).unique();
  return ctx.db.query("liveRatingRuns").withIndex("by_season").order("desc").first();
}

const rating = (ctx: QueryCtx, season: number, teamId: number) =>
  ctx.db.query("liveTeamRatings").withIndex("by_season_team", q => q.eq("season", season).eq("teamId", teamId)).unique();

export const importRatings = internalMutation({
  args: { run: v.object(liveRatingRunValidator), teams: v.array(v.object(liveTeamRatingFields)) },
  handler: async (ctx, { run, teams }) => {
    for (const team of teams) {
      const existing = await rating(ctx, run.season, team.teamId);
      const record = { season: run.season, ...team };
      if (existing) await ctx.db.replace(existing._id, record);
      else await ctx.db.insert("liveTeamRatings", record);
    }
    const saved = await ctx.db.query("liveRatingRuns").withIndex("by_season", q => q.eq("season", run.season)).unique();
    if (saved) await ctx.db.replace(saved._id, run);
    else await ctx.db.insert("liveRatingRuns", run);
    return { teams: teams.length };
  },
});

export const unfinishedGameIds = internalQuery({
  args: { season: v.number() },
  handler: async (ctx, { season }) => (await ctx.db.query("liveGamePredictions")
    .withIndex("by_season_game", q => q.eq("season", season)).collect())
    .filter(game => game.status !== "final" && game.status !== "cancelled").map(game => game.gameId),
});

/**
 * Predictions are made only before tip-off from the stored ratings (which never include the
 * game itself) and refresh until then; afterwards only the status and score change.
 */
export const upsertGames = internalMutation({
  args: { season: v.number(), games: v.array(v.object(liveGameInput)) },
  handler: async (ctx, { season, games }) => {
    const now = Date.now();
    const run = await latestRun(ctx, season);
    const counts = { created: 0, refreshed: 0, resultsUpdated: 0, skippedStarted: 0, unavailable: 0 };
    for (const game of games) {
      const existing = await ctx.db.query("liveGamePredictions")
        .withIndex("by_season_game", q => q.eq("season", season).eq("gameId", game.gameId)).unique();
      const started = Date.parse(game.startDate) <= now;
      if (started) {
        if (!existing) counts.skippedStarted++;
        else {
          await ctx.db.patch(existing._id, { status: game.status, homePoints: game.homePoints, awayPoints: game.awayPoints });
          counts.resultsUpdated++;
        }
        continue;
      }
      const [home, away] = await Promise.all([rating(ctx, season, game.homeTeamId), rating(ctx, season, game.awayTeamId)]);
      const missing = [!home && game.homeTeam, !away && game.awayTeam].filter(Boolean);
      const prediction = run && home && away
        ? { homeWinProbability: winProbability(liveModel, home, away, game.neutralSite ? "neutral" : "a"), unavailableReason: undefined }
        : { homeWinProbability: undefined, unavailableReason: run ? `No live features yet for ${missing.join(" and ")}.` : "Ratings have not been synced." };
      if (prediction.homeWinProbability === undefined) counts.unavailable++;
      const record = {
        ...game, season, weekStart: weekStart(game.startDate), modelId: liveMatchupModel.id,
        ...prediction, predictedAt: now, ratingsUpdatedAt: run?.updatedAt,
      };
      if (existing) {
        await ctx.db.replace(existing._id, record);
        counts.refreshed++;
      } else {
        await ctx.db.insert("liveGamePredictions", record);
        counts.created++;
      }
    }
    return counts;
  },
});

const modelSummary = {
  id: liveMatchupModel.id,
  source: liveMatchupModel.source,
  features: liveMatchupModel.features,
  coefficients: liveMatchupModel.coefficients,
  trainingSeasons: liveMatchupModel.trainingSeasons,
  homeEdgePoints: liveMatchupModel.homeEdgePoints,
  backtest: liveMatchupModel.backtest,
};

export const overview = query({
  args: {},
  handler: async ctx => {
    const run = await latestRun(ctx);
    if (!run) return { model: modelSummary, run: null, teams: [] };
    const teams = await ctx.db.query("liveTeamRatings").withIndex("by_season_team", q => q.eq("season", run.season)).collect();
    teams.sort((a, b) => a.teamName.localeCompare(b.teamName));
    return {
      model: modelSummary, run,
      teams: teams.map(({ teamId, teamName, conference, games, priorElo }) => ({ teamId, teamName, conference, games, hasPrior: priorElo !== undefined })),
    };
  },
});

export const predict = query({
  args: { teamAId: v.number(), teamBId: v.number(), venue: venueValidator },
  handler: async (ctx, { teamAId, teamBId, venue }) => {
    const run = await latestRun(ctx);
    if (!run) return null;
    const [a, b] = await Promise.all([rating(ctx, run.season, teamAId), rating(ctx, run.season, teamBId)]);
    if (!a || !b || teamAId === teamBId) return { season: run.season, a, b, probabilityA: null };
    const features = (team: TeamFeatures) => ({ quality: team.quality, elo: team.elo, netEff: team.netEff });
    return { season: run.season, a, b, probabilityA: winProbability(liveModel, features(a), features(b), venue) };
  },
});

export const weeks = query({
  args: {},
  handler: async ctx => {
    const run = await latestRun(ctx);
    if (!run) return null;
    const season = run.season;
    const games = await ctx.db.query("liveGamePredictions").withIndex("by_season_week", q => q.eq("season", season)).collect();
    const byWeek = new Map<string, typeof games>();
    for (const game of games) byWeek.set(game.weekStart, [...(byWeek.get(game.weekStart) ?? []), game]);
    return {
      season, run,
      overall: summarizePredictions(games),
      weeks: [...byWeek].map(([week, list]) => ({ weekStart: week, ...summarizePredictions(list) }))
        .sort((a, b) => b.weekStart.localeCompare(a.weekStart)),
    };
  },
});

export const week = query({
  args: { season: v.number(), weekStart: v.string() },
  handler: async (ctx, { season, weekStart }) => {
    const games = await ctx.db.query("liveGamePredictions")
      .withIndex("by_season_week", q => q.eq("season", season).eq("weekStart", weekStart)).collect();
    return games.sort((a, b) => a.startDate.localeCompare(b.startDate) || a.gameId - b.gameId);
  },
});
