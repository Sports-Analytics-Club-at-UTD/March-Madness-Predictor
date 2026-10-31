import { v } from "convex/values";
import { internalMutation, internalQuery, query } from "./_generated/server";
import { gameOddsValidator } from "./gameOddsModel";
import { resolveGameReference } from "./resolveGameReference";

export const get = query({
  args: { tournamentSlug: v.string(), gameId: v.string() },
  handler: async (ctx, args) => {
    const source = await resolveGameReference(ctx, args);
    if (!source) return null;
    return ctx.db.query("gameOdds")
      .withIndex("by_game", q => q.eq("tournamentSlug", source.tournamentSlug).eq("gameId", source.gameId)).unique();
  },
});

export const listStored = internalQuery({
  args: { tournamentSlug: v.string() },
  handler: async (ctx, { tournamentSlug }) => (await ctx.db.query("gameOdds")
    .withIndex("by_game", q => q.eq("tournamentSlug", tournamentSlug)).collect())
    .filter(record => record.outcomes.length === 2).map(record => record.gameId),
});

export const save = internalMutation({
  args: { records: v.array(gameOddsValidator) },
  handler: async (ctx, { records }) => {
    for (const record of records) {
      if (record.tournamentSlug !== "march-madness-men-2026") throw new Error("Only official 2026 games can be imported.");
      const tournament = await ctx.db.query("tournaments").withIndex("by_slug", q => q.eq("slug", record.tournamentSlug)).unique();
      const game = tournament?.games.find(game => game.id === record.gameId);
      if (!game || (record.outcomes.length !== 0 && (record.outcomes.length !== 2 ||
        ![game.topTeamId, game.bottomTeamId].every(id => record.outcomes.some(outcome => outcome.teamId === id))))) {
        throw new Error("Odds must belong to both teams in an existing game.");
      }
      for (const outcome of record.outcomes) {
        for (const price of [outcome.firstPrice, outcome.lastPrice]) {
          if (price !== undefined && (!Number.isFinite(price) || price < 0 || price > 1)) throw new Error("Invalid market price.");
        }
      }
      const existing = await ctx.db.query("gameOdds")
        .withIndex("by_game", q => q.eq("tournamentSlug", record.tournamentSlug).eq("gameId", record.gameId)).unique();
      if (existing) await ctx.db.replace(existing._id, record);
      else await ctx.db.insert("gameOdds", record);
    }
  },
});
