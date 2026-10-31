import { v } from "convex/values";
import { internalMutation, internalQuery } from "./_generated/server";
import { cbbdGameValidator } from "./cbbdModel";
import { mergeCbbdGames } from "./cbbdResults";

export const context = internalQuery({
  args: { season: v.number() },
  handler: async (ctx, { season }) => ({
    reference: await ctx.db.query("tournaments").withIndex("by_slug", q => q.eq("slug", `march-madness-men-${season}`)).unique(),
    saved: await ctx.db.query("cbbdSeasons").withIndex("by_season", q => q.eq("season", season)).unique(),
  }),
});

export const claimRefresh = internalMutation({
  args: { season: v.number(), force: v.boolean() },
  handler: async (ctx, { season, force }) => {
    const saved = await ctx.db.query("cbbdSeasons").withIndex("by_season", q => q.eq("season", season)).unique();
    const now = Date.now();
    if (!force && saved && now - saved.lastAttemptAt < 15 * 60_000) return false;
    if (saved) await ctx.db.patch(saved._id, { lastAttemptAt: now });
    else await ctx.db.insert("cbbdSeasons", { season, games: [], lastAttemptAt: now, updatedAt: 0, scheduleFetchedAt: 0 });
    return true;
  },
});

export const save = internalMutation({
  args: { season: v.number(), games: v.array(cbbdGameValidator), scheduleFetchedAt: v.number(), scoreboardAvailable: v.optional(v.boolean()) },
  handler: async (ctx, { season, games, scheduleFetchedAt, scoreboardAvailable }) => {
    const saved = await ctx.db.query("cbbdSeasons").withIndex("by_season", q => q.eq("season", season)).unique();
    const now = Date.now();
    const availability = scoreboardAvailable ?? saved?.scoreboardAvailable;
    const record = { season, games: mergeCbbdGames(saved?.games ?? [], games), scheduleFetchedAt,
      updatedAt: now, lastAttemptAt: now, ...(availability === undefined ? {} : { scoreboardAvailable: availability }) };
    if (saved) await ctx.db.replace(saved._id, record);
    else await ctx.db.insert("cbbdSeasons", record);
  },
});
