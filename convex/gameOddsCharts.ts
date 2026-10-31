import { v } from "convex/values";
import { internalMutation, query } from "./_generated/server";
import { kalshiChartValidator } from "./kalshiChartModel";
import { resolveGameReference } from "./resolveGameReference";

export const get = query({
  args: { tournamentSlug: v.string(), gameId: v.string() },
  handler: async (ctx, args) => {
    const source = await resolveGameReference(ctx, args);
    if (!source) return null;
    return ctx.db.query("gameOddsCharts")
      .withIndex("by_game", q => q.eq("tournamentSlug", source.tournamentSlug).eq("gameId", source.gameId)).unique();
  },
});

export const save = internalMutation({
  args: { chart: kalshiChartValidator },
  handler: async (ctx, { chart }) => {
    const existing = await ctx.db.query("gameOddsCharts")
      .withIndex("by_game", q => q.eq("tournamentSlug", chart.tournamentSlug).eq("gameId", chart.gameId)).unique();
    if (existing) await ctx.db.replace(existing._id, chart);
    else await ctx.db.insert("gameOddsCharts", chart);
  },
});
