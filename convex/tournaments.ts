import { v } from "convex/values";
import { internalMutation, query } from "./_generated/server";
import { marchMadness2026 } from "./data/marchMadness2026";
import { validateTournament } from "./tournamentModel";

export const getBySlug = query({
  args: { slug: v.string() },
  handler: async (ctx, { slug }) => ctx.db.query("tournaments").withIndex("by_slug", q => q.eq("slug", slug)).unique(),
});

// CLI-only seed: no unauthenticated browser writes to historical results.
export const seed2026 = internalMutation({
  args: {},
  handler: async ctx => {
    validateTournament(marchMadness2026);
    const existing = await ctx.db.query("tournaments").withIndex("by_slug", q => q.eq("slug", marchMadness2026.slug)).unique();
    if (existing) return existing._id;
    return ctx.db.insert("tournaments", marchMadness2026);
  },
});
