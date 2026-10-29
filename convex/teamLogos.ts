import { v } from "convex/values";
import { internalAction, internalMutation, internalQuery, query } from "./_generated/server";
import { internal } from "./_generated/api";

const teamLogo = {
  teamId: v.number(),
  teamName: v.string(),
  displayName: v.string(),
  sourceUrl: v.string(),
};

export const getByTeamName = query({
  args: { teamName: v.string() },
  handler: async (ctx, { teamName }) => {
    const logo = await ctx.db.query("teamLogos")
      .withIndex("by_name", q => q.eq("normalizedName", teamName.trim().toLowerCase())).first();
    return logo ? { ...logo, url: await ctx.storage.getUrl(logo.storageId) } : null;
  },
});

export const getByTeamId = query({
  args: { teamId: v.number() },
  handler: async (ctx, { teamId }) => {
    const logo = await ctx.db.query("teamLogos")
      .withIndex("by_team_id", q => q.eq("teamId", teamId)).unique();
    return logo ? { ...logo, url: await ctx.storage.getUrl(logo.storageId) } : null;
  },
});

export const listStored = internalQuery({
  args: {},
  handler: async ctx => (await ctx.db.query("teamLogos").collect())
    .map(({ teamId, sourceUrl }) => ({ teamId, sourceUrl })),
});

export const save = internalMutation({
  args: { ...teamLogo, storageId: v.id("_storage") },
  handler: async (ctx, args) => {
    const existing = await ctx.db.query("teamLogos")
      .withIndex("by_team_id", q => q.eq("teamId", args.teamId)).unique();
    const record = { ...args, normalizedName: args.teamName.trim().toLowerCase() };
    if (existing) {
      await ctx.db.patch(existing._id, record);
      if (existing.storageId !== args.storageId) await ctx.storage.delete(existing.storageId);
    } else {
      await ctx.db.insert("teamLogos", record);
    }
  },
});

export const importBatch = internalAction({
  args: { teams: v.array(v.object(teamLogo)) },
  handler: async (ctx, { teams }): Promise<{ imported: number; failures: { teamName: string; reason: string }[] }> => {
    if (teams.length > 20) throw new Error("Import at most 20 logos per batch.");
    const failures: { teamName: string; reason: string }[] = [];
    await Promise.all(teams.map(async team => {
      try {
        const url = new URL(team.sourceUrl);
        if (url.origin !== "https://cdn.collegefootballdata.com" || !url.pathname.startsWith("/cbb-logos/500/")) {
          throw new Error("Expected a 500px basketball logo from the CFBD CDN.");
        }
        const response = await fetch(url, { signal: AbortSignal.timeout(30_000) });
        if (!response.ok) throw new Error(`Logo download returned ${response.status}`);
        const blob = await response.blob();
        if (!blob.type.startsWith("image/") || blob.size === 0 || blob.size > 5_000_000) {
          throw new Error("Expected a nonempty image under 5 MB.");
        }
        const storageId = await ctx.storage.store(blob);
        try {
          await ctx.runMutation(internal.teamLogos.save, { ...team, storageId });
        } catch (error) {
          await ctx.storage.delete(storageId);
          throw error;
        }
      } catch (error) {
        failures.push({ teamName: team.teamName, reason: error instanceof Error ? error.message : "Import failed" });
      }
    }));
    return { imported: teams.length - failures.length, failures };
  },
});
