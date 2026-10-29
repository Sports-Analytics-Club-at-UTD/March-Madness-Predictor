import { v } from "convex/values";
import { internalMutation, internalQuery, query } from "./_generated/server";
import { marchMadness2026 } from "./data/marchMadness2026";
import { validateTournament, type Tournament } from "./tournamentModel";

export const list = query({
  args: {},
  handler: async ctx => {
    const brackets = await ctx.db.query("brackets").collect();
    return Promise.all(brackets.map(async bracket => ({ ...bracket, imageUrl: await ctx.storage.getUrl(bracket.logoStorageId) })));
  },
});

export const get = query({
  args: { id: v.id("brackets") },
  handler: async (ctx, { id }) => {
    const bracket = await ctx.db.get("brackets", id);
    if (!bracket) return null;
    return ctx.db.get("tournaments", bracket.tournamentId);
  },
});

export const generateUploadUrl = internalMutation({
  args: {},
  handler: async ctx => ctx.storage.generateUploadUrl(),
});

export const getSeedLogo = internalQuery({
  args: {},
  handler: async ctx => {
    const bracket = await ctx.db.query("brackets").withIndex("by_slug", q => q.eq("slug", "ut-upheaval")).unique();
    return bracket ? bracket.logoStorageId : null;
  },
});

function examplePrediction(favorite: string, slug: string, title: string, logoUrl: string): Tournament {
  const original = marchMadness2026;
  const winners = new Map<string, string>();
  const games = [...original.games].sort((a,b) => a.round - b.round || a.order - b.order).map(game => {
    const topTeamId = game.topSourceGameId ? winners.get(game.topSourceGameId)! : game.topTeamId;
    const bottomTeamId = game.bottomSourceGameId ? winners.get(game.bottomSourceGameId)! : game.bottomTeamId;
    const winnerTeamId = [topTeamId, bottomTeamId].includes(favorite) ? favorite
      : [topTeamId, bottomTeamId].includes(game.winnerTeamId) ? game.winnerTeamId : topTeamId;
    winners.set(game.id, winnerTeamId);
    const rest = { ...game };
    delete rest.topScore;
    delete rest.bottomScore;
    return { ...rest, topTeamId, bottomTeamId, winnerTeamId };
  });
  return { ...original, slug, name: `${title} — example prediction`, winnerTeamId: favorite,
    sourceUrls: original.sourceUrls, teams: original.teams.map(team => ({ ...team, logoUrl })), games };
}

export const seedExamples = internalMutation({
  args: { logoStorageId: v.id("_storage") },
  handler: async (ctx, { logoStorageId }) => {
    const logoUrl = await ctx.storage.getUrl(logoStorageId);
    if (!logoUrl) throw new Error("The default logo has not been uploaded.");
    const examples = [
      { slug: "ut-upheaval", title: "The UT Upheaval", description: "Example prediction · Texas wins it all", favorite: "texas" },
      { slug: "am-avenue", title: "A&M Avenue", description: "Example prediction · Texas A&M wins it all", favorite: "texas-a-m" },
      { slug: "2026-results", title: "The Real Deal", description: "2026 men's tournament · historical results", favorite: null },
    ];
    const ids = [];
    for (const example of examples) {
      const existing = await ctx.db.query("brackets").withIndex("by_slug", q => q.eq("slug", example.slug)).unique();
      if (existing) { ids.push(existing._id); continue; }
      const tournament = example.favorite
        ? examplePrediction(example.favorite, example.slug, example.title, logoUrl)
        : { ...marchMadness2026, slug: "2026-results-card", teams: marchMadness2026.teams.map(team => ({ ...team, logoUrl })) };
      validateTournament(tournament);
      const tournamentId = await ctx.db.insert("tournaments", tournament);
      ids.push(await ctx.db.insert("brackets", { slug: example.slug, title: example.title, description: example.description, logoStorageId, tournamentId }));
    }
    return ids;
  },
});
