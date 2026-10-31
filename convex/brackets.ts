import { v } from "convex/values";
import { internalMutation, internalQuery, query } from "./_generated/server";
import { marchMadness2026 } from "./data/marchMadness2026";
import { validateTournament, type Tournament } from "./tournamentModel";
import { compareBracket } from "./bracketComparison";

export const list = query({
  args: {},
  handler: async ctx => {
    const brackets = await ctx.db.query("brackets").collect();
    brackets.sort((a, b) => Number(b.slug === "2026-results") - Number(a.slug === "2026-results"));
    return Promise.all(brackets.map(async bracket => ({ ...bracket, imageUrl: await ctx.storage.getUrl(bracket.logoStorageId) })));
  },
});

// Publish the official results with the actual team assets already in storage.
export const publish2026 = internalMutation({
  args: {},
  handler: async ctx => {
    const aliases: Record<string, string> = {
      "Miami (Ohio)": "Miami (OH)",
      "Miami (FL)": "Miami",
      "Cal Baptist": "California Baptist",
      Penn: "Pennsylvania",
      "Long Island": "Long Island University",
      Hawaii: "Hawai'i",
      "Queens (N.C.)": "Queens University",
    };
    const teams = await Promise.all(marchMadness2026.teams.map(async team => {
      const name = (aliases[team.name] ?? team.name).toLowerCase();
      const logo = await ctx.db.query("teamLogos")
        .withIndex("by_name", q => q.eq("normalizedName", name)).unique();
      if (!logo) throw new Error(`Import the real logo for ${team.name} before publishing.`);
      const logoUrl = await ctx.storage.getUrl(logo.storageId);
      if (!logoUrl) throw new Error(`Stored logo missing for ${team.name}.`);
      return { ...team, logoUrl, storageId: logo.storageId };
    }));
    const tournament = {
      ...marchMadness2026,
      teams: teams.map(({ id, name, seed, logoUrl }) => ({ id, name, seed, logoUrl })),
    };
    validateTournament(tournament);
    const bracket = await ctx.db.query("brackets").withIndex("by_slug", q => q.eq("slug", "2026-results")).unique();
    const canonical = await ctx.db.query("tournaments").withIndex("by_slug", q => q.eq("slug", tournament.slug)).unique();
    const existingId = canonical?._id ?? bracket?.tournamentId;
    const tournamentId = existingId ?? await ctx.db.insert("tournaments", tournament);
    if (existingId) await ctx.db.replace(existingId, tournament);
    const champion = teams.find(team => team.id === tournament.winnerTeamId)!;
    const record = {
      slug: "2026-results",
      title: "2026 NCAA March Madness",
      description: "Official 2026 NCAA men's tournament results · Michigan national champions",
      logoStorageId: champion.storageId,
      tournamentId,
    };
    const bracketId = bracket?._id ?? await ctx.db.insert("brackets", record);
    if (bracket) await ctx.db.patch(bracketId, record);

    // Keep existing example predictions, but replace their placeholder images too.
    for (const [slug, favorite] of [["ut-upheaval", "texas"], ["am-avenue", "texas-a-m"]]) {
      const example = await ctx.db.query("brackets").withIndex("by_slug", q => q.eq("slug", slug)).unique();
      if (!example) continue;
      const prediction = await ctx.db.get("tournaments", example.tournamentId);
      if (!prediction) continue;
      const logoById = new Map(teams.map(team => [team.id, team.logoUrl]));
      await ctx.db.patch(prediction._id, {
        teams: prediction.teams.map(team => ({ ...team, logoUrl: logoById.get(team.id) ?? team.logoUrl })),
      });
      await ctx.db.patch(example._id, { logoStorageId: teams.find(team => team.id === favorite)!.storageId });
    }
    return { bracketId, tournamentId, teams: teams.length, games: tournament.games.length };
  },
});

export const get = query({
  args: { id: v.id("brackets") },
  handler: async (ctx, { id }) => {
    const bracket = await ctx.db.get("brackets", id);
    if (!bracket) return null;
    const tournament = await ctx.db.get("tournaments", bracket.tournamentId);
    if (!tournament) return null;
    const referenceSlug = `march-madness-men-${tournament.season}`;
    const reference = tournament.slug === referenceSlug ? tournament
      : await ctx.db.query("tournaments").withIndex("by_slug", q => q.eq("slug", referenceSlug)).unique();
    const results = await ctx.db.query("cbbdSeasons").withIndex("by_season", q => q.eq("season", tournament.season)).unique();
    return { ...tournament, gameComparisons: compareBracket(tournament, reference, results?.games ?? []) };
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
