// Copy this module into convex/ on the site branch, then sync the deployment.
// Uses existing tables: no schema or browser changes are required.
import { v } from "convex/values";
import { internalMutation, internalQuery } from "./_generated/server";
import { tournamentValidator, validateTournament } from "./tournamentModel";

export const getRoster = internalQuery({
  args: { season: v.number() },
  handler: async (ctx, { season }) => {
    const reference = await ctx.db.query("tournaments")
      .withIndex("by_slug", q => q.eq("slug", `march-madness-men-${season}`)).unique();
    if (!reference) throw new Error(`Import the official ${season} tournament reference first.`);
    const card = (await ctx.db.query("brackets").collect())
      .find(bracket => bracket.tournamentId === reference._id);
    if (!card) throw new Error("Publish an official-results card with a stored logo first.");
    if (!await ctx.storage.getUrl(card.logoStorageId)) throw new Error("Official card logo is missing from storage.");
    return { season, teams: reference.teams, logoStorageId: card.logoStorageId };
  },
});

export const importForecast = internalMutation({
  args: {
    items: v.array(v.object({
      tournament: tournamentValidator,
      title: v.string(),
      description: v.string(),
      logoStorageId: v.id("_storage"),
    })),
  },
  handler: async (ctx, { items }) => {
    if (!items.length) throw new Error("No brackets to import.");
    const slugs = new Set<string>();
    const result = [];
    // A single mutation is transactional: a failure rolls back every bracket.
    for (const item of items) {
      const tournament = item.tournament;
      if (!new RegExp(`^forecast-${tournament.season}-[a-f0-9]{64}$`).test(tournament.slug)) {
        throw new Error("Only forecast-owned slugs may be imported.");
      }
      if (slugs.has(tournament.slug)) throw new Error("Duplicate forecast slug in request.");
      slugs.add(tournament.slug);
      validateTournament(tournament);
      const reference = await ctx.db.query("tournaments")
        .withIndex("by_slug", q => q.eq("slug", `march-madness-men-${tournament.season}`)).unique();
      if (!reference) throw new Error("The official reference for this season is missing.");
      const canonical = new Map(reference.teams.map(team => [team.id, team]));
      for (const team of tournament.teams) {
        const known = canonical.get(team.id);
        if (!known || known.name !== team.name || known.seed !== team.seed) {
          throw new Error(`Team ${team.id} does not match the season's canonical roster.`);
        }
      }
      if (!await ctx.storage.getUrl(item.logoStorageId)) throw new Error("Card logo is missing from storage.");
      // The renderer requires a balanced main bracket with consecutive rounds.
      const games = new Map(tournament.games.map(game => [game.id, game]));
      const finalRound = Math.max(...tournament.games.map(game => game.round));
      if (tournament.teams.length < 4 || tournament.teams.length > 64 || tournament.teams.length !== 2 ** finalRound) {
        throw new Error("Expected a balanced main bracket of 4–64 teams.");
      }
      for (const game of tournament.games) {
        if (game.round < 1) throw new Error("First Four imports are not supported yet.");
        for (const source of [game.topSourceGameId, game.bottomSourceGameId]) {
          if (game.round === 1 ? source !== undefined : !source || games.get(source)?.round !== game.round - 1) {
            throw new Error("Expected direct round-one entrants and consecutive feeder games.");
          }
        }
      }
      const existing = await ctx.db.query("tournaments")
        .withIndex("by_slug", q => q.eq("slug", tournament.slug)).unique();
      const card = await ctx.db.query("brackets")
        .withIndex("by_slug", q => q.eq("slug", tournament.slug)).unique();
      if (card && card.tournamentId !== existing?._id) throw new Error("Existing card has a conflicting tournament reference.");
      // Refresh assets from the server's canonical records rather than trusting stale URLs.
      const record = { ...tournament, teams: tournament.teams.map(team => ({ ...canonical.get(team.id)! })) };
      const tournamentId = existing?._id ?? await ctx.db.insert("tournaments", record);
      if (existing) await ctx.db.replace(existing._id, record);
      const cardRecord = {
        slug: tournament.slug, title: item.title, description: item.description,
        logoStorageId: item.logoStorageId, tournamentId,
      };
      const bracketId = card?._id ?? await ctx.db.insert("brackets", cardRecord);
      if (card) await ctx.db.patch(card._id, cardRecord);
      result.push({ slug: tournament.slug, tournamentId, bracketId, updated: Boolean(existing) });
    }
    return result;
  },
});
