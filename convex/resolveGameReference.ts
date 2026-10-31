import type { QueryCtx } from "./_generated/server";
import { compareGame, type GameReference } from "./bracketComparison";

export async function resolveGameReference(ctx: QueryCtx, { tournamentSlug, gameId }: GameReference): Promise<GameReference | null> {
  const tournament = await ctx.db.query("tournaments").withIndex("by_slug", q => q.eq("slug", tournamentSlug)).unique();
  const game = tournament?.games.find(game => game.id === gameId);
  if (!tournament || !game) return null;
  const referenceSlug = `march-madness-men-${tournament.season}`;
  if (tournament.slug === referenceSlug) return { tournamentSlug, gameId };
  const reference = await ctx.db.query("tournaments").withIndex("by_slug", q => q.eq("slug", referenceSlug)).unique();
  const results = await ctx.db.query("cbbdSeasons").withIndex("by_season", q => q.eq("season", tournament.season)).unique();
  return reference && reference.season === tournament.season ? compareGame(game, reference, results?.games ?? []).oddsSource ?? null : null;
}
