import { v } from "convex/values";

export const gameOddsValidator = v.object({
  tournamentSlug: v.string(),
  gameId: v.string(),
  eventTicker: v.optional(v.string()),
  marketKind: v.optional(v.union(v.literal("game"), v.literal("championship"))),
  outcomes: v.array(v.object({
    teamId: v.string(),
    teamName: v.string(),
    ticker: v.string(),
    firstPrice: v.optional(v.number()),
    firstTimestamp: v.optional(v.number()),
    lastPrice: v.optional(v.number()),
    lastTimestamp: v.optional(v.number()),
  })),
  importedAt: v.number(),
});
