import { v, type Infer } from "convex/values";

export const kalshiChartValidator = v.object({
  tournamentSlug: v.string(),
  gameId: v.string(),
  eventTicker: v.string(),
  marketKind: v.union(v.literal("game"), v.literal("championship")),
  status: v.union(v.literal("live"), v.literal("closed"), v.literal("settled")),
  fetchedAt: v.number(),
  series: v.array(v.object({
    teamName: v.string(),
    ticker: v.string(),
    price: v.optional(v.number()),
    points: v.array(v.object({ timestamp: v.number(), price: v.number() })),
  })),
});

export type KalshiChart = Infer<typeof kalshiChartValidator>;
