import { v, type Infer } from "convex/values";

export const cbbdGameValidator = v.object({
  id: v.number(),
  round: v.number(),
  startDate: v.string(),
  status: v.union(v.literal("scheduled"), v.literal("in_progress"), v.literal("final"), v.literal("postponed"), v.literal("cancelled")),
  homeTeamId: v.string(),
  awayTeamId: v.string(),
  homeCbbdTeamId: v.number(),
  awayCbbdTeamId: v.number(),
  homePoints: v.optional(v.number()),
  awayPoints: v.optional(v.number()),
});
export type CbbdGame = Infer<typeof cbbdGameValidator>;

export const cbbdSeasonValidator = v.object({
  season: v.number(),
  games: v.array(cbbdGameValidator),
  lastAttemptAt: v.number(),
  updatedAt: v.number(),
  scheduleFetchedAt: v.number(),
  scoreboardAvailable: v.optional(v.boolean()),
});
