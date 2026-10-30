import { defineSchema, defineTable } from "convex/server";
import { tournamentValidator } from "./tournamentModel";
import { gameOddsValidator } from "./gameOddsModel";
import { v } from "convex/values";

export default defineSchema({
  gameOdds: defineTable(gameOddsValidator).index("by_game", ["tournamentSlug", "gameId"]),
  teamLogos: defineTable({
    teamId: v.number(),
    teamName: v.string(),
    normalizedName: v.string(),
    displayName: v.string(),
    sourceUrl: v.string(),
    storageId: v.id("_storage"),
  }).index("by_team_id", ["teamId"]).index("by_name", ["normalizedName"]),
  tournaments: defineTable(tournamentValidator).index("by_slug", ["slug"]),
  brackets: defineTable({
    slug: v.string(),
    title: v.string(),
    description: v.string(),
    logoStorageId: v.id("_storage"),
    tournamentId: v.id("tournaments"),
  }).index("by_slug", ["slug"]),
});
