import { defineSchema, defineTable } from "convex/server";
import { tournamentValidator } from "./tournamentModel";
import { v } from "convex/values";

export default defineSchema({
  tournaments: defineTable(tournamentValidator).index("by_slug", ["slug"]),
  brackets: defineTable({
    slug: v.string(),
    title: v.string(),
    description: v.string(),
    logoStorageId: v.id("_storage"),
    tournamentId: v.id("tournaments"),
  }).index("by_slug", ["slug"]),
});
