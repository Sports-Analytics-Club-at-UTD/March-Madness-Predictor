import { v, type Infer } from "convex/values";

export const tournamentValidator = v.object({
  slug: v.string(),
  name: v.string(),
  season: v.number(),
  winnerTeamId: v.string(),
  sourceUrls: v.array(v.string()),
  teams: v.array(v.object({
    id: v.string(),
    name: v.string(),
    logoUrl: v.string(),
    seed: v.number(),
  })),
  games: v.array(v.object({
    id: v.string(),
    round: v.number(),
    order: v.number(),
    label: v.string(),
    topTeamId: v.string(),
    bottomTeamId: v.string(),
    winnerTeamId: v.string(),
    topScore: v.optional(v.number()),
    bottomScore: v.optional(v.number()),
    topSourceGameId: v.optional(v.string()),
    bottomSourceGameId: v.optional(v.string()),
  })),
});

export type Tournament = Infer<typeof tournamentValidator>;
export type TournamentGame = Tournament["games"][number];

export function validateTournament(tournament: Tournament) {
  const teams = new Map(tournament.teams.map(team => [team.id, team]));
  const games = new Map(tournament.games.map(game => [game.id, game]));
  if (teams.size !== tournament.teams.length || games.size !== tournament.games.length) {
    throw new Error("Team and game IDs must be unique.");
  }
  if (teams.size < 2 || games.size !== teams.size - 1) {
    throw new Error("A completed single-elimination bracket needs one fewer game than teams.");
  }
  const usedSources = new Set<string>();
  const entrants = new Set<string>();
  const positions = new Set<string>();
  for (const game of games.values()) {
    if (!Number.isInteger(game.round) || game.round < 0 || !Number.isInteger(game.order) || game.order < 0) {
      throw new Error(`Invalid round or order for ${game.id}.`);
    }
    const position = `${game.round}:${game.order}`;
    if (positions.has(position)) throw new Error(`Duplicate bracket position ${position}.`);
    positions.add(position);
    if (!teams.has(game.topTeamId) || !teams.has(game.bottomTeamId) || game.topTeamId === game.bottomTeamId) {
      throw new Error(`Invalid teams in ${game.id}.`);
    }
    if (game.winnerTeamId !== game.topTeamId && game.winnerTeamId !== game.bottomTeamId) {
      throw new Error(`Winner must be a participating team in ${game.id}.`);
    }
    if (![game.topScore, game.bottomScore].every(score => score === undefined || (Number.isInteger(score) && score >= 0))) {
      throw new Error(`Invalid final scores in ${game.id}.`);
    }
    if (game.topScore !== undefined && game.bottomScore !== undefined) {
      if (game.topScore === game.bottomScore) throw new Error(`Tied final scores in ${game.id}.`);
      const winner = game.topScore > game.bottomScore ? game.topTeamId : game.bottomTeamId;
      if (game.winnerTeamId !== winner) throw new Error(`Winner disagrees with scores in ${game.id}.`);
    }
    for (const [sourceId, teamId] of [[game.topSourceGameId, game.topTeamId], [game.bottomSourceGameId, game.bottomTeamId]] as const) {
      if (!sourceId) {
        if (game.round > 1 || entrants.has(teamId)) throw new Error(`Invalid entry for ${teamId}.`);
        entrants.add(teamId);
        continue;
      }
      const source = games.get(sourceId);
      if (!source || source.round !== game.round - 1 || source.winnerTeamId !== teamId || usedSources.has(sourceId)) {
        throw new Error(`Invalid advancement from ${sourceId} to ${game.id}.`);
      }
      usedSources.add(sourceId);
    }
  }
  const finals = tournament.games.filter(game => !usedSources.has(game.id));
  if (finals.length !== 1 || finals[0].winnerTeamId !== tournament.winnerTeamId || entrants.size !== teams.size) {
    throw new Error("The bracket must have one championship and a matching tournament winner.");
  }
}
