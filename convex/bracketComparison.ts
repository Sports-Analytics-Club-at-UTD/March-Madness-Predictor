import type { Tournament, TournamentGame } from "./tournamentModel";
import type { CbbdGame } from "./cbbdModel";
import { confirmedWinner } from "./cbbdResults";

export type GameReference = { tournamentSlug: string; gameId: string };
export type GameComparison = {
  status: "pending" | "intact" | "break" | "hypothetical";
  oddsSource?: GameReference;
};
export type ComparedTournament = Tournament & { gameComparisons?: Record<string, GameComparison> };

export function compareGame(game: TournamentGame, reference: Tournament, results: CbbdGame[] = []): GameComparison {
  const sameTeams = (home: string, away: string) =>
    (home === game.topTeamId && away === game.bottomTeamId) || (away === game.topTeamId && home === game.bottomTeamId);
  const matches = reference.games.filter(candidate => candidate.round === game.round &&
    sameTeams(candidate.topTeamId, candidate.bottomTeamId));
  const oddsSource = matches.length === 1 ? { tournamentSlug: reference.slug, gameId: matches[0].id } : undefined;
  const actual = results.filter(result => result.round === game.round && sameTeams(result.homeTeamId, result.awayTeamId));
  const winner = actual.length === 1 ? confirmedWinner(actual[0]) : undefined;
  if (winner) return { status: game.winnerTeamId === winner ? "intact" : "break", ...(oddsSource ? { oddsSource } : {}) };
  const eliminated = results.some(result => {
    const winner = confirmedWinner(result);
    if (!winner || result.round >= game.round) return false;
    const loser = winner === result.homeTeamId ? result.awayTeamId : result.homeTeamId;
    return loser === game.topTeamId || loser === game.bottomTeamId;
  });
  return { status: eliminated ? "hypothetical" : "pending", ...(!eliminated && oddsSource ? { oddsSource } : {}) };
}

export function compareBracket(tournament: Tournament, reference: Tournament | null, results: CbbdGame[] = []): Record<string, GameComparison> {
  if (!reference || tournament.season !== reference.season) return {};
  return Object.fromEntries(tournament.games.map(game => [game.id, compareGame(game, reference, results)]));
}
