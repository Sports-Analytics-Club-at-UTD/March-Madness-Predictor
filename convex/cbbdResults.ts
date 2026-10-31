import type { GameInfo, ScoreboardGame } from "cbbd";
import type { CbbdGame } from "./cbbdModel";
import type { Tournament } from "./tournamentModel";

const aliases: Record<string, string> = {
  "Miami (Ohio)": "Miami (OH)", "Miami (FL)": "Miami", "Cal Baptist": "California Baptist",
  Penn: "Pennsylvania", "Long Island": "Long Island University", Hawaii: "Hawai'i", "Queens (N.C.)": "Queens University",
};
const schoolName = (name: string) => (aliases[name] ?? name).toLowerCase().replace(/[^a-z0-9]/g, "");
const roundPatterns = [/First Four/i, /1st Round|First Round|Round of 64/i, /2nd Round|Second Round|Round of 32/i,
  /Sweet 16|Sweet Sixteen/i, /Elite 8|Elite Eight/i, /Final Four|Semifinal/i, /National Championship/i];

export function confirmedWinner(game: CbbdGame): string | undefined {
  if (game.status !== "final" || game.homePoints === undefined || game.awayPoints === undefined ||
    !Number.isInteger(game.homePoints) || !Number.isInteger(game.awayPoints) ||
    game.homePoints < 0 || game.awayPoints < 0 || game.homePoints === game.awayPoints) return undefined;
  return game.homePoints > game.awayPoints ? game.homeTeamId : game.awayTeamId;
}

export function normalizeCbbdGames(games: GameInfo[], reference: Tournament): CbbdGame[] {
  return games.flatMap(game => {
    if (game.season !== reference.season || game.tournament?.toUpperCase() !== "NCAA") return [];
    const home = reference.teams.filter(team => schoolName(team.name) === schoolName(game.homeTeam));
    const away = reference.teams.filter(team => schoolName(team.name) === schoolName(game.awayTeam));
    const round = roundPatterns.findIndex(pattern => pattern.test(game.gameNotes ?? ""));
    // Skip unresolved participants and unknown rounds instead of guessing their identity.
    if (home.length !== 1 || away.length !== 1 || home[0].id === away[0].id || round < 0) return [];
    return [{ id: game.id, round, startDate: game.startDate, status: game.status,
      homeTeamId: home[0].id, awayTeamId: away[0].id,
      homeCbbdTeamId: game.homeTeamId, awayCbbdTeamId: game.awayTeamId,
      ...(game.homePoints == null ? {} : { homePoints: game.homePoints }),
      ...(game.awayPoints == null ? {} : { awayPoints: game.awayPoints }),
    }];
  });
}

export function mergeCbbdGames(saved: CbbdGame[], incoming: CbbdGame[]): CbbdGame[] {
  const games = new Map(saved.map(game => [game.id, game]));
  for (const game of incoming) {
    const previous = games.get(game.id);
    if (previous && confirmedWinner(previous) && !confirmedWinner(game)) continue;
    games.set(game.id, game);
  }
  return [...games.values()];
}

export function applyScoreboard(saved: CbbdGame[], scoreboard: ScoreboardGame[]): CbbdGame[] {
  const byId = new Map(scoreboard.map(game => [game.id, game]));
  const incoming = saved.flatMap(game => {
    const live = byId.get(game.id);
    if (!live || live.homeTeam.id !== game.homeCbbdTeamId || live.awayTeam.id !== game.awayCbbdTeamId) return [];
    const updated: CbbdGame = { ...game, status: live.status };
    delete updated.homePoints;
    delete updated.awayPoints;
    if (live.homeTeam.points != null) updated.homePoints = live.homeTeam.points;
    if (live.awayTeam.points != null) updated.awayPoints = live.awayTeam.points;
    return [updated];
  });
  return mergeCbbdGames(saved, incoming);
}

export function hasNewTournamentGame(saved: CbbdGame[], scoreboard: ScoreboardGame[]) {
  const knownGames = new Set(saved.map(game => game.id));
  const teams = new Set(saved.flatMap(game => [game.homeCbbdTeamId, game.awayCbbdTeamId]));
  return scoreboard.some(game => !knownGames.has(game.id) && teams.has(game.homeTeam.id) && teams.has(game.awayTeam.id));
}
