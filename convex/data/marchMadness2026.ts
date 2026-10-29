import type { Tournament, TournamentGame } from "../tournamentModel";

// Results: NCAA's official 2026 bracket, including the four play-in games.
const teams: Tournament["teams"] = [];
const games: TournamentGame[] = [];
const id = (name: string) => name.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/-$/, "");
function team(name: string, seed: number) {
  if (!teams.some(team => team.id === id(name))) teams.push({ id: id(name), name, seed, logoUrl: "/favicon.ico" });
  return id(name);
}
function game(gameId: string, round: number, order: number, label: string, topTeamId: string, bottomTeamId: string, scores: number[], topSourceGameId?: string, bottomSourceGameId?: string) {
  const result: TournamentGame = {
    id: gameId, round, order, label, topTeamId, bottomTeamId,
    topScore: scores[0], bottomScore: scores[1],
    winnerTeamId: scores[0] > scores[1] ? topTeamId : bottomTeamId,
    ...(topSourceGameId ? { topSourceGameId } : {}),
    ...(bottomSourceGameId ? { bottomSourceGameId } : {}),
  };
  games.push(result);
  return result;
}
const playIns = [
  game("first-four-0", 0, 0, "First Four", team("UMBC", 16), team("Howard", 16), [83, 86]),
  game("first-four-1", 0, 1, "First Four", team("Texas", 11), team("NC State", 11), [68, 66]),
  game("first-four-2", 0, 2, "First Four", team("Prairie View A&M", 16), team("Lehigh", 16), [67, 55]),
  game("first-four-3", 0, 3, "First Four", team("Miami (Ohio)", 11), team("SMU", 11), [89, 79]),
];
const regions: { name: string; names: string[]; scores: number[][][] }[] = [
  { name: "East", names: ["Duke", "Siena", "Ohio State", "TCU", "St. John's", "Northern Iowa", "Kansas", "Cal Baptist", "Louisville", "South Florida", "Michigan State", "North Dakota State", "UCLA", "UCF", "UConn", "Furman"], scores: [
    [[71,65],[64,66],[79,53],[68,60],[83,79],[92,67],[75,71],[82,71]],
    [[81,58],[67,65],[69,77],[57,73]], [[80,75],[63,67]], [[72,73]],
  ] },
  { name: "South", names: ["Florida", "Prairie View A&M", "Clemson", "Iowa", "Vanderbilt", "McNeese", "Nebraska", "Troy", "North Carolina", "VCU", "Illinois", "Penn", "Saint Mary's", "Texas A&M", "Houston", "Idaho"], scores: [
    [[114,55],[61,67],[78,68],[76,47],[78,82],[105,70],[50,63],[78,47]],
    [[72,73],[72,74],[55,76],[57,88]], [[77,71],[65,55]], [[59,71]],
  ] },
  { name: "West", names: ["Arizona", "Long Island", "Villanova", "Utah State", "Wisconsin", "High Point", "Arkansas", "Hawaii", "BYU", "Texas", "Gonzaga", "Kennesaw State", "Miami (FL)", "Missouri", "Purdue", "Queens (N.C.)"], scores: [
    [[92,58],[76,86],[82,83],[97,78],[71,79],[73,64],[80,66],[104,71]],
    [[78,66],[88,94],[74,68],[69,79]], [[109,88],[77,79]], [[79,64]],
  ] },
  { name: "Midwest", names: ["Michigan", "Howard", "Georgia", "Saint Louis", "Texas Tech", "Akron", "Alabama", "Hofstra", "Tennessee", "Miami (Ohio)", "Virginia", "Wright State", "Kentucky", "Santa Clara", "Iowa State", "Tennessee State"], scores: [
    [[101,80],[77,102],[91,71],[90,70],[78,56],[82,73],[89,84],[108,74]],
    [[95,72],[65,90],[79,72],[63,82]], [[90,77],[76,62]], [[95,62]],
  ] },
];
const seeds = [1,16,8,9,5,12,4,13,6,11,3,14,7,10,2,15];
const roundNames = ["First round", "Second round", "Sweet 16", "Elite Eight"];
const regionalFinals = regions.map((region, regionIndex) => {
  const entrants = region.names.map((name, index) => team(name, seeds[index]));
  let previous: TournamentGame[] = [];
  for (let round = 1; round <= 4; round++) {
    previous = region.scores[round - 1].map((scores, index) => {
      const top = round === 1 ? entrants[index * 2] : previous[index * 2].winnerTeamId;
      const bottom = round === 1 ? entrants[index * 2 + 1] : previous[index * 2 + 1].winnerTeamId;
      const topSource = round === 1 ? playIns.find(game => game.winnerTeamId === top) : previous[index * 2];
      const bottomSource = round === 1 ? playIns.find(game => game.winnerTeamId === bottom) : previous[index * 2 + 1];
      return game(`${region.name.toLowerCase()}-${round}-${index}`, round, regionIndex * (8 / 2 ** (round - 1)) + index,
        `${region.name} · ${roundNames[round - 1]}`, top, bottom, scores, topSource?.id, bottomSource?.id);
    });
  }
  return previous[0];
});
const semi1 = game("semifinal-0", 5, 0, "Final Four", regionalFinals[0].winnerTeamId, regionalFinals[1].winnerTeamId, [71,62], regionalFinals[0].id, regionalFinals[1].id);
const semi2 = game("semifinal-1", 5, 1, "Final Four", regionalFinals[2].winnerTeamId, regionalFinals[3].winnerTeamId, [73,91], regionalFinals[2].id, regionalFinals[3].id);
const final = game("championship", 6, 0, "National championship", semi1.winnerTeamId, semi2.winnerTeamId, [63,69], semi1.id, semi2.id);

export const marchMadness2026: Tournament = {
  slug: "march-madness-men-2026",
  name: "2026 NCAA Men's Basketball Tournament",
  season: 2026,
  winnerTeamId: final.winnerTeamId,
  sourceUrls: ["https://www.ncaa.com/march-madness-live/bracket", "https://www.ncaa.com/march-madness-live/bcg"],
  teams,
  games,
};
