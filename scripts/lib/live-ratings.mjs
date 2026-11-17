// Current-season team features for the live matchup model. The formulas mirror
// model/live_matchup_model.py (and kaggle_tuning/features.py): ridge quality,
// margin-aware Elo carried from last season, and mean net efficiency.

// CBBD school names that no Kaggle spelling covers (checked against the 2027 team list).
const ALIASES = {
  "app state": "appalachian st",
  "ualbany": "albany",
  "miami": "miami fl",
  "ul monroe": "ulm",
  "ut rio grande valley": "utrgv",
  "st thomas-minnesota": "st thomas mn",
  "queens university": "queens nc",
};

export function normalizeName(name) {
  return String(name).toLowerCase().normalize("NFKD").replace(/[̀-ͯ]/g, "")
    .replace(/[‘’'`]/g, "").replace(/[.,()]/g, " ").replace(/\s+/g, " ").trim();
}

/** Map CBBD teams to last season's Kaggle Elo priors by spelling. Ambiguous names stay unmatched. */
export function matchKaggleTeams(cbbdTeams, priorTeams) {
  const bySpelling = new Map();
  for (const prior of priorTeams) {
    for (const spelling of [prior.teamName, ...prior.spellings]) {
      const key = normalizeName(spelling);
      const ids = bySpelling.get(key) ?? new Set();
      ids.add(prior.kaggleTeamId);
      bySpelling.set(key, ids);
    }
  }
  const byId = new Map(priorTeams.map(prior => [prior.kaggleTeamId, prior]));
  const candidates = new Map();
  for (const team of cbbdTeams) {
    const names = [team.school, ALIASES[normalizeName(team.school)]].filter(Boolean).map(normalizeName);
    const ids = new Set(names.flatMap(name => [...(bySpelling.get(name) ?? [])]));
    if (ids.size === 1) candidates.set(team.id, [...ids][0]);
  }
  // A Kaggle team claimed by two CBBD teams is ambiguous for both.
  const claims = new Map();
  for (const kaggleId of candidates.values()) claims.set(kaggleId, (claims.get(kaggleId) ?? 0) + 1);
  const matched = new Map();
  const unmatched = [];
  for (const team of cbbdTeams) {
    const kaggleId = candidates.get(team.id);
    if (kaggleId !== undefined && claims.get(kaggleId) === 1) matched.set(team.id, byId.get(kaggleId));
    else unmatched.push(team.school);
  }
  return { matched, unmatched };
}

/** Final regular-season games between two Division I teams, in play order. */
export function ratingGames(games, divisionOneIds) {
  return games.filter(game => game.status === "final" && game.seasonType === "regular"
    && divisionOneIds.has(game.homeTeamId) && divisionOneIds.has(game.awayTeamId)
    && Number.isFinite(game.homePoints) && Number.isFinite(game.awayPoints)
    && game.homePoints !== game.awayPoints)
    .map(game => ({
      id: game.id, startDate: game.startDate, neutralSite: Boolean(game.neutralSite),
      homeTeamId: game.homeTeamId, awayTeamId: game.awayTeamId,
      homePoints: game.homePoints, awayPoints: game.awayPoints,
    }))
    .sort((a, b) => a.startDate.localeCompare(b.startDate) || a.id - b.id);
}

/** One row per team per game with the box-score fields net efficiency needs. */
export function boxRows(boxScores, gameIds) {
  return boxScores.filter(row => gameIds.has(row.gameId) && row.teamStats && row.opponentStats).map(row => {
    const t = row.teamStats, o = row.opponentStats;
    return {
      gameId: row.gameId, teamId: row.teamId, pts: t.points?.total, oppPts: o.points?.total,
      FGA: t.fieldGoals?.attempted, OR: t.rebounds?.offensive, TO: t.turnovers?.total, FTA: t.freeThrows?.attempted,
      oppFGA: o.fieldGoals?.attempted, oppOR: o.rebounds?.offensive, oppTO: o.turnovers?.total, oppFTA: o.freeThrows?.attempted,
    };
  }).filter(row => Object.values(row).every(Number.isFinite));
}

function solveSymmetric(a, b) {
  // Cholesky solve of a positive definite system (ridge normal equations).
  const n = b.length, l = a.map(row => row.slice());
  for (let j = 0; j < n; j++) {
    let d = l[j][j];
    for (let k = 0; k < j; k++) d -= l[j][k] ** 2;
    l[j][j] = Math.sqrt(d);
    for (let i = j + 1; i < n; i++) {
      let s = l[i][j];
      for (let k = 0; k < j; k++) s -= l[i][k] * l[j][k];
      l[i][j] = s / l[j][j];
    }
  }
  const y = new Array(n);
  for (let i = 0; i < n; i++) {
    let s = b[i];
    for (let k = 0; k < i; k++) s -= l[i][k] * y[k];
    y[i] = s / l[i][i];
  }
  const x = new Array(n);
  for (let i = n - 1; i >= 0; i--) {
    let s = y[i];
    for (let k = i + 1; k < n; k++) s -= l[k][i] * x[k];
    x[i] = s / l[i][i];
  }
  return x;
}

/** margin ~ q[home] - q[away] + h * (not neutral), ridge without intercept, q centred. */
export function ridgeQuality(games, { alpha, marginClip }) {
  const teams = [...new Set(games.flatMap(game => [game.homeTeamId, game.awayTeamId]))].sort((a, b) => a - b);
  const index = new Map(teams.map((team, i) => [team, i]));
  const n = teams.length + 1;
  const xtx = Array.from({ length: n }, () => new Array(n).fill(0));
  const xty = new Array(n).fill(0);
  for (const game of games) {
    const h = index.get(game.homeTeamId), a = index.get(game.awayTeamId), loc = game.neutralSite ? 0 : 1;
    const y = Math.max(-marginClip, Math.min(marginClip, game.homePoints - game.awayPoints));
    const cols = [[h, 1], [a, -1], ...(loc ? [[n - 1, loc]] : [])];
    for (const [i, xi] of cols) {
      xty[i] += xi * y;
      for (const [j, xj] of cols) xtx[i][j] += xi * xj;
    }
  }
  for (let i = 0; i < n; i++) xtx[i][i] += alpha;
  const coef = solveSymmetric(xtx, xty);
  const mean = coef.slice(0, -1).reduce((sum, q) => sum + q, 0) / teams.length;
  return { quality: new Map(teams.map((team, i) => [team, coef[i] - mean])), homeEdge: coef[n - 1] };
}

/** Margin-aware Elo, starting each team at 1500 + carry * (prior - 1500). */
export function eloRatings(games, priorElo, { k, home, carry }) {
  const ratings = new Map([...priorElo].map(([team, elo]) => [team, 1500 + carry * (elo - 1500)]));
  for (const game of games) {
    const homeWon = game.homePoints > game.awayPoints;
    const [w, l] = homeWon ? [game.homeTeamId, game.awayTeamId] : [game.awayTeamId, game.homeTeamId];
    const rw = ratings.get(w) ?? 1500, rl = ratings.get(l) ?? 1500;
    const hadj = game.neutralSite ? 0 : (homeWon ? home : -home);
    const expected = 1 / (1 + 10 ** ((rl - rw - hadj) / 400));
    const mov = Math.log1p(Math.abs(game.homePoints - game.awayPoints)) * 2.2 / ((rw - rl + hadj) * 0.001 + 2.2);
    const delta = k * mov * (1 - expected);
    ratings.set(w, rw + delta);
    ratings.set(l, rl - delta);
  }
  return ratings;
}

export function netEfficiency(rows) {
  const sums = new Map();
  for (const row of rows) {
    const poss = ((row.FGA - row.OR + row.TO + 0.475 * row.FTA) + (row.oppFGA - row.oppOR + row.oppTO + 0.475 * row.oppFTA)) / 2;
    if (!(poss > 0)) continue;
    const entry = sums.get(row.teamId) ?? { total: 0, games: 0 };
    entry.total += 100 * (row.pts - row.oppPts) / poss;
    entry.games += 1;
    sums.set(row.teamId, entry);
  }
  return new Map([...sums].map(([team, { total, games }]) => [team, { netEff: total / games, games }]));
}

/** Features for every team with at least one rated game and box score. */
export function fitLiveRatings({ games, box, priorElo, model }) {
  const { quality, homeEdge } = ridgeQuality(games, model.quality);
  const elo = eloRatings(games, priorElo, model.elo);
  const net = netEfficiency(box);
  const played = new Map();
  for (const game of games) for (const team of [game.homeTeamId, game.awayTeamId]) played.set(team, (played.get(team) ?? 0) + 1);
  const teams = new Map();
  for (const [teamId, q] of quality) {
    const n = net.get(teamId);
    if (!n) continue;
    teams.set(teamId, { games: played.get(teamId), boxGames: n.games, quality: q, elo: elo.get(teamId) ?? 1500, netEff: n.netEff });
  }
  return { teams, homeEdge };
}

/** The basketball season a date belongs to, named by its ending year as CBBD does. */
export function seasonForDate(date) {
  return date.getUTCMonth() >= 7 ? date.getUTCFullYear() + 1 : date.getUTCFullYear();
}
