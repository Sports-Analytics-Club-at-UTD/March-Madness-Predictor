import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import {
  boxRows, eloRatings, fitLiveRatings, matchKaggleTeams, netEfficiency, ratingGames, ridgeQuality, seasonForDate,
} from "../scripts/lib/live-ratings.mjs";

const model = JSON.parse(readFileSync(new URL("../convex/data/liveMatchupModel.json", import.meta.url), "utf8"));
const close = (actual, expected, tolerance = 1e-9) =>
  assert.ok(Math.abs(actual - expected) <= tolerance, `${actual} is not within ${tolerance} of ${expected}`);
const game = (id, home, away, homePoints, awayPoints, extra = {}) => ({
  id, startDate: `2026-11-0${id}T20:00:00.000Z`, status: "final", seasonType: "regular", neutralSite: true,
  homeTeamId: home, awayTeamId: away, homePoints, awayPoints, ...extra,
});

test("CBBD schools match last season's Kaggle priors by spelling and alias", () => {
  const priors = [
    { kaggleTeamId: 1274, teamName: "Miami FL", spellings: ["miami (fl)", "miami fl"], elo: 1700 },
    { kaggleTeamId: 1275, teamName: "Miami OH", spellings: ["miami (oh)", "miami ohio"], elo: 1450 },
    { kaggleTeamId: 1361, teamName: "San Jose St", spellings: ["san jose st", "san jose state"], elo: 1400 },
    { kaggleTeamId: 1385, teamName: "St John's", spellings: ["st john's", "st. john's"], elo: 1800 },
  ];
  const teams = [
    { id: 1, school: "Miami" }, { id: 2, school: "Miami (OH)" }, { id: 3, school: "San José State" },
    { id: 4, school: "St. John's" }, { id: 5, school: "West Florida" },
  ];
  const { matched, unmatched } = matchKaggleTeams(teams, priors);
  assert.equal(matched.get(1).kaggleTeamId, 1274);
  assert.equal(matched.get(2).kaggleTeamId, 1275);
  assert.equal(matched.get(3).kaggleTeamId, 1361);
  assert.equal(matched.get(4).kaggleTeamId, 1385);
  assert.deepEqual(unmatched, ["West Florida"]);
});

test("ambiguous spellings and duplicate claims stay unmatched", () => {
  const priors = [
    { kaggleTeamId: 1, teamName: "Alpha", spellings: ["shared"], elo: 1500 },
    { kaggleTeamId: 2, teamName: "Beta", spellings: ["shared"], elo: 1500 },
    { kaggleTeamId: 3, teamName: "Gamma", spellings: ["gamma", "gamma u"], elo: 1500 },
  ];
  const { matched, unmatched } = matchKaggleTeams(
    [{ id: 10, school: "Shared" }, { id: 11, school: "Gamma" }, { id: 12, school: "Gamma U" }], priors);
  assert.equal(matched.size, 0);
  assert.deepEqual(unmatched, ["Shared", "Gamma", "Gamma U"]);
});

test("ratings use only final, decided, regular-season games between D1 teams, in order", () => {
  const d1 = new Set([1, 2, 3]);
  const rated = ratingGames([
    game(3, 1, 2, 70, 60),
    game(1, 2, 3, 80, 75),
    game(2, 1, 9, 90, 40),
    game(4, 1, 3, 0, 0, { status: "scheduled" }),
    game(5, 1, 3, 65, 60, { seasonType: "postseason" }),
    game(6, 2, 3, 61, 61),
  ], d1);
  assert.deepEqual(rated.map(g => g.id), [1, 3]);
});

test("ridge quality solves the penalized least squares exactly", () => {
  // One neutral game, margin 10, alpha 1: (X'X + I) q = X'y gives q = +-10/3.
  const { quality, homeEdge } = ridgeQuality([game(1, 1, 2, 70, 60)], model.quality);
  close(quality.get(1), 10 / 3);
  close(quality.get(2), -10 / 3);
  assert.equal(homeEdge, 0);
  // Margins are clipped at 60 points before fitting.
  const clipped = ridgeQuality([game(1, 1, 2, 130, 40)], model.quality);
  close(clipped.quality.get(1), 20);
  // A home game adds the home column: X = [1, -1, 1].
  const home = ridgeQuality([game(1, 1, 2, 70, 60, { neutralSite: false })], model.quality);
  close(home.quality.get(1), 2.5);
  close(home.homeEdge, 2.5);
});

test("Elo carries last season's rating and updates by margin", () => {
  const prior = new Map([[1, 1600]]);
  const start = eloRatings([], prior, model.elo);
  close(start.get(1), 1575);
  const after = eloRatings([game(1, 3, 4, 70, 60)], new Map(), model.elo);
  close(after.get(3), 1500 + 20 * Math.log(11) * 0.5);
  close(after.get(4), 1500 - 20 * Math.log(11) * 0.5);
});

test("net efficiency averages per-game margins per 100 Kaggle-formula possessions", () => {
  const box = boxRows([{
    gameId: 1, teamId: 1,
    teamStats: { points: { total: 80 }, fieldGoals: { attempted: 60 }, rebounds: { offensive: 10 }, turnovers: { total: 10 }, freeThrows: { attempted: 20 } },
    opponentStats: { points: { total: 70 }, fieldGoals: { attempted: 60 }, rebounds: { offensive: 10 }, turnovers: { total: 10 }, freeThrows: { attempted: 20 } },
  }, { gameId: 2, teamId: 1, teamStats: null, opponentStats: null }], new Set([1, 2]));
  assert.equal(box.length, 1);
  close(netEfficiency(box).get(1).netEff, 100 * 10 / 69.5);
});

test("teams without a box score get no live features", () => {
  const { teams } = fitLiveRatings({
    games: [game(1, 1, 2, 70, 60)],
    box: [{ gameId: 1, teamId: 1, pts: 70, oppPts: 60, FGA: 60, OR: 10, TO: 10, FTA: 20, oppFGA: 60, oppOR: 10, oppTO: 10, oppFTA: 20 }],
    priorElo: new Map(), model,
  });
  assert.deepEqual([...teams.keys()], [1]);
  assert.equal(teams.get(1).games, 1);
});

test("seasons are named by their ending year", () => {
  assert.equal(seasonForDate(new Date("2026-11-17T00:00:00Z")), 2027);
  assert.equal(seasonForDate(new Date("2027-03-20T00:00:00Z")), 2027);
});
