/* eslint-disable @typescript-eslint/no-require-imports -- CommonJS is needed to register the TypeScript test loader before loading source files. */
const fs = require("node:fs");
const assert = require("node:assert/strict");
const { test } = require("node:test");
const ts = require("typescript");

// Use the project's existing compiler so the checks also run on older Node versions.
require.extensions[".ts"] = (module, filename) => {
  const { outputText } = ts.transpileModule(fs.readFileSync(filename, "utf8"), {
    compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2020 },
  });
  module._compile(outputText, filename);
};
const { marchMadness2026 } = require("../convex/data/marchMadness2026.ts");
const { validateTournament } = require("../convex/tournamentModel.ts");
const { buildTournamentBracket } = require("../src/lib/tournament-bracket.ts");
const { compareBracket, compareGame } = require("../convex/bracketComparison.ts");
const { confirmedWinner, normalizeCbbdGames, applyScoreboard, mergeCbbdGames, hasNewTournamentGame } = require("../convex/cbbdResults.ts");
const finalResults = marchMadness2026.games.map((game, index) => ({
  id: index, round: game.round, startDate: "2026-03-19T18:00:00Z", status: "final",
  homeTeamId: game.topTeamId, awayTeamId: game.bottomTeamId,
  homeCbbdTeamId: index * 2, awayCbbdTeamId: index * 2 + 1,
  homePoints: game.topScore, awayPoints: game.bottomScore,
}));

function modelPrediction(overrides, slug = "test-model") {
  const predicted = structuredClone(marchMadness2026);
  predicted.slug = slug;
  const winners = new Map();
  predicted.games = [...predicted.games].sort((a, b) => a.round - b.round || a.order - b.order).map(game => {
    const topTeamId = game.topSourceGameId ? winners.get(game.topSourceGameId) : game.topTeamId;
    const bottomTeamId = game.bottomSourceGameId ? winners.get(game.bottomSourceGameId) : game.bottomTeamId;
    const winnerTeamId = overrides[game.id] ?? ([topTeamId, bottomTeamId].includes(game.winnerTeamId) ? game.winnerTeamId : topTeamId);
    winners.set(game.id, winnerTeamId);
    const result = { ...game, id: `model-${game.id}`, topTeamId, bottomTeamId, winnerTeamId };
    if (game.topSourceGameId) result.topSourceGameId = `model-${game.topSourceGameId}`;
    if (game.bottomSourceGameId) result.bottomSourceGameId = `model-${game.bottomSourceGameId}`;
    delete result.topScore;
    delete result.bottomScore;
    return result;
  });
  predicted.winnerTeamId = predicted.games.at(-1).winnerTeamId;
  validateTournament(predicted);
  return predicted;
}

test("2026 sample is a complete, consistent 68-team tournament", () => {
  validateTournament(marchMadness2026);
  assert.equal(marchMadness2026.teams.length, 68);
  assert.equal(marchMadness2026.games.length, 67);
  assert.equal(marchMadness2026.winnerTeamId, "michigan");
  assert.deepEqual([0,1,2,3,4,5,6].map(round => marchMadness2026.games.filter(game => game.round === round).length), [4,32,16,8,4,2,1]);
  const final = marchMadness2026.games.find(game => game.id === "championship");
  assert.deepEqual([final.topTeamId, final.topScore, final.bottomTeamId, final.bottomScore], ["uconn",63,"michigan",69]);
});

test("invalid winners, advancement, IDs, and missing games are rejected", () => {
  for (const modify of [
    t => { t.winnerTeamId = "uconn"; },
    t => { t.games[0].winnerTeamId = t.games[0].topTeamId; },
    t => { t.games.at(-1).topSourceGameId = t.games.at(-1).id; },
    t => { t.games[1].id = t.games[0].id; },
    t => { t.games.pop(); },
  ]) {
    const tournament = structuredClone(marchMadness2026);
    modify(tournament);
    assert.throws(() => validateTournament(tournament));
  }
});

test("both halves connect inward to the correct winner's row", () => {
  const { nodes, edges } = buildTournamentBracket(marchMadness2026);
  assert.equal(nodes.length, 67);
  assert.equal(edges.length, 58);
  const byId = new Map(nodes.map(node => [node.id, node]));
  for (const node of nodes) {
    assert.equal(node.draggable, false);
    assert.equal(node.deletable, false);
    assert.equal(node.connectable, false);
    assert.equal(node.style.pointerEvents, "all", "locked games must receive hover events");
    assert.ok(Number.isFinite(node.position.x) && Number.isFinite(node.position.y));
    assert.equal(edges.filter(edge => edge.source === node.id).length, node.id === "championship" || node.data.label === "First Four" || node.data.label.includes("Elite Eight") ? 0 : 1);
  }
  for (const edge of edges) {
    const source = byId.get(edge.source);
    const target = byId.get(edge.target);
    assert.ok(Math.abs(source.position.x - target.position.x) >= 504, `${edge.id} needs room for a full card and connector`);
    assert.ok(source.data.direction === "right" ? source.position.x > target.position.x : source.position.x < target.position.x);
    assert.equal(source.data[source.data.winner === "top" ? "topTeam" : "bottomTeam"].id,
      target.data[edge.targetHandle === "top" ? "topTeam" : "bottomTeam"].id);
    assert.equal(edge.type, "smoothstep");
  }
  const left = nodes.filter(node => node.data.direction === "left" && node.data.label !== "First Four");
  const right = nodes.filter(node => node.data.direction === "right" && node.data.label !== "First Four");
  assert.equal(left.filter(node => node.data.label.includes("First round")).length, 16);
  assert.equal(right.filter(node => node.data.label.includes("First round")).length, 16);
  const final = byId.get("championship");
  assert.equal(byId.get("semifinal-0").position.y, final.position.y);
  assert.equal(byId.get("semifinal-1").position.y, final.position.y);
  for (const region of ["east", "west"]) assert.ok(byId.get(`${region}-4-0`).position.y < final.position.y);
  for (const region of ["south", "midwest"]) assert.ok(byId.get(`${region}-4-0`).position.y > final.position.y);
  const playIns = nodes.filter(node => node.data.label === "First Four");
  assert.equal(new Set(playIns.map(node => node.position.y)).size, 1);
  assert.ok(playIns.every(node => node.position.y > Math.max(...nodes.filter(other => other.data.label !== "First Four").map(other => other.position.y))));
  assert.ok(edges.every(edge => !byId.get(edge.target).data.semifinal));
  assert.equal(edges.filter(edge => edge.target === "championship").length, 2);
  assert.ok(left.every(node => node.position.x < byId.get("championship").position.x));
  assert.ok(right.every(node => node.position.x > byId.get("championship").position.x));
  for (let i = 0; i < nodes.length; i++) {
    for (let j = i + 1; j < nodes.length; j++) {
      const horizontalGap = Math.abs(nodes[i].position.x - nodes[j].position.x);
      const verticalGap = Math.abs(nodes[i].position.y - nodes[j].position.y);
      assert.ok(horizontalGap >= 384 || verticalGap >= 200, `${nodes[i].id} overlaps ${nodes[j].id}`);
    }
  }
});

test("predicted winners validate and generate a bracket without scores", () => {
  const predicted = structuredClone(marchMadness2026);
  predicted.games.forEach(game => { delete game.topScore; delete game.bottomScore; });
  validateTournament(predicted);
  const { nodes, edges } = buildTournamentBracket(predicted);
  assert.equal(nodes.length, 67);
  assert.equal(edges.length, 58);
  for (const node of nodes) {
    assert.equal(node.data.topTeam.score, undefined);
    assert.equal(node.data.bottomTeam.score, undefined);
    assert.equal(node.data.showScores, false);
  }
  predicted.games[0].winnerTeamId = "not-a-participant";
  assert.throws(() => validateTournament(predicted), /participating team/);
});

test("score visibility is opt-in even for historical results", () => {
  assert.ok(buildTournamentBracket(marchMadness2026).nodes.every(node => node.data.showScores === false));
  assert.ok(buildTournamentBracket(marchMadness2026, true).nodes.every(node => node.data.showScores === true));
  const invalid = structuredClone(marchMadness2026);
  invalid.games[0].topScore = -1;
  assert.throws(() => validateTournament(invalid), /Invalid final scores/);
  invalid.games[0].topScore = invalid.games[0].bottomScore;
  assert.throws(() => validateTournament(invalid), /Tied final scores/);
});

test("prediction brackets reuse real markets even with different game IDs and an incorrect winner", () => {
  const predicted = modelPrediction({ "west-3-1": "texas" });
  const comparison = compareBracket(predicted, marchMadness2026, finalResults);
  assert.deepEqual(comparison["model-east-1-0"], {
    status: "intact", oddsSource: { tournamentSlug: marchMadness2026.slug, gameId: "east-1-0" },
  });
  assert.deepEqual(comparison["model-west-3-1"], {
    status: "break", oddsSource: { tournamentSlug: marchMadness2026.slug, gameId: "west-3-1" },
  });
  assert.deepEqual(comparison["model-west-4-0"], { status: "hypothetical" });
  assert.equal(comparison["model-east-4-0"].status, "intact", "an unrelated branch keeps its odds");
  const nodes = buildTournamentBracket({ ...predicted, gameComparisons: comparison }).nodes;
  assert.equal(nodes.find(node => node.id === "model-west-3-1").data.bracketStatus, "break");
  assert.equal(nodes.find(node => node.id === "model-west-4-0").data.bracketStatus, "hypothetical");
});

test("real matchups can regain markets after an earlier hypothetical branch", () => {
  const predicted = modelPrediction({ "east-1-0": "siena", "east-2-0": "tcu", "east-3-0": "st-john-s" });
  const comparison = compareBracket(predicted, marchMadness2026, finalResults);
  assert.equal(comparison["model-east-1-0"].status, "break");
  assert.equal(comparison["model-east-2-0"].status, "hypothetical");
  assert.equal(comparison["model-semifinal-0"].status, "intact");
  assert.equal(comparison["model-semifinal-0"].oddsSource.gameId, "semifinal-0");
});

test("matchup comparison ignores team ordering but rejects different rounds, seasons and ambiguous matches", () => {
  const real = marchMadness2026.games.find(game => game.id === "east-1-0");
  assert.equal(compareGame({ ...real, topTeamId: real.bottomTeamId, bottomTeamId: real.topTeamId }, marchMadness2026, finalResults).status, "intact");
  assert.equal(compareGame({ ...real, round: 2 }, marchMadness2026, finalResults).status, "hypothetical");
  assert.equal(compareGame(real, { ...marchMadness2026, games: [real, real] }).oddsSource, undefined);
  assert.deepEqual(compareBracket({ ...marchMadness2026, season: 2027 }, marchMadness2026), {});
  assert.deepEqual(compareBracket(marchMadness2026, null), {});
  assert.ok(Object.values(compareBracket(marchMadness2026, marchMadness2026, finalResults)).every(game => game.status === "intact"));
});

test("CBBD finals, rather than live leads or reference picks, determine divergence", () => {
  const predicted = modelPrediction({ "west-3-1": "texas" });
  const live = finalResults.map(game => ({ ...game, status: "scheduled", homePoints: undefined, awayPoints: undefined }));
  const index = marchMadness2026.games.findIndex(game => game.id === "west-3-1");
  live[index] = { ...live[index], status: "in_progress", homePoints: 90, awayPoints: 10 };
  let comparison = compareBracket(predicted, marchMadness2026, live);
  assert.equal(comparison["model-west-3-1"].status, "pending");
  assert.equal(comparison["model-west-4-0"].status, "pending");
  assert.ok(comparison["model-west-3-1"].oddsSource, "pending games retain known markets");
  live[index] = { ...finalResults[index] };
  const reference = structuredClone(marchMadness2026);
  reference.games.find(game => game.id === "west-3-1").winnerTeamId = "texas";
  comparison = compareBracket(predicted, reference, live);
  assert.equal(comparison["model-west-3-1"].status, "break", "CBBD overrides a stale reference result");
  assert.equal(comparison["model-west-4-0"].status, "hypothetical");
  assert.equal(comparison["model-east-1-0"].status, "pending", "unplayed branches aren't marked broken");
  for (const status of ["scheduled", "in_progress", "postponed", "cancelled"]) {
    assert.equal(confirmedWinner({ ...finalResults[index], status }), undefined);
  }
  assert.equal(confirmedWinner({ ...finalResults[index], homePoints: 70, awayPoints: 70 }), undefined);
  assert.equal(confirmedWinner({ ...finalResults[index], homePoints: undefined }), undefined);
  assert.ok(Object.values(compareBracket(predicted, reference)).every(game => game.status === "pending"));
});

test("CBBD normalization requires NCAA season, known teams and a recognized round", () => {
  const game = { id: 123, season: 2026, tournament: "NCAA", gameNotes: "West Region - 1st Round", status: "final",
    startDate: "2026-03-20T18:00:00Z", homeTeam: "Kansas", awayTeam: "California Baptist", homeTeamId: 1, awayTeamId: 2,
    homePoints: 68, awayPoints: 60 };
  const rows = normalizeCbbdGames([game, { ...game, season: 2025 }, { ...game, tournament: "NIT" },
    { ...game, awayTeam: "TBD" }, { ...game, gameNotes: "Unknown round" }], marchMadness2026);
  assert.equal(rows.length, 1);
  assert.equal(rows[0].awayTeamId, "cal-baptist");
  assert.equal(rows[0].round, 1);
  assert.equal(confirmedWinner(rows[0]), "kansas");
});

test("scoreboard updates preserve finals, reject a different game, and accept final corrections", () => {
  const saved = finalResults[0];
  assert.deepEqual(applyScoreboard([saved], []), [saved]);
  const feed = { id: saved.id, status: "in_progress",
    homeTeam: { id: saved.homeCbbdTeamId, points: 99 }, awayTeam: { id: saved.awayCbbdTeamId, points: 1 } };
  assert.deepEqual(applyScoreboard([saved], [feed]), [saved], "a stale live row must not undo a final");
  const pending = { ...saved, status: "scheduled", homePoints: undefined, awayPoints: undefined };
  const updated = applyScoreboard([pending], [{ ...feed, status: "final" }]);
  assert.equal(confirmedWinner(updated[0]), saved.homeTeamId);
  assert.deepEqual(applyScoreboard([pending], [{ ...feed, homeTeam: { id: -1, points: 99 } }]), [pending]);
  const corrected = { ...saved, homePoints: 99, awayPoints: 1 };
  assert.equal(confirmedWinner(mergeCbbdGames([saved], [corrected])[0]), saved.homeTeamId);
  assert.equal(hasNewTournamentGame([saved], [feed]), false);
  assert.equal(hasNewTournamentGame([saved], [{ ...feed, id: -1 }]), true, "new rounds between known tournament teams refresh the schedule");
  assert.equal(hasNewTournamentGame([saved], [{ ...feed, id: -1, homeTeam: { id: -999, points: 0 } }]), false);
});
