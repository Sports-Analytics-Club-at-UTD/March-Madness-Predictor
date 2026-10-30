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
