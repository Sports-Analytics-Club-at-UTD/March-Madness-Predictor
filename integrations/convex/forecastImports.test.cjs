/* eslint-disable @typescript-eslint/no-require-imports -- Register the TypeScript loader before loading site modules. */
/* Copy into tests/ in the site checkout. No network or Convex deployment needed. */
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const Module = require("node:module");
const { test } = require("node:test");
const ts = require("typescript");

require.extensions[".ts"] = (module, filename) => {
  const { outputText } = ts.transpileModule(fs.readFileSync(filename, "utf8"), {
    compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2020 },
  });
  module._compile(outputText, filename);
};
const modulePath = path.resolve(__dirname, "../convex/forecastImports.ts");
const load = Module._load;
Module._load = function (request, parent, ...rest) {
  if (parent?.filename === modulePath && request === "./_generated/server") {
    return { internalQuery: value => value, internalMutation: value => value };
  }
  return load.call(this, request, parent, ...rest);
};
let handlers;
try { handlers = require(modulePath); } finally { Module._load = load; }
const { marchMadness2026 } = require("../convex/data/marchMadness2026.ts");
const { buildTournamentBracket } = require("../src/lib/tournament-bracket.ts");

function fixture() {
  const teams = [0, 1, 2, 3].map(i => ({ id: `team-${i}`, name: `Team ${i}`, seed: i + 1, logoUrl: "/favicon.ico" }));
  const tournament = {
    slug: `forecast-2026-${"a".repeat(64)}`, name: "Model", season: 2026,
    winnerTeamId: "team-0", sourceUrls: [], teams,
    games: [
      { id: "g1", round: 1, order: 0, label: "First round", topTeamId: "team-0", bottomTeamId: "team-1", winnerTeamId: "team-0" },
      { id: "g2", round: 1, order: 1, label: "First round", topTeamId: "team-2", bottomTeamId: "team-3", winnerTeamId: "team-2" },
      { id: "g3", round: 2, order: 0, label: "Final", topTeamId: "team-0", bottomTeamId: "team-2", winnerTeamId: "team-0", topSourceGameId: "g1", bottomSourceGameId: "g2" },
    ],
  };
  const reference = { ...structuredClone(tournament), _id: "reference", slug: "march-madness-men-2026" };
  const tables = {
    tournaments: [reference],
    brackets: [{ _id: "official-card", slug: "2026-results", tournamentId: "reference", logoStorageId: "logo" }],
  };
  let sequence = 0;
  const ctx = {
    storage: { getUrl: async id => id === "logo" ? "https://example.test/logo" : null },
    db: {
      query(table) {
        return {
          collect: async () => structuredClone(tables[table]),
          withIndex(_index, filter) {
            let field, expected;
            filter({ eq(key, value) { field = key; expected = value; } });
            return { unique: async () => structuredClone(tables[table].find(row => row[field] === expected) ?? null) };
          },
        };
      },
      async insert(table, record) { const _id = `insert-${++sequence}`; tables[table].push({ ...structuredClone(record), _id }); return _id; },
      async replace(id, record) {
        const rows = Object.values(tables).find(rows => rows.some(row => row._id === id));
        rows[rows.findIndex(row => row._id === id)] = { ...structuredClone(record), _id: id };
      },
      async patch(id, record) {
        const row = Object.values(tables).flat().find(row => row._id === id);
        Object.assign(row, structuredClone(record));
      },
    },
  };
  return { ctx, tables, item: { tournament, title: "Model", description: "Run 1", logoStorageId: "logo" } };
}

test("roster reads canonical teams and stored card logo", async () => {
  const { ctx } = fixture();
  const roster = await handlers.getRoster.handler(ctx, { season: 2026 });
  assert.equal(roster.teams.length, 4);
  assert.equal(roster.logoStorageId, "logo");
  await assert.rejects(handlers.getRoster.handler(ctx, { season: 2025 }), /reference/);
});

test("repeat imports reuse IDs and leave official records unchanged", async () => {
  const { ctx, tables, item } = fixture();
  const original = structuredClone(tables.tournaments[0]);
  const first = await handlers.importForecast.handler(ctx, { items: [item] });
  item.title = "Updated title";
  const second = await handlers.importForecast.handler(ctx, { items: [item] });
  assert.equal(first[0].bracketId, second[0].bracketId);
  assert.equal(first[0].tournamentId, second[0].tournamentId);
  assert.equal(second[0].updated, true);
  assert.equal(tables.brackets.length, 2);
  assert.equal(tables.brackets[1].title, "Updated title");
  assert.deepEqual(tables.tournaments[0], original);
  const { nodes } = buildTournamentBracket(item.tournament);
  assert.equal(nodes.length, 3);
  assert.ok(nodes.every(node => Number.isFinite(node.position.x) && Number.isFinite(node.position.y)));
});

test("reject official slugs, bad advancement, unknown teams, missing logos and conflicting cards", async () => {
  const cases = [
    item => { item.tournament.slug = "march-madness-men-2026"; },
    item => { item.tournament.games[2].topTeamId = "team-1"; },
    item => { item.tournament.teams[0].name = "Unknown team"; },
    item => { item.logoStorageId = "missing"; },
  ];
  for (const mutate of cases) {
    const { ctx, item } = fixture();
    mutate(item);
    await assert.rejects(handlers.importForecast.handler(ctx, { items: [item] }));
  }
  const { ctx, tables, item } = fixture();
  tables.brackets.push({ _id: "conflict", slug: item.tournament.slug, tournamentId: "reference" });
  await assert.rejects(handlers.importForecast.handler(ctx, { items: [item] }), /conflicting/);
});

test("render a 64-team main bracket using the actual demo layout", () => {
  const tournament = structuredClone(marchMadness2026);
  tournament.games = tournament.games.filter(game => game.round > 0);
  for (const game of tournament.games.filter(game => game.round === 1)) {
    delete game.topSourceGameId;
    delete game.bottomSourceGameId;
  }
  const entrants = new Set(tournament.games.filter(game => game.round === 1).flatMap(game => [game.topTeamId, game.bottomTeamId]));
  tournament.teams = tournament.teams.filter(team => entrants.has(team.id));
  const { nodes, edges } = buildTournamentBracket(tournament);
  assert.equal(nodes.length, 63);
  assert.ok(edges.length > 0);
  assert.ok(nodes.every(node => Number.isFinite(node.position.x) && Number.isFinite(node.position.y)));
});

// Allows cross-language verification of an actual Python-produced payload.
if (process.env.FORECAST_IMPORT_PAYLOAD) {
  test("Python payload passes the real site validator, mutation and layout", async () => {
    const { ctx, tables } = fixture();
    const payload = JSON.parse(fs.readFileSync(process.env.FORECAST_IMPORT_PAYLOAD, "utf8"));
    tables.tournaments[0].teams = payload.items[0].tournament.teams;
    for (const item of payload.items) item.logoStorageId = "logo";
    await handlers.importForecast.handler(ctx, payload);
    for (const item of payload.items) {
      const { nodes } = buildTournamentBracket(item.tournament);
      assert.equal(nodes.length, item.tournament.games.length);
      assert.ok(nodes.every(node => Number.isFinite(node.position.x) && Number.isFinite(node.position.y)));
    }
  });
}
