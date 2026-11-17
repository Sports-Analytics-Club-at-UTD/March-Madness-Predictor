/* eslint-disable @typescript-eslint/no-require-imports -- CommonJS is needed to register the TypeScript test loader before loading source files. */
const fs = require("node:fs");
const assert = require("node:assert/strict");
const { test } = require("node:test");
const ts = require("typescript");

require.extensions[".ts"] = (module, filename) => {
  const { outputText } = ts.transpileModule(fs.readFileSync(filename, "utf8"), {
    compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2020, esModuleInterop: true },
  });
  module._compile(outputText, filename);
};
const { winProbability, liveModel, weekStart, gradePrediction, summarizePredictions } = require("../convex/liveMatchupModel.ts");

const close = (actual, expected, tolerance = 1e-6) =>
  assert.ok(Math.abs(actual - expected) <= tolerance, `${actual} is not within ${tolerance} of ${expected}`);

// Live 2027 features from `pnpm live:sync --dump` on 2026-11-17; expected values from
// `uv run model/live_matchup_model.py parity` (sklearn) on the same dump.
const duke = { quality: 22.85813391045937, elo: 1905.2210038584824, netEff: 44.04037300462996 };
const houston = { quality: 16.604774596821507, elo: 1855.2753728880575, netEff: 48.833174121378676 };

test("matches the Python model's probability for the same features", () => {
  close(winProbability(liveModel, duke, houston), 0.705702);
  close(winProbability(liveModel, duke, houston, "a"), 0.816507);
});

test("probabilities are complementary across team order and venue", () => {
  close(winProbability(liveModel, duke, houston) + winProbability(liveModel, houston, duke), 1, 1e-12);
  close(winProbability(liveModel, duke, houston, "a") + winProbability(liveModel, houston, duke, "b"), 1, 1e-12);
  close(winProbability(liveModel, duke, duke), 0.5, 1e-12);
});

test("home court and a better rating raise the win probability", () => {
  assert.ok(winProbability(liveModel, duke, duke, "a") > 0.5);
  assert.ok(winProbability(liveModel, duke, duke, "b") < 0.5);
  assert.ok(winProbability(liveModel, { ...houston, quality: houston.quality + 5 }, duke) > winProbability(liveModel, houston, duke));
  assert.ok(winProbability(liveModel, { ...houston, elo: houston.elo + 100 }, duke) > winProbability(liveModel, houston, duke));
});

test("the exported model is the standardized three-feature logit without an intercept", () => {
  assert.deepEqual(liveModel.features, ["quality", "elo", "net_eff"]);
  // One standard deviation of quality alone moves the log odds by its coefficient.
  const a = { quality: liveModel.scalerScale[0], elo: 0, netEff: 0 }, b = { quality: 0, elo: 0, netEff: 0 };
  close(Math.log(winProbability(liveModel, a, b) / (1 - winProbability(liveModel, a, b))), liveModel.coefficients[0], 1e-9);
});

test("weeks start on Monday in Eastern time", () => {
  assert.equal(weekStart("2026-11-17T03:00:00Z"), "2026-11-16"); // Monday 10pm ET
  assert.equal(weekStart("2026-11-16T04:30:00Z"), "2026-11-09"); // Sunday 11:30pm ET
  assert.equal(weekStart("2026-11-22T23:00:00Z"), "2026-11-16");
  assert.equal(weekStart("2027-01-01T00:00:00Z"), "2026-12-28");
});

test("only final games with a prediction are graded", () => {
  const win = gradePrediction({ status: "final", homePoints: 70, awayPoints: 60, homeWinProbability: 0.8 });
  assert.equal(win.correct, true);
  close(win.brier, 0.04, 1e-12);
  assert.equal(gradePrediction({ status: "final", homePoints: 60, awayPoints: 70, homeWinProbability: 0.8 }).correct, false);
  assert.equal(gradePrediction({ status: "scheduled", homeWinProbability: 0.8 }), null);
  assert.equal(gradePrediction({ status: "final", homePoints: 70, awayPoints: 60 }), null);
  assert.equal(gradePrediction({ status: "postponed", homePoints: 0, awayPoints: 0, homeWinProbability: 0.6 }), null);
  const summary = summarizePredictions([
    { status: "final", homePoints: 70, awayPoints: 60, homeWinProbability: 0.8 },
    { status: "final", homePoints: 60, awayPoints: 70, homeWinProbability: 0.6 },
    { status: "scheduled", homeWinProbability: 0.5 },
    { status: "scheduled" },
  ]);
  assert.equal(summary.games, 4);
  assert.equal(summary.predicted, 3);
  assert.equal(summary.graded, 2);
  assert.equal(summary.correct, 1);
  close(summary.brier, (0.04 + 0.36) / 2, 1e-12);
  assert.equal(summarizePredictions([]).brier, null);
});
