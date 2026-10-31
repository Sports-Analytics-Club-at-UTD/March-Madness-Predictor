/* eslint-disable @typescript-eslint/no-require-imports -- Load the project's TypeScript helpers without adding a test runner. */
const fs = require("node:fs");
const assert = require("node:assert/strict");
const { test } = require("node:test");
const ts = require("typescript");

require.extensions[".ts"] = (module, filename) => {
  module._compile(ts.transpileModule(fs.readFileSync(filename, "utf8"), {
    compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022 },
  }).outputText, filename);
};
const { chartPoints, marketState, parseMarketPrice } = require("../convex/kalshiHistory.ts");
const { oddsDomain, oddsPath, priceAtTime } = require("../src/lib/odds-chart.ts");

test("live and archived candle fields yield sorted prices without inventing missing trades", () => {
  assert.deepEqual(chartPoints([
    { end_period_ts: 120, price: { close: "0.25" } },
    { end_period_ts: 60, price: { close_dollars: "0.00" } },
    { end_period_ts: 180, price: { close: null } },
    { end_period_ts: 240, price: { close: "" } },
    { end_period_ts: 300, price: { close: "1.01" } },
  ]), [{ timestamp: 60, price: 0 }, { timestamp: 120, price: 0.25 }]);
  assert.equal(parseMarketPrice("0.0000"), 0);
  assert.equal(parseMarketPrice("not a price"), undefined);
});

test("only finalized yes/no outcomes are settled; closed markets can still update", () => {
  assert.equal(marketState([{ status: "active", result: "" }]), "live");
  assert.equal(marketState([{ status: "closed", result: "yes" }]), "closed");
  assert.equal(marketState([{ status: "finalized", result: "yes" }, { status: "finalized", result: "no" }]), "settled");
  assert.equal(marketState([{ status: "finalized", result: "scalar" }]), "closed");
});

test("the chart preserves the 0–100% scale and doesn't reveal future prices when scrubbing", () => {
  const points = [{ timestamp: 60, price: 0 }, { timestamp: 120, price: 1 }];
  const domain = oddsDomain([{ points }]);
  assert.deepEqual(domain, { start: 60, end: 120 });
  assert.equal(oddsPath(points, domain), "M8.00,210.00 L420.00,20.00");
  assert.equal(priceAtTime(points, 59), undefined);
  assert.deepEqual(priceAtTime(points, 90), points[0]);
  assert.equal(oddsDomain([{ points: [] }]), null);
  assert.deepEqual(oddsDomain([{ points: [points[0]] }]), { start: 60, end: 120 });
});
