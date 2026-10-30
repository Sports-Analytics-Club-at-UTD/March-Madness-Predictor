import { test } from "node:test";
import assert from "node:assert/strict";
import { matchGameEvent, normalizeTeam, summarizePrices } from "../scripts/lib/kalshi-matching.mjs";

const event = { title: "Siena at Duke", event_ticker: "KXNCAAMBGAME-26MAR19SIEDUKE", series_ticker: "KXNCAAMBGAME" };

test("game match requires both teams, correct series, year and round date, and a unique event", () => {
  assert.equal(matchGameEvent([event], "Duke", "Siena", 1), event);
  assert.equal(matchGameEvent([event], "Duke", "Siena", 2), null);
  assert.equal(matchGameEvent([event], "Duke", "Texas", 1), null);
  assert.equal(matchGameEvent([event, event], "Duke", "Siena", 1), null);
  assert.equal(matchGameEvent([{ ...event, event_ticker: "KXNCAAMBGAME-25MAR19SIEDUKE" }], "Duke", "Siena", 1), null);
  assert.equal(matchGameEvent([{ ...event, series_ticker: "KXNCAAWBGAME" }], "Duke", "Siena", 1), null);
});

test("school aliases keep similarly named schools separate", () => {
  assert.equal(normalizeTeam("Miami (Ohio)"), normalizeTeam("Miami (OH)"));
  assert.notEqual(normalizeTeam("Miami (Ohio)"), normalizeTeam("Miami (FL)"));
  assert.equal(normalizeTeam("Hawaii"), normalizeTeam("Hawai'i"));
  assert.equal(normalizeTeam("Michigan State"), normalizeTeam("Michigan St."));
  const firstFour = { ...event, title: "First Four: Texas vs NC St.", event_ticker: "KXNCAAMBGAME-26MAR17TEXNCST" };
  assert.equal(matchGameEvent([firstFour], "NC State", "Texas", 0), firstFour);
});

test("historical prices preserve zero, skip missing values, and sort by timestamp", () => {
  assert.deepEqual(summarizePrices([
    { end_period_ts: 10800, price: { open: "0.20", close: "0.99" } },
    { end_period_ts: 3600, price: { open: "0.00", close: "0.05" } },
    { end_period_ts: 14400, price: { open: null, close: null } },
  ]), { firstPrice: 0, firstTimestamp: 0, lastPrice: 0.99, lastTimestamp: 10800 });
  assert.deepEqual(summarizePrices([]), {});
  assert.deepEqual(summarizePrices([{end_period_ts: 3600, price: { open_dollars: "0.4", close_dollars: "0.5" }}]),
    { firstPrice: 0.4, firstTimestamp: 0, lastPrice: 0.5, lastTimestamp: 3600 });
});
