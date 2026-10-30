const aliases = {
  "nc state": "nc st", "miami ohio": "miami oh", "cal baptist": "california baptist",
  "long island": "liu", hawaii: "hawaii", "queens n c": "queens university",
  "ohio state": "ohio st", "michigan state": "michigan st", "north dakota state": "north dakota st",
  "utah state": "utah st", "kennesaw state": "kennesaw st", "wright state": "wright st",
  "iowa state": "iowa st", "tennessee state": "tennessee st",
};

/** @param {string} name */
export function normalizeTeam(name) {
  const normalized = name.toLowerCase().replace(/[’']/g, "").replace(/[^a-z0-9]+/g, " ").trim();
  return aliases[/** @type {keyof typeof aliases} */ (normalized)] ?? normalized;
}

// Official 2026 round dates prevent matching regular-season games or another tournament.
const roundDates = [["0317", "0318"], ["0319", "0320"], ["0321", "0322"], ["0326", "0327"], ["0328", "0329"], ["0404"], ["0406"]];

/** @param {{title: string, event_ticker: string, series_ticker: string}[]} events
 * @param {string} topName @param {string} bottomName @param {number} round */
export function matchGameEvent(events, topName, bottomName, round) {
  const wanted = [normalizeTeam(topName), normalizeTeam(bottomName)].sort().join("|");
  const matches = events.filter(event => {
    if (event.series_ticker !== "KXNCAAMBGAME") return false;
    const date = event.event_ticker.match(/-26(MAR|APR)(\d{2})/);
    if (!date || !roundDates[round]?.includes(`${date[1] === "MAR" ? "03" : "04"}${date[2]}`)) return false;
    const names = event.title.replace(/^First Four:\s*/i, "").split(/ at | vs\.? /i);
    return names.length === 2 && names.map(normalizeTeam).sort().join("|") === wanted;
  });
  return matches.length === 1 ? matches[0] : null;
}

/** @param {{end_period_ts: number, price: {open?: string | null, close?: string | null, open_dollars?: string | null, close_dollars?: string | null}}[]} candles */
export function summarizePrices(candles) {
  const sorted = [...candles].sort((a, b) => a.end_period_ts - b.end_period_ts);
  /** @param {string | null | undefined} value */
  const valid = value => value != null && value !== "" && Number.isFinite(Number(value)) && Number(value) >= 0 && Number(value) <= 1;
  const first = sorted.find(candle => valid(candle.price.open_dollars ?? candle.price.open));
  const last = sorted.findLast(candle => valid(candle.price.close_dollars ?? candle.price.close));
  return {
    ...(first ? { firstPrice: Number(first.price.open_dollars ?? first.price.open), firstTimestamp: first.end_period_ts - 3600 } : {}),
    ...(last ? { lastPrice: Number(last.price.close_dollars ?? last.price.close), lastTimestamp: last.end_period_ts } : {}),
  };
}
