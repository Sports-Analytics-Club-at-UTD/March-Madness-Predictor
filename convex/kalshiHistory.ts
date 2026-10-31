type Candle = {
  end_period_ts: number;
  price: { close?: string | null; close_dollars?: string | null };
};

export function parseMarketPrice(value: string | undefined | null) {
  if (value == null || value.trim() === "") return undefined;
  const price = Number(value);
  return Number.isFinite(price) && price >= 0 && price <= 1 ? price : undefined;
}

export function chartPoints(candles: Candle[]) {
  const prices = new Map<number, number>();
  for (const candle of candles) {
    const price = parseMarketPrice(candle.price.close_dollars ?? candle.price.close);
    if (price !== undefined && Number.isFinite(candle.end_period_ts)) prices.set(candle.end_period_ts, price);
  }
  return [...prices].sort(([a], [b]) => a - b).map(([timestamp, price]) => ({ timestamp, price }));
}

export function marketState(markets: { status: string; result: string }[]): "live" | "closed" | "settled" {
  if (markets.every(market => market.status === "finalized" && ["yes", "no"].includes(market.result))) return "settled";
  return markets.some(market => market.status === "active") ? "live" : "closed";
}
