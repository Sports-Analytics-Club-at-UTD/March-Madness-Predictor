"use node";

import { ConvexError, v } from "convex/values";
import { Configuration, HistoricalApi, MarketApi } from "kalshi-typescript";
import { action } from "./_generated/server";
import { api, internal } from "./_generated/api";
import { chartPoints, marketState, parseMarketPrice } from "./kalshiHistory";
import type { KalshiChart } from "./kalshiChartModel";

export const refreshGameChart = action({
  args: { tournamentSlug: v.string(), gameId: v.string() },
  handler: async (ctx, args): Promise<"found" | "not_found"> => {
    const odds = await ctx.runQuery(api.gameOdds.get, args);
    if (!odds?.eventTicker || odds.outcomes.length !== 2) return "not_found";
    const source = { tournamentSlug: odds.tournamentSlug, gameId: odds.gameId };
    const cached = await ctx.runQuery(api.gameOddsCharts.get, source);
    if (cached && (cached.status === "settled" || Date.now() - cached.fetchedAt < 15_000)) return "found";
    const apiKey = process.env.KALSHI_API_ID;
    const privateKey = process.env.KALSHI_PRIVATE_KEY;
    if (!apiKey || !privateKey) throw new ConvexError("Kalshi server credentials are not configured.");
    try {
      const config = new Configuration({ apiKey, privateKeyPem: privateKey.replace(/\\n/g, "\n"), baseOptions: { timeout: 15_000 } });
      const live = new MarketApi(config);
      const historical = new HistoricalApi(config);
      const series = await Promise.all(odds.outcomes.map(async outcome => {
        let archived = false;
        const market = await live.getMarket(outcome.ticker).then(response => response.data.market).catch(async error => {
          if ((error as { response?: { status: number } }).response?.status !== 404) throw error;
          archived = true;
          return (await historical.getHistoricalMarket(outcome.ticker)).data.market;
        });
        const settled = marketState([market]) === "settled";
        const now = Date.now() / 1000;
        const end = settled ? Date.parse(market.close_time) / 1000 : Math.min(now, Date.parse(market.close_time) / 1000);
        const start = Math.max(Date.parse(market.open_time) / 1000, end - 6 * 3600);
        const endPeriod = Math.ceil(end / 60) * 60;
        let points: { timestamp: number; price: number }[] = [];
        if (start < end) {
          try {
            const candles = archived
              ? (await historical.getMarketCandlesticksHistorical(outcome.ticker, Math.floor(start), endPeriod, 1)).data.candlesticks
              : (await live.getMarketCandlesticks(odds.eventTicker!.split("-")[0], outcome.ticker, Math.floor(start), endPeriod, 1)).data.candlesticks;
            points = chartPoints(candles);
          } catch (error) {
            if (!archived && (error as { response?: { status: number } }).response?.status === 404) {
              points = chartPoints((await historical.getMarketCandlesticksHistorical(outcome.ticker, Math.floor(start), endPeriod, 1)).data.candlesticks);
            } else throw error;
          }
        }
        const price = settled ? (market.result === "yes" ? 1 : 0)
          : Number(market.volume_fp) > 0 ? parseMarketPrice(market.last_price_dollars) : undefined;
        if (settled && price !== undefined) {
          const settlement = Date.parse(market.settlement_ts ?? market.close_time) / 1000;
          points.push({ timestamp: Math.max(endPeriod, settlement), price });
        } else if (price !== undefined) {
          points.push({ timestamp: Math.max(points.at(-1)?.timestamp ?? 0, now), price });
        }
        return {
          market,
          chart: { teamName: outcome.teamName, ticker: outcome.ticker, ...(price === undefined ? {} : { price }), points },
        };
      }));
      const chart: KalshiChart = {
        ...source, eventTicker: odds.eventTicker, marketKind: odds.marketKind ?? "game",
        status: marketState(series.map(item => item.market)), fetchedAt: Date.now(), series: series.map(item => item.chart),
      };
      await ctx.runMutation(internal.gameOddsCharts.save, { chart });
      return "found";
    } catch {
      // SDK errors contain signed headers; expose only a safe error to the browser.
      throw new ConvexError("Kalshi prices are temporarily unavailable. Please try again.");
    }
  },
});
