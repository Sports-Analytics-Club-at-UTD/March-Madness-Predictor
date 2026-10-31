"use client";

import { useEffect, useState } from "react";
import { useAction, useQuery } from "convex/react";
import { api } from "../../convex/_generated/api";
import { KalshiOddsChart } from "@/components/kalshi-odds-chart";
import { Skeleton } from "@/components/ui/skeleton";

export function GameOddsPreview({ tournamentSlug, gameId, hypothetical = false }: { tournamentSlug: string; gameId: string; hypothetical?: boolean }) {
  const odds = useQuery(api.gameOdds.get, { tournamentSlug, gameId });
  const chart = useQuery(api.gameOddsCharts.get, { tournamentSlug, gameId });
  const refresh = useAction(api.kalshi.refreshGameChart);
  const [error, setError] = useState(false);
  const [retry, setRetry] = useState(0);

  useEffect(() => {
    if (!odds?.eventTicker || chart?.status === "settled") return;
    let active = true;
    let timer: ReturnType<typeof setTimeout> | undefined;
    async function update() {
      if (document.visibilityState === "hidden") {
        if (active) timer = setTimeout(update, 15_000);
        return;
      }
      try {
        await refresh({ tournamentSlug, gameId });
        if (active) {
          setError(false);
          timer = setTimeout(update, 15_000);
        }
      } catch {
        if (active) setError(true);
      }
    }
    void update();
    return () => { active = false; clearTimeout(timer); };
  }, [odds?.eventTicker, chart?.status, refresh, tournamentSlug, gameId, retry]);

  if (odds === null || (odds && (!odds.eventTicker || odds.outcomes.length !== 2))) return (
    <div role="status" className="flex min-h-60 flex-col items-center justify-center gap-2 px-6 text-center text-sm text-muted-foreground">
      <p className={hypothetical ? "font-medium text-amber-700 dark:text-amber-400" : ""}>{hypothetical ? "Hypothetical matchup" : "Couldn't find game"}</p>
      {hypothetical && <p>This matchup diverges from the real bracket, so no game odds are available.</p>}
    </div>
  );
  if (!chart && !error) return (
    <div role="status" className="space-y-5 p-4">
      <span className="sr-only">Loading Kalshi prices…</span>
      <Skeleton className="h-5 w-28 bg-muted motion-reduce:animate-none" />
      <Skeleton className="h-48 w-full bg-muted motion-reduce:animate-none" />
    </div>
  );
  return (
    <>
      {chart && <KalshiOddsChart chart={chart} />}
      {error && <div role="status" className="flex min-h-12 items-center justify-between gap-3 px-4 py-3 text-xs text-muted-foreground">
        <span>{chart ? "Price update unavailable. Showing saved prices." : "Kalshi prices are temporarily unavailable."}</span>
        <button type="button" onClick={() => setRetry(value => value + 1)} className="shrink-0 rounded px-2 py-1 text-foreground underline underline-offset-4 focus-visible:outline-2">Try again</button>
      </div>}
    </>
  );
}
