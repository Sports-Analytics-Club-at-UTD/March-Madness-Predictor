"use client";

import { useQuery } from "convex/react";
import { api } from "../../convex/_generated/api";
import { CardContent } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";

export function GameOddsPreview({ tournamentSlug, gameId }: { tournamentSlug: string; gameId: string }) {
  const odds = useQuery(api.gameOdds.get, { tournamentSlug, gameId });
  if (odds === undefined) return (
    <CardContent role="status" className="space-y-3">
      <span className="sr-only">Loading Kalshi odds…</span>
      <Skeleton className="h-4 w-36 motion-reduce:animate-none" />
      <Skeleton className="h-8 w-full motion-reduce:animate-none" />
      <Skeleton className="h-8 w-full motion-reduce:animate-none" />
    </CardContent>
  );
  if (!odds?.eventTicker || odds.outcomes.length !== 2) return (
    <CardContent role="status" className="flex min-h-32 items-center justify-center text-sm text-muted-foreground">
      {"Couldn't find game"}
    </CardContent>
  );
  const timestamps = odds.outcomes.flatMap(outcome => outcome.lastTimestamp === undefined ? [] : [outcome.lastTimestamp]);
  const formatPrice = (price: number | undefined) => price === undefined ? "—" : `${(price * 100).toFixed(1)}%`;
  return (
    <CardContent className="space-y-3">
      <p className="font-medium">Kalshi · Historical prices</p>
      <table className="w-full text-left text-xs">
        <caption className="sr-only">Historical implied win probabilities from Kalshi trades</caption>
        <thead className="text-muted-foreground"><tr>
          <th scope="col" className="pb-2 font-normal">Team</th>
          <th scope="col" className="pb-2 text-right font-normal">First trade</th>
          <th scope="col" className="pb-2 text-right font-normal">Last trade</th>
        </tr></thead>
        <tbody>{odds.outcomes.map(outcome => (
          <tr key={outcome.ticker}>
            <th scope="row" className="max-w-36 truncate py-1.5 pr-2 font-medium" title={outcome.teamName}>{outcome.teamName}</th>
            <td className="text-right tabular-nums">{formatPrice(outcome.firstPrice)}</td>
            <td className="text-right tabular-nums">{formatPrice(outcome.lastPrice)}</td>
          </tr>
        ))}</tbody>
      </table>
      <p className="text-xs leading-relaxed text-muted-foreground">
        {odds.marketKind === "championship" ? "Tournament champion market · history after finalists were known." : "Game winner markets · hourly trade history."}
        {timestamps.length > 0 ? ` Last period: ${new Date(Math.max(...timestamps) * 1000).toLocaleString("en-US", { timeZone: "America/New_York", dateStyle: "medium", timeStyle: "short" })} ET.` : " Price history unavailable."}
      </p>
      <p className="break-all text-[10px] text-muted-foreground">{odds.eventTicker}</p>
    </CardContent>
  );
}
