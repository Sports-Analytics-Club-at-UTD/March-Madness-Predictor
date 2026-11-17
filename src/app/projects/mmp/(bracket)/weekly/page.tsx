"use client";

import { useState } from "react";
import { useQuery } from "convex/react";
import type { FunctionReturnType } from "convex/server";
import { Check, X } from "lucide-react";
import { api } from "../../../../../convex/_generated/api";
import { gradePrediction } from "../../../../../convex/liveMatchupModel";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Skeleton } from "@/components/ui/skeleton";
import { LiveModelNotes, formatUpdated, percent } from "@/components/live-model-notes";

type Game = FunctionReturnType<typeof api.liveMatchups.week>[number];
type Summary = { graded: number; correct: number; brier: number | null; predicted: number; games: number };

const day = (iso: string) => new Intl.DateTimeFormat("en-US", { timeZone: "America/New_York", weekday: "long", month: "short", day: "numeric" }).format(new Date(iso));
const time = (iso: string) => new Intl.DateTimeFormat("en-US", { timeZone: "America/New_York", hour: "numeric", minute: "2-digit", timeZoneName: "short" }).format(new Date(iso));
const weekLabel = (start: string) => new Intl.DateTimeFormat("en-US", { timeZone: "UTC", month: "short", day: "numeric" }).format(new Date(`${start}T12:00:00Z`));

function record(summary: Summary) {
  if (!summary.graded) return `${summary.predicted} predicted, none final yet`;
  return `${summary.correct} of ${summary.graded} correct (${percent(summary.correct / summary.graded)}), Brier ${summary.brier!.toFixed(3)}`;
}

function GameRow({ game }: { game: Game }) {
  const p = game.homeWinProbability;
  const grade = gradePrediction(game);
  const pick = p === undefined ? null : p >= 0.5 ? { team: game.homeTeam, p } : { team: game.awayTeam, p: 1 - p };
  const scored = game.status !== "scheduled" && game.homePoints !== undefined && game.awayPoints !== undefined;
  return (
    <li className="grid gap-1 py-2.5 sm:grid-cols-[6rem_1fr_13rem] sm:items-center sm:gap-4">
      <span className="text-xs text-muted-foreground">{time(game.startDate)}</span>
      <span>
        {game.awayTeam} {scored && <span className="tabular-nums">{game.awayPoints}</span>}
        <span className="text-muted-foreground"> {game.neutralSite ? "vs" : "at"} </span>
        {game.homeTeam} {scored && <span className="tabular-nums">{game.homePoints}</span>}
        {game.status !== "scheduled" && game.status !== "final" && <span className="ml-2 text-xs text-muted-foreground">{game.status.replace("_", " ")}</span>}
      </span>
      {pick ? (
        <span className="flex items-center gap-1.5 text-sm sm:justify-end">
          {grade && (grade.correct
            ? <Check aria-label="Correct pick" className="size-4 text-primary" />
            : <X aria-label="Wrong pick" className="size-4 text-destructive" />)}
          <span>{pick.team}</span>
          <span className="tabular-nums text-muted-foreground">{percent(pick.p)}</span>
        </span>
      ) : <span className="text-xs text-muted-foreground sm:text-right">{game.unavailableReason ?? "No prediction"}</span>}
    </li>
  );
}

function Week({ season, weekStart, filter }: { season: number; weekStart: string; filter: string }) {
  const games = useQuery(api.liveMatchups.week, { season, weekStart });
  if (games === undefined) return <Skeleton className="h-64 rounded-xl" />;
  const needle = filter.trim().toLowerCase();
  const shown = needle ? games.filter(game => `${game.homeTeam} ${game.awayTeam}`.toLowerCase().includes(needle)) : games;
  if (!shown.length) return <p role="status" className="text-sm text-muted-foreground">No games match “{filter}”.</p>;
  const byDay = new Map<string, Game[]>();
  for (const game of shown) byDay.set(day(game.startDate), [...(byDay.get(day(game.startDate)) ?? []), game]);
  return (
    <div className="space-y-6">
      {[...byDay].map(([label, list]) => (
        <section key={label} aria-label={label}>
          <h3 className="border-b border-border pb-1 text-sm font-medium">{label} <span className="font-normal text-muted-foreground">· {list.length} games</span></h3>
          <ul className="divide-y divide-border text-sm">{list.map(game => <GameRow key={game.gameId} game={game} />)}</ul>
        </section>
      ))}
    </div>
  );
}

export default function WeeklyPredictionsPage() {
  const overview = useQuery(api.liveMatchups.overview);
  const data = useQuery(api.liveMatchups.weeks);
  const [selected, setSelected] = useState<string | null>(null);
  const [filter, setFilter] = useState("");
  if (data === undefined || overview === undefined) return <div role="status" className="p-16"><span className="sr-only">Loading predictions…</span><Skeleton className="h-40 max-w-3xl rounded-xl" /></div>;
  const weeks = data?.weeks ?? [];
  // Default to the week that is under way (or the next one), otherwise the latest stored week.
  const today = new Date().toISOString().slice(0, 10);
  const current = [...weeks].reverse().find(week => week.weekStart >= addDays(today, -6))?.weekStart ?? weeks[0]?.weekStart;
  const active = weeks.find(week => week.weekStart === selected) ?? weeks.find(week => week.weekStart === current);

  return (
    <div className="h-full overflow-y-auto px-4 pb-16 pt-16 sm:px-16">
      <div className="mx-auto max-w-4xl space-y-6">
        <header className="space-y-1">
          <h1 className="text-3xl font-normal tracking-tight">Weekly predictions</h1>
          <p className="text-sm text-muted-foreground">
            Model win probabilities for upcoming Division I men&apos;s games, made before tip-off and graded once games are final.
          </p>
          {data && <p className="text-xs text-muted-foreground">
            Season so far: {record(data.overall)}.{data.run && ` Ratings updated ${formatUpdated(data.run.updatedAt)}.`}
          </p>}
        </header>
        {!data || !weeks.length ? (
          <p role="status" className="rounded-xl bg-card p-4 text-sm ring-1 ring-foreground/10">
            No predictions stored yet. Run <code>pnpm live:sync</code> to pull the coming week&apos;s games from CBBD.
          </p>
        ) : <>
          <nav aria-label="Weeks" className="flex flex-wrap gap-2">
            {weeks.map(week => (
              <Button key={week.weekStart} size="sm" variant={week.weekStart === active?.weekStart ? "secondary" : "ghost"}
                aria-pressed={week.weekStart === active?.weekStart} onClick={() => setSelected(week.weekStart)}>
                Week of {weekLabel(week.weekStart)}
              </Button>
            ))}
          </nav>
          {active && <>
            <div className="flex flex-wrap items-center justify-between gap-3 rounded-xl bg-card p-4 text-sm ring-1 ring-foreground/10">
              <span><strong className="font-medium">Week of {weekLabel(active.weekStart)}:</strong> {active.games} games, {record(active)}.</span>
              <Input aria-label="Filter by team" placeholder="Filter by team" value={filter} onChange={event => setFilter(event.target.value)} className="max-w-56" />
            </div>
            <Week season={data.season} weekStart={active.weekStart} filter={filter} />
          </>}
        </>}
        <LiveModelNotes model={overview.model} run={overview.run} />
      </div>
    </div>
  );
}

function addDays(isoDay: string, days: number) {
  const date = new Date(`${isoDay}T12:00:00Z`);
  date.setUTCDate(date.getUTCDate() + days);
  return date.toISOString().slice(0, 10);
}
