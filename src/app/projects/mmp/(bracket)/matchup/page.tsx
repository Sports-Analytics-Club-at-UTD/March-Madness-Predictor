"use client";

import { useState } from "react";
import Image from "next/image";
import { useQuery } from "convex/react";
import { ArrowLeftRight } from "lucide-react";
import { api } from "../../../../../../convex/_generated/api";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { LiveModelNotes, formatUpdated, percent } from "@/components/live-model-notes";

type Venue = "neutral" | "a" | "b";
const EARLY_GAMES = 5;
const selectClass = "h-9 w-full rounded-lg border border-input bg-card px-2.5 text-sm outline-none focus-visible:border-ring focus-visible:ring-3 focus-visible:ring-ring/50";

function TeamLogo({ teamId }: { teamId: number }) {
  const logo = useQuery(api.teamLogos.getByTeamId, { teamId });
  if (!logo?.url) return <span aria-hidden="true" className="size-12" />;
  return <Image src={logo.url} alt="" width={48} height={48} unoptimized className="size-12 object-contain" />;
}

export default function LiveMatchupPage() {
  const overview = useQuery(api.liveMatchups.overview);
  const [teamA, setTeamA] = useState<number | null>(null);
  const [teamB, setTeamB] = useState<number | null>(null);
  const [venue, setVenue] = useState<Venue>("neutral");
  const ready = teamA !== null && teamB !== null && teamA !== teamB;
  const prediction = useQuery(api.liveMatchups.predict, ready ? { teamAId: teamA, teamBId: teamB, venue } : "skip");

  if (overview === undefined) return <div role="status" className="space-y-4 p-16"><span className="sr-only">Loading teams…</span><Skeleton className="h-40 max-w-3xl rounded-xl" /></div>;
  const { run, teams, model } = overview;
  const name = (id: number | null) => teams.find(team => team.teamId === id)?.teamName ?? "";
  const swap = () => {
    setTeamA(teamB);
    setTeamB(teamA);
    setVenue(venue === "a" ? "b" : venue === "b" ? "a" : "neutral");
  };

  return (
    <div className="h-full overflow-y-auto px-4 pb-16 pt-16 sm:px-16">
      <div className="mx-auto max-w-3xl space-y-6">
        <header className="space-y-1">
          <h1 className="text-3xl font-normal tracking-tight">Live matchup</h1>
          <p className="text-sm text-muted-foreground">
            Pick any two Division I men&apos;s teams to see the model&apos;s win probability from this season&apos;s results.
          </p>
          {run && <p className="text-xs text-muted-foreground">
            {run.season - 1}–{String(run.season).slice(2)} ratings from {run.gamesUsed.toLocaleString()} completed games, updated {formatUpdated(run.updatedAt)}.
          </p>}
        </header>

        {!run ? (
          <p role="status" className="rounded-xl bg-card p-4 text-sm ring-1 ring-foreground/10">
            Live ratings have not been synced yet. Run <code>pnpm live:sync</code> to compute them from CBBD.
          </p>
        ) : <>
          <section aria-label="Choose teams" className="grid items-end gap-3 sm:grid-cols-[1fr_auto_1fr]">
            <label className="space-y-1 text-sm">
              <span className="text-muted-foreground">Team A</span>
              <select className={selectClass} value={teamA ?? ""} onChange={event => setTeamA(event.target.value ? Number(event.target.value) : null)}>
                <option value="">Choose a team</option>
                {teams.map(team => <option key={team.teamId} value={team.teamId}>{team.teamName}</option>)}
              </select>
            </label>
            <Button variant="outline" size="icon-lg" onClick={swap} aria-label="Swap teams" title="Swap teams"><ArrowLeftRight aria-hidden="true" /></Button>
            <label className="space-y-1 text-sm">
              <span className="text-muted-foreground">Team B</span>
              <select className={selectClass} value={teamB ?? ""} onChange={event => setTeamB(event.target.value ? Number(event.target.value) : null)}>
                <option value="">Choose a team</option>
                {teams.map(team => <option key={team.teamId} value={team.teamId}>{team.teamName}</option>)}
              </select>
            </label>
          </section>

          <fieldset className="flex flex-wrap items-center gap-2 text-sm">
            <legend className="sr-only">Venue</legend>
            <span className="mr-1 text-muted-foreground">Venue</span>
            {([["neutral", "Neutral court"], ["a", teamA !== null ? `At ${name(teamA)}` : "Team A home"], ["b", teamB !== null ? `At ${name(teamB)}` : "Team B home"]] as const).map(([value, label]) => (
              <Button key={value} size="sm" variant={venue === value ? "secondary" : "ghost"} aria-pressed={venue === value} onClick={() => setVenue(value)}>{label}</Button>
            ))}
          </fieldset>

          {teamA !== null && teamA === teamB && <p role="status" className="text-sm text-muted-foreground">Choose two different teams.</p>}
          {!ready && teamA !== teamB && <p role="status" className="text-sm text-muted-foreground">Choose both teams to see a prediction.</p>}
          {ready && prediction === undefined && <Skeleton className="h-56 rounded-xl" />}
          {ready && prediction && (prediction.probabilityA === null || !prediction.a || !prediction.b) && (
            <p role="alert" className="rounded-xl bg-card p-4 text-sm ring-1 ring-foreground/10">
              No live features yet for {[!prediction.a && name(teamA), !prediction.b && name(teamB)].filter(Boolean).join(" and ")}:
              they have no completed Division I game with a box score this season, so the model cannot rate them.
            </p>
          )}
          {ready && prediction?.a && prediction.b && prediction.probabilityA !== null && (() => {
            const { a, b, probabilityA } = prediction;
            const favorite = probabilityA >= 0.5 ? a : b;
            const early = [a, b].filter(team => team.games < EARLY_GAMES);
            const noPrior = [a, b].filter(team => team.priorElo === undefined);
            return (
              <section aria-label="Prediction" className="space-y-5 rounded-xl bg-card p-5 ring-1 ring-foreground/10">
                <div className="grid grid-cols-[1fr_auto_1fr] items-center gap-4">
                  {[a, b].map((team, index) => {
                    const p = index === 0 ? probabilityA : 1 - probabilityA;
                    return (
                      <div key={team.teamId} className={`flex flex-col items-center gap-2 text-center ${index === 1 ? "col-start-3" : ""}`}>
                        <TeamLogo teamId={team.teamId} />
                        <span className="font-medium">{team.teamName}</span>
                        <span className={`text-4xl tabular-nums tracking-tight ${team === favorite ? "text-primary" : "text-muted-foreground"}`}>{percent(p)}</span>
                      </div>
                    );
                  })}
                  <span className="col-start-2 row-start-1 text-sm text-muted-foreground">vs</span>
                </div>
                <div className="flex h-2 overflow-hidden rounded-full bg-muted" role="img" aria-label={`${a.teamName} ${percent(probabilityA)}, ${b.teamName} ${percent(1 - probabilityA)}`}>
                  <div className="bg-primary" style={{ width: `${100 * probabilityA}%` }} />
                </div>
                <p className="text-center text-sm">
                  The model picks <strong>{favorite.teamName}</strong> to win{venue === "neutral" ? " on a neutral court" : ` at ${venue === "a" ? a.teamName : b.teamName}`}.
                </p>
                <table className="w-full text-sm tabular-nums">
                  <caption className="sr-only">Live features used for this prediction</caption>
                  <thead className="text-xs text-muted-foreground">
                    <tr><th scope="col" className="py-1 text-left font-normal">Live feature</th><th scope="col" className="text-right font-normal">{a.teamName}</th><th scope="col" className="text-right font-normal">{b.teamName}</th></tr>
                  </thead>
                  <tbody className="divide-y divide-border">
                    <tr><th scope="row" className="py-1.5 text-left font-normal">Rated games</th><td className="text-right">{a.games}</td><td className="text-right">{b.games}</td></tr>
                    <tr><th scope="row" className="py-1.5 text-left font-normal">Adjusted margin (pts)</th><td className="text-right">{a.quality.toFixed(1)}</td><td className="text-right">{b.quality.toFixed(1)}</td></tr>
                    <tr><th scope="row" className="py-1.5 text-left font-normal">Elo</th><td className="text-right">{a.elo.toFixed(0)}</td><td className="text-right">{b.elo.toFixed(0)}</td></tr>
                    <tr><th scope="row" className="py-1.5 text-left font-normal">Net efficiency (per 100)</th><td className="text-right">{a.netEff.toFixed(1)}</td><td className="text-right">{b.netEff.toFixed(1)}</td></tr>
                  </tbody>
                </table>
                {(early.length > 0 || noPrior.length > 0) && (
                  <ul className="space-y-1 rounded-lg bg-muted p-3 text-xs text-muted-foreground">
                    {early.map(team => <li key={`early-${team.teamId}`}>{team.teamName} has {team.games} rated game{team.games === 1 ? "" : "s"}; its ratings rest on few games and may move a lot.</li>)}
                    {noPrior.map(team => <li key={`prior-${team.teamId}`}>{team.teamName} has no 2025–26 Elo, so its Elo started at 1500.</li>)}
                  </ul>
                )}
              </section>
            );
          })()}
          {run.teamsWithoutFeatures.length > 0 && (
            <p className="text-xs text-muted-foreground">Not rated yet (no completed game with a box score): {run.teamsWithoutFeatures.join(", ")}.</p>
          )}
        </>}
        <LiveModelNotes model={model} run={run} />
      </div>
    </div>
  );
}
