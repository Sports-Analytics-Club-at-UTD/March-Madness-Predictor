"use client";

import { useQuery } from "convex/react";
import dynamic from "next/dynamic";
import { api } from "../../../../convex/_generated/api";
import { TournamentBracketSkeleton } from "@/components/tournament-bracket-skeleton";
import { useBracketWorkspace } from "@/components/bracket-workspace";
import { CbbdResultSync } from "@/components/cbbd-result-sync";

const TournamentBracket = dynamic(
  () => import("@/components/tournament-bracket").then(module => module.TournamentBracket),
  { loading: () => <TournamentBracketSkeleton /> },
);

export default function MmpPage() {
  const { selectedId, brackets } = useBracketWorkspace();
  const tournament = useQuery(api.brackets.get, selectedId ? { id: selectedId } : "skip");
  if (brackets === undefined || (selectedId && tournament === undefined)) return <TournamentBracketSkeleton />;
  if (!selectedId || tournament === null) return <p role="status" className="p-16 text-sm text-muted-foreground">Select a bracket from the sidebar.</p>;
  if (!tournament) return null;
  return <>
    <TournamentBracket key={selectedId} tournament={tournament} showScores={brackets.find(bracket => bracket._id === selectedId)?.slug === "2026-results"} />
    <CbbdResultSync season={tournament.season} />
  </>;
}
