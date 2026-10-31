"use client";

import { Handle, Position, type Node, type NodeProps } from "@xyflow/react";
import { PreviewCard } from "@base-ui/react/preview-card";
import { useState } from "react";
import { Card } from "@/components/ui/card";
import { GameOddsPreview } from "@/components/game-odds-preview";
import { TournamentBracketCard, type TournamentBracketCardProps } from "@/components/tournament-bracket-card";
import type { GameComparison } from "../../convex/bracketComparison";

export type TournamentBracketNodeType = Node<TournamentBracketCardProps & { tournamentSlug: string; bracketStatus?: GameComparison["status"]; label?: string; semifinal?: boolean; direction?: "left" | "right" | "final" }, "tournamentBracket">;
const handleClass = "pointer-events-none! size-1! border-0! bg-transparent!";

export function TournamentBracketNode({ id, data }: NodeProps<TournamentBracketNodeType>) {
  const [previewOpen, setPreviewOpen] = useState(false);
  const headerHeight = data.label ? 24 : 0;
  const inputSide = data.direction === "right" ? Position.Right : Position.Left;
  const outputSide = data.direction === "right" ? Position.Left : Position.Right;
  return (
    <div className="relative">
      {data.label && <p className="flex h-6 items-center justify-between gap-2 px-1 text-sm font-medium text-muted-foreground">
        <span className="truncate">{data.label}</span>
        {data.bracketStatus === "break" && <span className="shrink-0 text-xs font-semibold text-red-600 dark:text-red-400">Bracket breaks here</span>}
        {data.bracketStatus === "hypothetical" && <span className="shrink-0 text-xs font-semibold text-amber-700 dark:text-amber-400">Hypothetical matchup</span>}
      </p>}
      <Handle id="top" type="target" position={data.semifinal ? Position.Top : inputSide} isConnectable={false} className={handleClass} style={{ top: data.semifinal ? headerHeight : headerHeight + 88 }} />
      <Handle id="bottom" type="target" position={data.semifinal ? Position.Bottom : data.direction === "final" ? Position.Right : inputSide} isConnectable={false} className={handleClass} style={{ top: data.semifinal ? headerHeight + 176 : headerHeight + 88 }} />
      <PreviewCard.Root open={previewOpen} onOpenChange={setPreviewOpen}>
        <PreviewCard.Trigger
          render={<div />}
          tabIndex={0}
          aria-label={`Preview odds for ${data.topTeam.name} vs ${data.bottomTeam.name}${data.bracketStatus === "break" ? ". Bracket breaks here: predicted winner differs from the actual result" : data.bracketStatus === "hypothetical" ? ". Hypothetical matchup: no game market" : ""}`}
          delay={1000}
          className={`rounded-2xl focus-visible:outline-2 focus-visible:outline-offset-4 focus-visible:outline-ring ${data.bracketStatus === "break" ? "ring-2 ring-red-500 outline-none" : data.bracketStatus === "hypothetical" ? "outline-2 outline-dashed outline-offset-2 outline-amber-500" : "outline-none"}`}
        >
          <TournamentBracketCard topTeam={data.topTeam} bottomTeam={data.bottomTeam} winner={data.winner} showScores={data.showScores} />
        </PreviewCard.Trigger>
        <PreviewCard.Portal>
          <PreviewCard.Positioner side="top" align="center" sideOffset={12} className="z-50">
            <PreviewCard.Popup
              render={<Card />}
              aria-label={`Kalshi odds for ${data.topTeam.name} vs ${data.bottomTeam.name}`}
              className="nodrag nopan nowheel min-h-48 w-[560px] max-w-[calc(100vw-2rem)] gap-0 overflow-hidden border-border bg-card py-0 text-card-foreground shadow-xl ring-1 ring-border"
            >
              {previewOpen && <GameOddsPreview tournamentSlug={data.tournamentSlug} gameId={id} hypothetical={data.bracketStatus === "hypothetical"} />}
            </PreviewCard.Popup>
          </PreviewCard.Positioner>
        </PreviewCard.Portal>
      </PreviewCard.Root>
      <Handle id="winner" type="source" position={outputSide} isConnectable={false} className={handleClass} style={{ top: headerHeight + 88 }} />
    </div>
  );
}
