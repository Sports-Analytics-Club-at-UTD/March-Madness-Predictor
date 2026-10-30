"use client";

import { Handle, Position, type Node, type NodeProps } from "@xyflow/react";
import { PreviewCard } from "@base-ui/react/preview-card";
import { useState } from "react";
import { Card } from "@/components/ui/card";
import { GameOddsPreview } from "@/components/game-odds-preview";
import { TournamentBracketCard, type TournamentBracketCardProps } from "@/components/tournament-bracket-card";

export type TournamentBracketNodeType = Node<TournamentBracketCardProps & { tournamentSlug: string; label?: string; semifinal?: boolean; direction?: "left" | "right" | "final" }, "tournamentBracket">;
const handleClass = "pointer-events-none! size-1! border-0! bg-transparent!";

export function TournamentBracketNode({ id, data }: NodeProps<TournamentBracketNodeType>) {
  const [previewOpen, setPreviewOpen] = useState(false);
  const headerHeight = data.label ? 24 : 0;
  const inputSide = data.direction === "right" ? Position.Right : Position.Left;
  const outputSide = data.direction === "right" ? Position.Left : Position.Right;
  return (
    <div className="relative">
      {data.label && <p className="h-6 px-1 text-sm font-medium text-muted-foreground">{data.label}</p>}
      <Handle id="top" type="target" position={data.semifinal ? Position.Top : inputSide} isConnectable={false} className={handleClass} style={{ top: data.semifinal ? headerHeight : headerHeight + 88 }} />
      <Handle id="bottom" type="target" position={data.semifinal ? Position.Bottom : data.direction === "final" ? Position.Right : inputSide} isConnectable={false} className={handleClass} style={{ top: data.semifinal ? headerHeight + 176 : headerHeight + 88 }} />
      <PreviewCard.Root open={previewOpen} onOpenChange={setPreviewOpen}>
        <PreviewCard.Trigger
          render={<div />}
          tabIndex={0}
          aria-label={`Preview odds for ${data.topTeam.name} vs ${data.bottomTeam.name}`}
          delay={1000}
          className="rounded-2xl outline-none focus-visible:ring-2 focus-visible:ring-ring"
        >
          <TournamentBracketCard topTeam={data.topTeam} bottomTeam={data.bottomTeam} winner={data.winner} showScores={data.showScores} />
        </PreviewCard.Trigger>
        <PreviewCard.Portal>
          <PreviewCard.Positioner side="top" align="center" sideOffset={12} className="z-50">
            <PreviewCard.Popup
              render={<Card />}
              aria-label={`Kalshi odds for ${data.topTeam.name} vs ${data.bottomTeam.name}`}
              className="nodrag nopan nowheel min-h-48 w-96 max-w-[calc(100vw-2rem)] shadow-xl"
            >
              {previewOpen && <GameOddsPreview tournamentSlug={data.tournamentSlug} gameId={id} />}
            </PreviewCard.Popup>
          </PreviewCard.Positioner>
        </PreviewCard.Portal>
      </PreviewCard.Root>
      <Handle id="winner" type="source" position={outputSide} isConnectable={false} className={handleClass} style={{ top: headerHeight + 88 }} />
    </div>
  );
}
