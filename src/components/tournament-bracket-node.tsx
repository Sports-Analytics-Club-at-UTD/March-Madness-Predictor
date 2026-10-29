"use client";

import { Handle, Position, type Node, type NodeProps } from "@xyflow/react";
import { TournamentBracketCard, type TournamentBracketCardProps } from "@/components/tournament-bracket-card";

export type TournamentBracketNodeType = Node<TournamentBracketCardProps & { label?: string; semifinal?: boolean; direction?: "left" | "right" | "final" }, "tournamentBracket">;
const handleClass = "pointer-events-none! size-1! border-0! bg-transparent!";

export function TournamentBracketNode({ data }: NodeProps<TournamentBracketNodeType>) {
  const headerHeight = data.label ? 24 : 0;
  const inputSide = data.direction === "right" ? Position.Right : Position.Left;
  const outputSide = data.direction === "right" ? Position.Left : Position.Right;
  return (
    <div className="relative">
      {data.label && <p className="h-6 px-1 text-sm font-medium text-muted-foreground">{data.label}</p>}
      <Handle id="top" type="target" position={data.semifinal ? Position.Top : inputSide} isConnectable={false} className={handleClass} style={{ top: data.semifinal ? headerHeight : headerHeight + 88 }} />
      <Handle id="bottom" type="target" position={data.semifinal ? Position.Bottom : data.direction === "final" ? Position.Right : inputSide} isConnectable={false} className={handleClass} style={{ top: data.semifinal ? headerHeight + 176 : headerHeight + 88 }} />
      <TournamentBracketCard topTeam={data.topTeam} bottomTeam={data.bottomTeam} winner={data.winner} showScores={data.showScores} />
      <Handle id="winner" type="source" position={outputSide} isConnectable={false} className={handleClass} style={{ top: headerHeight + 88 }} />
    </div>
  );
}
