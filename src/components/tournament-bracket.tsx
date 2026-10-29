"use client";

import { useMemo } from "react";
import { Background, BackgroundVariant, ReactFlow, ViewportPortal } from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import type { Tournament } from "../../convex/tournamentModel";
import { buildTournamentBracket } from "@/lib/tournament-bracket";
import { TournamentBracketNode } from "@/components/tournament-bracket-node";
import { MAX_ZOOM, MIN_ZOOM, MmpCanvasToolbar } from "@/components/mmp-canvas-toolbar";

const nodeTypes = { tournamentBracket: TournamentBracketNode };

export function TournamentBracket({ tournament, showScores = false }: { tournament: Tournament; showScores?: boolean }) {
  const { nodes, edges, labels } = useMemo(() => buildTournamentBracket(tournament, showScores), [tournament, showScores]);
  return (
    <ReactFlow
      nodes={nodes}
      edges={edges}
      nodeTypes={nodeTypes}
      nodesDraggable={false}
      nodesConnectable={false}
      edgesReconnectable={false}
      elementsSelectable={false}
      deleteKeyCode={null}
      fitView
      fitViewOptions={{ padding: 0.15, minZoom: 0.65, maxZoom: 1 }}
      style={{ backgroundColor: "var(--background)" }}
      aria-label={`${tournament.name} bracket`}
      panOnDrag
      zoomOnScroll
      zoomOnPinch
      minZoom={MIN_ZOOM}
      maxZoom={MAX_ZOOM}
      proOptions={{ hideAttribution: true }}
    >
      <Background variant={BackgroundVariant.Dots} gap={32} size={1.5} color="var(--border)" />
      <ViewportPortal>
        {labels.map(label => (
          <div key={label.text} className={`pointer-events-none absolute w-96 text-base font-semibold text-muted-foreground ${label.align === "right" ? "text-right" : "text-left"}`} style={{ transform: `translate(${label.x}px, ${label.y}px)` }}>
            {label.text}
          </div>
        ))}
      </ViewportPortal>
      <MmpCanvasToolbar />
    </ReactFlow>
  );
}
