import { MarkerType, type Edge } from "@xyflow/react";
import type { ComparedTournament } from "../../convex/bracketComparison";
import type { TournamentBracketNodeType } from "@/components/tournament-bracket-node";

export function buildTournamentBracket(tournament: ComparedTournament, showScores = false) {
  const teams = new Map(tournament.teams.map(team => [team.id, team]));
  const games = new Map(tournament.games.map(game => [game.id, game]));
  const sorted = [...tournament.games].sort((a, b) => a.round - b.round || a.order - b.order);
  const sources = new Set(tournament.games.flatMap(game => [game.topSourceGameId, game.bottomSourceGameId].filter(Boolean)));
  const final = tournament.games.find(game => !sources.has(game.id))!;
  const directions = new Map<string, "left" | "right" | "final">([[final.id, "final"]]);
  const positions = new Map<string, { x: number; y: number }>();
  const centerX = final.round * 544;

  function assignSide(id: string | undefined, side: "left" | "right") {
    if (!id) return;
    const game = games.get(id)!;
    directions.set(id, side);
    assignSide(game.topSourceGameId, side);
    assignSide(game.bottomSourceGameId, side);
  }
  assignSide(final.topSourceGameId, "left");
  assignSide(final.bottomSourceGameId, "right");

  for (const side of ["left", "right"] as const) {
    const sideGames = sorted.filter(game => directions.get(game.id) === side);
    sideGames.filter(game => game.round === 1).forEach((game, index) => {
      positions.set(game.id, { x: side === "left" ? 544 : centerX * 2 - 544, y: index * 256 });
    });
    for (const game of sideGames.filter(game => game.round > 1)) {
      const top = positions.get(game.topSourceGameId!)!;
      const bottom = positions.get(game.bottomSourceGameId!)!;
      const x = game.round * 544;
      positions.set(game.id, { x: side === "left" ? x : centerX * 2 - x, y: (top.y + bottom.y) / 2 });
    }
    for (const game of sideGames.filter(game => game.round === 0)) {
      const target = sideGames.find(target => target.topSourceGameId === game.id || target.bottomSourceGameId === game.id)!;
      positions.set(game.id, { x: side === "left" ? 0 : centerX * 2, y: positions.get(target.id)!.y });
    }
  }
  const leftY = positions.get(final.topSourceGameId ?? "")?.y ?? 0;
  const rightY = positions.get(final.bottomSourceGameId ?? "")?.y ?? 0;
  positions.set(final.id, { x: centerX, y: (leftY + rightY) / 2 });

  const labels: { text: string; x: number; y: number; align: "left" | "right" }[] = [];
  const detachedPlayIns = new Set<string>();
  const semifinals = new Set<string>();
  const leftSemi = games.get(final.topSourceGameId ?? "");
  const rightSemi = games.get(final.bottomSourceGameId ?? "");
  if (final.round >= 3 && leftSemi && rightSemi) {
    const regionalRoots = [leftSemi.topSourceGameId, leftSemi.bottomSourceGameId, rightSemi.topSourceGameId, rightSemi.bottomSourceGameId];
    if (regionalRoots.every(id => id && games.has(id))) {
      const groups = regionalRoots.map((rootId, index) => {
        const members = new Set<string>();
        function collect(id: string | undefined) {
          if (!id) return;
          const game = games.get(id)!;
          members.add(id);
          collect(game.topSourceGameId);
          collect(game.bottomSourceGameId);
        }
        collect(rootId);
        return { root: games.get(rootId!)!, members, side: index < 2 ? "left" as const : "right" as const, lower: index % 2 === 1 };
      });
      const cardWidth = 384;
      const regionHeight = Math.max(...groups.map(group => sorted.filter(game => group.members.has(game.id) && game.round === 1).length)) * 256 - 56;
      const lowerY = regionHeight + 400;
      const finalsY = regionHeight + 108;
      const roundX = (round: number) => (round - 1) * 504;
      const innerX = roundX(final.round - 2);
      const middleX = innerX + 504;
      const rightEdge = middleX * 2;
      for (const group of groups) {
        const baseY = group.lower ? lowerY : 0;
        const regionalGames = sorted.filter(game => group.members.has(game.id));
        const firstRound = regionalGames.filter(game => game.round === 1);
        firstRound.forEach((game, index) => positions.set(game.id, { x: group.side === "left" ? 0 : rightEdge, y: baseY + index * 256 }));
        for (const game of regionalGames.filter(game => game.round > 1)) {
          const x = roundX(game.round);
          positions.set(game.id, { x: group.side === "left" ? x : rightEdge - x,
            y: (positions.get(game.topSourceGameId!)!.y + positions.get(game.bottomSourceGameId!)!.y) / 2 });
        }
        labels.push({ text: group.root.label.split(" · ")[0], x: group.side === "left" ? 0 : rightEdge, y: baseY - 56, align: group.side });
      }
      semifinals.add(leftSemi.id);
      semifinals.add(rightSemi.id);
      positions.set(leftSemi.id, { x: innerX, y: finalsY });
      positions.set(rightSemi.id, { x: rightEdge - innerX, y: finalsY });
      positions.set(final.id, { x: middleX, y: finalsY });
      const playIns = sorted.filter(game => game.round === 0);
      const footerY = lowerY + regionHeight + 120;
      const footerWidth = playIns.length * (cardWidth + 32) - 32;
      playIns.forEach((game, index) => {
        positions.set(game.id, { x: (rightEdge + cardWidth - footerWidth) / 2 + index * (cardWidth + 32), y: footerY });
        detachedPlayIns.add(game.id);
      });
      if (playIns.length) labels.push({ text: "First Four", x: middleX, y: footerY - 56, align: "left" });
    }
  }

  const nodes: TournamentBracketNodeType[] = sorted.map(game => ({
    id: game.id,
    type: "tournamentBracket",
    position: positions.get(game.id)!,
    draggable: false,
    deletable: false,
    connectable: false,
    // Locked nodes still need pointer events for the game preview.
    style: { pointerEvents: "all" },
    data: {
      tournamentSlug: tournament.slug,
      bracketStatus: tournament.gameComparisons?.[game.id]?.status,
      label: game.label,
      semifinal: semifinals.has(game.id),
      direction: directions.get(game.id)!,
      showScores,
      topTeam: { ...teams.get(game.topTeamId)!, score: game.topScore },
      bottomTeam: { ...teams.get(game.bottomTeamId)!, score: game.bottomScore },
      winner: game.winnerTeamId === game.topTeamId ? "top" : "bottom",
    },
  }));
  const edges: Edge[] = [];
  for (const game of sorted) {
    for (const [source, targetHandle] of [[game.topSourceGameId, "top"], [game.bottomSourceGameId, "bottom"]] as const) {
      if (!source || detachedPlayIns.has(source) || semifinals.has(game.id)) continue;
      edges.push({
        id: `${source}-${game.id}`,
        source,
        sourceHandle: "winner",
        target: game.id,
        targetHandle,
        type: "smoothstep",
        selectable: false,
        deletable: false,
        style: { stroke: "var(--muted-foreground)", strokeWidth: 1.5 },
        markerEnd: { type: MarkerType.ArrowClosed, color: "var(--muted-foreground)", width: 16, height: 16 },
      });
    }
  }
  return { nodes, edges, labels };
}
