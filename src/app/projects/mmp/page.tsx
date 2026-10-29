"use client";

import { ReactFlow } from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import { MAX_ZOOM, MIN_ZOOM, MmpCanvasToolbar } from "@/components/mmp-canvas-toolbar";

export default function MmpPage() {
  return (
    <ReactFlow
      className="bg-background"
      style={{ backgroundColor: "var(--background)" }}
      aria-label="March Madness canvas"
      panOnDrag
      zoomOnScroll
      zoomOnPinch
      minZoom={MIN_ZOOM}
      maxZoom={MAX_ZOOM}
      proOptions={{ hideAttribution: true }}
    >
      <MmpCanvasToolbar />
    </ReactFlow>
  );
}
