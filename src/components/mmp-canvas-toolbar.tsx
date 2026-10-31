"use client";

import { useTheme } from "@/components/theme-provider";
import type { MouseEvent } from "react";
import { Maximize, Moon, Sun, ZoomIn, ZoomOut } from "lucide-react";
import { Panel, useReactFlow, useViewport } from "@xyflow/react";
import { Button } from "@/components/ui/button";
import { toggleThemeWithReveal } from "@/lib/theme-transition";

export const MIN_ZOOM = 0.01;
export const MAX_ZOOM = 4;

export function MmpCanvasToolbar() {
  const flow = useReactFlow();
  const { zoom } = useViewport();
  const { resolvedTheme, setTheme } = useTheme();

  function changeZoom(delta: number) {
    void flow.zoomTo(Math.min(MAX_ZOOM, Math.max(MIN_ZOOM, Math.round((flow.getZoom() + delta) * 100) / 100)));
  }

  function fitContent() {
    if (flow.getNodes().length === 0) {
      void flow.setViewport({ x: 0, y: 0, zoom: 1 });
      return;
    }
    void flow.fitView({ padding: 0.15, minZoom: MIN_ZOOM, maxZoom: MAX_ZOOM });
  }

  function toggleTheme(event: MouseEvent<HTMLButtonElement>) {
    void toggleThemeWithReveal(event, resolvedTheme === "dark" ? "light" : "dark", setTheme);
  }

  const buttonClass = "size-12 rounded-full hover:bg-foreground/10 [&_svg]:size-6 sm:w-16";

  return (
    <Panel position="bottom-center" className="mb-6!">
      <div role="toolbar" aria-label="Canvas controls" className="flex items-center rounded-full border border-border/50 bg-muted/90 p-2 shadow-lg backdrop-blur-xl">
        <Button variant="ghost" size="icon" className={buttonClass} disabled={zoom >= MAX_ZOOM} onClick={() => changeZoom(0.2)} aria-label="Zoom in by 20%" title="Zoom in by 20%"><ZoomIn /></Button>
        <span aria-hidden="true" className="h-8 border-l-2 border-dashed border-foreground/40" />
        <Button variant="ghost" size="icon" className={buttonClass} onClick={fitContent} aria-label="Fit all content" title="Fit all content"><Maximize /></Button>
        <span aria-hidden="true" className="h-8 border-l-2 border-dashed border-foreground/40" />
        <Button variant="ghost" size="icon" className={buttonClass} disabled={zoom <= MIN_ZOOM} onClick={() => changeZoom(-0.2)} aria-label="Zoom out by 20%" title="Zoom out by 20%"><ZoomOut /></Button>
        <span aria-hidden="true" className="h-8 border-l-2 border-dashed border-foreground/40" />
        <Button variant="ghost" size="icon" className={buttonClass} onClick={toggleTheme} aria-label="Toggle light and dark theme" title="Toggle light and dark theme"><Moon className="dark:hidden" /><Sun className="hidden dark:block" /></Button>
      </div>
    </Panel>
  );
}
