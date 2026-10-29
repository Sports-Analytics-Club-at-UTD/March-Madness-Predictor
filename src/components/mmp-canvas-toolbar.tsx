"use client";

import { useRef, useSyncExternalStore, type MouseEvent } from "react";
import { Maximize, Moon, Sun, ZoomIn, ZoomOut } from "lucide-react";
import { Panel, useReactFlow, useViewport } from "@xyflow/react";
import { Button } from "@/components/ui/button";

export const MIN_ZOOM = 0.01;
export const MAX_ZOOM = 4;

function subscribeToTheme(onChange: () => void) {
  const observer = new MutationObserver(onChange);
  observer.observe(document.documentElement, { attributes: true, attributeFilter: ["class"] });
  return () => observer.disconnect();
}

function getTheme() {
  return document.documentElement.classList.contains("dark");
}

function getServerTheme() {
  return false;
}

export function MmpCanvasToolbar() {
  const flow = useReactFlow();
  const { zoom } = useViewport();
  const isDark = useSyncExternalStore(subscribeToTheme, getTheme, getServerTheme);
  const transitioning = useRef(false);

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

  async function toggleTheme(event: MouseEvent<HTMLButtonElement>) {
    if (transitioning.current) return;
    const root = document.documentElement;
    const updateTheme = () => root.classList.toggle("dark");
    if (!document.startViewTransition || window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
      updateTheme();
      return;
    }

    const bounds = event.currentTarget.getBoundingClientRect();
    const x = bounds.left + bounds.width / 2;
    const y = bounds.top + bounds.height / 2;
    const radius = Math.hypot(Math.max(x, window.innerWidth - x), Math.max(y, window.innerHeight - y));
    transitioning.current = true;
    root.dataset.themeReveal = "true";
    const transition = document.startViewTransition(() => { updateTheme(); });
    try {
      await transition.ready;
      await root.animate(
        { clipPath: [`circle(0px at ${x}px ${y}px)`, `circle(${radius}px at ${x}px ${y}px)`] },
        { duration: 550, easing: "cubic-bezier(0.4, 0, 0.2, 1)", pseudoElement: "::view-transition-new(root)" },
      ).finished;
    } catch {
      // A skipped transition still applies its theme update.
    } finally {
      await transition.finished.catch(() => {});
      delete root.dataset.themeReveal;
      transitioning.current = false;
    }
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
        <Button variant="ghost" size="icon" className={buttonClass} onClick={toggleTheme} aria-label={`Switch to ${isDark ? "light" : "dark"} mode`} title={`Switch to ${isDark ? "light" : "dark"} mode`}>{isDark ? <Moon /> : <Sun />}</Button>
      </div>
    </Panel>
  );
}
