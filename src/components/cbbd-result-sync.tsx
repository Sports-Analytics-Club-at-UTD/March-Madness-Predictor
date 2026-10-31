"use client";

import { useEffect, useState } from "react";
import { useAction } from "convex/react";
import { api } from "../../convex/_generated/api";

export function CbbdResultSync({ season }: { season: number }) {
  const refresh = useAction(api.cbbd.refreshSeason);
  const [error, setError] = useState(false);
  useEffect(() => {
    let active = true;
    let timer: ReturnType<typeof setTimeout> | undefined;
    async function update() {
      if (document.visibilityState === "hidden") {
        if (active) timer = setTimeout(update, 15 * 60_000);
        return;
      }
      try {
        const status = await refresh({ season });
        if (!active) return;
        setError(false);
        if (status === "complete") return;
      } catch {
        if (active) setError(true);
      }
      if (active) timer = setTimeout(update, 15 * 60_000);
    }
    void update();
    return () => { active = false; clearTimeout(timer); };
  }, [season, refresh]);
  return error ? <p role="status" className="absolute right-4 top-4 z-10 max-w-xs rounded-lg border bg-background/90 px-3 py-2 text-xs text-muted-foreground">Couldn’t refresh CBBD results. Showing saved results.</p> : null;
}
