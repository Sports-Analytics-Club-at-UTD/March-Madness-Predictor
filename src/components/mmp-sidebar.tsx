"use client";

import { Sidebar, SidebarContent, useSidebar } from "@/components/ui/sidebar";
import { Skeleton } from "@/components/ui/skeleton";
import { PredictionBracketCard } from "@/components/prediction-bracket-card";
import { useBracketWorkspace } from "@/components/bracket-workspace";

export function MmpSidebar() {
  const { brackets, selectedId, selectBracket } = useBracketWorkspace();
  const { isMobile, setOpenMobile } = useSidebar();
  return (
    <Sidebar collapsible="offcanvas">
      <SidebarContent aria-label="Brackets" aria-busy={brackets === undefined} className="gap-5 px-3 py-4">
        {brackets === undefined && <div role="status" className="space-y-5">
          <span className="sr-only">Loading brackets…</span>
          {[0, 1, 2].map(index => <Skeleton key={index} className="h-32 w-full rounded-xl motion-reduce:animate-none" />)}
        </div>}
        {brackets?.length === 0 && <p className="text-sm text-muted-foreground">No brackets available yet.</p>}
        {brackets?.map(bracket => (
          <div key={bracket._id}>
            <PredictionBracketCard title={bracket.title} imageUrl={bracket.imageUrl ?? "/next-favicon.ico"} bracketId={bracket._id} selected={selectedId === bracket._id} onSelect={id => {
              selectBracket(id);
              if (isMobile) setOpenMobile(false);
            }} />
          </div>
        ))}
      </SidebarContent>
    </Sidebar>
  );
}
