"use client";

import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { ArrowLeft, BookOpen, CalendarDays, Swords } from "lucide-react";
import { Sidebar, SidebarContent, SidebarHeader, useSidebar } from "@/components/ui/sidebar";
import { Skeleton } from "@/components/ui/skeleton";
import { PredictionBracketCard } from "@/components/prediction-bracket-card";
import { useBracketWorkspace } from "@/components/bracket-workspace";

export function MmpSidebar() {
  const { brackets, selectedId, selectBracket } = useBracketWorkspace();
  const { isMobile, setOpenMobile } = useSidebar();
  const pathname = usePathname();
  const router = useRouter();
  const liveLinks = [
    { href: "/projects/mmp/matchup", label: "Live matchup", Icon: Swords },
    { href: "/projects/mmp/weekly", label: "Weekly predictions", Icon: CalendarDays },
  ];
  return (
    <Sidebar collapsible="offcanvas">
      <SidebarHeader className="px-3 pt-3 pb-0">
        <Link href="/projects" aria-label="Back to projects" title="Back to projects" className="flex size-8 items-center justify-center rounded-lg text-accent transition-colors hover:bg-accent/10 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent">
          <ArrowLeft aria-hidden="true" className="size-4" />
        </Link>
        <Link href="/projects/mmp/writeup" className="mt-3 flex items-center gap-2 rounded-lg border border-sidebar-border px-3 py-2 text-sm font-medium transition-colors hover:bg-foreground/5 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent">
          <BookOpen aria-hidden="true" className="size-4 text-accent" />
          How the model works
        </Link>
      </SidebarHeader>
      <nav aria-label="Live season" className="grid gap-1 px-3 pt-3">
        {liveLinks.map(({ href, label, Icon }) => (
          <Link key={href} href={href} aria-current={pathname === href ? "page" : undefined}
            onClick={() => { if (isMobile) setOpenMobile(false); }}
            className={`flex items-center gap-2 rounded-lg px-2 py-1.5 text-sm transition-colors hover:bg-muted focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent ${pathname === href ? "bg-muted font-medium" : "text-muted-foreground"}`}>
            <Icon aria-hidden="true" className="size-4" />{label}
          </Link>
        ))}
      </nav>
      <SidebarContent aria-label="Brackets" aria-busy={brackets === undefined} className="gap-5 px-3 py-4">
        {brackets === undefined && <div role="status" className="space-y-5">
          <span className="sr-only">Loading brackets…</span>
          {[0, 1, 2].map(index => <Skeleton key={index} className="h-32 w-full rounded-xl motion-reduce:animate-none" />)}
        </div>}
        {brackets?.length === 0 && <p className="text-sm text-muted-foreground">No brackets available yet.</p>}
        {brackets?.map(bracket => (
          <div key={bracket._id}>
            <PredictionBracketCard title={bracket.title} imageUrl={bracket.imageUrl ?? "/next-favicon.ico"} bracketId={bracket._id} selected={pathname === "/projects/mmp" && selectedId === bracket._id} onSelect={id => {
              selectBracket(id);
              if (pathname !== "/projects/mmp") router.push("/projects/mmp");
              if (isMobile) setOpenMobile(false);
            }} />
          </div>
        ))}
      </SidebarContent>
    </Sidebar>
  );
}
