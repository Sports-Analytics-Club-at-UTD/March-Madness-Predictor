import type { ReactNode } from "react";
import { MmpSidebar } from "@/components/mmp-sidebar";
import { SidebarInset, SidebarProvider, SidebarTrigger } from "@/components/ui/sidebar";
import { TooltipProvider } from "@/components/ui/tooltip";

// The interactive canvas shell opts out of instant-navigation validation.
export const instant = false;

export default function MmpLayout({ children }: { children: ReactNode }) {
  return (
    <TooltipProvider>
      <SidebarProvider className="h-svh min-h-0 overflow-hidden">
        <MmpSidebar />
        <SidebarInset className="min-w-0 overflow-hidden">
          <div className="absolute inset-0">{children}</div>
          <SidebarTrigger className="absolute left-4 top-4 z-10 bg-background/80 backdrop-blur-sm" />
        </SidebarInset>
      </SidebarProvider>
    </TooltipProvider>
  );
}
