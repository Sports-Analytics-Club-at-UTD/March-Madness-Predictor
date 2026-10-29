import { Skeleton } from "@/components/ui/skeleton";

export function TournamentBracketSkeleton() {
  return (
    <div role="status" aria-busy="true" className="flex h-full items-center justify-center overflow-hidden p-12">
      <span className="sr-only">Loading bracket…</span>
      <div aria-hidden="true" className="flex w-full min-w-[640px] max-w-5xl items-center gap-8">
        {[4, 2, 1, 2, 4].map((count, column) => (
          <div key={column} className="flex flex-1 flex-col justify-around gap-8">
            {Array.from({ length: count }, (_, row) => (
              <div key={row} className="space-y-2 rounded-xl border bg-background p-3">
                <Skeleton className="h-3 w-2/3 motion-reduce:animate-none" />
                <Skeleton className="h-8 w-full motion-reduce:animate-none" />
                <Skeleton className="h-8 w-full motion-reduce:animate-none" />
              </div>
            ))}
          </div>
        ))}
      </div>
    </div>
  );
}
