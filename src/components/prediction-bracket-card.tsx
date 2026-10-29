"use client";

import Image from "next/image";
import type { Id } from "../../convex/_generated/dataModel";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";

type PredictionBracketCardProps = {
  imageUrl: string;
  title: string;
  bracketId: Id<"brackets">;
  onSelect: (id: Id<"brackets">) => void;
  selected?: boolean;
};

export function PredictionBracketCard({ imageUrl, title, bracketId, onSelect, selected = false }: PredictionBracketCardProps) {
  return (
    <Button variant="ghost" onClick={() => onSelect(bracketId)} aria-pressed={selected} aria-label={`Show ${title}`} className="h-auto w-full rounded-xl p-0 text-left whitespace-normal">
      <Card className={`relative min-h-32 w-full justify-center overflow-hidden border py-6 ${selected ? "border-foreground/40 bg-muted" : "border-border bg-sidebar hover:bg-muted"}`}>
        <span className="relative z-10 max-w-[75%] px-4 text-2xl font-normal leading-snug tracking-tight">{title}</span>
        <Image src={imageUrl} alt="" width={96} height={96} unoptimized className="absolute -right-5 top-1/2 size-24 -translate-y-1/2 object-contain opacity-50" />
      </Card>
    </Button>
  );
}
