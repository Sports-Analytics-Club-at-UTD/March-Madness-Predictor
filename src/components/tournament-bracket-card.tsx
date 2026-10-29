import Image from "next/image";
import { Card } from "@/components/ui/card";

export type BracketTeam = {
  name: string;
  logoUrl: string;
  seed?: number;
  score?: number;
};

export type TournamentBracketCardProps = {
  topTeam: BracketTeam;
  bottomTeam: BracketTeam;
  winner: "top" | "bottom";
  showScores?: boolean;
};

export function TournamentBracketCard({ topTeam, bottomTeam, winner, showScores = false }: TournamentBracketCardProps) {
  return (
    <Card className="w-96 gap-0 overflow-hidden rounded-2xl bg-background py-0 shadow-sm">
      {([topTeam, bottomTeam] as const).map((team, index) => {
        const won = winner === (index === 0 ? "top" : "bottom");
        return (
          <div key={index} className={`flex h-22 items-center gap-4 px-7 ${won ? "bg-background" : "bg-sidebar text-sidebar-foreground"} ${index === 0 ? "border-b border-dashed border-foreground/60" : ""}`}>
            <Image src={team.logoUrl} alt={`${team.name} logo`} width={40} height={40} unoptimized draggable={false} className="size-10 shrink-0 object-contain" />
            <span className="min-w-0 flex-1 truncate text-xl" title={team.name}>
              {team.seed !== undefined && <span className="mr-1.5 text-sm text-muted-foreground">{team.seed}</span>}
              {team.name}
            </span>
            {showScores && team.score !== undefined && <span className="text-base tabular-nums">{team.score}</span>}
            <span aria-label={won ? "Winner" : "Loser"} className={`shrink-0 text-3xl font-medium ${won ? "text-[#247a61] dark:text-[#46967d]" : "text-red-600 dark:text-red-400"}`}>
              {won ? "W" : "L"}
            </span>
          </div>
        );
      })}
    </Card>
  );
}
