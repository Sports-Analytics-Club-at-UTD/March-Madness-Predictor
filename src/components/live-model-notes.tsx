import type { FunctionReturnType } from "convex/server";
import type { api } from "../../convex/_generated/api";

type Overview = NonNullable<FunctionReturnType<typeof api.liveMatchups.overview>>;

export const percent = (p: number) => `${(100 * p).toFixed(1)}%`;

export function formatUpdated(timestamp: number) {
  return new Intl.DateTimeFormat("en-US", {
    timeZone: "America/New_York", month: "short", day: "numeric", hour: "numeric", minute: "2-digit", timeZoneName: "short",
  }).format(timestamp);
}

/** What the model is and where it is weak, from the exported artifact itself. */
export function LiveModelNotes({ model, run }: Pick<Overview, "model" | "run">) {
  const years = Object.values(model.backtest);
  const briers = years.map(year => year.brier), accuracy = years.map(year => year.accuracy);
  return (
    <details className="rounded-xl bg-card p-4 text-sm text-muted-foreground ring-1 ring-foreground/10">
      <summary className="cursor-pointer font-medium text-foreground">About the model and its limits</summary>
      <div className="mt-3 space-y-2 leading-relaxed">
        <p>
          A logistic regression trained on men&apos;s NCAA tournament games from {model.trainingSeasons[0]}–{model.trainingSeasons[1]}.
          It compares the two teams on three ratings computed from this season&apos;s CBBD results: a ridge-adjusted scoring
          margin, a margin-aware Elo rating carried over at 75% from last season, and net points per 100 possessions.
          It is the logistic member of the club&apos;s Kaggle model without seeds and Massey rankings (which do not
          exist before Selection Sunday), the XGBoost member, or betting-market odds.
        </p>
        <p>
          In walk-forward tests on past tournaments (2015–2025) it scored a Brier of {Math.min(...briers).toFixed(3)}–{Math.max(...briers).toFixed(3)}{" "}
          and picked {percent(Math.min(...accuracy))}–{percent(Math.max(...accuracy))} of winners.
        </p>
        <ul className="list-disc space-y-1 pl-5">
          <li>Early-season ratings rest on a few games each and lean on last season&apos;s Elo, so they shift quickly.</li>
          <li>It was trained on neutral-court games between tournament teams. Regular-season games, especially lopsided ones, are outside that range.</li>
          <li>
            Home court is an adjustment, not something the model learned: the home team gains {model.homeEdgePoints.toFixed(1)} points
            of adjusted margin (the average full-season home edge) and the Elo home bonus.
          </li>
          <li>Injuries, lineups and travel are not included, except as they show up in results.</li>
          {run && run.teamsWithoutPrior.length > 0 && <li>No 2025–26 Elo for {run.teamsWithoutPrior.join(", ")}; they started at the 1500 average.</li>}
        </ul>
      </div>
    </details>
  );
}
