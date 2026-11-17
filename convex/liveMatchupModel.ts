import { v } from "convex/values";
import liveMatchupModel from "./data/liveMatchupModel.json";

export type LiveModel = {
  scalerMean: readonly number[];
  scalerScale: readonly number[];
  coefficients: readonly number[];
  elo: { readonly home: number };
  homeEdgePoints: number;
};
export type TeamFeatures = { quality: number; elo: number; netEff: number };
/** "neutral", or which side of the (a, b) pair is at home. */
export type Venue = "neutral" | "a" | "b";

export const venueValidator = v.union(v.literal("neutral"), v.literal("a"), v.literal("b"));

export const liveTeamRatingFields = {
  teamId: v.number(),
  teamName: v.string(),
  conference: v.optional(v.string()),
  games: v.number(),
  boxGames: v.number(),
  quality: v.number(),
  elo: v.number(),
  netEff: v.number(),
  priorElo: v.optional(v.number()),
};

export const liveTeamRatingValidator = { season: v.number(), ...liveTeamRatingFields };

export const liveRatingRunValidator = {
  season: v.number(),
  modelId: v.string(),
  updatedAt: v.number(),
  gamesUsed: v.number(),
  lastGameDate: v.optional(v.string()),
  /** This season's ridge home term; informational only (inflated early in the season). */
  fittedHomeEdge: v.number(),
  /** D1 schools without a usable last-season Elo; they start at 1500. */
  teamsWithoutPrior: v.array(v.string()),
  /** D1 schools with no rated game or box score yet. */
  teamsWithoutFeatures: v.array(v.string()),
};

export const liveGameInput = {
  gameId: v.number(),
  startDate: v.string(),
  status: v.string(),
  neutralSite: v.boolean(),
  homeTeamId: v.number(),
  homeTeam: v.string(),
  awayTeamId: v.number(),
  awayTeam: v.string(),
  homePoints: v.optional(v.number()),
  awayPoints: v.optional(v.number()),
};

export const liveGamePredictionValidator = {
  ...liveGameInput,
  season: v.number(),
  weekStart: v.string(),
  modelId: v.string(),
  /** Absent when either team had no live features before tip-off. */
  homeWinProbability: v.optional(v.number()),
  unavailableReason: v.optional(v.string()),
  predictedAt: v.number(),
  ratingsUpdatedAt: v.optional(v.number()),
};

type GameResult = { status: string; homePoints?: number; awayPoints?: number; homeWinProbability?: number };

/** Graded outcome once a game is final; null while it is not or when there was no prediction. */
export function gradePrediction(game: GameResult) {
  if (game.status !== "final" || game.homeWinProbability === undefined
    || game.homePoints === undefined || game.awayPoints === undefined || game.homePoints === game.awayPoints) return null;
  const homeWon = game.homePoints > game.awayPoints;
  const p = game.homeWinProbability;
  return { correct: (p >= 0.5) === homeWon, brier: (p - (homeWon ? 1 : 0)) ** 2 };
}

export function summarizePredictions(games: GameResult[]) {
  const graded = games.map(gradePrediction).filter(grade => grade !== null);
  return {
    games: games.length,
    predicted: games.filter(game => game.homeWinProbability !== undefined).length,
    graded: graded.length,
    correct: graded.filter(grade => grade.correct).length,
    brier: graded.length ? graded.reduce((sum, grade) => sum + grade.brier, 0) / graded.length : null,
  };
}

const sigmoid = (x: number) => 1 / (1 + Math.exp(-x));

function logit(model: LiveModel, diff: number[]) {
  return diff.reduce((sum, d, i) => sum + model.coefficients[i] * (d - model.scalerMean[i]) / model.scalerScale[i], 0);
}

/**
 * P(team a beats team b), averaged over both team orders like kaggle_tuning's backtest.
 * The model was trained on neutral-court tournament games. At a home venue the home team's
 * quality gains the long-run ridge home edge (points) and its Elo gains the Elo model's home
 * bonus, the same terms those ratings use when they are fitted; net efficiency is unchanged.
 */
export function winProbability(model: LiveModel, a: TeamFeatures, b: TeamFeatures, venue: Venue = "neutral") {
  const sign = venue === "a" ? 1 : venue === "b" ? -1 : 0;
  const diff = [a.quality - b.quality + sign * model.homeEdgePoints, a.elo - b.elo + sign * model.elo.home, a.netEff - b.netEff];
  const forward = sigmoid(logit(model, diff));
  const reverse = sigmoid(logit(model, diff.map(d => -d)));
  return (forward + 1 - reverse) / 2;
}

export const liveModel: LiveModel = liveMatchupModel;
export const liveModelId = liveMatchupModel.id;

/** Monday (America/New_York) of the week containing the ISO timestamp, as YYYY-MM-DD. */
export function weekStart(iso: string) {
  const parts = new Intl.DateTimeFormat("en-US", {
    timeZone: "America/New_York", year: "numeric", month: "2-digit", day: "2-digit", weekday: "short",
  }).formatToParts(new Date(iso));
  const get = (type: string) => parts.find(part => part.type === type)!.value;
  const day = new Date(Date.UTC(Number(get("year")), Number(get("month")) - 1, Number(get("day"))));
  const offset = (["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"].indexOf(get("weekday")));
  day.setUTCDate(day.getUTCDate() - offset);
  return day.toISOString().slice(0, 10);
}
