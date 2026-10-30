/* eslint-disable */
/**
 * Generated `api` utility.
 *
 * THIS CODE IS AUTOMATICALLY GENERATED.
 *
 * To regenerate, run `npx convex dev`.
 * @module
 */

import type * as brackets from "../brackets.js";
import type * as data_marchMadness2026 from "../data/marchMadness2026.js";
import type * as gameOdds from "../gameOdds.js";
import type * as gameOddsModel from "../gameOddsModel.js";
import type * as teamLogos from "../teamLogos.js";
import type * as tournamentModel from "../tournamentModel.js";
import type * as tournaments from "../tournaments.js";

import type {
  ApiFromModules,
  FilterApi,
  FunctionReference,
} from "convex/server";

declare const fullApi: ApiFromModules<{
  brackets: typeof brackets;
  "data/marchMadness2026": typeof data_marchMadness2026;
  gameOdds: typeof gameOdds;
  gameOddsModel: typeof gameOddsModel;
  teamLogos: typeof teamLogos;
  tournamentModel: typeof tournamentModel;
  tournaments: typeof tournaments;
}>;

/**
 * A utility for referencing Convex functions in your app's public API.
 *
 * Usage:
 * ```js
 * const myFunctionReference = api.myModule.myFunction;
 * ```
 */
export declare const api: FilterApi<
  typeof fullApi,
  FunctionReference<any, "public">
>;

/**
 * A utility for referencing Convex functions in your app's internal API.
 *
 * Usage:
 * ```js
 * const myFunctionReference = internal.myModule.myFunction;
 * ```
 */
export declare const internal: FilterApi<
  typeof fullApi,
  FunctionReference<any, "internal">
>;

export declare const components: {};
