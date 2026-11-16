
"""Run season-wide tournament-success score simulations without retraining.

Baselines come from the best SHAP ensemble written by combo_eval.py by default.
Pass --bundle to use a Ridge inference bundle instead.
"""

import argparse
import contextlib
import io
import json
from pathlib import Path
import runpy

import joblib
import numpy as np
import pandas as pd

from monte_carlo import simulate_season_success, simulate_season_from_scores
from backend_data.export import export_forecast
from backend_data.kalshi import source_artifact


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--bundle", type=Path, help="Ridge bundle. Omit to use combo_eval SHAP-ensemble scores.")
    parser.add_argument("--n-simulations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--expected-teams", type=int)
    parser.add_argument("--features-csv", type=Path, help="Already engineered team-season features (Ridge mode).")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--training-output-dir", type=Path, default=Path("model_outputs"))
    parser.add_argument("--output-dir", type=Path, default=Path("model_outputs/monte_carlo"))
    parser.add_argument("--ensemble-dir", type=Path, default=Path("ensemble_bracket_evaluations"),
                        help="combo_eval.py output directory (SHAP mode).")
    parser.add_argument("--shap-weights", type=Path,
                        help="Specific top_20_shap_weights CSV. Defaults to the Rank_1 ensemble.")
    parser.add_argument("--backtest", action="store_true",
                        help="SHAP mode: also simulate every historical season for diagnostic plots.")
    parser.add_argument("--residual-mode", choices=["stratified", "global"], default="stratified",
                        help="SHAP mode: local stratified/heteroskedastic residuals vs flat global pool.")
    parser.add_argument("--pool-size", type=int, default=100,
                        help="SHAP mode: number of entries in bracket pool for EV/leverage optimizer.")
    parser.add_argument("--no-plots", action="store_true", help="SHAP mode: skip visualizations.")
    parser.add_argument("--kalshi-csv", type=Path, help="Optional historical Kalshi CSV included in validated JSON.")
    parser.add_argument("--kalshi-mapping", type=Path, help="Reviewed team aliases and event-to-game mappings JSON.")
    args = parser.parse_args()
    if args.kalshi_mapping and not args.kalshi_csv:
        parser.error("--kalshi-mapping requires --kalshi-csv")
    if args.bundle is None:
        run_shap(args, parser)
        return
    bundle = joblib.load(args.bundle)
    if args.features_csv:
        features = pd.read_csv(args.features_csv)
    else:
        cache = args.training_output_dir / "tournament_wins" / f"tournament_wins_{args.season}.csv"
        if not cache.exists():
            parser.error(f"Target cache missing: {cache}. Supply --features-csv or generate the cache first.")
        # Reuse exactly the training feature builder; no training main or API
        # call is needed because the target cache was verified above.
        with contextlib.redirect_stdout(io.StringIO()):
            training = runpy.run_path(str(Path(__file__).parent / "models" / "ridge_shap.py"))
            features = training["build_team_features"](
                str(args.data_dir), args.season, str(args.training_output_dir)
            )
    table = simulate_season_success(
        features, bundle, args.season, n_simulations=args.n_simulations,
        seed=args.seed, expected_teams=args.expected_teams,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / f"team_success_scores_{args.season}.csv"
    table.to_csv(path, index=False)
    # Keep the CSV's requested schema; describe interpretation and provenance
    # alongside it so they accompany any shared results.
    metadata = {
        "interpretation": "Continuous tournament-success scores and threshold frequencies; not calibrated matchup or bracket probabilities.",
        "season": args.season,
        "teams_processed": len(table),
        "n_simulations_per_team": args.n_simulations,
        "seed": args.seed,
        "random_streams": "Independent SeedSequence child streams in alphabetical team order.",
        "bundle_path": str(args.bundle),
        "training_seasons": bundle["training_seasons"],
        "validation_season": bundle["validation_season"],
        "residual_source": bundle["residual_source"],
        "residual_count": len(bundle["validation_residuals"]),
        "feature_count": len(bundle["feature_names"]),
        "missing_features_zero_filled": [name for name in bundle["feature_names"] if name not in features.columns],
        "validation_checks": [
            "Expected team count and unique team names",
            "All rows belong to the selected season",
            "Feature order and fitted model/scaler dimensions",
            "No NaN or infinite summary values",
            "Threshold frequencies are bounded and nonincreasing",
        ],
        "assumptions": [
            "The same empirical residual pool applies to each team.",
            "Scores are continuous, without rounding or clipping.",
            "The bundle's held-out residual pool calibrates these outputs; it is not an independent probability-validation set.",
            "Player-season totals require verification of pre-tournament provenance.",
        ],
    }
    if args.season in bundle["validation_seasons"]:
        metadata["validation_checks"].append("Complete saved validation roster")
    if bundle["residual_source"] == "held_out_season" and bundle["validation_season"] == args.season:
        metadata["validation_checks"].append("Matching saved baseline prediction for every team")
    path.with_suffix(".metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    export_backend_results(args, table, metadata, "ridge")
    print(f"Saved {len(table)} teams to {path}")
    print("Validation passed. Results are tournament-success scores / threshold frequencies.")
    if metadata["missing_features_zero_filled"]:
        print(f"Zero-filled absent features: {metadata['missing_features_zero_filled']}")
    print(table[["team", "ridge_baseline_prediction", "simulated_mean"]].head(10).to_string(index=False))


def export_backend_results(args, table, metadata, mode, bracket_df=None, parsed_tree=None):
    mapping_path = getattr(args, "kalshi_mapping", None)
    mapping = json.loads(mapping_path.read_text(encoding="utf-8")) if mapping_path else None
    stem = f"team_success_scores{'_shap' if mode == 'shap' else ''}_{metadata['season']}"
    artifacts = [source_artifact(args.output_dir / f"{stem}.csv", "text/csv"),
                 source_artifact(args.output_dir / f"{stem}.metadata.json", "application/json")]
    if bracket_df is not None:
        artifacts.append(source_artifact(args.output_dir / f"optimal_bracket_{metadata['season']}.csv", "text/csv"))
    if mapping_path:
        artifacts.append(source_artifact(mapping_path, "application/json"))
    path = export_forecast(
        table, metadata, mode, args.output_dir, bracket_df=bracket_df, parsed_tree=parsed_tree,
        kalshi_csv=getattr(args, "kalshi_csv", None), kalshi_mapping=mapping,
        source_artifacts=artifacts,
    )
    print(f"Saved validated backend JSON to {path}")


def _simulate_shap_season(history, scores, season, field, actual_wins, args,
                          all_seasons=None, expected_teams=None):
    """Calibrate on other seasons, run stratified bootstrap + 63-game bracket tree."""
    import shap_monte_carlo as smc
    slope, intercept, flat_residuals, in_sample = smc.calibrate(history, season)
    baselines = scores[scores["team"].isin(field)].copy()
    baselines["baseline_prediction"] = slope * baselines["ensemble_score"] + intercept
    baselines["actual_wins"] = baselines["team"].map(actual_wins) if actual_wins else np.nan
    baselines_input = baselines[["team", "season", "ensemble_score", "baseline_prediction", "actual_wins"]].drop(columns="season")

    if args.residual_mode == "stratified":
        residuals_to_use = smc.build_stratified_residuals(history, season, baselines)
    else:
        residuals_to_use = flat_residuals

    table, samples = simulate_season_from_scores(
        baselines_input, residuals_to_use, season, n_simulations=args.n_simulations, seed=args.seed,
        expected_teams=expected_teams, return_samples=True,
    )

    # Fit out-of-fold pairwise matchup model & simulate 63-game bracket tree
    matchup_model, matchup_diag = (None, {})
    if all_seasons is not None:
        matchup_model, matchup_diag = smc.fit_matchup_model(all_seasons, args.data_dir, history, season)
    tree_res = smc.run_bracket_tree_simulation(
        season, args.data_dir, baselines, matchup_model,
        n_simulations=args.n_simulations, seed=args.seed,
    )
    parsed_tree = None
    if tree_res is not None:
        tree_df, parsed_tree = tree_res
        table = table.merge(tree_df, on="team", how="left")

    calib = {
        "slope": slope,
        "intercept": intercept,
        "residual_mode": args.residual_mode,
        "residual_count": int(len(flat_residuals)),
        "in_sample_calibration": in_sample,
        "residuals": flat_residuals,
        "matchup_model": matchup_diag,
    }
    return smc.add_ranks(table), samples, calib, parsed_tree


def run_shap(args, parser):
    import shap_monte_carlo as smc

    try:
        df_shap, info = smc.load_shap_recipe(args.ensemble_dir, args.shap_weights)
    except (FileNotFoundError, ValueError) as e:
        parser.error(str(e))
    weights = dict(zip(df_shap["feature"], df_shap["shap_importance"]))
    label = info.get("ensemble_name", Path(info["shap_weights_path"]).stem)
    member_shap = smc.load_member_shap_by_season(args.training_output_dir, info.get("ensemble_members", []))
    print(f"[SHAP] Baseline source: {info['shap_weights_path']}")
    print(f"[SHAP] Ensemble: {label} ({'per-model z-score blend' if member_shap else 'blended SHAP weights'})")

    all_seasons = smc.discover_seasons(args.data_dir)
    wins_dirs = [args.data_dir, args.training_output_dir]
    print("[SHAP] Scoring historical seasons for calibration...")
    history = smc.build_history(all_seasons, args.data_dir, wins_dirs, weights, member_shap)
    if history.empty or history[history["season"] != args.season].empty:
        parser.error("No other seasons with cached tournament_wins found for calibration.")

    actual_wins = smc.load_actual_wins(args.season, wins_dirs)
    field = smc.tournament_field(args.season, args.data_dir, actual_wins)
    if not field:
        parser.error(f"No tournament field for {args.season}: need tournament_wins cache or "
                     f"{args.data_dir}/{args.season}/tournament_games.csv.")
    scores, missing = smc.score_season(args.data_dir, args.season, weights, member_shap)
    unscored = sorted(field - set(scores["team"]))
    if unscored:
        print(f"[SHAP][WARNING] {len(unscored)} field teams have no feature data: {unscored}")
    expected = args.expected_teams

    table, samples, calib, parsed_tree = _simulate_shap_season(
        history, scores, args.season, field, actual_wins, args,
        all_seasons=all_seasons, expected_teams=expected,
    )

    # Simulate historical seasons for out-of-fold Isotonic probability calibration & backtest
    print("[SHAP] Running out-of-fold historical simulations for Isotonic calibration & backtest...")
    hist_tables_by_season = {}
    for s in sorted(history["season"].unique()):
        s_hist = history[history["season"] == s]
        wins = dict(zip(s_hist["team"], s_hist["actual_wins"]))
        t_s, _, _, _ = _simulate_shap_season(
            history, s_hist[["team", "season", "ensemble_score"]],
            int(s), set(wins), wins, args, all_seasons=all_seasons,
        )
        hist_tables_by_season[int(s)] = t_s

    oof_for_target = pd.concat(
        [t for s, t in hist_tables_by_season.items() if s != args.season], ignore_index=True
    )
    iso_calibrators = smc.fit_isotonic_calibrators(oof_for_target)
    table = smc.apply_isotonic_calibration(table, iso_calibrators)
    table = table.rename(columns={"baseline_prediction": "shap_baseline_prediction"})

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"team_success_scores_shap_{args.season}.csv"
    table.to_csv(path, index=False)

    # Run Expected Value (EV) & Game-Theory Leverage Bracket Pool Optimizer
    bracket_df, pool_summary = None, None
    if parsed_tree is not None:
        print(f"[SHAP] Optimizing 63-game bracket for {args.pool_size}-entry pool...")
        bracket_df, pool_summary, _, _ = smc.optimize_bracket_pool(
            parsed_tree, scores, pool_size=args.pool_size,
            n_simulations=args.n_simulations, seed=args.seed,
        )
        bracket_path = out_dir / f"optimal_bracket_{args.season}.csv"
        bracket_df.to_csv(bracket_path, index=False)
        print(f"[SHAP] Saved 63-game optimized bracket to {bracket_path}")
        print(f"       Max-EV Chalk Champion:     {pool_summary['max_ev_champion']} "
              f"(Exp Pts: {pool_summary['max_ev_expected_points']:.1f}, Pool Win%: {pool_summary['max_ev_pool_win_prob']:.1%}, "
              f"Actual Pts: {pool_summary['actual_max_ev_points']:.0f})")
        print(f"       Leverage Optimal Champion: {pool_summary['leverage_champion']} "
              f"(Exp Pts: {pool_summary['leverage_expected_points']:.1f}, Pool Win%: {pool_summary['leverage_pool_win_prob']:.1%}, "
              f"Actual Pts: {pool_summary['actual_leverage_points']:.0f})")

    plot_root = out_dir / "shap_plots"
    if not args.no_plots:
        print("[SHAP] Rendering visualizations...")
        plot_table = table.rename(columns={"shap_baseline_prediction": "baseline_prediction"})
        smc.render_season_plots(
            plot_table, samples, args.season, calib["slope"], calib["intercept"],
            str(plot_root / str(args.season)), bracket_df=bracket_df, pool_summary=pool_summary,
        )
        model_dir = plot_root / "model"
        model_dir.mkdir(parents=True, exist_ok=True)
        smc.plot_calibration_scatter(history, args.season, calib["slope"], calib["intercept"],
                                     str(model_dir / f"calibration_excl_{args.season}.jpg"))
        smc.plot_residual_pool(calib["residuals"], str(model_dir / f"residual_pool_excl_{args.season}.jpg"))
        smc.plot_shap_recipe(df_shap, label, str(model_dir / "shap_recipe.jpg"))

    backtest_summary = None
    if args.backtest:
        print("[SHAP] Aggregating leave-one-season-out backtest metrics...")
        cal_tables, metrics = [], []
        for s in sorted(hist_tables_by_season):
            oof_other = pd.concat([t for k, t in hist_tables_by_season.items() if k != s], ignore_index=True)
            cal_s = smc.fit_isotonic_calibrators(oof_other)
            t_cal = smc.apply_isotonic_calibration(hist_tables_by_season[s], cal_s)
            cal_tables.append(t_cal)
            metrics.append(smc.season_metrics(t_cal))
        all_tables = pd.concat(cal_tables, ignore_index=True)
        df_metrics = pd.DataFrame(metrics)
        all_tables.to_csv(out_dir / "shap_backtest_team_scores.csv", index=False)
        df_metrics.to_csv(out_dir / "shap_backtest_season_metrics.csv", index=False)
        if not args.no_plots:
            smc.render_backtest_plots(all_tables, df_metrics, str(plot_root / "backtest"))
        backtest_summary = {
            k: float(df_metrics[k].mean())
            for k in ["MAE", "RMSE", "Spearman", "Coverage_90", "Champion_P_ge_6", "Brier_ge_1", "Brier_ge_4", "Brier_ge_6"]
            if k in df_metrics
        }
        print(f"[SHAP] Backtest means: {backtest_summary}")

    metadata = {
        "interpretation": "Tournament-success scores, isotonic-calibrated frequencies, 63-game bracket-tree survival probabilities, and EV-optimized bracket picks.",
        "baseline_source": "combo_eval.py best SHAP ensemble (replaces Ridge prediction)",
        "scoring_mode": "per_model_zscore_blend" if member_shap else "blended_shap_weights",
        "season": args.season,
        "teams_processed": len(table),
        "n_simulations_per_team": args.n_simulations,
        "seed": args.seed,
        "random_streams": "Independent SeedSequence child streams in alphabetical team order.",
        **info,
        "calibration": {k: v for k, v in calib.items() if k != "residuals"},
        "calibration_seasons": sorted(int(s) for s in history["season"].unique() if s != args.season),
        "shap_feature_count": len(weights),
        "shap_features_missing_from_data": missing,
        "field_teams_without_features": unscored,
        "actual_results_available": bool(actual_wins),
        "bracket_pool_optimization": pool_summary,
        "backtest_means": backtest_summary,
        "assumptions": [
            "Per-model z-score harmonization matches combo_eval Phase 2/3 when member SHAP files exist.",
            "Stratified local residual pools condition error variance on team strength (ensemble_score neighborhood).",
            "Isotonic regression calibrates threshold probabilities out-of-fold with monotone round enforcement.",
            "Pairwise matchup probabilities are fitted via out-of-fold Logistic Regression on score and seed differentials.",
            "63-game bracket-tree Monte Carlo enforces single-elimination bracket topology (1 champion per simulation).",
            "Exact dynamic programming finds the globally optimal 63-game bracket for both Max-EV and Pool-Leverage objectives.",
        ],
    }
    path.with_suffix(".metadata.json").write_text(json.dumps(metadata, indent=2, default=str) + "\n")
    export_backend_results(args, table, metadata, "shap", bracket_df, parsed_tree)
    print(f"Saved {len(table)} teams to {path}")
    display_cols = [c for c in ["team", "seed", "ensemble_score", "simulated_mean", "tree_expected_wins",
                                "probability_ge_6_wins", "calibrated_prob_ge_6_wins", "tree_prob_ge_6_wins"]
                    if c in table.columns]
    print(table[display_cols].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
