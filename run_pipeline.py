#!/usr/bin/env python3
"""Master pipeline runner for March Madness Predictor.

Orchestrates and customizes end-to-end tournament forecasting across:
  1. models.py        - Model training, validation log aggregation, & global comparisons.
  2. combo_eval.py    - Exhaustive combinatorial ensemble evaluation & SHAP recipe extraction.
  3. run_monte_carlo.py / shap_monte_carlo.py - Calibrated simulations, 63-game bracket trees,
                        and Game-Theory Leverage / Max-EV bracket pool optimization.

Examples:
  # Run entire pipeline from end to end for 2024 season
  python run_pipeline.py --all --season 2024

  # Run only combinatorial evaluation and Monte Carlo (skipping model training)
  python run_pipeline.py --stages combo_eval monte_carlo --season 2024

  # Run only Monte Carlo simulation using the Rank 1 ensemble with 10k simulations & backtest
  python run_pipeline.py --stages monte_carlo --season 2024 --shap-rank 1 --backtest

  # Quick dry run to inspect the planned commands without executing
  python run_pipeline.py --all --season 2024 --dry-run

  # Run specific models only, then parse logs and plot leaderboard
  python run_pipeline.py --stages models --models lasso_shap ridge_shap
"""

import argparse
import glob
import os
import re
import subprocess
import sys
import time
from pathlib import Path

# Safe stdout encoding for Windows terminals
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


class Colors:
    """ANSI terminal styling helpers (graceful fallback)."""
    HEADER = "\033[95m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    END = "\033[0m"


def print_banner(title: str, subtitle: str = ""):
    width = 80
    print("\n" + "=" * width)
    print(f"  {title.upper()}")
    if subtitle:
        print(f"  {subtitle}")
    print("=" * width)


def resolve_python_executable(root_dir: Path, custom_python: str = None) -> str:
    """Detects .venv python or active virtualenv before falling back to sys.executable."""
    if custom_python:
        return custom_python
    if sys.prefix != sys.base_prefix:
        return sys.executable
    venv_win = root_dir / ".venv" / "Scripts" / "python.exe"
    if venv_win.exists():
        return str(venv_win)
    venv_unix = root_dir / ".venv" / "bin" / "python"
    if venv_unix.exists():
        return str(venv_unix)
    return sys.executable


def discover_available_models(models_dir: Path):
    if not models_dir.exists():
        return []
    scripts = sorted([
        f.stem for f in models_dir.glob("*.py")
        if not f.name.lower().startswith("ft")
    ])
    return scripts


def discover_top_ensembles(ensemble_dir: Path):
    weights_dir = ensemble_dir / "top_20_shap_weights"
    if not weights_dir.exists():
        return []
    files = sorted(weights_dir.glob("Rank_*_SHAP.csv"))
    ranks = []
    for f in files:
        m = re.search(r"Rank_(\d+)_", f.name)
        if m:
            ranks.append((int(m.group(1)), f))
    ranks.sort(key=lambda x: x[0])
    return ranks


def discover_seasons_in_data(data_dir: Path):
    if not data_dir.exists():
        return []
    seasons = sorted([
        int(p.name) for p in data_dir.iterdir()
        if p.is_dir() and re.fullmatch(r"\d{4}", p.name)
    ])
    return seasons


# ---------------------------------------------------------------------------
# Stage 1: Models Execution & Log Aggregation
# ---------------------------------------------------------------------------
def run_stage_models(args, root_dir: Path, py_exe: str) -> dict:
    t0 = time.time()
    models_dir = root_dir / args.models_dir
    outputs_dir = root_dir / args.output_dir
    available = discover_available_models(models_dir)

    print_banner("Stage 1: Model Training & Evaluation", f"Directory: {models_dir}")
    print(f"Discovered {len(available)} model architectures: {', '.join(available)}")

    selected_models = []
    if "all" in [m.lower() for m in args.models]:
        selected_models = available
    else:
        for user_m in args.models:
            clean = user_m.strip().lower().replace(".py", "")
            matches = [m for m in available if clean == m.lower() or clean == m.lower().replace("_shap", "")]
            if matches:
                selected_models.extend(matches)
            else:
                print(f"{Colors.YELLOW}[Warning] Model '{user_m}' not recognized. Skipping.{Colors.END}")
        selected_models = sorted(list(set(selected_models)))

    if not selected_models and not args.skip_model_training:
        print(f"{Colors.RED}[Error] No valid models selected to run.{Colors.END}")
        return {"status": "FAILED", "elapsed": time.time() - t0, "artifacts": []}

    # 1. Execute individual model training scripts if requested
    if args.skip_model_training:
        print(f"\n{Colors.YELLOW}[Option] --skip-model-training requested. Bypassing training runs.{Colors.END}")
    else:
        print(f"\nRunning training for {len(selected_models)} models: {', '.join(selected_models)}")
        for idx, model_name in enumerate(selected_models, 1):
            script_path = models_dir / f"{model_name}.py"
            cmd = [py_exe, str(script_path)]
            print(f"\n  ({idx}/{len(selected_models)}) Running {model_name}...")
            print(f"  Command: {' '.join(cmd)}")
            if args.dry_run:
                continue

            try:
                subprocess.run(cmd, cwd=str(root_dir), check=True)
                print(f"  {Colors.GREEN}[OK] Completed {model_name}{Colors.END}")
            except subprocess.CalledProcessError as e:
                print(f"  {Colors.RED}[X] Failed {model_name} with exit code {e.returncode}{Colors.END}")
                if args.fail_fast:
                    return {"status": "FAILED", "elapsed": time.time() - t0, "artifacts": []}

    # 2. Parse training logs and generate global comparison charts via models.py
    print(f"\n[Aggregation] Parsing validation logs and generating global comparison charts...")
    artifacts = []
    if not args.dry_run:
        try:
            import models as models_module
            df_metrics = models_module.parse_training_logs(str(outputs_dir))
            if not df_metrics.empty:
                models_module.generate_global_comparison_plots(df_metrics, str(outputs_dir))
                artifacts = [
                    str(outputs_dir / "GLOBAL_MODEL_METRICS.csv"),
                    str(outputs_dir / "GLOBAL_MODEL_COMPARISON_NDCG.jpg"),
                    str(outputs_dir / "GLOBAL_MODEL_COMPARISON_SPEARMAN.jpg"),
                    str(outputs_dir / "GLOBAL_MODEL_COMPARISON_MARE.jpg"),
                    str(outputs_dir / "GLOBAL_MODEL_COMPARISON_OVERALL_RANK.jpg"),
                ]
            else:
                print(f"{Colors.YELLOW}[Warning] No training logs found to parse in {outputs_dir}.{Colors.END}")
        except Exception as e:
            print(f"{Colors.RED}[Error] Global log parsing / plotting failed: {e}{Colors.END}")
            if args.fail_fast:
                return {"status": "FAILED", "elapsed": time.time() - t0, "artifacts": []}

    return {"status": "SUCCESS", "elapsed": time.time() - t0, "artifacts": artifacts}


# ---------------------------------------------------------------------------
# Stage 2: Combinatorial Ensemble Evaluation
# ---------------------------------------------------------------------------
def run_stage_combo_eval(args, root_dir: Path, py_exe: str) -> dict:
    t0 = time.time()
    combo_script = root_dir / "combo_eval.py"
    ensemble_dir = root_dir / args.ensemble_dir

    print_banner("Stage 2: Combinatorial Ensemble Evaluation", f"Script: {combo_script}")
    if not combo_script.exists():
        print(f"{Colors.RED}[Error] combo_eval.py not found at {combo_script}{Colors.END}")
        return {"status": "FAILED", "elapsed": time.time() - t0, "artifacts": []}

    cmd = [py_exe, str(combo_script)]
    print(f"Executing combinatorial search across all model combinations...")
    print(f"Command: {' '.join(cmd)}")

    if args.dry_run:
        return {"status": "DRY_RUN", "elapsed": 0.0, "artifacts": []}

    try:
        subprocess.run(cmd, cwd=str(root_dir), check=True)
    except subprocess.CalledProcessError as e:
        print(f"{Colors.RED}[X] combo_eval.py failed with return code {e.returncode}{Colors.END}")
        return {"status": "FAILED", "elapsed": time.time() - t0, "artifacts": []}

    top_ranks = discover_top_ensembles(ensemble_dir)
    artifacts = [
        str(ensemble_dir / "ALL_ENSEMBLE_COMBINATIONS_METRICS.csv"),
        str(ensemble_dir / "TOP_10_ENSEMBLES_ACCURACY.jpg"),
        str(ensemble_dir / "TOP_10_ENSEMBLES_BLIND_ERROR.jpg"),
        str(ensemble_dir / "TOP_10_ENSEMBLES_NDCG.jpg"),
        str(ensemble_dir / "TOP_10_ENSEMBLES_SPEARMAN.jpg"),
        str(ensemble_dir / "TOP_10_ENSEMBLES_MARE.jpg"),
    ]
    if top_ranks:
        print(f"\n{Colors.GREEN}[OK] Discovered {len(top_ranks)} exported SHAP recipes under {ensemble_dir / 'top_20_shap_weights'}{Colors.END}")
        for r, p in top_ranks[:5]:
            print(f"    - Rank {r}: {p.name}")

    return {"status": "SUCCESS", "elapsed": time.time() - t0, "artifacts": artifacts}


# ---------------------------------------------------------------------------
# Stage 3: Monte Carlo Simulation & Optimization
# ---------------------------------------------------------------------------
def run_stage_monte_carlo(args, root_dir: Path, py_exe: str) -> dict:
    t0 = time.time()
    mc_script = root_dir / "run_monte_carlo.py"
    data_dir = root_dir / args.data_dir
    mc_out_dir = root_dir / args.mc_output_dir
    ensemble_dir = root_dir / args.ensemble_dir

    print_banner("Stage 3: Monte Carlo Simulation & Bracket Optimization", f"Script: {mc_script}")

    all_data_seasons = discover_seasons_in_data(data_dir)
    target_seasons = []
    for s in args.seasons:
        if str(s).lower() == "all":
            target_seasons = all_data_seasons
            break
        try:
            target_seasons.append(int(s))
        except ValueError:
            print(f"{Colors.YELLOW}[Warning] Invalid season '{s}' ignored.{Colors.END}")
    if not target_seasons:
        target_seasons = [2024]
    target_seasons = sorted(list(set(target_seasons)))

    resolved_shap_weights = None
    if not args.bundle:
        if args.shap_weights:
            resolved_shap_weights = Path(args.shap_weights)
            if not resolved_shap_weights.is_absolute():
                resolved_shap_weights = root_dir / resolved_shap_weights
        else:
            top_ranks = discover_top_ensembles(ensemble_dir)
            rank_map = dict(top_ranks)
            if args.shap_rank in rank_map:
                resolved_shap_weights = rank_map[args.shap_rank]
                print(f"Using Rank {args.shap_rank} SHAP recipe: {resolved_shap_weights.name}")
            elif top_ranks:
                resolved_shap_weights = top_ranks[0][1]
                print(f"{Colors.YELLOW}[Warning] Rank {args.shap_rank} not found. Defaulting to Rank 1: {resolved_shap_weights.name}{Colors.END}")

    total_artifacts = []
    for season in target_seasons:
        print(f"\n--- Simulating Season {season} ---")
        cmd = [
            py_exe, str(mc_script),
            "--season", str(season),
            "--n-simulations", str(args.n_simulations),
            "--seed", str(args.seed),
            "--data-dir", str(data_dir),
            "--output-dir", str(mc_out_dir),
        ]
        for flag, value in (("--kalshi-csv", args.kalshi_csv), ("--kalshi-mapping", args.kalshi_mapping)):
            if value:
                resolved = value if value.is_absolute() else root_dir / value
                cmd.extend([flag, str(resolved)])

        if args.bundle:
            cmd.extend(["--bundle", str(args.bundle)])
        else:
            cmd.extend([
                "--ensemble-dir", str(ensemble_dir),
                "--residual-mode", str(args.residual_mode),
                "--pool-size", str(args.pool_size),
            ])
            if resolved_shap_weights and resolved_shap_weights.exists():
                cmd.extend(["--shap-weights", str(resolved_shap_weights)])
            if args.backtest:
                cmd.append("--backtest")
            if args.no_plots:
                cmd.append("--no-plots")

        print(f"Command: {' '.join(cmd)}")
        if args.dry_run:
            continue

        previous_json = set(mc_out_dir.glob(f"forecast_{season}_*.json"))
        try:
            subprocess.run(cmd, cwd=str(root_dir), check=True)
            print(f"{Colors.GREEN}[OK] Simulation completed for Season {season}{Colors.END}")
            total_artifacts.extend(str(p) for p in sorted(
                set(mc_out_dir.glob(f"forecast_{season}_*.json")) - previous_json
            ))

            if not args.bundle:
                total_artifacts.append(str(mc_out_dir / f"team_success_scores_shap_{season}.csv"))
                total_artifacts.append(str(mc_out_dir / f"team_success_scores_shap_{season}.metadata.json"))
                opt_bracket = mc_out_dir / f"optimal_bracket_{season}.csv"
                if opt_bracket.exists():
                    total_artifacts.append(str(opt_bracket))
            else:
                total_artifacts.append(str(mc_out_dir / f"team_success_scores_{season}.csv"))
                total_artifacts.append(str(mc_out_dir / f"team_success_scores_{season}.metadata.json"))

        except subprocess.CalledProcessError as e:
            print(f"{Colors.RED}[X] Simulation failed for Season {season} with code {e.returncode}{Colors.END}")
            if args.fail_fast:
                return {"status": "FAILED", "elapsed": time.time() - t0, "artifacts": total_artifacts}

    return {"status": "SUCCESS", "elapsed": time.time() - t0, "artifacts": total_artifacts}


# ---------------------------------------------------------------------------
# CLI Argument Parsing & Runner Setup
# ---------------------------------------------------------------------------
def parse_arguments():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    stage_group = parser.add_argument_group("Stage Selection")
    stage_group.add_argument(
        "--all", action="store_true",
        help="Execute all pipeline stages in sequence (models -> combo_eval -> monte_carlo).",
    )
    stage_group.add_argument(
        "--stages", nargs="+",
        choices=["models", "combo_eval", "combo-eval", "monte_carlo", "monte-carlo"],
        help="Explicitly specify which stage(s) to run (e.g. --stages combo_eval monte_carlo).",
    )
    stage_group.add_argument(
        "--run-models", action="store_true",
        help="Run Stage 1: Model training, validation, and global comparison plots.",
    )
    stage_group.add_argument(
        "--run-combo-eval", action="store_true",
        help="Run Stage 2: Combinatorial ensemble evaluation and top SHAP extraction.",
    )
    stage_group.add_argument(
        "--run-monte-carlo", action="store_true",
        help="Run Stage 3: Monte Carlo simulation, bracket tree simulation, and pool optimization.",
    )

    models_group = parser.add_argument_group("Stage 1: Models Options")
    models_group.add_argument(
        "--models", nargs="+", default=["all"],
        help="Model architecture(s) to train (e.g. 'all', or 'lasso_shap' 'ridge_shap' 'rf_shap'). Default: all.",
    )
    models_group.add_argument(
        "--skip-model-training", action="store_true",
        help="Skip executing model training scripts; only parse existing logs and generate comparison plots.",
    )
    models_group.add_argument(
        "--models-dir", type=Path, default=Path("models"),
        help="Directory containing individual model scripts. Default: models.",
    )

    combo_group = parser.add_argument_group("Stage 2: Combo Eval Options")
    combo_group.add_argument(
        "--ensemble-dir", type=Path, default=Path("ensemble_bracket_evaluations"),
        help="Output directory for combinatorial evaluation. Default: ensemble_bracket_evaluations.",
    )

    mc_group = parser.add_argument_group("Stage 3: Monte Carlo Options")
    mc_group.add_argument(
        "--season", "--seasons", dest="seasons", nargs="+", default=["2024"],
        help="Tournament season(s) to simulate (e.g. 2024, or 2023 2024, or 'all'). Default: 2024.",
    )
    mc_group.add_argument(
        "--shap-rank", type=int, default=1,
        help="Ensemble rank (1-20) from top_20_shap_weights to simulate. Default: 1.",
    )
    mc_group.add_argument(
        "--shap-weights", type=Path,
        help="Explicit path to a SHAP weights CSV (overrides --shap-rank).",
    )
    mc_group.add_argument(
        "--bundle", type=Path,
        help="Path to Ridge inference bundle (.joblib). Switches Monte Carlo from SHAP mode to Ridge mode.",
    )
    mc_group.add_argument(
        "--kalshi-csv", type=Path,
        help="Optional historical Kalshi CSV for the backend JSON export.",
    )
    mc_group.add_argument(
        "--kalshi-mapping", type=Path,
        help="Reviewed aliases and game mappings JSON (requires --kalshi-csv).",
    )
    mc_group.add_argument(
        "--n-simulations", type=int, default=10000,
        help="Number of Monte Carlo simulation trials per team / tournament. Default: 10000.",
    )
    mc_group.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for repeatable child RNG streams. Default: 42.",
    )
    mc_group.add_argument(
        "--residual-mode", choices=["stratified", "global"], default="stratified",
        help="Error pool mode: local stratified/heteroskedastic vs flat global pool. Default: stratified.",
    )
    mc_group.add_argument(
        "--pool-size", type=int, default=100,
        help="Bracket pool size for Expected Value & Game-Theory Leverage optimizer. Default: 100.",
    )
    mc_group.add_argument(
        "--backtest", action="store_true",
        help="Run leave-one-season-out backtesting across all historical seasons.",
    )
    mc_group.add_argument(
        "--no-plots", action="store_true",
        help="Disable rendering of matplotlib diagnostic plots.",
    )
    mc_group.add_argument(
        "--mc-output-dir", type=Path, default=Path("model_outputs/monte_carlo"),
        help="Output directory for Monte Carlo tables and optimal brackets. Default: model_outputs/monte_carlo.",
    )

    gen_group = parser.add_argument_group("General & Execution Controls")
    gen_group.add_argument(
        "--python", dest="python_exe", type=str, default=None,
        help="Custom Python interpreter executable path. Default: auto-detects .venv/Scripts/python.exe.",
    )
    gen_group.add_argument(
        "--data-dir", type=Path, default=Path("data"),
        help="Root directory for historical season and bracket data. Default: data.",
    )
    gen_group.add_argument(
        "--output-dir", type=Path, default=Path("model_outputs"),
        help="Root directory for model outputs and logs. Default: model_outputs.",
    )
    gen_group.add_argument(
        "--dry-run", action="store_true",
        help="Print execution steps and command lines without running them.",
    )
    gen_group.add_argument(
        "--fail-fast", action="store_true",
        help="Stop immediately if any stage fails instead of proceeding.",
    )
    gen_group.add_argument(
        "--list-models", action="store_true",
        help="List available model scripts in the models directory and exit.",
    )
    gen_group.add_argument(
        "--list-ensembles", action="store_true",
        help="List discovered top-20 ensemble recipes and exit.",
    )

    args = parser.parse_args()
    if args.kalshi_mapping and not args.kalshi_csv:
        parser.error("--kalshi-mapping requires --kalshi-csv")
    return args


# ---------------------------------------------------------------------------
# Main Orchestration Loop
# ---------------------------------------------------------------------------
def main():
    args = parse_arguments()
    root_dir = Path(__file__).resolve().parent
    py_exe = resolve_python_executable(root_dir, args.python_exe)

    if args.list_models:
        available = discover_available_models(root_dir / args.models_dir)
        print(f"\nAvailable Model Scripts in '{args.models_dir}':")
        for m in available:
            print(f"  - {m}")
        return

    if args.list_ensembles:
        ensembles = discover_top_ensembles(root_dir / args.ensemble_dir)
        print(f"\nAvailable Top Ensemble Recipes in '{args.ensemble_dir}/top_20_shap_weights':")
        for rank, path in ensembles:
            print(f"  - Rank {rank:2d}: {path.name}")
        return

    active_stages = set()
    if args.all:
        active_stages = {"models", "combo_eval", "monte_carlo"}
    else:
        if args.stages:
            for s in args.stages:
                norm = s.lower().replace("-", "_")
                active_stages.add(norm)
        if args.run_models:
            active_stages.add("models")
        if args.run_combo_eval:
            active_stages.add("combo_eval")
        if args.run_monte_carlo:
            active_stages.add("monte_carlo")

    if not active_stages:
        print(f"{Colors.YELLOW}[Info] No execution stages selected.{Colors.END}")
        print("Usage quickstart:")
        print("  python run_pipeline.py --all --season 2024")
        print("  python run_pipeline.py --stages models --skip-model-training")
        print("  python run_pipeline.py --stages combo_eval")
        print("  python run_pipeline.py --stages monte_carlo --season 2024 --shap-rank 1")
        print("  python run_pipeline.py --help")
        return

    pipeline_start = time.time()
    stage_results = {}

    print_banner(
        "March Madness Predictor: Master Execution Pipeline",
        f"Active Stages: {', '.join(sorted(active_stages))} | Interpreter: {py_exe} | Dry Run: {args.dry_run}",
    )

    # 1. Models Stage
    if "models" in active_stages:
        stage_results["models"] = run_stage_models(args, root_dir, py_exe)
        if stage_results["models"]["status"] == "FAILED" and args.fail_fast:
            print(f"\n{Colors.RED}[Pipeline Terminated] Stage 'models' failed and --fail-fast was specified.{Colors.END}")
            sys.exit(1)
    else:
        stage_results["models"] = {"status": "SKIPPED", "elapsed": 0.0, "artifacts": []}

    # 2. Combo Eval Stage
    if "combo_eval" in active_stages:
        stage_results["combo_eval"] = run_stage_combo_eval(args, root_dir, py_exe)
        if stage_results["combo_eval"]["status"] == "FAILED" and args.fail_fast:
            print(f"\n{Colors.RED}[Pipeline Terminated] Stage 'combo_eval' failed and --fail-fast was specified.{Colors.END}")
            sys.exit(1)
    else:
        stage_results["combo_eval"] = {"status": "SKIPPED", "elapsed": 0.0, "artifacts": []}

    # 3. Monte Carlo Stage
    if "monte_carlo" in active_stages:
        stage_results["monte_carlo"] = run_stage_monte_carlo(args, root_dir, py_exe)
        if stage_results["monte_carlo"]["status"] == "FAILED" and args.fail_fast:
            print(f"\n{Colors.RED}[Pipeline Terminated] Stage 'monte_carlo' failed and --fail-fast was specified.{Colors.END}")
            sys.exit(1)
    else:
        stage_results["monte_carlo"] = {"status": "SKIPPED", "elapsed": 0.0, "artifacts": []}

    total_elapsed = time.time() - pipeline_start
    print_banner("Pipeline Execution Summary", f"Total Duration: {total_elapsed:.1f}s")

    for stage_name, res in stage_results.items():
        status = res["status"]
        color = Colors.GREEN if status == "SUCCESS" else (Colors.YELLOW if status == "SKIPPED" else Colors.RED)
        print(f"  Stage {stage_name.ljust(12)}: {color}{status.ljust(9)}{Colors.END} ({res['elapsed']:.1f}s)")
        if res.get("artifacts"):
            for art in res["artifacts"]:
                print(f"    - {art}")

    print("\n" + "=" * 80)
    print(f"{Colors.GREEN}[COMPLETE] Master pipeline execution finished.{Colors.END}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
