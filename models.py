import os
import sys
import subprocess
import re
import numpy as np
import pandas as pd

# --- MATPLOTLIB BACKEND FIX ---
# Forces non-interactive rendering to prevent Tkinter / OS-level thread crashes
import matplotlib
matplotlib.use('Agg')
# ------------------------------

import matplotlib.pyplot as plt
import seaborn as sns

def execute_model_scripts(root_dir, models_dir):
    if not os.path.exists(models_dir):
        print(f"[Error] Models directory not found at {models_dir}")
        sys.exit(1)
        
    # Filter out FT Transformer scripts if they are still in the directory
    scripts = sorted([
        f for f in os.listdir(models_dir) 
        if f.endswith('.py') and not f.lower().startswith('ft')
    ])
    
    print("=" * 80)
    print(f"Master Execution Pipeline Initiated. Discovered {len(scripts)} models.")
    print("=" * 80)
    
    for script in scripts:
        script_path = os.path.join(models_dir, script)
        print(f"\n>>>>> EXECUTING: {script} >>>>>")
        try:
            subprocess.run([sys.executable, script_path], cwd=root_dir, check=True)
            print(f"<<<<< COMPLETED: {script} <<<<<")
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] Script {script} failed with exit code {e.returncode}. Continuing to next model...")

def parse_training_logs(outputs_dir):
    print("\n[Aggregation] Parsing validation logs across all models...")
    records = []
    
    # Updated regex patterns to safely catch 'nan' text when models mathematically collapse
    re_year = re.compile(r"Target Year:\s*(\d+)", re.IGNORECASE)
    re_ndcg = re.compile(r"NDCG.*?([\-0-9\.]+|nan|NaN)", re.IGNORECASE)
    re_spearman = re.compile(r"Spearman.*?([\-0-9\.]+|nan|NaN)", re.IGNORECASE)
    re_mare = re.compile(r"(?:MARE|Rank Error).*?([\-0-9\.]+|nan|NaN)", re.IGNORECASE)
    
    if not os.path.exists(outputs_dir):
        print(f"[Error] Outputs directory not found at {outputs_dir}")
        return pd.DataFrame()
        
    for model_name in os.listdir(outputs_dir):
        # Exclude metadata folders and forcefully exclude FT Transformer artifacts
        if not os.path.isdir(os.path.join(outputs_dir, model_name)): continue
        if model_name.lower() in ["tournament_wins", "bracket_evaluations", "ft_transformer", "ft"]:
            continue
            
        model_dir = os.path.join(outputs_dir, model_name)
            
        for run_folder in os.listdir(model_dir):
            run_dir = os.path.join(model_dir, run_folder)
            log_path = os.path.join(run_dir, "training_summary.log")
            
            if not os.path.isdir(run_dir) or not os.path.exists(log_path):
                continue
            if run_folder == "all_years":
                continue
                
            with open(log_path, 'r') as f:
                content = f.read()
                
            year_match = re_year.search(content)
            ndcg_match = re_ndcg.search(content)
            spearman_match = re_spearman.search(content)
            mare_match = re_mare.search(content)
            
            if year_match and ndcg_match and spearman_match and mare_match:
                
                # Safely parse strings, converting mathematical collapses (NaN) to 0.0 penalty scores
                sp_val = spearman_match.group(1).lower()
                sp_float = 0.0 if sp_val == 'nan' else float(sp_val)
                
                ndcg_val = ndcg_match.group(1).lower()
                ndcg_float = 0.0 if ndcg_val == 'nan' else float(ndcg_val)
                
                mare_val = mare_match.group(1).lower()
                # 34.0 error positions is mathematically equivalent to random guessing for a 68 team field
                mare_float = 34.0 if mare_val == 'nan' else float(mare_val) 
                
                # Format names cleanly (e.g., lasso_rf_meta -> Lasso Rf Meta)
                clean_model_title = model_name.replace('_', ' ').title()
                
                records.append({
                    'Model': clean_model_title,
                    'Year': int(year_match.group(1)),
                    'NDCG': ndcg_float,
                    'Spearman': sp_float,
                    'MARE': mare_float
                })
                
    df = pd.DataFrame(records)
    if not df.empty:
        df = df.sort_values(by=['Model', 'Year']).reset_index(drop=True)
    return df

def generate_global_comparison_plots(df, outputs_dir):
    if df.empty:
        print("[Warning] No metric records found. Skipping global plots.")
        return
        
    num_models = len(df['Model'].unique())
    print(f"  [Plot] Generating 4 separate Comparison Plots for {num_models} models...")
    
    os.makedirs(outputs_dir, exist_ok=True)
    sns.set_theme(style="whitegrid")
    unique_years = sorted(df['Year'].unique())
    
    # Expanded marker list to guarantee support for up to 15 models without crashing
    markers = ['o', 's', '^', 'D', 'v', 'p', '*', 'X', 'P', 'H', 'd', '<', '>', '8', 'h']
    # If the number of models exceeds 10, we must use tab20 so Seaborn doesn't run out of colors
    active_palette = 'tab20' if num_models > 10 else 'tab10'
    
    # --- PLOT 1: NDCG ---
    plt.figure(figsize=(14, 7))
    sns.lineplot(
        data=df, x='Year', y='NDCG', hue='Model', style='Model', 
        markers=markers[:num_models], dashes=False, 
        lw=2.5, markersize=9, palette=active_palette
    )
    plt.title('Global Model Comparison: Ranking Fidelity (NDCG) Over Time', fontsize=16, fontweight='bold')
    plt.xlabel('Holdout Target Season', fontsize=12)
    plt.ylabel('NDCG (Higher is Better)', fontsize=12)
    plt.xticks(unique_years)
    plt.legend(title='Architecture', bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()
    ndcg_path = os.path.join(outputs_dir, "GLOBAL_MODEL_COMPARISON_NDCG.jpg")
    plt.savefig(ndcg_path, dpi=300, bbox_inches='tight', format='jpg')
    plt.close()
    print(f"  [Output] NDCG plot saved to: {ndcg_path}")
    
    # --- PLOT 2: SPEARMAN ---
    plt.figure(figsize=(14, 7))
    sns.lineplot(
        data=df, x='Year', y='Spearman', hue='Model', style='Model', 
        markers=markers[:num_models], dashes=False, 
        lw=2.5, markersize=9, palette=active_palette
    )
    plt.title('Global Model Comparison: Spearman Rank Correlation Over Time', fontsize=16, fontweight='bold')
    plt.xlabel('Holdout Target Season', fontsize=12)
    plt.ylabel('Spearman Correlation (Higher is Better)', fontsize=12)
    plt.xticks(unique_years)
    plt.legend(title='Architecture', bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()
    spearman_path = os.path.join(outputs_dir, "GLOBAL_MODEL_COMPARISON_SPEARMAN.jpg")
    plt.savefig(spearman_path, dpi=300, bbox_inches='tight', format='jpg')
    plt.close()
    print(f"  [Output] Spearman plot saved to: {spearman_path}")
    
    # --- PLOT 3: MARE ---
    plt.figure(figsize=(14, 7))
    sns.lineplot(
        data=df, x='Year', y='MARE', hue='Model', style='Model', 
        markers=markers[:num_models], dashes=False, 
        lw=2.5, markersize=9, palette=active_palette
    )
    plt.title('Global Model Comparison: Mean Absolute Rank Error (MARE) Over Time', fontsize=16, fontweight='bold')
    plt.xlabel('Holdout Target Season', fontsize=12)
    plt.ylabel('Rank Error Positions (Lower is Better)', fontsize=12)
    plt.xticks(unique_years)
    plt.legend(title='Architecture', bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()
    mare_path = os.path.join(outputs_dir, "GLOBAL_MODEL_COMPARISON_MARE.jpg")
    plt.savefig(mare_path, dpi=300, bbox_inches='tight', format='jpg')
    plt.close()
    print(f"  [Output] MARE plot saved to: {mare_path}")
    
    # --- PLOT 4: OVERALL LEADERBOARD RANKING (Composite Borda Count) ---
    print("  [Aggregation] Calculating Master Borda Count Rankings...")
    # Calculate rank placements (1 to N) for each metric per year
    df['ndcg_points'] = df.groupby('Year')['NDCG'].rank(ascending=True, method='average')
    df['spearman_points'] = df.groupby('Year')['Spearman'].rank(ascending=True, method='average')
    
    # For MARE, lower is better, so we rank descending to give high points to low errors
    df['mare_points'] = df.groupby('Year')['MARE'].rank(ascending=False, method='average')
    
    # Sum the points across all metrics
    df['total_points'] = df['ndcg_points'] + df['spearman_points'] + df['mare_points']
    
    # Average the points across all years to get the final master tier score
    leaderboard = df.groupby('Model')['total_points'].mean().reset_index()
    leaderboard = leaderboard.sort_values(by='total_points', ascending=False)
    
    plt.figure(figsize=(12, 8))
    # Map hue directly to Model and disable the legend to satisfy Seaborn requirements
    sns.barplot(
        x='total_points', y='Model', hue='Model', data=leaderboard, 
        palette='viridis', edgecolor='black', zorder=2, legend=False
    )
    plt.title('Master Overall Model Performance Leaderboard\n(Composite Score based on Seasonal Ranking Placements)', fontsize=14, fontweight='bold')
    plt.xlabel('Average Composite Tier Score (Higher is Better)', fontsize=11)
    plt.ylabel('')
    plt.grid(axis='x', linestyle='--', alpha=0.7, zorder=1)
    
    # Annotate the exact scores onto the bars
    for index, row in enumerate(leaderboard.itertuples()):
        plt.text(row.total_points + 0.1, index, f'{row.total_points:.1f} pts', color='black', va="center")
        
    plt.tight_layout()
    rank_path = os.path.join(outputs_dir, "GLOBAL_MODEL_COMPARISON_OVERALL_RANK.jpg")
    plt.savefig(rank_path, dpi=300, bbox_inches='tight', format='jpg')
    plt.close()
    print(f"  [Output] Overall Leaderboard plot saved to: {rank_path}")
    
    # Save Metrics CSV (Clean up the temporary point columns first)
    df_clean = df.drop(columns=['ndcg_points', 'spearman_points', 'mare_points', 'total_points'])
    csv_path = os.path.join(outputs_dir, "GLOBAL_MODEL_METRICS.csv")
    df_clean.to_csv(csv_path, index=False)
    print(f"  [Output] Aggregated metrics CSV saved to: {csv_path}")

def main():
    # Safely resolve absolute paths regardless of invocation directory
    root_dir = os.path.abspath(os.path.dirname(__file__)).strip()
    models_dir = os.path.join(root_dir, "models")
    outputs_dir = os.path.join(root_dir, "model_outputs")
    
    # Unleashed Model Execution
    execute_model_scripts(root_dir, models_dir)
    
    df_metrics = parse_training_logs(outputs_dir)
    generate_global_comparison_plots(df_metrics, outputs_dir)
    
    print("\n" + "=" * 80)
    print("[COMPLETE] Master Framework Aggregation Finished.")
    print("=" * 80)

if __name__ == "__main__":
    main()