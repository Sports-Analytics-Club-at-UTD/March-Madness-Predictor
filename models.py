import os
import sys
import subprocess
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def execute_model_scripts(root_dir, models_dir):
    if not os.path.exists(models_dir):
        print(f"[Error] Models directory not found at {models_dir}")
        sys.exit(1)
        
    scripts = sorted([f for f in os.listdir(models_dir) if f.endswith('.py')])
    
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
        model_dir = os.path.join(outputs_dir, model_name)
        if not os.path.isdir(model_dir) or model_name == "tournament_wins":
            continue
            
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
                
                records.append({
                    'Model': model_name.replace('_', ' ').title(),
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
        
    print(f"  [Plot] Generating Global Comparison Plot for {len(df['Model'].unique())} models...")
    
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(3, 1, figsize=(14, 18), sharex=True)
    unique_years = sorted(df['Year'].unique())
    markers = ['o', 's', '^', 'D', 'v', 'p', '*', 'X']
    
    sns.lineplot(
        data=df, x='Year', y='NDCG', hue='Model', style='Model', 
        markers=markers[:len(df['Model'].unique())], dashes=False, 
        lw=2.5, markersize=8, ax=axes[0], palette='tab10'
    )
    axes[0].set_title('Global Model Comparison: Ranking Fidelity (NDCG) Over Time', fontsize=16, fontweight='bold')
    axes[0].set_ylabel('NDCG (Higher is Better)', fontsize=12)
    axes[0].legend(title='Architecture', bbox_to_anchor=(1.02, 1), loc='upper left')
    
    sns.lineplot(
        data=df, x='Year', y='Spearman', hue='Model', style='Model', 
        markers=markers[:len(df['Model'].unique())], dashes=False, 
        lw=2.5, markersize=8, ax=axes[1], palette='tab10', legend=False
    )
    axes[1].set_title('Global Model Comparison: Spearman Rank Correlation Over Time', fontsize=16, fontweight='bold')
    axes[1].set_ylabel('Spearman Correlation (Higher is Better)', fontsize=12)
    
    sns.lineplot(
        data=df, x='Year', y='MARE', hue='Model', style='Model', 
        markers=markers[:len(df['Model'].unique())], dashes=False, 
        lw=2.5, markersize=8, ax=axes[2], palette='tab10', legend=False
    )
    axes[2].set_title('Global Model Comparison: Mean Absolute Rank Error (MARE) Over Time', fontsize=16, fontweight='bold')
    axes[2].set_xlabel('Holdout Target Season', fontsize=12)
    axes[2].set_ylabel('Rank Error Positions (Lower is Better)', fontsize=12)
    axes[2].set_xticks(unique_years)
    
    plt.tight_layout()
    plot_path = os.path.join(outputs_dir, "GLOBAL_MODEL_COMPARISON.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    csv_path = os.path.join(outputs_dir, "GLOBAL_MODEL_METRICS.csv")
    df.to_csv(csv_path, index=False)
    
    print(f"  [Output] Multi-panel comparison plot saved to: {plot_path}")
    print(f"  [Output] Aggregated metrics CSV saved to: {csv_path}")

def main():
    root_dir = os.path.abspath(os.path.dirname(__file__))
    models_dir = os.path.join(root_dir, "models")
    outputs_dir = os.path.join(root_dir, "model_outputs")
    
    execute_model_scripts(root_dir, models_dir) # Uncomment this if you want it to run everything again
    
    df_metrics = parse_training_logs(outputs_dir)
    generate_global_comparison_plots(df_metrics, outputs_dir)
    
    print("\n" + "=" * 80)
    print("[COMPLETE] Master Framework Aggregation Finished.")
    print("=" * 80)

if __name__ == "__main__":
    main()