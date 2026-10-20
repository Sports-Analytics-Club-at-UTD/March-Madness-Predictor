import os
import glob
import re
import itertools
import numpy as np
import pandas as pd
import requests

# --- MATPLOTLIB BACKEND FIX ---
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import ndcg_score
from scipy.stats import spearmanr
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("CBBD_API_KEY")
BASE_URL = "https://api.collegebasketballdata.com"
HEADERS = {"Authorization": f"Bearer {API_KEY}"} if API_KEY else {}

def fetch_tournament_wins(season, data_dir):
    """Fetches postseason NCAA tournament wins and caches them to determine global ranking fidelity."""
    cache_dir = os.path.join(data_dir, "tournament_wins")
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"tournament_wins_{season}.csv")
    
    if os.path.exists(cache_file):
        df_wins = pd.read_csv(cache_file)
        return dict(zip(df_wins['team'], df_wins['wins']))
        
    print(f"    [API] Cache miss. Querying /games endpoint for {season} total wins...")
    url = f"{BASE_URL}/games"
    params = {"season": season, "seasonType": "postseason", "tournament": "NCAA"}
    
    try:
        response = requests.get(url, params=params, headers=HEADERS, timeout=60)
        response.raise_for_status()
        games = response.json()
    except Exception as e:
        print(f"    [Error] Could not retrieve postseason games: {e}")
        return {}

    wins = {}
    for game in games:
        home, away = game.get("homeTeam"), game.get("awayTeam")
        if home not in wins: wins[home] = 0
        if away not in wins: wins[away] = 0
        if game.get("homeWinner") is True: wins[home] += 1
        elif game.get("awayWinner") is True: wins[away] += 1

    df_wins = pd.DataFrame(list(wins.items()), columns=['team', 'wins'])
    df_wins.to_csv(cache_file, index=False)
    return wins

def fetch_bracket_games(year, data_dir):
    """Loads bracket matchups, fuzzy-matches columns, and normalizes teams."""
    year_dir = os.path.join(data_dir, str(year))
    os.makedirs(year_dir, exist_ok=True)
    csv_path = os.path.join(year_dir, "tournament_games.csv")
    
    df = pd.DataFrame()
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
        except Exception:
            pass
            
    if df.empty:
        url = f"{BASE_URL}/games"
        params = {"season": year, "seasonType": "postseason", "tournament": "NCAA"}
        try:
            response = requests.get(url, params=params, headers=HEADERS, timeout=30)
            response.raise_for_status()
            games = response.json()
            if not games: return pd.DataFrame()
            df = pd.DataFrame(games)
            df.to_csv(csv_path, index=False)
        except Exception:
            return pd.DataFrame()

    df.columns = df.columns.str.replace(r'[^\x00-\x7F]+', '', regex=True).str.strip()
    col_map = {}
    for c in df.columns:
        clean_c = re.sub(r'[\s_\-]', '', str(c)).lower()
        if clean_c in ['hometeam', 'home']: col_map[c] = 'homeTeam'
        elif clean_c in ['awayteam', 'away']: col_map[c] = 'awayTeam'
        elif clean_c in ['homewinner', 'winnerhome']: col_map[c] = 'homeWinner'
        elif clean_c in ['awaywinner', 'winneraway']: col_map[c] = 'awayWinner'
            
    df.rename(columns=col_map, inplace=True)
    return df

def build_team_features_for_scoring(data_dir, season):
    """Rebuilds the raw statistical feature matrix for a given season."""
    season_dir = os.path.join(data_dir, str(season))
    if not os.path.exists(season_dir): return pd.DataFrame()
        
    teams = sorted([d for d in os.listdir(season_dir) if os.path.isdir(os.path.join(season_dir, d))])
    engineered_rows = []
    skip_cols = {'gameId', 'season', 'teamId', 'opponentId', 'teamSeed', 'opponentSeed', 
                 'totalMinutes', 'games', 'wins', 'losses', 'starts'}
    
    for team_safe in teams:
        team_dir = os.path.join(season_dir, team_safe)
        game_file = os.path.join(team_dir, f"{team_safe}_game_stats.csv")
        player_file = os.path.join(team_dir, f"{team_safe}_players_season_stats.csv")
        
        if not os.path.exists(game_file) or not os.path.exists(player_file): continue
            
        df_g = pd.read_csv(game_file)
        if df_g.empty: continue
            
        actual_team_name = str(df_g['team'].iloc[0]).strip()
        features = {'team': actual_team_name}
        
        df_reg = df_g[df_g['seasonType'] == 'regular']
        if df_reg.empty: df_reg = df_g 
            
        for c in df_reg.select_dtypes(include=[np.number]).columns:
            if c not in skip_cols and not c.endswith('EloEnd'):
                features[f"game_mean_{c}"] = df_reg[c].mean()
                features[f"game_std_{c}"] = df_reg[c].std()
                
        df_p = pd.read_csv(player_file)
        if not df_p.empty and 'minutes' in df_p.columns:
            top5 = df_p.nlargest(5, 'minutes')
            for c in top5.select_dtypes(include=[np.number]).columns:
                if c not in skip_cols and 'Id' not in c:
                    features[f"starters_mean_{c}"] = top5[c].mean()
                    features[f"roster_max_{c}"] = df_p[c].max()
                    
        engineered_rows.append(features)
        
    df = pd.DataFrame(engineered_rows).fillna(0)
    return df

def generate_top_10_plot(df_results, metric_col, title, ylabel, out_path, ascending=False):
    """Generic plotting function to generate the top 10 time series for any metric."""
    print(f"  [Plotting] Generating Top 10 plot for: {metric_col}...")
    top_10 = df_results.sort_values(by=f'Global_{metric_col}', ascending=ascending).head(10)
    
    plot_data = []
    for rank_idx, row in enumerate(top_10.iterrows(), 1):
        idx, row_data = row
        short_name = f"Rank {rank_idx}: {row_data['Model_Count']}-Model Blend"
        
        for y_data in row_data['Yearly_Data']:
            plot_data.append({
                'Ensemble': short_name,
                'Year': y_data['Year'],
                metric_col: y_data[metric_col]
            })
            
    df_plot = pd.DataFrame(plot_data)
    
    plt.figure(figsize=(14, 7))
    sns.set_theme(style="whitegrid")
    
    unique_years = sorted(df_plot['Year'].unique())
    markers = ['o', 's', '^', 'D', 'v', 'p', '*', 'X', 'P', 'H']
    
    sns.lineplot(
        data=df_plot, x='Year', y=metric_col, hue='Ensemble', style='Ensemble',
        markers=markers, dashes=False, lw=2.5, markersize=9, palette='tab10'
    )
    
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel('Tournament Year', fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.xticks(unique_years)
    
    if metric_col == 'Accuracy':
        plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
        
    plt.legend(title='Ultimate Meta-Architectures', bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight', format='jpg')
    plt.close()

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir) if os.path.basename(script_dir) == "models" else script_dir
    
    data_dir = os.path.join(project_root, "data")
    models_dir = os.path.join(project_root, "model_outputs")
    eval_dir = os.path.join(project_root, "ensemble_bracket_evaluations")
    os.makedirs(eval_dir, exist_ok=True)
    
    print("=" * 80)
    print("Initiating Master Ensemble Combinator (Full Multivariable Evaluation)")
    print("=" * 80)
    
    model_paths = glob.glob(os.path.join(models_dir, "*", "Train_*_Val_*"))
    
    master_scores = {}
    bracket_games = {}
    actual_team_wins = {}
    discovered_models = set()
    
    print("\n[Phase 1] Ingesting SHAP Feature Importances & Baseline Realities...")
    for path in model_paths:
        parts = path.split(os.sep)
        model_name = parts[-2].replace('_', ' ').title()
        folder_name = parts[-1]
        
        if model_name.lower().startswith('ft'): continue
            
        match = re.search(r"Val_(\d{4})", folder_name)
        if not match: continue
        year = int(match.group(1))
        
        shap_path = os.path.join(path, "shap_feature_importance.csv")
        if not os.path.exists(shap_path): continue
        
        shap_df = pd.read_csv(shap_path)
        shap_dict = dict(zip(shap_df['feature'], shap_df['shap_importance']))
        
        if year not in master_scores:
            master_scores[year] = {}
            # Cache the overall win realities for NDCG/Spearman/MARE
            actual_team_wins[year] = fetch_tournament_wins(year, data_dir)
            
            # Cache the actual bracket for Accuracy
            df_games = fetch_bracket_games(year, data_dir)
            if not df_games.empty and 'homeTeam' in df_games.columns and 'homeWinner' in df_games.columns:
                df_games['homeTeam'] = df_games['homeTeam'].astype(str).str.strip()
                df_games['awayTeam'] = df_games['awayTeam'].astype(str).str.strip()
                df_games['Actual_Winner'] = np.where(df_games['homeWinner'].astype(str).str.lower() == 'true', df_games['homeTeam'], df_games['awayTeam'])
                bracket_games[year] = df_games
            else:
                bracket_games[year] = pd.DataFrame()
                
        if bracket_games[year].empty: continue
            
        discovered_models.add(model_name)
        master_scores[year][model_name] = {}
        
        df_features = build_team_features_for_scoring(data_dir, year)
        if df_features.empty: continue
        
        teams = df_features['team'].values
        X_raw = df_features.drop(columns=['team'])
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_raw)
        df_scaled = pd.DataFrame(X_scaled, columns=X_raw.columns)
        df_scaled['team'] = teams
        
        for _, row in df_scaled.iterrows():
            team = str(row['team']).strip()
            score = sum(float(row[feat]) * float(imp) for feat, imp in shap_dict.items() if feat in df_scaled.columns)
            master_scores[year][model_name][team] = score

    discovered_models = sorted(list(discovered_models))
    print(f"[Engine] Found {len(discovered_models)} active base models.")
    print("[Engine] Z-Score Normalizing base scores to allow for fair mathematical blending...")
    
    for year in master_scores:
        for model in master_scores[year]:
            scores = list(master_scores[year][model].values())
            if not scores: continue
            mean_s = np.mean(scores)
            std_s = np.std(scores) if np.std(scores) > 0 else 1.0
            for t in master_scores[year][model]:
                master_scores[year][model][t] = (master_scores[year][model][t] - mean_s) / std_s

    all_combinations = []
    for i in range(2, len(discovered_models) + 1):
        all_combinations.extend(list(itertools.combinations(discovered_models, i)))
        
    print(f"\n[Phase 2] Generated {len(all_combinations)} possible ensemble architectures.")
    print("[Engine] Calculating global metrics (Accuracy, NDCG, Spearman, MARE) across all years...")

    ensemble_results = []
    total_combos = len(all_combinations)
    
    for idx, combo in enumerate(all_combinations, 1):
        combo_name = " + ".join(combo)
        combo_correct = 0
        combo_total = 0
        combo_yearly_acc = []
        
        if idx % 100 == 0 or idx == 1 or idx == total_combos:
            print(f"  [Progress] Simulating architecture {idx:4}/{total_combos} ({idx/total_combos:.1%}) -> Size: {len(combo)} models")
            
        for year, df_games in bracket_games.items():
            if df_games.empty: continue
            if not all(m in master_scores[year] for m in combo): continue
                
            # Metric 1: Head-to-Head Bracket Accuracy
            correct_this_year = 0
            total_this_year = 0
            
            for _, game in df_games.iterrows():
                home = game.get('homeTeam')
                away = game.get('awayTeam')
                actual = game.get('Actual_Winner')
                
                missing = False
                for m in combo:
                    if home not in master_scores[year][m] or away not in master_scores[year][m]:
                        missing = True
                        break
                if missing: continue
                
                home_ensemble_score = np.mean([master_scores[year][m][home] for m in combo])
                away_ensemble_score = np.mean([master_scores[year][m][away] for m in combo])
                
                pred = home if home_ensemble_score > away_ensemble_score else away
                if pred == actual: correct_this_year += 1
                total_this_year += 1
                
            # Metrics 2-4: Global Ranking Fidelity (NDCG, Spearman, MARE)
            y_true = []
            y_pred = []
            for team, true_wins in actual_team_wins[year].items():
                missing = False
                for m in combo:
                    if team not in master_scores[year][m]:
                        missing = True
                        break
                if not missing:
                    y_true.append(true_wins)
                    y_pred.append(np.mean([master_scores[year][m][team] for m in combo]))
            
            ndcg_val = 0.0
            spearman_corr = 0.0
            mare = 34.0
            
            if len(y_true) > 1:
                ndcg_val = ndcg_score([y_true], [y_pred])
                spearman_corr, _ = spearmanr(y_true, y_pred)
                if np.isnan(spearman_corr): spearman_corr = 0.0
                
                actual_rank = pd.Series(y_true).rank(method='average', ascending=False).values
                pred_rank = pd.Series(y_pred).rank(method='average', ascending=False).values
                mare = np.mean(np.abs(actual_rank - pred_rank))
                
            if total_this_year > 0:
                acc = correct_this_year / total_this_year
                combo_correct += correct_this_year
                combo_total += total_this_year
                
                combo_yearly_acc.append({
                    'Year': year, 
                    'Accuracy': acc,
                    'NDCG': ndcg_val,
                    'Spearman': spearman_corr,
                    'MARE': mare
                })
                
        if combo_total > 0:
            global_acc = combo_correct / combo_total
            global_ndcg = np.mean([y['NDCG'] for y in combo_yearly_acc])
            global_spearman = np.mean([y['Spearman'] for y in combo_yearly_acc])
            global_mare = np.mean([y['MARE'] for y in combo_yearly_acc])
            
            ensemble_results.append({
                'Ensemble_Name': combo_name,
                'Model_Count': len(combo),
                'Total_Games_Eval': combo_total,
                'Global_Accuracy': global_acc,
                'Global_NDCG': global_ndcg,
                'Global_Spearman': global_spearman,
                'Global_MARE': global_mare,
                'Yearly_Data': combo_yearly_acc
            })

    print("\n[Phase 3] Extraction & Visualization")
    df_results = pd.DataFrame(ensemble_results)
    if df_results.empty:
        print("[Error] No evaluations completed. Cannot generate output.")
        return
    
    # 1. Output the ENTIRE dataset of all 4,000+ combos to CSV
    csv_path = os.path.join(eval_dir, "ALL_ENSEMBLE_COMBINATIONS_METRICS.csv")
    df_results[['Ensemble_Name', 'Model_Count', 'Total_Games_Eval', 'Global_Accuracy', 'Global_NDCG', 'Global_Spearman', 'Global_MARE']].sort_values(by='Global_Accuracy', ascending=False).to_csv(csv_path, index=False)
    print(f"  [I/O] Saved complete metrics for all {len(df_results)} ensembles to: {csv_path}")

    # 2. Generate the 4 Top 10 Plots
    generate_top_10_plot(
        df_results, 'Accuracy', 
        title='Top 10 Ensembles by Bracket Accuracy', 
        ylabel='Bracket Prediction Accuracy (% of Games)', 
        out_path=os.path.join(eval_dir, "TOP_10_ENSEMBLES_ACCURACY.jpg"), 
        ascending=False
    )
    
    generate_top_10_plot(
        df_results, 'NDCG', 
        title='Top 10 Ensembles by Ranking Fidelity (NDCG)', 
        ylabel='NDCG Score (Higher is Better)', 
        out_path=os.path.join(eval_dir, "TOP_10_ENSEMBLES_NDCG.jpg"), 
        ascending=False
    )
    
    generate_top_10_plot(
        df_results, 'Spearman', 
        title='Top 10 Ensembles by Spearman Rank Correlation', 
        ylabel='Spearman Correlation (Higher is Better)', 
        out_path=os.path.join(eval_dir, "TOP_10_ENSEMBLES_SPEARMAN.jpg"), 
        ascending=False
    )
    
    generate_top_10_plot(
        df_results, 'MARE', 
        title='Top 10 Ensembles by Mean Absolute Rank Error (MARE)', 
        ylabel='Rank Error Positions (Lower is Better)', 
        out_path=os.path.join(eval_dir, "TOP_10_ENSEMBLES_MARE.jpg"), 
        ascending=True # Because for Error, lower is better
    )
    
    print("\n" + "=" * 80)
    print("[COMPLETE] Master Multivariable Combinatorial Pipeline resolved successfully.")
    print("=" * 80)

if __name__ == "__main__":
    main()