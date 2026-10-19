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
    print(f"    [TRACE] Executing fetch_tournament_wins() for Season: {season}")
    cache_dir = os.path.join(data_dir, "tournament_wins")
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"tournament_wins_{season}.csv")
    
    if os.path.exists(cache_file):
        print(f"    [I/O] Cache Hit! Loading historical target wins from: {cache_file}")
        df_wins = pd.read_csv(cache_file)
        wins_dict = dict(zip(df_wins['team'], df_wins['wins']))
        print(f"    [DEBUG] Loaded {len(wins_dict)} team win targets for {season}.")
        return wins_dict
        
    print(f"    [API] Cache Miss. Initiating HTTP GET to /games endpoint for {season}...")
    url = f"{BASE_URL}/games"
    params = {"season": season, "seasonType": "postseason", "tournament": "NCAA"}
    
    try:
        response = requests.get(url, params=params, headers=HEADERS, timeout=60)
        response.raise_for_status()
        games = response.json()
        print(f"    [API] Success! Retrieved {len(games)} postseason game records for {season}.")
    except Exception as e:
        print(f"    [ERROR] API Failure during tournament_wins fetch: {e}")
        return {}

    wins = {}
    print(f"    [TRACE] Parsing {len(games)} games to determine exact win totals per team...")
    for game in games:
        home, away = game.get("homeTeam"), game.get("awayTeam")
        if home not in wins: wins[home] = 0
        if away not in wins: wins[away] = 0
        if game.get("homeWinner") is True: wins[home] += 1
        elif game.get("awayWinner") is True: wins[away] += 1

    print(f"    [I/O] Caching calculated wins for {len(wins)} teams to disk...")
    df_wins = pd.DataFrame(list(wins.items()), columns=['team', 'wins'])
    df_wins.to_csv(cache_file, index=False)
    return wins

def fetch_bracket_games(year, data_dir):
    print(f"    [TRACE] Executing fetch_bracket_games() for Season: {year}")
    year_dir = os.path.join(data_dir, str(year))
    os.makedirs(year_dir, exist_ok=True)
    csv_path = os.path.join(year_dir, "tournament_games.csv")
    
    df = pd.DataFrame()
    if os.path.exists(csv_path):
        try:
            print(f"    [I/O] Reading local tournament bracket: {csv_path}")
            df = pd.read_csv(csv_path)
            print(f"    [DEBUG] Successfully loaded {len(df)} bracket games.")
        except Exception as e:
            print(f"    [ERROR] Failed to read {csv_path}: {e}")
            
    if df.empty:
        print(f"    [API] No local bracket found. Querying endpoint for {year} NCAA Tournament...")
        url = f"{BASE_URL}/games"
        params = {"season": year, "seasonType": "postseason", "tournament": "NCAA"}
        try:
            response = requests.get(url, params=params, headers=HEADERS, timeout=30)
            response.raise_for_status()
            games = response.json()
            if not games: 
                print(f"    [WARNING] API returned 0 games for {year}.")
                return pd.DataFrame()
            df = pd.DataFrame(games)
            print(f"    [I/O] Writing newly fetched bracket ({len(df)} games) to {csv_path}")
            df.to_csv(csv_path, index=False)
        except Exception as e:
            print(f"    [ERROR] Bracket API fetch failed: {e}")
            return pd.DataFrame()

    print(f"    [TRACE] Stripping Unicode/BOM characters from column headers...")
    df.columns = df.columns.str.replace(r'[^\x00-\x7F]+', '', regex=True).str.strip()
    
    col_map = {}
    for c in df.columns:
        clean_c = re.sub(r'[\s_\-]', '', str(c)).lower()
        if clean_c in ['hometeam', 'home']: col_map[c] = 'homeTeam'
        elif clean_c in ['awayteam', 'away']: col_map[c] = 'awayTeam'
        elif clean_c in ['homewinner', 'winnerhome']: col_map[c] = 'homeWinner'
        elif clean_c in ['awaywinner', 'winneraway']: col_map[c] = 'awayWinner'
            
    print(f"    [DEBUG] Applying fuzzy column map: {col_map}")
    df.rename(columns=col_map, inplace=True)
    return df

def build_team_features_for_scoring(data_dir, season):
    print(f"      [TRACE] Initializing build_team_features_for_scoring() -> Year: {season}")
    season_dir = os.path.join(data_dir, str(season))
    if not os.path.exists(season_dir): 
        print(f"      [WARNING] Directory missing: {season_dir}")
        return pd.DataFrame()
        
    teams = sorted([d for d in os.listdir(season_dir) if os.path.isdir(os.path.join(season_dir, d))])
    print(f"      [DEBUG] Located {len(teams)} team subdirectories for {season}.")
    
    engineered_rows = []
    skip_cols = {'gameId', 'season', 'teamId', 'opponentId', 'teamSeed', 'opponentSeed', 
                 'totalMinutes', 'games', 'wins', 'losses', 'starts'}
    
    for i, team_safe in enumerate(teams):
        team_dir = os.path.join(season_dir, team_safe)
        game_file = os.path.join(team_dir, f"{team_safe}_game_stats.csv")
        player_file = os.path.join(team_dir, f"{team_safe}_players_season_stats.csv")
        
        if not os.path.exists(game_file) or not os.path.exists(player_file): 
            continue
            
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
        
        # Periodic output to show engine is alive during heavy I/O
        if (i + 1) % 100 == 0:
            print(f"        [INFO] Processed features for {i + 1}/{len(teams)} teams...")
            
    df_out = pd.DataFrame(engineered_rows).fillna(0)
    print(f"      [DEBUG] Final engineered feature matrix shape for {season}: {df_out.shape}")
    return df_out

def simulate_blind_bracket(df_games, ensemble_scores):
    if 'startDate' in df_games.columns:
        df_games = df_games.sort_values('startDate').tail(63).reset_index(drop=True)
    else:
        df_games = df_games.tail(63).reset_index(drop=True)
        
    if len(df_games) != 63:
        return 0.0 
        
    round_1_games = df_games.iloc[:32]
    
    actual_survival = {}
    for team in set(round_1_games['homeTeam']).union(set(round_1_games['awayTeam'])):
        actual_survival[team] = 0
        
    for _, game in df_games.iterrows():
        winner = game['Actual_Winner']
        if winner in actual_survival:
            actual_survival[winner] += 1
            
    game_path_map = {}
    for i in range(32, 63):
        home_team = df_games.iloc[i]['homeTeam']
        away_team = df_games.iloc[i]['awayTeam']
        home_origin_idx = None
        away_origin_idx = None
        
        for j in range(i):
            if df_games.iloc[j]['Actual_Winner'] == home_team:
                home_origin_idx = j
            elif df_games.iloc[j]['Actual_Winner'] == away_team:
                away_origin_idx = j
                
        game_path_map[i] = {'left_source': home_origin_idx, 'right_source': away_origin_idx}

    simulated_winners = {}
    predicted_survival = {t: 0 for t in actual_survival.keys()}
    
    for i in range(32):
        home = round_1_games.iloc[i]['homeTeam']
        away = round_1_games.iloc[i]['awayTeam']
        home_score = ensemble_scores.get(home, 0)
        away_score = ensemble_scores.get(away, 0)
        winner = home if home_score > away_score else away
        simulated_winners[i] = winner
        predicted_survival[winner] += 1
        
    for i in range(32, 63):
        path = game_path_map.get(i)
        if not path or path['left_source'] is None or path['right_source'] is None:
            continue
            
        home_sim = simulated_winners.get(path['left_source'])
        away_sim = simulated_winners.get(path['right_source'])
        
        if not home_sim or not away_sim: continue
            
        home_score = ensemble_scores.get(home_sim, 0)
        away_score = ensemble_scores.get(away_sim, 0)
        winner = home_sim if home_score > away_score else away_sim
        simulated_winners[i] = winner
        predicted_survival[winner] += 1
        
    errors = []
    for team in actual_survival:
        errors.append(abs(actual_survival[team] - predicted_survival[team]))
        
    return np.mean(errors)

def generate_top_10_plot(df_results, metric_col, title, ylabel, out_path, ascending=False):
    print(f"  [Plotting] Generating Top 10 plot for: {metric_col}...")
    top_10 = df_results.sort_values(by=f'Global_{metric_col}', ascending=ascending).head(10)
    
    plot_data = []
    for rank_idx, row in enumerate(top_10.iterrows(), 1):
        idx, row_data = row
        short_name = f"Rank {rank_idx}: {row_data['Model_Count']}-Model Blend"
        for y_data in row_data['Yearly_Data']:
            plot_data.append({'Ensemble': short_name, 'Year': y_data['Year'], metric_col: y_data[metric_col]})
            
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
    print(f"  [I/O] Saved Plot: {out_path}")
    plt.close()

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir) if os.path.basename(script_dir) == "models" else script_dir
    
    data_dir = os.path.join(project_root, "data")
    models_dir = os.path.join(project_root, "model_outputs")
    eval_dir = os.path.join(project_root, "ensemble_bracket_evaluations")
    os.makedirs(eval_dir, exist_ok=True)
    
    print("=" * 80)
    print("Initiating Master Ensemble Combinator (Hyper-Verbose Diagnostics)")
    print("=" * 80)
    print(f"[SYSTEM] Project Root: {project_root}")
    print(f"[SYSTEM] Loading Model Definitions from: {models_dir}")
    print(f"[SYSTEM] Target Analytics Directory: {eval_dir}\n")
    
    model_paths = glob.glob(os.path.join(models_dir, "*", "Train_*_Val_*"))
    print(f"[ENGINE] Detected {len(model_paths)} total historical training artifacts.")
    
    master_scores = {}
    master_shap_dicts = {} 
    bracket_games = {}
    actual_team_wins = {}
    discovered_models = set()
    
    print("\n" + "-" * 50)
    print("PHASE 1: BASELINE INGESTION & FEATURE MAPPING")
    print("-" * 50)
    for path in model_paths:
        parts = path.split(os.sep)
        model_name = parts[-2].replace('_', ' ').title()
        folder_name = parts[-1]
        
        if model_name.lower().startswith('ft'): 
            print(f"  [SKIP] Skipping FT Transformer model mapping: {folder_name}")
            continue
            
        match = re.search(r"Val_(\d{4})", folder_name)
        if not match: continue
        year = int(match.group(1))
        
        shap_path = os.path.join(path, "shap_feature_importance.csv")
        if not os.path.exists(shap_path): 
            print(f"  [WARNING] Missing SHAP file at {shap_path}")
            continue
            
        print(f"\n  [INGEST] Loading Architecture: {model_name} | Validation Target: {year}")
        shap_df = pd.read_csv(shap_path)
        shap_dict = dict(zip(shap_df['feature'], shap_df['shap_importance']))
        print(f"    [INFO] Acquired {len(shap_dict)} SHAP feature weights.")
        
        if model_name not in master_shap_dicts:
            master_shap_dicts[model_name] = shap_dict 
        else:
            if year > 2020: master_shap_dicts[model_name] = shap_dict
            
        if year not in master_scores:
            print(f"    [INFO] Registering Season {year} into Master Score Registry...")
            master_scores[year] = {}
            actual_team_wins[year] = fetch_tournament_wins(year, data_dir)
            df_games = fetch_bracket_games(year, data_dir)
            if not df_games.empty and 'homeTeam' in df_games.columns and 'homeWinner' in df_games.columns:
                df_games['homeTeam'] = df_games['homeTeam'].astype(str).str.strip()
                df_games['awayTeam'] = df_games['awayTeam'].astype(str).str.strip()
                df_games['Actual_Winner'] = np.where(df_games['homeWinner'].astype(str).str.lower() == 'true', df_games['homeTeam'], df_games['awayTeam'])
                bracket_games[year] = df_games
                print(f"    [SUCCESS] Bracket loaded and standardized for {year}.")
            else:
                print(f"    [ERROR] Critical column missing in bracket for {year}. Rejecting dataset.")
                bracket_games[year] = pd.DataFrame()
                
        if bracket_games[year].empty: continue
            
        discovered_models.add(model_name)
        master_scores[year][model_name] = {}
        
        df_features = build_team_features_for_scoring(data_dir, year)
        if df_features.empty: continue
        
        teams = df_features['team'].values
        X_raw = df_features.drop(columns=['team'])
        
        print(f"    [MATH] Applying StandardScaler to {X_raw.shape[1]} features across {len(teams)} teams.")
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_raw)
        df_scaled = pd.DataFrame(X_scaled, columns=X_raw.columns)
        df_scaled['team'] = teams
        
        print(f"    [MATH] Executing Base Dot Product Multiplication...")
        for _, row in df_scaled.iterrows():
            team = str(row['team']).strip()
            score = sum(float(row[feat]) * float(imp) for feat, imp in shap_dict.items() if feat in df_scaled.columns)
            master_scores[year][model_name][team] = score

    discovered_models = sorted(list(discovered_models))
    print("\n" + "-" * 50)
    print("PHASE 2: GLOBAL Z-SCORE NORMALIZATION")
    print("-" * 50)
    print(f"[ENGINE] Mathematical harmonization starting for {len(discovered_models)} models...")
    
    for year in master_scores:
        print(f"\n  [NORMALIZE] Harmonizing scales for Season {year}")
        for model in master_scores[year]:
            scores = list(master_scores[year][model].values())
            if not scores: continue
            mean_s = np.mean(scores)
            std_s = np.std(scores) if np.std(scores) > 0 else 1.0
            print(f"    [MATH] Model: {model:25} | Raw Mean: {mean_s:8.4f} | Raw Std: {std_s:8.4f} | Teams: {len(scores)}")
            for t in master_scores[year][model]:
                master_scores[year][model][t] = (master_scores[year][model][t] - mean_s) / std_s

    all_combinations = []
    for i in range(2, len(discovered_models) + 1):
        combo_subset = list(itertools.combinations(discovered_models, i))
        print(f"  [COMBINE] Generated {len(combo_subset):4} unique {i}-model combinations.")
        all_combinations.extend(combo_subset)
        
    print(f"\n[ENGINE] Total permutations scheduled for simulation: {len(all_combinations)}")
    print("\n" + "-" * 50)
    print("PHASE 3: TOURNAMENT SIMULATION ENGINE (H2H & BLIND PATHING)")
    print("-" * 50)

    ensemble_results = []
    total_combos = len(all_combinations)
    
    for idx, combo in enumerate(all_combinations, 1):
        combo_name = " + ".join(combo)
        combo_correct = 0
        combo_total = 0
        combo_yearly_acc = []
        
        # Hyper-verbose trace every 50 loops
        if idx % 50 == 0 or idx == 1 or idx == total_combos:
            print(f"\n  =======================================================")
            print(f"  [SIMULATOR] Architecture {idx:4}/{total_combos} ({idx/total_combos:.1%})")
            print(f"  [RECIPE] {combo_name}")
            print(f"  =======================================================")
            
        for year, df_games in bracket_games.items():
            if df_games.empty: continue
            if not all(m in master_scores[year] for m in combo): continue
                
            correct_this_year = 0
            total_this_year = 0
            
            # 1. H2H Matchups
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
                
            # 2. Ranking Fidelity
            y_true, y_pred = [], []
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
                
            # 3. Blind Bracket Topological Error
            combined_team_scores = {}
            for team in actual_team_wins[year].keys():
                missing = False
                for m in combo:
                    if team not in master_scores[year][m]:
                        missing = True
                        break
                if not missing:
                    combined_team_scores[team] = np.mean([master_scores[year][m][team] for m in combo])
                    
            blind_error = simulate_blind_bracket(df_games, combined_team_scores)
                
            if total_this_year > 0:
                combo_correct += correct_this_year
                combo_total += total_this_year
                
                acc_val = correct_this_year / total_this_year
                
                # Deep Trace Print
                if idx % 50 == 0 or idx == 1:
                    print(f"    [YEAR {year}] Acc: {acc_val:.1%} | NDCG: {ndcg_val:.3f} | Spear: {spearman_corr:.3f} | BBPE: {blind_error:.2f}")
                    
                combo_yearly_acc.append({
                    'Year': year, 
                    'Accuracy': acc_val,
                    'NDCG': ndcg_val,
                    'Spearman': spearman_corr,
                    'MARE': mare,
                    'Blind_Error': blind_error
                })
                
        if combo_total > 0:
            ensemble_results.append({
                'Ensemble_Name': combo_name,
                'Models': combo,
                'Model_Count': len(combo),
                'Total_Games_Eval': combo_total,
                'Global_Accuracy': combo_correct / combo_total,
                'Global_NDCG': np.mean([y['NDCG'] for y in combo_yearly_acc]),
                'Global_Spearman': np.mean([y['Spearman'] for y in combo_yearly_acc]),
                'Global_MARE': np.mean([y['MARE'] for y in combo_yearly_acc]),
                'Global_Blind_Error': np.mean([y['Blind_Error'] for y in combo_yearly_acc]),
                'Yearly_Data': combo_yearly_acc
            })

    print("\n" + "-" * 50)
    print("PHASE 4: METRIC EXTRACTION & PLOTTING")
    print("-" * 50)
    df_results = pd.DataFrame(ensemble_results)
    if df_results.empty:
        print("[ERROR] Fatal Halt. DataFrame is empty. No evaluations completed.")
        return
    
    csv_path = os.path.join(eval_dir, "ALL_ENSEMBLE_COMBINATIONS_METRICS.csv")
    print(f"  [I/O] Exporting mega-matrix of all metrics to {csv_path}")
    df_results[['Ensemble_Name', 'Model_Count', 'Global_Accuracy', 'Global_Blind_Error', 'Global_NDCG', 'Global_Spearman', 'Global_MARE']].sort_values(by='Global_Blind_Error', ascending=True).to_csv(csv_path, index=False)

    generate_top_10_plot(df_results, 'Accuracy', 'Top 10 Ensembles by Bracket Accuracy', 'Accuracy (% of Games)', os.path.join(eval_dir, "TOP_10_ENSEMBLES_ACCURACY.jpg"), ascending=False)
    generate_top_10_plot(df_results, 'NDCG', 'Top 10 Ensembles by Ranking Fidelity (NDCG)', 'NDCG Score', os.path.join(eval_dir, "TOP_10_ENSEMBLES_NDCG.jpg"), ascending=False)
    generate_top_10_plot(df_results, 'Spearman', 'Top 10 Ensembles by Spearman Rank Correlation', 'Spearman Correlation', os.path.join(eval_dir, "TOP_10_ENSEMBLES_SPEARMAN.jpg"), ascending=False)
    generate_top_10_plot(df_results, 'MARE', 'Top 10 Ensembles by Mean Absolute Rank Error', 'Rank Error Positions', os.path.join(eval_dir, "TOP_10_ENSEMBLES_MARE.jpg"), ascending=True)
    generate_top_10_plot(df_results, 'Blind_Error', 'Top 10 Ensembles by Blind Bracket Placement Error', 'Average Rounds Off-Target', os.path.join(eval_dir, "TOP_10_ENSEMBLES_BLIND_ERROR.jpg"), ascending=True)

    print("\n  [I/O] Generating Blended SHAP Importances for Top 20 Elite Architectures...")
    top_20 = df_results.sort_values(by='Global_Blind_Error', ascending=True).head(20)
    shap_out_dir = os.path.join(eval_dir, "top_20_shap_weights")
    os.makedirs(shap_out_dir, exist_ok=True)
    
    for i, row in enumerate(top_20.itertuples(), 1):
        blended_shap = {}
        for m in row.Models:
            for feat, imp in master_shap_dicts[m].items():
                if feat not in blended_shap: blended_shap[feat] = []
                blended_shap[feat].append(imp)
                
        final_shap = {k: np.mean(v) for k, v in blended_shap.items()}
        df_shap = pd.DataFrame(list(final_shap.items()), columns=['feature', 'shap_importance'])
        df_shap = df_shap.sort_values(by='shap_importance', ascending=False)
        
        safe_name = f"Rank_{i}_" + "_".join([m.replace(" ", "") for m in row.Models])[:50]
        out_f = os.path.join(shap_out_dir, f"{safe_name}_SHAP.csv")
        df_shap.to_csv(out_f, index=False)
        print(f"    [I/O] Extracted Recipe -> {out_f}")

    print("\n" + "=" * 80)
    print("[COMPLETE] Master Multivariable Combinatorial Pipeline resolved successfully.")
    print("=" * 80)

if __name__ == "__main__":
    main()