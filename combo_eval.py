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
    cache_dir = os.path.join(data_dir, "tournament_wins")
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"tournament_wins_{season}.csv")
    
    if os.path.exists(cache_file):
        df_wins = pd.read_csv(cache_file)
        return dict(zip(df_wins['team'], df_wins['wins']))
        
    print(f"    [API] Cache Miss. Initiating HTTP GET to /games endpoint for {season}...")
    url = f"{BASE_URL}/games"
    params = {"season": season, "seasonType": "postseason", "tournament": "NCAA"}
    
    try:
        response = requests.get(url, params=params, headers=HEADERS, timeout=60)
        response.raise_for_status()
        games = response.json()
    except Exception as e:
        print(f"    [ERROR] API Failure during tournament_wins fetch: {e}")
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
    year_dir = os.path.join(data_dir, str(year))
    os.makedirs(year_dir, exist_ok=True)
    csv_path = os.path.join(year_dir, "tournament_games.csv")
    
    df = pd.DataFrame()
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
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
                return pd.DataFrame()
            df = pd.DataFrame(games)
            df.to_csv(csv_path, index=False)
        except Exception as e:
            print(f"    [ERROR] Bracket API fetch failed: {e}")
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
    season_dir = os.path.join(data_dir, str(season))
    if not os.path.exists(season_dir): 
        return pd.DataFrame()
        
    teams = sorted([d for d in os.listdir(season_dir) if os.path.isdir(os.path.join(season_dir, d))])
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

        # --- Advanced Tempo-Free Four Factors & Efficiency Metrics ---
        poss_series = df_reg['teamStats.possessions'] if 'teamStats.possessions' in df_reg.columns else pd.Series(dtype=float)
        if poss_series.empty or poss_series.isna().all() or (poss_series <= 0).all():
            fga = df_reg.get('teamStats.fieldGoals.attempted', 0)
            orb = df_reg.get('teamStats.rebounds.offensive', 0)
            tov = df_reg.get('teamStats.turnovers.total', 0)
            fta = df_reg.get('teamStats.freeThrows.attempted', 0)
            poss_est = np.maximum(fga - orb + tov + 0.475 * fta, 1.0)
            tot_poss = float(poss_est.sum())
        else:
            tot_poss = float(np.maximum(poss_series.fillna(0), 1.0).sum())

        pts_tot = float(df_reg['teamStats.points.total'].sum()) if 'teamStats.points.total' in df_reg.columns else (
            float(df_reg['homePoints'].sum() + df_reg['awayPoints'].sum()) if 'homePoints' in df_reg.columns else 0.0
        )
        opp_pts_tot = float(df_reg['opponentStats.points.total'].sum()) if 'opponentStats.points.total' in df_reg.columns else 0.0

        if tot_poss > 0:
            features['tempo_off_rating'] = (pts_tot / tot_poss) * 100.0
            features['tempo_def_rating'] = (opp_pts_tot / tot_poss) * 100.0
            features['tempo_net_rating'] = features['tempo_off_rating'] - features['tempo_def_rating']
        else:
            features['tempo_off_rating'] = 100.0
            features['tempo_def_rating'] = 100.0
            features['tempo_net_rating'] = 0.0

        if 'pace' in df_reg.columns:
            features['tempo_pace_mean'] = float(df_reg['pace'].mean())

        # Four Factors (Offense & Defense)
        ff_off_efg = df_reg.get('teamStats.fourFactors.effectiveFieldGoalPct')
        features['four_factors_off_efg'] = float(ff_off_efg.mean()) if ff_off_efg is not None and not ff_off_efg.isna().all() else 50.0
        ff_def_efg = df_reg.get('opponentStats.fourFactors.effectiveFieldGoalPct')
        features['four_factors_def_efg'] = float(ff_def_efg.mean()) if ff_def_efg is not None and not ff_def_efg.isna().all() else 50.0

        ff_off_tov = df_reg.get('teamStats.fourFactors.turnoverRatio')
        features['four_factors_off_tov'] = float(ff_off_tov.mean()) if ff_off_tov is not None and not ff_off_tov.isna().all() else 18.0
        ff_def_tov = df_reg.get('opponentStats.fourFactors.turnoverRatio')
        features['four_factors_def_tov'] = float(ff_def_tov.mean()) if ff_def_tov is not None and not ff_def_tov.isna().all() else 18.0

        ff_off_orb = df_reg.get('teamStats.fourFactors.offensiveReboundPct')
        features['four_factors_off_orb'] = float(ff_off_orb.mean()) if ff_off_orb is not None and not ff_off_orb.isna().all() else 30.0
        ff_def_orb = df_reg.get('opponentStats.fourFactors.offensiveReboundPct')
        features['four_factors_def_drb'] = float((100.0 - ff_def_orb).mean()) if ff_def_orb is not None and not ff_def_orb.isna().all() else 70.0

        ff_off_ftr = df_reg.get('teamStats.fourFactors.freeThrowRate')
        features['four_factors_off_ftr'] = float(ff_off_ftr.mean()) if ff_off_ftr is not None and not ff_off_ftr.isna().all() else 25.0
        ff_def_ftr = df_reg.get('opponentStats.fourFactors.freeThrowRate')
        features['four_factors_def_ftr'] = float(ff_def_ftr.mean()) if ff_def_ftr is not None and not ff_def_ftr.isna().all() else 25.0

        # Rotation Shrinkage & Starting Lineup Concentration
        if not df_p.empty and 'minutes' in df_p.columns:
            tot_p_min = float(df_p['minutes'].sum())
            top7_min = float(df_p.nlargest(7, 'minutes')['minutes'].sum())
            features['rotation_top7_minutes_ratio'] = top7_min / max(tot_p_min, 1.0)
            if 'points' in df_p.columns:
                tot_p_pts = float(df_p['points'].sum())
                top5_pts = float(df_p.nlargest(5, 'points')['points'].sum())
                features['rotation_top5_points_ratio'] = top5_pts / max(tot_p_pts, 1.0)
                    
        engineered_rows.append(features)
            
    df_out = pd.DataFrame(engineered_rows).fillna(0)
    return df_out

def get_feature_polarity(feature_name: str) -> float:
    """
    Returns +1.0 for metrics that correlate positively with team strength/success,
    and -1.0 for metrics where higher values denote negative outcomes (e.g., opponent stats, turnovers, fouls).
    """
    fl = str(feature_name).lower()
    if 'opponentstats' in fl:
        # If opponent commits turnovers or fouls, that benefits our team (+1.0)
        if any(f in fl for f in ['turnover', 'foul']):
            return 1.0
        return -1.0
    # Team errors and fouls hurt the team (-1.0)
    if any(f in fl for f in ['turnover', 'fouls', 'calc_turnover_ratio', 'def_efg', 'def_ftr', 'tempo_def_rating', 'four_factors_off_tov']):
        return -1.0
    return 1.0

# Global cache for bracket structures to eliminate redundant pandas .iloc indexing
_BRACKET_STRUCTURE_CACHE = {}

def get_bracket_structure(df_games):
    cache_key = id(df_games)
    if cache_key in _BRACKET_STRUCTURE_CACHE:
        return _BRACKET_STRUCTURE_CACHE[cache_key]
        
    if 'startDate' in df_games.columns:
        df_63 = df_games.sort_values('startDate').tail(63).reset_index(drop=True)
    else:
        df_63 = df_games.tail(63).reset_index(drop=True)
        
    if len(df_63) != 63:
        _BRACKET_STRUCTURE_CACHE[cache_key] = None
        return None
        
    r1 = df_63.iloc[:32]
    round_1_pairs = [(str(r['homeTeam']).strip(), str(r['awayTeam']).strip()) for _, r in r1.iterrows()]
    actual_winners = [str(r['Actual_Winner']).strip() for _, r in df_63.iterrows()]
    
    actual_survival = {}
    for h, a in round_1_pairs:
        actual_survival[h] = 0
        actual_survival[a] = 0
    for w in actual_winners:
        if w in actual_survival:
            actual_survival[w] += 1
            
    game_path_map = {}
    for i in range(32, 63):
        home_team = str(df_63.iloc[i]['homeTeam']).strip()
        away_team = str(df_63.iloc[i]['awayTeam']).strip()
        home_origin_idx = None
        away_origin_idx = None
        for j in range(i):
            if actual_winners[j] == home_team:
                home_origin_idx = j
            elif actual_winners[j] == away_team:
                away_origin_idx = j
        game_path_map[i] = (home_origin_idx, away_origin_idx)
        
    h2h_games = []
    for _, g in df_games.iterrows():
        h = str(g.get('homeTeam', '')).strip()
        a = str(g.get('awayTeam', '')).strip()
        w = str(g.get('Actual_Winner', '')).strip()
        if h and a and w:
            h2h_games.append((h, a, w))
            
    parsed = {
        'round_1_pairs': round_1_pairs,
        'game_path_map': game_path_map,
        'actual_survival': actual_survival,
        'h2h_games': h2h_games,
    }
    _BRACKET_STRUCTURE_CACHE[cache_key] = parsed
    return parsed

def compute_stacking_weights(combo, master_scores, actual_team_wins):
    """Solve for optimal non-negative least squares (NNLS) stacking weights across historical seasons."""
    from scipy.optimize import nnls
    X_rows, y_rows = [], []
    for year, wins_dict in actual_team_wins.items():
        if year not in master_scores: continue
        if not all(m in master_scores[year] for m in combo): continue
        for team, tw in wins_dict.items():
            if all(team in master_scores[year][m] for m in combo):
                X_rows.append([master_scores[year][m][team] for m in combo])
                y_rows.append(float(tw))
    if not X_rows or len(X_rows) < len(combo):
        return np.ones(len(combo)) / len(combo)
    X = np.array(X_rows, dtype=float)
    y = np.array(y_rows, dtype=float)
    w, _ = nnls(X, y)
    if w.sum() > 0:
        return w / w.sum()
    return np.ones(len(combo)) / len(combo)

def simulate_blind_bracket(df_games, ensemble_scores):
    """Simulates 63-game single-elimination bracket using pre-cached structure for 1000x speedup."""
    parsed = get_bracket_structure(df_games)
    if parsed is None:
        return 0.0, 0.5, 0.5, 0.0
        
    round_1_pairs = parsed['round_1_pairs']
    game_path_map = parsed['game_path_map']
    actual_survival = parsed['actual_survival']
    
    simulated_winners = {}
    predicted_survival = {t: 0 for t in actual_survival}
    game_confidences = []
    round_pts_map = [10] * 32 + [20] * 16 + [40] * 8 + [80] * 4 + [160] * 2 + [320] * 1
    
    for i, (home, away) in enumerate(round_1_pairs):
        home_score = ensemble_scores.get(home, 0.0)
        away_score = ensemble_scores.get(away, 0.0)
        diff = abs(home_score - away_score)
        conf = float(1.0 / (1.0 + np.exp(-0.85 * diff)))
        game_confidences.append(conf)
        winner = home if home_score > away_score else away
        simulated_winners[i] = winner
        predicted_survival[winner] = predicted_survival.get(winner, 0) + 1
        
    for i in range(32, 63):
        left_src, right_src = game_path_map[i]
        if left_src is None or right_src is None:
            game_confidences.append(0.5)
            continue
        home_sim = simulated_winners.get(left_src)
        away_sim = simulated_winners.get(right_src)
        if not home_sim or not away_sim:
            game_confidences.append(0.5)
            continue
            
        home_score = ensemble_scores.get(home_sim, 0.0)
        away_score = ensemble_scores.get(away_sim, 0.0)
        diff = abs(home_score - away_score)
        conf = float(1.0 / (1.0 + np.exp(-0.85 * diff)))
        game_confidences.append(conf)
        winner = home_sim if home_score > away_score else away_sim
        simulated_winners[i] = winner
        predicted_survival[winner] = predicted_survival.get(winner, 0) + 1
        
    errors = [abs(actual_survival[team] - predicted_survival.get(team, 0)) for team in actual_survival]
    blind_error = float(np.mean(errors)) if errors else 0.0
    mean_game_conf = float(np.mean(game_confidences)) if game_confidences else 0.5
    bracket_conf = float(np.exp(np.mean(np.log(np.maximum(game_confidences, 1e-6))))) if game_confidences else 0.5
    expected_pts = float(sum(pts * conf for pts, conf in zip(round_pts_map[:len(game_confidences)], game_confidences)))
    return blind_error, bracket_conf, mean_game_conf, expected_pts

def generate_top_ensemble_plot(df_results, metric_col, title, ylabel, out_path, top_n=10, ascending=False):
    print(f"  [Plotting] Generating Top {top_n} plot for: {metric_col}...")
    top_df = df_results.sort_values(by=f'Global_{metric_col}', ascending=ascending).head(top_n)
    
    plot_data = []
    for rank_idx, row in enumerate(top_df.iterrows(), 1):
        idx, row_data = row
        short_name = f"Rank {rank_idx}: {row_data['Model_Count']}-Model Blend"
        for y_data in row_data['Yearly_Data']:
            if metric_col in y_data:
                plot_data.append({'Ensemble': short_name, 'Year': y_data['Year'], metric_col: y_data[metric_col]})
            
    df_plot = pd.DataFrame(plot_data)
    if df_plot.empty:
        return
    plt.figure(figsize=(15, 8))
    sns.set_theme(style="whitegrid")
    
    unique_years = sorted(df_plot['Year'].unique())
    base_markers = ['o', 's', '^', 'D', 'v', 'p', '*', 'X', 'P', 'H', 'd', '<', '>', '8', 'h']
    markers = (base_markers * 3)[:top_n]
    palette = sns.color_palette("tab10" if top_n <= 10 else "husl", top_n)
    
    sns.lineplot(
        data=df_plot, x='Year', y=metric_col, hue='Ensemble', style='Ensemble',
        markers=markers, dashes=False, lw=2.2, markersize=8, palette=palette
    )
    
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel('Tournament Year', fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.xticks(unique_years)
    
    if 'Accuracy' in metric_col or 'Confidence' in metric_col:
        plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
        
    plt.legend(title=f'Top {top_n} Meta-Architectures', bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight', format='jpg')
    print(f"  [I/O] Saved Plot: {out_path}")
    plt.close()

def generate_top_10_plot(df_results, metric_col, title, ylabel, out_path, ascending=False):
    generate_top_ensemble_plot(df_results, metric_col, title, ylabel, out_path, top_n=10, ascending=ascending)

def main(args=None):
    import argparse
    parser = argparse.ArgumentParser(description="Multi-Objective Combinatorial Ensemble Evaluator")
    parser.add_argument("--directional", action="store_true", help="Apply directional polarity to feature weights (invert opponent stats, turnovers, fouls)")
    parser.add_argument("--stacking", action="store_true", help="Solve for optimal non-negative least squares stacking weights for each ensemble combination.")
    parsed_args, _ = parser.parse_known_args(args)
    use_directional = parsed_args.directional
    use_stacking = parsed_args.stacking

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir) if os.path.basename(script_dir) == "models" else script_dir
    
    data_dir = os.path.join(project_root, "data")
    models_dir = os.path.join(project_root, "model_outputs")
    eval_dir = os.path.join(project_root, "ensemble_bracket_evaluations")
    os.makedirs(eval_dir, exist_ok=True)
    
    print("=" * 80)
    print("Initiating Master Ensemble Combinator (Hyper-Verbose Diagnostics)")
    if use_directional:
        print("  [Directional Scoring Mode Active]: Inverting opponent stats and team turnovers/fouls.")
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
    season_scaled_cache = {}
    
    print("\n" + "-" * 50)
    print("PHASE 1: BASELINE INGESTION & FEATURE MAPPING")
    print("-" * 50)
    for path in model_paths:
        parts = path.split(os.sep)
        model_name = parts[-2].replace('_', ' ').title()
        folder_name = parts[-1]
        
        retired_prefixes = ('ft', 'elo baseline', 'tabnet', 'lstm', 'logistic regression')
        if any(model_name.lower().startswith(p) for p in retired_prefixes):
            if not model_name.lower().startswith('lasso lstm'):
                continue
            
        match = re.search(r"Val_(\d{4})", folder_name)
        if not match: continue
        year = int(match.group(1))
        
        shap_path = os.path.join(path, "shap_feature_importance.csv")
        if not os.path.exists(shap_path): 
            continue
            
        shap_df = pd.read_csv(shap_path)
        shap_dict = dict(zip(shap_df['feature'], shap_df['shap_importance']))
        
        if model_name not in master_shap_dicts:
            master_shap_dicts[model_name] = shap_dict 
        else:
            if year > 2020: master_shap_dicts[model_name] = shap_dict
            
        if year not in master_scores:
            master_scores[year] = {}
            actual_team_wins[year] = fetch_tournament_wins(year, data_dir)
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
        
        # Optimize by caching scaled features once per season instead of re-reading files per model
        if year not in season_scaled_cache:
            df_features = build_team_features_for_scoring(data_dir, year)
            if not df_features.empty:
                teams = df_features['team'].values
                X_raw = df_features.drop(columns=['team'])
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X_raw)
                df_scaled = pd.DataFrame(X_scaled, columns=X_raw.columns)
                df_scaled['team'] = teams
                season_scaled_cache[year] = df_scaled
            else:
                season_scaled_cache[year] = pd.DataFrame()
                
        df_scaled = season_scaled_cache[year]
        if df_scaled.empty: continue
        
        for _, row in df_scaled.iterrows():
            team = str(row['team']).strip()
            score = sum(
                float(row[feat]) * float(imp) * (get_feature_polarity(feat) if use_directional else 1.0)
                for feat, imp in shap_dict.items() if feat in df_scaled.columns
            )
            master_scores[year][model_name][team] = score

    discovered_models = sorted(list(discovered_models))
    print(f"\n[ENGINE] Successfully ingested {len(discovered_models)} active model architectures: {', '.join(discovered_models)}")
    print("\n" + "-" * 50)
    print("PHASE 2: GLOBAL Z-SCORE NORMALIZATION")
    print("-" * 50)
    
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
        combo_subset = list(itertools.combinations(discovered_models, i))
        print(f"  [COMBINE] Generated {len(combo_subset):4} unique {i}-model combinations.")
        all_combinations.extend(combo_subset)
        
    print(f"\n[ENGINE] Total permutations scheduled for simulation: {len(all_combinations)}")
    print("\n" + "-" * 50)
    print("PHASE 3: TOURNAMENT SIMULATION ENGINE (H2H & BLIND PATHING WITH CONFIDENCE METRICS)")
    print("-" * 50)

    # Pre-parse all bracket structures
    for year, df_games in bracket_games.items():
        if not df_games.empty:
            get_bracket_structure(df_games)

    ensemble_results = []
    total_combos = len(all_combinations)
    
    for idx, combo in enumerate(all_combinations, 1):
        combo_name = " + ".join(combo)
        combo_correct = 0
        combo_total = 0
        combo_yearly_acc = []
        if use_stacking:
            weights = compute_stacking_weights(combo, master_scores, actual_team_wins)
        else:
            weights = np.ones(len(combo)) / len(combo)
        
        if idx % 50 == 0 or idx == 1 or idx == total_combos:
            print(f"  [SIMULATOR] Architecture {idx:4}/{total_combos} ({idx/total_combos:.1%}) -> {combo_name[:60]}")
            
        for year, df_games in bracket_games.items():
            if df_games.empty: continue
            if not all(m in master_scores[year] for m in combo): continue
            parsed_bracket = get_bracket_structure(df_games)
            if not parsed_bracket: continue
                
            correct_this_year = 0
            total_this_year = 0
            h2h_confidences = []
            
            # 1. H2H Matchups with Per-Game Confidence (fast list iteration)
            for home, away, actual in parsed_bracket['h2h_games']:
                missing = False
                for m in combo:
                    if home not in master_scores[year][m] or away not in master_scores[year][m]:
                        missing = True
                        break
                if missing: continue
                
                home_ensemble_score = float(sum(w * master_scores[year][m][home] for w, m in zip(weights, combo)))
                away_ensemble_score = float(sum(w * master_scores[year][m][away] for w, m in zip(weights, combo)))
                
                diff = abs(home_ensemble_score - away_ensemble_score)
                game_conf = float(1.0 / (1.0 + np.exp(-0.85 * diff)))
                h2h_confidences.append(game_conf)
                
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
                    y_pred.append(float(sum(w * master_scores[year][m][team] for w, m in zip(weights, combo))))
            
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
                
            # 3. Blind Bracket Topological Error & Bracket-Wide Confidence Score
            combined_team_scores = {}
            for team in actual_team_wins[year].keys():
                missing = False
                for m in combo:
                    if team not in master_scores[year][m]:
                        missing = True
                        break
                if not missing:
                    combined_team_scores[team] = float(sum(w * master_scores[year][m][team] for w, m in zip(weights, combo)))
                    
            blind_error, bracket_conf, blind_game_conf, expected_pts = simulate_blind_bracket(df_games, combined_team_scores)
            mean_game_conf = float(np.mean(h2h_confidences)) if h2h_confidences else blind_game_conf
                
            if total_this_year > 0:
                combo_correct += correct_this_year
                combo_total += total_this_year
                acc_val = correct_this_year / total_this_year
                
                combo_yearly_acc.append({
                    'Year': year, 
                    'Accuracy': acc_val,
                    'Per_Game_Confidence': mean_game_conf,
                    'Per_Bracket_Confidence': bracket_conf,
                    'Expected_Bracket_Points': expected_pts,
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
                'Stacking_Weights': list(weights),
                'Total_Games_Eval': combo_total,
                'Global_Accuracy': combo_correct / combo_total,
                'Global_Per_Game_Confidence': float(np.mean([y['Per_Game_Confidence'] for y in combo_yearly_acc])),
                'Global_Per_Bracket_Confidence': float(np.mean([y['Per_Bracket_Confidence'] for y in combo_yearly_acc])),
                'Global_Expected_Points': float(np.mean([y['Expected_Bracket_Points'] for y in combo_yearly_acc])),
                'Global_NDCG': float(np.mean([y['NDCG'] for y in combo_yearly_acc])),
                'Global_Spearman': float(np.mean([y['Spearman'] for y in combo_yearly_acc])),
                'Global_MARE': float(np.mean([y['MARE'] for y in combo_yearly_acc])),
                'Global_Blind_Error': float(np.mean([y['Blind_Error'] for y in combo_yearly_acc])),
                'Yearly_Data': combo_yearly_acc
            })

    print("\n" + "-" * 50)
    print("PHASE 4: METRIC EXTRACTION & TOP-25 EXPORT")
    print("-" * 50)
    df_results = pd.DataFrame(ensemble_results)
    if df_results.empty:
        print("[ERROR] Fatal Halt. DataFrame is empty. No evaluations completed.")
        return

    # Calculate multi-metric composite performance rank (Borda count across all core metrics)
    df_results['acc_rank'] = df_results['Global_Accuracy'].rank(ascending=False, method='average')
    df_results['blind_rank'] = df_results['Global_Blind_Error'].rank(ascending=True, method='average')
    df_results['ndcg_rank'] = df_results['Global_NDCG'].rank(ascending=False, method='average')
    df_results['spear_rank'] = df_results['Global_Spearman'].rank(ascending=False, method='average')
    df_results['mare_rank'] = df_results['Global_MARE'].rank(ascending=True, method='average')
    df_results['conf_rank'] = df_results['Global_Per_Bracket_Confidence'].rank(ascending=False, method='average')
    
    df_results['Composite_Rank_Score'] = (
        df_results['blind_rank'] * 1.5 +
        df_results['acc_rank'] * 1.5 +
        df_results['conf_rank'] * 1.0 +
        df_results['ndcg_rank'] * 1.0 +
        df_results['spear_rank'] * 1.0 +
        df_results['mare_rank'] * 1.0
    )
    df_results = df_results.sort_values(by=['Global_Blind_Error', 'Composite_Rank_Score'], ascending=[True, True]).reset_index(drop=True)
    
    # 1. Export ALL ensemble combinations
    csv_path = os.path.join(eval_dir, "ALL_ENSEMBLE_COMBINATIONS_METRICS.csv")
    print(f"  [I/O] Exporting mega-matrix of all metrics to {csv_path}")
    all_cols = [
        'Ensemble_Name', 'Model_Count', 'Global_Accuracy', 
        'Global_Per_Game_Confidence', 'Global_Per_Bracket_Confidence', 'Global_Expected_Points',
        'Global_Blind_Error', 'Global_NDCG', 'Global_Spearman', 'Global_MARE', 'Composite_Rank_Score'
    ]
    df_results[all_cols].to_csv(csv_path, index=False)

    # 2. Export 25 Best Performing Ensembles Table
    top_25 = df_results.head(25).copy()
    top_25['Rank'] = np.arange(1, len(top_25) + 1)
    top_25_csv = os.path.join(eval_dir, "TOP_25_BEST_PERFORMING_METRICS.csv")
    top_25_cols = [
        'Rank', 'Ensemble_Name', 'Model_Count', 'Global_Accuracy', 
        'Global_Per_Game_Confidence', 'Global_Per_Bracket_Confidence', 'Global_Expected_Points',
        'Global_Blind_Error', 'Global_NDCG', 'Global_Spearman', 'Global_MARE', 'Composite_Rank_Score'
    ]
    top_25[top_25_cols].to_csv(top_25_csv, index=False)
    print(f"  [I/O] Exported Top 25 Best Performing Metrics to {top_25_csv}")

    # 3. Export Top 10 Plots (Legacy) and Top 25 Plots (Upgraded)
    generate_top_10_plot(df_results, 'Accuracy', 'Top 10 Ensembles by Bracket Accuracy', 'Accuracy (% of Games)', os.path.join(eval_dir, "TOP_10_ENSEMBLES_ACCURACY.jpg"), ascending=False)
    generate_top_10_plot(df_results, 'NDCG', 'Top 10 Ensembles by Ranking Fidelity (NDCG)', 'NDCG Score', os.path.join(eval_dir, "TOP_10_ENSEMBLES_NDCG.jpg"), ascending=False)
    generate_top_10_plot(df_results, 'Spearman', 'Top 10 Ensembles by Spearman Rank Correlation', 'Spearman Correlation', os.path.join(eval_dir, "TOP_10_ENSEMBLES_SPEARMAN.jpg"), ascending=False)
    generate_top_10_plot(df_results, 'MARE', 'Top 10 Ensembles by Mean Absolute Rank Error', 'Rank Error Positions', os.path.join(eval_dir, "TOP_10_ENSEMBLES_MARE.jpg"), ascending=True)
    generate_top_10_plot(df_results, 'Blind_Error', 'Top 10 Ensembles by Blind Bracket Placement Error', 'Average Rounds Off-Target', os.path.join(eval_dir, "TOP_10_ENSEMBLES_BLIND_ERROR.jpg"), ascending=True)

    # High-dimension Top 25 visualizations
    generate_top_ensemble_plot(df_results, 'Accuracy', 'Top 25 Ensembles by Bracket Accuracy', 'Accuracy (% of Games)', os.path.join(eval_dir, "TOP_25_ENSEMBLES_ACCURACY.jpg"), top_n=25, ascending=False)
    generate_top_ensemble_plot(df_results, 'Per_Bracket_Confidence', 'Top 25 Ensembles by Per-Bracket Confidence Score', 'Per-Bracket Confidence (Geometric Mean)', os.path.join(eval_dir, "TOP_25_ENSEMBLES_CONFIDENCE.jpg"), top_n=25, ascending=False)
    generate_top_ensemble_plot(df_results, 'Blind_Error', 'Top 25 Ensembles by Blind Bracket Placement Error', 'Average Rounds Off-Target', os.path.join(eval_dir, "TOP_25_ENSEMBLES_BLIND_ERROR.jpg"), top_n=25, ascending=True)
    generate_top_ensemble_plot(df_results, 'NDCG', 'Top 25 Ensembles by Ranking Fidelity (NDCG)', 'NDCG Score', os.path.join(eval_dir, "TOP_25_ENSEMBLES_NDCG.jpg"), top_n=25, ascending=False)
    generate_top_ensemble_plot(df_results, 'Spearman', 'Top 25 Ensembles by Spearman Rank Correlation', 'Spearman Correlation', os.path.join(eval_dir, "TOP_25_ENSEMBLES_SPEARMAN.jpg"), top_n=25, ascending=False)
    generate_top_ensemble_plot(df_results, 'MARE', 'Top 25 Ensembles by Mean Absolute Rank Error', 'Rank Error Positions', os.path.join(eval_dir, "TOP_25_ENSEMBLES_MARE.jpg"), top_n=25, ascending=True)

    # 4. Generate Blended SHAP Importances for Top 25 Elite Architectures
    print("\n  [I/O] Generating Blended SHAP Importances for Top 25 Elite Architectures...")
    shap_25_dir = os.path.join(eval_dir, "top_25_shap_weights")
    shap_20_dir = os.path.join(eval_dir, "top_20_shap_weights")
    os.makedirs(shap_25_dir, exist_ok=True)
    os.makedirs(shap_20_dir, exist_ok=True)
    
    for i, row in enumerate(top_25.itertuples(), 1):
        blended_shap = {}
        row_weights = getattr(row, 'Stacking_Weights', [1.0 / len(row.Models)] * len(row.Models))
        for w, m in zip(row_weights, row.Models):
            for feat, imp in master_shap_dicts[m].items():
                blended_shap[feat] = blended_shap.get(feat, 0.0) + w * imp
                
        final_shap = {k: v * (get_feature_polarity(k) if use_directional else 1.0) for k, v in blended_shap.items()}
        df_shap = pd.DataFrame(list(final_shap.items()), columns=['feature', 'shap_importance'])
        df_shap = df_shap.sort_values(by='shap_importance', ascending=False)
        
        safe_name = f"Rank_{i}_" + "_".join([m.replace(" ", "") for m in row.Models])[:50]
        out_25 = os.path.join(shap_25_dir, f"{safe_name}_SHAP.csv")
        df_shap.to_csv(out_25, index=False)
        if i <= 20:
            out_20 = os.path.join(shap_20_dir, f"{safe_name}_SHAP.csv")
            df_shap.to_csv(out_20, index=False)
        print(f"    [I/O] Extracted Recipe Rank {i} -> {out_25}")

    print("\n" + "=" * 80)
    print("[COMPLETE] Master Multivariable Combinatorial Pipeline resolved successfully.")
    print(f"Top 25 Metrics saved to: {top_25_csv}")
    print("=" * 80)

if __name__ == "__main__":
    main()
