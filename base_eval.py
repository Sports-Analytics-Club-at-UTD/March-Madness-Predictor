import os
import glob
import re
import numpy as np
import pandas as pd
import requests

# --- MATPLOTLIB BACKEND FIX ---
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("CBBD_API_KEY")
BASE_URL = "https://api.collegebasketballdata.com"
HEADERS = {"Authorization": f"Bearer {API_KEY}"} if API_KEY else {}

def fetch_bracket_games(year, data_dir):
    """
    Looks for the tournament bracket in data/<year>/tournament_games.csv.
    If it doesn't exist, queries the API, saves the CSV locally, and returns the dataframe.
    Aggressively strips Excel BOMs and fuzzy-matches column names to handle schema changes.
    """
    year_dir = os.path.join(data_dir, str(year))
    os.makedirs(year_dir, exist_ok=True)
    csv_path = os.path.join(year_dir, "tournament_games.csv")
    
    df = pd.DataFrame()
    
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            print(f"  [Data] Cache hit! Successfully loaded {len(df)} games from {csv_path}")
        except Exception as e:
            print(f"  [Error] Could not read local {csv_path}: {e}. Falling back to API...")
    
    if df.empty:
        print(f"  [API] Cache miss. Querying /games endpoint for {year} postseason bracket...")
        url = f"{BASE_URL}/games"
        params = {"season": year, "seasonType": "postseason", "tournament": "NCAA"}
        
        try:
            response = requests.get(url, params=params, headers=HEADERS, timeout=30)
            response.raise_for_status()
            games = response.json()
            
            if not games:
                print(f"  [Warning] API returned no tournament games for {year}.")
                return pd.DataFrame()
                
            df = pd.DataFrame(games)
            df.to_csv(csv_path, index=False)
            print(f"  [API] Successfully retrieved and saved {len(df)} postseason games to {csv_path}.")
        except Exception as e:
            print(f"  [Error] Could not fetch bracket for {year}: {e}")
            return pd.DataFrame()

    # 1. Clean invisible BOM characters injected by Excel
    df.columns = df.columns.str.replace(r'[^\x00-\x7F]+', '', regex=True).str.strip()
    
    # 2. Fuzzy Column Mapping: Handles schema changes across different years/API versions
    col_map = {}
    for c in df.columns:
        clean_c = re.sub(r'[\s_\-]', '', str(c)).lower()
        if clean_c in ['hometeam', 'home']:
            col_map[c] = 'homeTeam'
        elif clean_c in ['awayteam', 'away']:
            col_map[c] = 'awayTeam'
        elif clean_c in ['homewinner', 'winnerhome']:
            col_map[c] = 'homeWinner'
        elif clean_c in ['awaywinner', 'winneraway']:
            col_map[c] = 'awayWinner'
            
    df.rename(columns=col_map, inplace=True)
    return df

def build_team_features_for_scoring(data_dir, season):
    """Rebuilds the raw statistical feature matrix for a given season."""
    season_dir = os.path.join(data_dir, str(season))
    if not os.path.exists(season_dir):
        return pd.DataFrame()
        
    teams = sorted([d for d in os.listdir(season_dir) if os.path.isdir(os.path.join(season_dir, d))])
    engineered_rows = []
    skip_cols = {'gameId', 'season', 'teamId', 'opponentId', 'teamSeed', 'opponentSeed', 
                 'totalMinutes', 'games', 'wins', 'losses', 'starts'}
    
    for team_safe in teams:
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
        
    df = pd.DataFrame(engineered_rows).fillna(0)
    return df

def generate_bracket_matchup_plot(games_df, model_name, year, out_dir, accuracy, mean_game_conf=0.5, bracket_conf=0.5):
    """Creates a visual grid showing Correct (Green) vs Incorrect (Red) predictions with confidence."""
    plt.figure(figsize=(14, max(8, len(games_df) * 0.28)))
    
    games_df = games_df.sort_values(by='Predicted_Correct', ascending=False).reset_index(drop=True)
    y_positions = np.arange(len(games_df))
    colors = ['#2ca02c' if row['Predicted_Correct'] else '#d62728' for _, row in games_df.iterrows()]
    
    plt.barh(y_positions, [1]*len(games_df), color=colors, alpha=0.15, edgecolor=colors, height=0.7)
    
    for idx, row in games_df.iterrows():
        matchup_text = f"{row['homeTeam']} vs {row['awayTeam']}"
        conf_str = f" [Conf: {row['Game_Confidence']:.1%}]" if 'Game_Confidence' in row else ""
        outcome_text = f"Pick: {row['Predicted_Winner']} (Actual: {row['Actual_Winner']}){conf_str}"
        
        plt.text(0.02, idx, matchup_text, va='center', fontsize=9.5, fontweight='bold', color='#333333')
        plt.text(0.48, idx, outcome_text, va='center', fontsize=9.5, color=colors[idx], fontweight='bold')
        
    plt.title(f'{model_name} Bracket Predictions ({year})\nAccuracy: {accuracy:.1%} | Game Conf: {mean_game_conf:.1%} | Bracket Conf: {bracket_conf:.1%}', fontsize=15, fontweight='bold')
    plt.xlim(0, 1)
    plt.yticks([])
    plt.xticks([])
    plt.gca().invert_yaxis() 
    sns.despine(left=True, bottom=True)
    
    os.makedirs(out_dir, exist_ok=True)
    plot_path = os.path.join(out_dir, f"{model_name.replace(' ', '_')}_{year}_Bracket.jpg")
    plt.savefig(plot_path, dpi=120, bbox_inches='tight', format='jpg')
    plt.close()

def main(args=None):
    import argparse
    from combo_eval import get_feature_polarity
    parser = argparse.ArgumentParser(description="Base Model Evaluation Engine")
    parser.add_argument("--directional", action="store_true", help="Apply directional polarity to feature weights")
    parsed_args, _ = parser.parse_known_args(args)
    use_directional = parsed_args.directional
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir) if os.path.basename(script_dir) == "models" else script_dir
    
    data_dir = os.path.join(project_root, "data")
    models_dir = os.path.join(project_root, "model_outputs")
    eval_dir = os.path.join(project_root, "bracket_evaluations")
    os.makedirs(eval_dir, exist_ok=True)
    
    print("=" * 80)
    print("Initiating Local/API Bracket Simulation & SHAP Dot-Product Evaluation Engine")
    print("=" * 80)
    
    model_paths = glob.glob(os.path.join(models_dir, "*", "Train_*_Val_*"))
    evaluations = []
    team_records = []
    season_scaled_cache = {}
    
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
            
        print(f"\n[Evaluating] {model_name} -> Season {year}")
        
        shap_df = pd.read_csv(shap_path)
        shap_dict = dict(zip(shap_df['feature'], shap_df['shap_importance']))
        
        df_games = fetch_bracket_games(year, data_dir)
        if df_games.empty:
            continue
            
        if 'homeTeam' in df_games.columns and 'awayTeam' in df_games.columns:
            df_games['homeTeam'] = df_games['homeTeam'].astype(str).str.strip()
            df_games['awayTeam'] = df_games['awayTeam'].astype(str).str.strip()
        else:
            print(f"  [Error] Bracket for {year} is missing 'homeTeam' or 'awayTeam' columns.")
            continue
            
        if 'homeWinner' in df_games.columns:
            df_games['Actual_Winner'] = np.where(df_games['homeWinner'].astype(str).str.lower() == 'true', df_games['homeTeam'], df_games['awayTeam'])
        else:
            print(f"  [Warning] Missing 'homeWinner' column. Cannot evaluate accuracy for {year}.")
            continue
        
        # Cache features and scaling once per season
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
        
        team_scores = {}
        for _, row in df_scaled.iterrows():
            team = str(row['team']).strip()
            score = 0.0
            for feat, imp in shap_dict.items():
                if feat in df_scaled.columns:
                    score += float(row[feat]) * float(imp) * (get_feature_polarity(feat) if use_directional else 1.0)
            team_scores[team] = score
            
        correct_predictions = 0
        total_games = 0
        predictions_log = []
        game_confidences = []
        
        for _, game in df_games.iterrows():
            home = game.get('homeTeam')
            away = game.get('awayTeam')
            actual_winner = game.get('Actual_Winner')
            
            if home not in team_scores or away not in team_scores:
                continue
                
            home_score = team_scores[home]
            away_score = team_scores[away]
            diff = abs(home_score - away_score)
            game_conf = float(1.0 / (1.0 + np.exp(-0.85 * diff)))
            game_confidences.append(game_conf)
            
            predicted_winner = home if home_score > away_score else away
            is_correct = (predicted_winner == actual_winner)
            
            if is_correct: correct_predictions += 1
            total_games += 1
            
            predictions_log.append({
                'homeTeam': home,
                'awayTeam': away,
                'Actual_Winner': actual_winner,
                'Predicted_Winner': predicted_winner,
                'Predicted_Correct': is_correct,
                'Game_Confidence': game_conf
            })
            
            team_records.append({'Model': model_name, 'Year': year, 'Team': actual_winner, 'Type': 'Actual_Wins', 'Count': 1})
            team_records.append({'Model': model_name, 'Year': year, 'Team': predicted_winner, 'Type': 'Predicted_Wins', 'Count': 1})
            
        if total_games == 0: 
            print(f"  [Warning] Intersection mismatch! Teams in bracket did not match teams in data folders.")
            continue
            
        accuracy = correct_predictions / total_games
        mean_game_conf = float(np.mean(game_confidences)) if game_confidences else 0.5
        bracket_conf = float(np.exp(np.mean(np.log(np.maximum(game_confidences, 1e-6))))) if game_confidences else 0.5
        
        evaluations.append({
            'Model': model_name,
            'Year': year,
            'Accuracy': accuracy,
            'Per_Game_Confidence': mean_game_conf,
            'Per_Bracket_Confidence': bracket_conf,
            'Total_Games': total_games,
            'Correct_Games': correct_predictions
        })
        
        df_preds = pd.DataFrame(predictions_log)
        if year >= 2021:
            visual_dir = os.path.join(eval_dir, "bracket_visualizations", model_name)
            generate_bracket_matchup_plot(df_preds, model_name, year, visual_dir, accuracy, mean_game_conf, bracket_conf)

    if not evaluations:
        print("\n[Warning] No evaluations completed. Ensure CSVs possess expected column headers. Exiting.")
        return
        
    df_eval = pd.DataFrame(evaluations).sort_values(by=['Model', 'Year'])
    num_models = len(df_eval['Model'].unique())
    unique_years = sorted(df_eval['Year'].unique())
    
    # Export metrics CSV
    metrics_csv = os.path.join(eval_dir, "BASE_MODELS_PERFORMANCE_METRICS.csv")
    df_eval.to_csv(metrics_csv, index=False)
    print(f"\n  [I/O] Saved Base Models Performance Metrics to: {metrics_csv}")
    
    # Plot Accuracy
    print("\n[Plotting] Generating Global Bracket Accuracy Time Series...")
    plt.figure(figsize=(14, 7))
    sns.set_theme(style="whitegrid")
    markers = ['o', 's', '^', 'D', 'v', 'p', '*', 'X', 'P', 'H', 'd', '<', '>', '8', 'h']
    active_palette = 'tab20' if num_models > 10 else 'tab10'
    
    sns.lineplot(
        data=df_eval, x='Year', y='Accuracy', hue='Model', style='Model',
        markers=markers[:num_models], dashes=False, lw=2.5, markersize=8, palette=active_palette
    )
    plt.title('Bracket Prediction Accuracy Over Time (Dot-Product Heuristic)', fontsize=16, fontweight='bold')
    plt.xlabel('Tournament Year', fontsize=12)
    plt.ylabel('Prediction Accuracy (% of Games)', fontsize=12)
    plt.xticks(unique_years)
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
    plt.legend(title='Architecture', bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()
    acc_path = os.path.join(eval_dir, "BRACKET_ACCURACY_OVER_TIME.jpg")
    plt.savefig(acc_path, dpi=300, bbox_inches='tight', format='jpg')
    plt.close()
    print(f"  -> Saved: {acc_path}")

    # Plot Bracket Confidence Over Time
    print("[Plotting] Generating Bracket Confidence Over Time...")
    plt.figure(figsize=(14, 7))
    sns.lineplot(
        data=df_eval, x='Year', y='Per_Bracket_Confidence', hue='Model', style='Model',
        markers=markers[:num_models], dashes=False, lw=2.5, markersize=8, palette=active_palette
    )
    plt.title('Model Per-Bracket Confidence Score Over Time (Geometric Mean Likelihood)', fontsize=16, fontweight='bold')
    plt.xlabel('Tournament Year', fontsize=12)
    plt.ylabel('Bracket Confidence Score', fontsize=12)
    plt.xticks(unique_years)
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
    plt.legend(title='Architecture', bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()
    conf_path = os.path.join(eval_dir, "BRACKET_CONFIDENCE_OVER_TIME.jpg")
    plt.savefig(conf_path, dpi=300, bbox_inches='tight', format='jpg')
    plt.close()
    print(f"  -> Saved: {conf_path}")

    print("[Plotting] Generating Top 10 Teams Actual vs Predicted Wins...")
    df_teams = pd.DataFrame(team_records)
    agg_teams = df_teams.groupby(['Team', 'Type'])['Count'].sum().reset_index()
    pivot_teams = agg_teams.pivot(index='Team', columns='Type', values='Count').fillna(0)
    
    pivot_teams['Actual_Wins'] = pivot_teams['Actual_Wins'] / num_models
    pivot_teams['Predicted_Wins'] = pivot_teams['Predicted_Wins'] / num_models
    
    top_10 = pivot_teams.sort_values(by='Actual_Wins', ascending=False).head(10)
    top_10_melted = top_10.reset_index().melt(id_vars='Team', value_vars=['Actual_Wins', 'Predicted_Wins'], var_name='Metric', value_name='Wins')
    
    plt.figure(figsize=(14, 7))
    sns.barplot(data=top_10_melted, x='Team', y='Wins', hue='Metric', palette=['#1f77b4', '#ff7f0e'], edgecolor='black')
    plt.title('Top 10 Teams: Actual vs Predicted Tournament Wins (Averaged Across All Models)', fontsize=16, fontweight='bold')
    plt.xlabel('Team', fontsize=12)
    plt.ylabel('Total Tournament Wins', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='')
    plt.tight_layout()
    team_path = os.path.join(eval_dir, "TOP_10_TEAMS_PRED_VS_ACTUAL.jpg")
    plt.savefig(team_path, dpi=300, bbox_inches='tight', format='jpg')
    plt.close()
    print(f"  -> Saved: {team_path}")
    
    print("\n" + "=" * 80)
    print("[COMPLETE] Bracket Simulation Output generated successfully.")
    print("=" * 80)

if __name__ == "__main__":
    main()
