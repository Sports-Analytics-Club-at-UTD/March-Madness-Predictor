import os
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, ndcg_score
from scipy.stats import spearmanr
from dotenv import load_dotenv
import warnings

warnings.filterwarnings('ignore')
load_dotenv()

API_KEY = os.getenv("CBBD_API_KEY")
BASE_URL = "https://api.collegebasketballdata.com"
HEADERS = {"Authorization": f"Bearer {API_KEY}"}

def fetch_tournament_wins(season, output_dir):
    """Fetches postseason NCAA tournament wins and caches them."""
    cache_dir = os.path.join(output_dir, "tournament_wins")
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"tournament_wins_{season}.csv")
    
    print(f"      [API] Checking cache for {season} tournament wins at {cache_file}...")
    if os.path.exists(cache_file):
        print(f"      [API] Cache hit! Loading tournament wins for {season} from CSV.")
        df_wins = pd.read_csv(cache_file)
        return dict(zip(df_wins['team'], df_wins['wins']))
        
    print(f"      [API] Cache miss. Querying /games endpoint for {season} postseason...")
    url = f"{BASE_URL}/games"
    params = {"season": season, "seasonType": "postseason", "tournament": "NCAA"}
    
    try:
        response = requests.get(url, params=params, headers=HEADERS, timeout=60)
        response.raise_for_status()
        games = response.json()
    except Exception as e:
        print(f"      [Error] Could not retrieve postseason games: {e}")
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

def build_team_features(data_dir, season, output_dir):
    """Parses team subdirectories to extract regular-season metrics, specifically retaining Elo."""
    season_dir = os.path.join(data_dir, str(season))
    print(f"    [Data] Searching for team directories in {season_dir}...")
    
    if not os.path.exists(season_dir):
        return pd.DataFrame()
        
    tourney_wins = fetch_tournament_wins(season, output_dir)
    teams = sorted([d for d in os.listdir(season_dir) if os.path.isdir(os.path.join(season_dir, d))])
    
    if len(teams) == 0:
        return pd.DataFrame()
    
    engineered_rows = []
    skip_cols = {'gameId', 'season', 'teamId', 'opponentId', 'teamSeed', 'opponentSeed', 
                 'totalMinutes', 'games', 'wins', 'losses', 'starts'}
    
    for team_safe in teams:
        team_dir = os.path.join(season_dir, team_safe)
        game_file = os.path.join(team_dir, f"{team_safe}_game_stats.csv")
        
        if not os.path.exists(game_file):
            continue
            
        df_g = pd.read_csv(game_file)
        if df_g.empty: 
            continue
            
        actual_team_name = df_g['team'].iloc[0]
        
        features = {
            'team': actual_team_name,
            'season': int(season),
            'target_wins': tourney_wins.get(actual_team_name, 0)
        }
        
        df_reg = df_g[df_g['seasonType'] == 'regular']
        if df_reg.empty: 
            df_reg = df_g 
            
        # We explicitly ALLOW 'EloEnd' here because this baseline requires pre-tournament Elo
        num_cols = df_reg.select_dtypes(include=[np.number]).columns
        for c in num_cols:
            if c not in skip_cols:
                features[f"game_mean_{c}"] = df_reg[c].mean()
                
        engineered_rows.append(features)
        
    df_engineered = pd.DataFrame(engineered_rows).fillna(0)
    df_engineered = df_engineered.loc[:, (df_engineered != 0).any(axis=0)]
    return df_engineered

def compute_recency_weights(seasons_series, target_season, half_life=3.0):
    delta = target_season - seasons_series.astype(float)
    weights = np.power(0.5, delta / half_life)
    return (weights / weights.mean()).values

def plot_accuracy_over_time(metrics_df, out_dir):
    print("\n  [Plot] Rendering Baseline Global Accuracy Over Time...")
    
    plt.figure(figsize=(10, 5))
    plt.plot(metrics_df['Year'], metrics_df['MARE'], marker='^', color='#d62728', lw=2.5, markersize=8)
    plt.title('Baseline: Mean Absolute Rank Error (MARE) Over Time', fontsize=14)
    plt.xlabel('Validation Season', fontsize=11)
    plt.ylabel('Placement Difference (Actual vs Predicted Rank)', fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xticks(metrics_df['Year'])
    plt.savefig(os.path.join(out_dir, "rank_error_over_time.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    plt.figure(figsize=(10, 5))
    plt.plot(metrics_df['Year'], metrics_df['NDCG'], marker='o', label='NDCG', color='#1f77b4', lw=2)
    plt.plot(metrics_df['Year'], metrics_df['Spearman'], marker='s', label='Spearman Rank Corr', color='#ff7f0e', lw=2)
    plt.title('Baseline: Ranking Fidelity Over Time', fontsize=14)
    plt.xlabel('Validation Season', fontsize=11)
    plt.ylabel('Score (Higher is Better)', fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xticks(metrics_df['Year'])
    plt.legend()
    plt.savefig(os.path.join(out_dir, "accuracy_over_time.png"), dpi=300, bbox_inches='tight')
    plt.close()

def main():
    data_dir = "data"
    output_base = "model_outputs"
    baseline_base = os.path.join(output_base, "elo_baseline")
    os.makedirs(output_base, exist_ok=True)
    os.makedirs(baseline_base, exist_ok=True)
    
    print("=" * 80)
    print("Executing Pure Elo / Linear Baseline Walk-Forward Model")
    print("=" * 80)
    
    existing_seasons = sorted([int(f) for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f)) and f.isdigit()])
    
    all_data_dict = {}
    master_df = pd.DataFrame()
    yearly_metrics = []
    
    print("\n[Phase 1] Feature Extraction")
    for season in existing_seasons:
        df_season = build_team_features(data_dir, season, output_base)
        if not df_season.empty and len(df_season) >= 10:
            all_data_dict[season] = df_season
            master_df = pd.concat([master_df, df_season], ignore_index=True)
            
    if master_df.empty:
        print("\n[Fatal Error] No team data available.")
        return

    master_df = master_df.fillna(0)
    aligned_cols = master_df.columns.tolist()
    for season in all_data_dict:
        all_data_dict[season] = all_data_dict[season].reindex(columns=aligned_cols, fill_value=0)

    available_seasons = sorted(list(all_data_dict.keys()))
    
    print("\n[Phase 2] Executing Baseline Walk-Forward Validation")
    for i in range(1, len(available_seasons)):
        val_yr = available_seasons[i]
        train_years = available_seasons[:i]
        
        run_name = f"Train_{train_years[0]}_to_{train_years[-1]}_Val_{val_yr}"
        out_dir = os.path.join(baseline_base, run_name)
        os.makedirs(out_dir, exist_ok=True)
        
        print("\n" + "-" * 75)
        print(f"[BASELINE EXECUTION] {run_name}")
        
        df_train = pd.concat([all_data_dict[yr] for yr in train_years], ignore_index=True)
        df_val = all_data_dict[val_yr]
        
        train_weights = compute_recency_weights(df_train['season'], target_season=val_yr, half_life=3.0)
        
        # Dynamically identify the best single ranking feature available
        potential_elo_cols = [c for c in df_train.columns if 'elo' in c.lower() or 'netrating' in c.lower() or 'adj_em' in c.lower()]
        if potential_elo_cols:
            target_feature = potential_elo_cols[0]
            print(f"  [Feature Sync] Using single heuristic column: {target_feature}")
        else:
            # Fallback to point differential or simple score if Elo is not present
            target_feature = [c for c in df_train.columns if 'score' in c.lower()][0]
            print(f"  [Feature Sync] Elo not found. Falling back to: {target_feature}")
            
        X_train = df_train[[target_feature]]
        y_train = df_train['target_wins']
        X_val = df_val[[target_feature]]
        y_val = df_val['target_wins']
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        print(f"  [Model] Fitting Ridge Regression mapping {target_feature} -> Target Wins...")
        model = Ridge(alpha=1.0, random_state=42)
        model.fit(X_train_scaled, y_train, sample_weight=train_weights)
        
        preds = model.predict(X_val_scaled)
        
        mae = mean_absolute_error(y_val, preds)
        mse = mean_squared_error(y_val, preds)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_val, preds)
        
        ndcg_val = ndcg_score([y_val.values], [preds]) if len(y_val) > 1 else 0
        spearman_corr, _ = spearmanr(y_val, preds)
        
        actual_rank = y_val.rank(method='average', ascending=False).values
        pred_rank = pd.Series(preds).rank(method='average', ascending=False).values
        mare = np.mean(np.abs(actual_rank - pred_rank))
        
        print(f"  [Metrics] {val_yr} Holdout -> MAE: {mae:.3f} | RMSE: {rmse:.3f}")
        print(f"  [Metrics] Ranking -> NDCG: {ndcg_val:.4f} | Rank Corr: {spearman_corr:.4f} | MARE: {mare:.2f} positions")
        
        yearly_metrics.append({
            'Year': val_yr,
            'NDCG': ndcg_val,
            'Spearman': spearman_corr,
            'MARE': mare
        })
        
        with open(os.path.join(out_dir, "training_summary.log"), "w") as f:
            f.write("=" * 60 + "\n")
            f.write(f"BASELINE EXECUTION LOG: {run_name}\n")
            f.write("=" * 60 + "\n")
            f.write(f"Heuristic Feature Used: {target_feature}\n")
            f.write(f"Validation Target Year: {val_yr}\n")
            f.write(f"Training Observations: {len(X_train)}\n")
            f.write("\n--- PERFORMANCE METRICS ---\n")
            f.write(f"Validation Mean Absolute Error (MAE): {mae:.4f} wins\n")
            f.write(f"Validation Root Mean Squared Error (RMSE): {rmse:.4f}\n")
            f.write(f"Validation R-squared (R2): {r2:.4f}\n")
            f.write(f"Validation NDCG: {ndcg_val:.4f}\n")
            f.write(f"Validation Spearman Rank Correlation: {spearman_corr:.4f}\n")
            f.write(f"Validation Mean Absolute Rank Error: {mare:.4f} positions\n")
            
    if yearly_metrics:
        df_metrics = pd.DataFrame(yearly_metrics)
        plot_accuracy_over_time(df_metrics, baseline_base)
        
    print("\n" + "=" * 80)
    print(f"[COMPLETE] Baseline Execution Finished. Check outputs to establish your 'Mendoza Line'.")
    print(f"Directory: {os.path.abspath(baseline_base)}")
    print("=" * 80)

if __name__ == "__main__":
    main()