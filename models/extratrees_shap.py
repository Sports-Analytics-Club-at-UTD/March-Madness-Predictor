import os
import time
import requests
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import ExtraTreesRegressor
import shap
import optuna
import umap.umap_ as umap

# --- MATPLOTLIB BACKEND FIX ---
import matplotlib
matplotlib.use('Agg')
# ------------------------------

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, ndcg_score
from scipy.stats import spearmanr
from dotenv import load_dotenv
import warnings

warnings.filterwarnings('ignore')
load_dotenv()

API_KEY = os.getenv("CBBD_API_KEY")
BASE_URL = "https://api.collegebasketballdata.com"
HEADERS = {"Authorization": f"Bearer {API_KEY}"} if API_KEY else {}

def fetch_tournament_wins(season, output_dir):
    cache_dir = os.path.join(output_dir, "tournament_wins")
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"tournament_wins_{season}.csv")
    if os.path.exists(cache_file):
        df_wins = pd.read_csv(cache_file)
        return dict(zip(df_wins['team'], df_wins['wins']))
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
    season_dir = os.path.join(data_dir, str(season))
    if not os.path.exists(season_dir): return pd.DataFrame()
    tourney_wins = fetch_tournament_wins(season, output_dir)
    teams = sorted([d for d in os.listdir(season_dir) if os.path.isdir(os.path.join(season_dir, d))])
    if len(teams) == 0: return pd.DataFrame()
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
        actual_team_name = df_g['team'].iloc[0]
        features = {'team': actual_team_name, 'season': int(season), 'target_wins': tourney_wins.get(actual_team_name, 0)}
        df_reg = df_g[df_g['seasonType'] == 'regular']
        if df_reg.empty: df_reg = df_g 
        num_cols = df_reg.select_dtypes(include=[np.number]).columns
        for c in num_cols:
            if c not in skip_cols and not c.endswith('EloEnd'):
                features[f"game_mean_{c}"] = df_reg[c].mean()
                features[f"game_std_{c}"] = df_reg[c].std()
        df_p = pd.read_csv(player_file)
        if not df_p.empty and 'minutes' in df_p.columns:
            top5 = df_p.nlargest(5, 'minutes')
            p_num_cols = top5.select_dtypes(include=[np.number]).columns
            for c in p_num_cols:
                if c not in skip_cols and 'Id' not in c:
                    features[f"starters_mean_{c}"] = top5[c].mean()
                    features[f"roster_max_{c}"] = df_p[c].max()
        engineered_rows.append(features)
    df_engineered = pd.DataFrame(engineered_rows).fillna(0)
    if df_engineered.empty: return df_engineered
    return df_engineered.loc[:, (df_engineered != 0).any(axis=0)]

def compute_recency_weights(seasons_series, target_season, half_life=3.0):
    delta = target_season - seasons_series.astype(float)
    weights = np.power(0.5, delta / half_life)
    return (weights / weights.mean()).values

def optimize_extra_trees(X_train, y_train, sample_weights=None):
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 250),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 8),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', 1.0]),
            'random_state': 42,
            'n_jobs': 1
        }
        kf = KFold(n_splits=3, shuffle=True, random_state=42)
        losses = []
        for tr_idx, val_idx in kf.split(X_train):
            X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[tr_idx], y_train.iloc[val_idx]
            w_tr = sample_weights[tr_idx] if sample_weights is not None else None
            model = ExtraTreesRegressor(**params)
            model.fit(X_tr, y_tr, sample_weight=w_tr)
            preds = model.predict(X_val)
            losses.append(mean_squared_error(y_val, preds))
        return np.mean(losses)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=15)
    return study.best_params

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    data_dir = os.path.join(project_root, "data")
    output_base = os.path.join(project_root, "model_outputs")
    et_base = os.path.join(output_base, "extra_trees")
    
    os.makedirs(output_base, exist_ok=True)
    os.makedirs(et_base, exist_ok=True)
    
    if not os.path.exists(data_dir): return
    existing_seasons = sorted([
        int(f) for f in os.listdir(data_dir) 
        if os.path.isdir(os.path.join(data_dir, f)) and f.isdigit()
    ])
    
    print("=" * 80)
    print("Expanding Window Walk-Forward Model: Extremely Randomized Trees (ExtraTrees)")
    print(f"Discovered Seasons: {existing_seasons}")
    print("=" * 80)
    
    all_data_dict = {}
    master_df = pd.DataFrame()
    yearly_metrics = []
    
    for season in existing_seasons:
        df_season = build_team_features(data_dir, season, output_base)
        if not df_season.empty and len(df_season) >= 10:
            all_data_dict[season] = df_season
            master_df = pd.concat([master_df, df_season], ignore_index=True)
            
    if master_df.empty: return
    master_df = master_df.fillna(0)
    aligned_cols = master_df.columns.tolist()
    for season in all_data_dict:
        all_data_dict[season] = all_data_dict[season].reindex(columns=aligned_cols, fill_value=0)
        
    available_seasons = sorted(list(all_data_dict.keys()))
    
    for i in range(1, len(available_seasons)):
        val_yr = available_seasons[i]
        train_years = available_seasons[:i]
        run_name = f"Train_{train_years[0]}_to_{train_years[-1]}_Val_{val_yr}"
        out_dir = os.path.join(et_base, run_name)
        os.makedirs(out_dir, exist_ok=True)
        
        df_train = pd.concat([all_data_dict[yr] for yr in train_years], ignore_index=True)
        df_val = all_data_dict[val_yr]
        train_weights = compute_recency_weights(df_train['season'], target_season=val_yr, half_life=3.0)
        
        drop_cols = ['target_wins', 'team', 'season']
        X_train = df_train.drop(columns=drop_cols, errors='ignore')
        y_train = df_train['target_wins']
        X_val = df_val.drop(columns=drop_cols, errors='ignore')
        y_val = df_val['target_wins']
        
        best_params = optimize_extra_trees(X_train, y_train, sample_weights=train_weights)
        model = ExtraTreesRegressor(**best_params, random_state=42, n_jobs=1)
        model.fit(X_train, y_train, sample_weight=train_weights)
        
        preds = model.predict(X_val)
        actuals = y_val.values
        
        val_mse = mean_squared_error(actuals, preds)
        val_mae = mean_absolute_error(actuals, preds)
        val_r2 = r2_score(actuals, preds)
        val_ndcg = ndcg_score([actuals], [preds])
        val_spearman, _ = spearmanr(actuals, preds)
        if np.isnan(val_spearman): val_spearman = 0.0
        
        actual_ranks = pd.Series(actuals).rank(ascending=False, method='average')
        pred_ranks = pd.Series(preds).rank(ascending=False, method='average')
        val_mare = np.mean(np.abs(actual_ranks - pred_ranks))
        
        yearly_metrics.append({
            'Year': val_yr, 'NDCG': val_ndcg, 'Spearman': val_spearman, 'MARE': val_mare
        })
        
        with open(os.path.join(out_dir, "training_summary.log"), 'w') as f:
            f.write(f"Target Year: {val_yr}\n")
            f.write(f"MSE: {val_mse:.4f}\n")
            f.write(f"MAE: {val_mae:.4f}\n")
            f.write(f"R2: {val_r2:.4f}\n")
            f.write(f"NDCG: {val_ndcg:.4f}\n")
            f.write(f"Spearman: {val_spearman:.4f}\n")
            f.write(f"MARE: {val_mare:.4f}\n")
            
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_val)
        vals = np.abs(shap_values).mean(0)
        fi = pd.DataFrame(list(zip(X_val.columns, vals)), columns=['feature', 'shap_importance'])
        fi.sort_values(by=['shap_importance'], ascending=False, inplace=True)
        fi.to_csv(os.path.join(out_dir, "shap_feature_importance.csv"), index=False)
        print(f"  [ExtraTrees] Val Year {val_yr} -> NDCG: {val_ndcg:.4f} | Spearman: {val_spearman:.4f} | MARE: {val_mare:.2f}")

    all_out = os.path.join(et_base, "all_years")
    os.makedirs(all_out, exist_ok=True)
    latest_season = master_df['season'].max()
    all_weights = compute_recency_weights(master_df['season'], target_season=latest_season, half_life=3.0)
    X_all = master_df.drop(columns=['target_wins', 'team', 'season'], errors='ignore')
    y_all = master_df['target_wins']
    best_params_all = optimize_extra_trees(X_all, y_all, sample_weights=all_weights)
    master_model = ExtraTreesRegressor(**best_params_all, random_state=42, n_jobs=1)
    master_model.fit(X_all, y_all, sample_weight=all_weights)
    joblib.dump(master_model, os.path.join(all_out, "extra_trees_master_model.joblib"))
    
    explainer_all = shap.TreeExplainer(master_model)
    shap_vals_all = explainer_all.shap_values(X_all)
    vals_all = np.abs(shap_vals_all).mean(0)
    fi_all = pd.DataFrame(list(zip(X_all.columns, vals_all)), columns=['feature', 'shap_importance'])
    fi_all.sort_values(by=['shap_importance'], ascending=False, inplace=True)
    fi_all.to_csv(os.path.join(all_out, "shap_feature_importance.csv"), index=False)
    print("\n[COMPLETE] ExtraTrees Modeling Pipeline finished successfully.")

if __name__ == "__main__":
    main()
