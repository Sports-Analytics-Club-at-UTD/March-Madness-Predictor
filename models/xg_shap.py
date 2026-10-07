import os
import time
import requests
import json
import numpy as np
import pandas as pd
import xgboost as xgb
import shap
import optuna
import umap.umap_ as umap
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import ndcg_score
from scipy.stats import spearmanr
from dotenv import load_dotenv
import warnings

warnings.filterwarnings('ignore')
load_dotenv()

API_KEY = os.getenv("CBBD_API_KEY")
BASE_URL = "https://api.collegebasketballdata.com"
HEADERS = {"Authorization": f"Bearer {API_KEY}"}

def fetch_tournament_wins(season, output_dir):
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
        print(f"      [API] Successfully retrieved {len(games)} postseason games.")
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

    print(f"      [API] Parsed target wins for {len(wins)} teams. Saving to cache...")
    df_wins = pd.DataFrame(list(wins.items()), columns=['team', 'wins'])
    df_wins.to_csv(cache_file, index=False)
    return wins

def build_team_features(data_dir, season, output_dir):
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
    
    cols_processed, p_cols_processed = 0, 0
    
    for team_safe in teams:
        team_dir = os.path.join(season_dir, team_safe)
        game_file = os.path.join(team_dir, f"{team_safe}_game_stats.csv")
        player_file = os.path.join(team_dir, f"{team_safe}_players_season_stats.csv")
        
        if not os.path.exists(game_file) or not os.path.exists(player_file):
            continue
            
        df_g = pd.read_csv(game_file)
        if df_g.empty: 
            continue
            
        actual_team_name = df_g['team'].iloc[0]
        features = {'team': actual_team_name, 'season': int(season), 'target_wins': tourney_wins.get(actual_team_name, 0)}
        
        df_reg = df_g[df_g['seasonType'] == 'regular']
        if df_reg.empty: df_reg = df_g 
            
        num_cols = df_reg.select_dtypes(include=[np.number]).columns
        cols_processed = 0
        for c in num_cols:
            if c not in skip_cols and not c.endswith('EloEnd'):
                features[f"game_mean_{c}"] = df_reg[c].mean()
                features[f"game_std_{c}"] = df_reg[c].std()
                cols_processed += 2
                
        df_p = pd.read_csv(player_file)
        p_cols_processed = 0
        if not df_p.empty and 'minutes' in df_p.columns:
            top5 = df_p.nlargest(5, 'minutes')
            for c in top5.select_dtypes(include=[np.number]).columns:
                if c not in skip_cols and 'Id' not in c:
                    features[f"starters_mean_{c}"] = top5[c].mean()
                    features[f"roster_max_{c}"] = df_p[c].max()
                    p_cols_processed += 2
                    
        engineered_rows.append(features)
        
    print(f"    [Data] Extracted {cols_processed} game features and {p_cols_processed} player features per team.")
    df_engineered = pd.DataFrame(engineered_rows).fillna(0)
    
    if df_engineered.empty:
        return df_engineered
        
    df_engineered = df_engineered.loc[:, (df_engineered != 0).any(axis=0)]
    return df_engineered

def compute_recency_weights(seasons_series, target_season, half_life=3.0):
    print(f"  [Weights] Calculating exponential decay weights (Target: {target_season}, Half-life: {half_life} yrs)...")
    delta = target_season - seasons_series.astype(float)
    weights = np.power(0.5, delta / half_life)
    normalized_weights = weights / weights.mean()
    return normalized_weights.values

def generate_eda(df, out_dir, title_prefix=""):
    os.makedirs(out_dir, exist_ok=True)
    numeric_df = df.select_dtypes(include=[np.number])
    
    plt.figure(figsize=(7, 4))
    sns.countplot(x='target_wins', data=df, palette='magma')
    plt.title(f'{title_prefix} Target Distribution (Tournament Wins)')
    plt.savefig(os.path.join(out_dir, "target_distribution.png"), bbox_inches='tight', dpi=200)
    plt.close()
    
    correlations = numeric_df.corr()['target_wins'].abs().sort_values(ascending=False)
    top_features = correlations.index[1:6].tolist()
    
    if top_features:
        num_features = len(top_features)
        fig, axes = plt.subplots(num_features, num_features, figsize=(15, 15))
        plt.subplots_adjust(hspace=0.4, wspace=0.4)
        for i in range(num_features):
            for j in range(num_features):
                ax = axes[i, j]
                if i == j:
                    ax.hist(df[top_features[i]], bins=20, color='royalblue', edgecolor='black', alpha=0.7)
                    ax.set_title(top_features[i][:15], fontsize=8)
                else:
                    scatter = ax.scatter(df[top_features[j]], df[top_features[i]], 
                                         c=df['target_wins'], cmap='viridis', s=10, alpha=0.6)
                if i == num_features - 1: ax.set_xlabel(top_features[j][:15], fontsize=8)
                if j == 0: ax.set_ylabel(top_features[i][:15], fontsize=8)
                    
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        fig.colorbar(scatter, cax=cbar_ax, label='Target Wins')
        fig.suptitle(f"{title_prefix} Top Correlated Features Pairwise Scatter", fontsize=16)
        plt.savefig(os.path.join(out_dir, "top_features_scatter.png"), bbox_inches='tight', dpi=200)
        plt.close()
        
    top_15 = correlations.index[0:16].tolist()
    plt.figure(figsize=(12, 10))
    sns.heatmap(df[top_15].corr(), annot=True, cmap='vlag', fmt=".2f", cbar=True)
    plt.title(f"{title_prefix} Correlation Matrix (Top 15 Features)")
    plt.savefig(os.path.join(out_dir, "correlation_matrix.png"), bbox_inches='tight', dpi=200)
    plt.close()

def plot_learning_curves(evals_result, out_dir, run_name):
    print("  [Plot] Generating Training vs Validation Learning Curves (NDCG)...")
    epochs = len(evals_result['validation_0']['ndcg'])
    x_axis = range(1, epochs + 1)
    
    plt.figure(figsize=(8, 5))
    train_ndcg = evals_result['validation_0']['ndcg']
    val_ndcg = evals_result['validation_1']['ndcg']
    best_iter = int(np.argmax(val_ndcg))
    
    plt.plot(x_axis, train_ndcg, label='Train NDCG', color='#1f77b4', lw=2)
    plt.plot(x_axis, val_ndcg, label='Validation NDCG', color='#d62728', lw=2)
    plt.axvline(best_iter + 1, linestyle='--', color='gray', alpha=0.7, label=f'Max Val NDCG (Iter {best_iter + 1})')
    plt.title(f'{run_name} - Learning Curve (NDCG)')
    plt.xlabel('Boosting Iterations')
    plt.ylabel('NDCG Score (Higher is Better)')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "learning_curves.png"), dpi=300, bbox_inches='tight')
    plt.close()

def plot_accuracy_over_time(metrics_df, out_dir):
    print("\n  [Plot] Rendering Model Global Accuracy Over Time...")
    
    plt.figure(figsize=(10, 5))
    plt.plot(metrics_df['Year'], metrics_df['MARE'], marker='^', color='#d62728', lw=2.5, markersize=8)
    plt.title('Mean Absolute Rank Error (MARE) Over Time', fontsize=14)
    plt.xlabel('Validation Season', fontsize=11)
    plt.ylabel('Placement Difference (Actual vs Predicted Rank)', fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xticks(metrics_df['Year'])
    
    mare_path = os.path.join(out_dir, "rank_error_over_time.png")
    plt.savefig(mare_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    plt.figure(figsize=(10, 5))
    plt.plot(metrics_df['Year'], metrics_df['NDCG'], marker='o', label='NDCG', color='#1f77b4', lw=2)
    plt.plot(metrics_df['Year'], metrics_df['Spearman'], marker='s', label='Spearman Rank Corr', color='#ff7f0e', lw=2)
    plt.title('Model Ranking Fidelity Over Time', fontsize=14)
    plt.xlabel('Validation Season', fontsize=11)
    plt.ylabel('Score (Higher is Better)', fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xticks(metrics_df['Year'])
    plt.legend()
    
    acc_path = os.path.join(out_dir, "accuracy_over_time.png")
    plt.savefig(acc_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  [Plot] MARE Error Plot saved to: {mare_path}")
    print(f"  [Plot] Rank Fidelity Plot saved to: {acc_path}")

def optimize_xgboost(df_train, val_yr):
    print("  [Optuna] Initializing XGBoost Ranker Search Space...")
    
    def objective(trial):
        params = {
            'objective': 'rank:pairwise',
            'eval_metric': 'ndcg',
            'max_depth': trial.suggest_int('max_depth', 2, 6),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.12, log=True),
            'n_estimators': trial.suggest_int('n_estimators', 60, 250),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
            'random_state': 42
        }
        
        kf = KFold(n_splits=3, shuffle=True, random_state=42)
        ndcg_scores = []
        
        for tr_idx, val_idx in kf.split(df_train):
            train_fold = df_train.iloc[tr_idx].sort_values('season')
            val_fold = df_train.iloc[val_idx].sort_values('season')
            
            qid_tr = train_fold['season']
            y_tr = train_fold['target_wins']
            X_tr = train_fold.drop(columns=['target_wins', 'team', 'season'], errors='ignore')
            w_tr_groups = compute_recency_weights(pd.Series(qid_tr.unique()), target_season=val_yr, half_life=3.0)
            
            qid_val = val_fold['season']
            y_val = val_fold['target_wins']
            X_val = val_fold.drop(columns=['target_wins', 'team', 'season'], errors='ignore')
            
            model = xgb.XGBRanker(**params)
            model.fit(
                X_tr, y_tr, qid=qid_tr, sample_weight=w_tr_groups,
                eval_set=[(X_val, y_val)], eval_qid=[qid_val], verbose=False
            )
            
            preds = model.predict(X_val)
            
            fold_ndcgs = []
            for season_grp in qid_val.unique():
                mask = qid_val == season_grp
                if mask.sum() > 1:
                    fold_ndcgs.append(ndcg_score([y_val[mask].values], [preds[mask]]))
            if fold_ndcgs:
                ndcg_scores.append(np.mean(fold_ndcgs))
            
        return np.mean(ndcg_scores) if ndcg_scores else 0

    study = optuna.create_study(direction='maximize')
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(objective, n_trials=20)
    print(f"  [Optuna] Search Complete. Best CV NDCG: {study.best_value:.4f}")
    return study.best_params

def generate_3d_umap(df, out_dir):
    print("  [UMAP] Initiating 3D Projection Engine...")
    X = df.drop(columns=['team', 'season', 'target_wins'], errors='ignore').select_dtypes(include=[np.number])
    X_scaled = StandardScaler().fit_transform(X)
    
    reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
    embedding = reducer.fit_transform(X_scaled)
    
    le = LabelEncoder()
    team_colors = le.fit_transform(df['team'])
    
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.scatter(embedding[:, 0], embedding[:, 1], df['target_wins'], c=team_colors, cmap='nipy_spectral', alpha=0.75, s=40, edgecolors='black', linewidth=0.4)
    ax.set_xlabel('UMAP 1 (Play Style/Efficiency)')
    ax.set_ylabel('UMAP 2 (Physicality/Turnover Profile)')
    ax.set_zlabel('Actual Tournament Wins')
    plt.title('3D UMAP Manifold: Team Signatures Elevated by Postseason Success')
    plt.savefig(os.path.join(out_dir, "3d_umap_overall.png"), dpi=300, bbox_inches='tight')
    plt.close()

def main():
    data_dir = "data"
    output_base = "model_outputs"
    xgboost_base = os.path.join(output_base, "xgboost")
    os.makedirs(output_base, exist_ok=True)
    os.makedirs(xgboost_base, exist_ok=True)
    
    print(f"[System] Ensured output directories exist:")
    print(f"         Root: {os.path.abspath(output_base)}")
    print(f"         Models: {os.path.abspath(xgboost_base)}")
    
    existing_seasons = sorted([int(f) for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f)) and f.isdigit()])
    print("=" * 80)
    print("Expanding Window Walk-Forward Model (Rank:Pairwise) with Deep Diagnostics")
    print(f"Discovered Seasons in Database: {existing_seasons}")
    print("=" * 80)
    
    all_data_dict = {}
    master_df = pd.DataFrame()
    yearly_metrics = []
    
    print("\n[Phase 1] Feature Extraction & Alignment")
    for season in existing_seasons:
        print(f"\n  >>> Initializing feature build for season: {season} <<<")
        df_season = build_team_features(data_dir, season, output_base)
        if not df_season.empty and len(df_season) >= 10:
            all_data_dict[season] = df_season
            master_df = pd.concat([master_df, df_season], ignore_index=True)
            print(f"  [Success] Extracted {len(df_season)} teams with {df_season.shape[1]} features for {season}.")
            
    if master_df.empty:
        print("\n[Fatal Error] No team data available. Pipeline terminated.")
        return

    print("\n[Data] Aligning feature space across all historical periods...")
    master_df = master_df.fillna(0)
    aligned_cols = master_df.columns.tolist()
    
    for season in all_data_dict:
        all_data_dict[season] = all_data_dict[season].reindex(columns=aligned_cols, fill_value=0)

    available_seasons = sorted(list(all_data_dict.keys()))
    
    print("\n[Phase 2] Executing Out-of-Time Walk-Forward Validation")
    for i in range(1, len(available_seasons)):
        val_yr = available_seasons[i]
        train_years = available_seasons[:i]
        
        run_name = f"Train_{train_years[0]}_to_{train_years[-1]}_Val_{val_yr}"
        out_dir = os.path.join(xgboost_base, run_name)
        os.makedirs(out_dir, exist_ok=True)
        
        print("\n" + "=" * 75)
        print(f"[EXECUTION] {run_name}")
        print("=" * 75)
        
        df_train = pd.concat([all_data_dict[yr] for yr in train_years], ignore_index=True)
        df_val = all_data_dict[val_yr]
        
        df_train = df_train.sort_values('season')
        df_val = df_val.sort_values('season')
        
        generate_eda(df_train, out_dir, title_prefix=f"{train_years[0]}-{train_years[-1]}")
        
        drop_cols = ['target_wins', 'team', 'season']
        qid_train = df_train['season']
        y_train = df_train['target_wins']
        X_train = df_train.drop(columns=drop_cols, errors='ignore')
        
        qid_val = df_val['season']
        y_val = df_val['target_wins']
        X_val = df_val.drop(columns=drop_cols, errors='ignore')
        
        group_weights = compute_recency_weights(pd.Series(qid_train.unique()), target_season=val_yr, half_life=3.0)
        best_params = optimize_xgboost(df_train, val_yr)
        
        print("\n  [XGBRanker] Fitting Regressor with Dual Evaluation Sets...")
        model = xgb.XGBRanker(**best_params, objective='rank:pairwise', eval_metric='ndcg', random_state=42)
        
        eval_set = [(X_train, y_train), (X_val, y_val)]
        model.fit(
            X_train, y_train, qid=qid_train,
            sample_weight=group_weights,
            eval_set=eval_set, eval_qid=[qid_train, qid_val],
            verbose=False
        )
        
        evals_result = model.evals_result()
        
        df_iter_log = pd.DataFrame({
            'iteration': range(1, len(evals_result['validation_0']['ndcg']) + 1),
            'train_ndcg': evals_result['validation_0']['ndcg'],
            'val_ndcg': evals_result['validation_1']['ndcg']
        })
        df_iter_log.to_csv(os.path.join(out_dir, "training_iteration_log.csv"), index=False)
        plot_learning_curves(evals_result, out_dir, run_name)
        
        preds = model.predict(X_val)
        ndcg_val = ndcg_score([y_val.values], [preds]) if len(y_val) > 1 else 0
        spearman_corr, _ = spearmanr(y_val, preds)
        
        actual_rank = y_val.rank(method='average', ascending=False).values
        pred_rank = pd.Series(preds).rank(method='average', ascending=False).values
        mare = np.mean(np.abs(actual_rank - pred_rank))
        
        print(f"  [Metrics] Ranking -> NDCG: {ndcg_val:.4f} | Rank Corr: {spearman_corr:.4f} | MARE: {mare:.2f} positions")
        
        yearly_metrics.append({
            'Year': val_yr,
            'NDCG': ndcg_val,
            'Spearman': spearman_corr,
            'MARE': mare
        })
        
        summary_log_path = os.path.join(out_dir, "training_summary.log")
        with open(summary_log_path, "w") as f:
            f.write("=" * 60 + "\n")
            f.write(f"MODEL EXECUTION LOG (RANK:PAIRWISE): {run_name}\n")
            f.write("=" * 60 + "\n")
            f.write(f"Validation Target Year: {val_yr}\n")
            f.write(f"Training Observations: {len(X_train)}\n")
            f.write(f"Validation Observations: {len(X_val)}\n")
            f.write("--- HYPERPARAMETERS (OPTUNA) ---\n")
            for param, val in best_params.items():
                f.write(f"{param}: {val}\n")
            f.write("\n--- PERFORMANCE METRICS ---\n")
            f.write(f"Validation Normalized Discounted Cumulative Gain (NDCG): {ndcg_val:.4f}\n")
            f.write(f"Validation Spearman Rank Correlation: {spearman_corr:.4f}\n")
            f.write(f"Validation Mean Absolute Rank Error (MARE): {mare:.4f} positions\n")
            f.write(f"Optimal Iteration (Max Val NDCG): {int(np.argmax(evals_result['validation_1']['ndcg']) + 1)}\n")
            
        json_path = os.path.join(out_dir, "xgboost_model.json")
        model.save_model(json_path)
        
        # --- AGGRESSIVE SHAP COMPATIBILITY PATCH FOR XGBOOST 2.1+ ---
        import re
        with open(json_path, 'r') as f:
            model_data = json.load(f)
            
        base_score_str = model_data.get('learner', {}).get('learner_model_param', {}).get('base_score', '0.5')
        
        # Robust extraction of the float value, handling brackets and scientific notation
        match = re.search(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', str(base_score_str))
        clean_base_score = float(match.group()) if match else 0.5
        
        patched_booster = xgb.Booster()
        patched_booster.load_model(json_path)
        patched_booster.set_param({"base_score": clean_base_score})
        
        print("\n  [Module] SHAP Attribution Initialization...")
        explainer = shap.TreeExplainer(patched_booster)
        # --------------------------------------------------
        
        print(f"  [SHAP] Processing {X_val.shape[0]} Validation Vectors...")
        shap_values = explainer.shap_values(X_val)
        
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_val, show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "shap_summary.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        vals = np.abs(shap_values).mean(0)
        fi = pd.DataFrame(list(zip(X_val.columns, vals)), columns=['feature', 'shap_importance'])
        fi.sort_values(by=['shap_importance'], ascending=False, inplace=True)
        fi.to_csv(os.path.join(out_dir, "shap_feature_importance.csv"), index=False)
        
    if yearly_metrics:
        df_metrics = pd.DataFrame(yearly_metrics)
        plot_accuracy_over_time(df_metrics, xgboost_base)
        
    print("\n" + "=" * 80)
    print("[Phase 3] Executing Master Aggregate Pipeline (All Historical Years Combined)")
    print("=" * 80)
    
    all_out = os.path.join(xgboost_base, "all_years")
    os.makedirs(all_out, exist_ok=True)
    
    generate_eda(master_df, all_out, title_prefix="All-Time")
    generate_3d_umap(master_df, all_out)
    
    latest_season = master_df['season'].max()
    master_df = master_df.sort_values('season') 
    
    qid_all = master_df['season']
    X_all = master_df.drop(columns=['target_wins', 'team', 'season'], errors='ignore')
    y_all = master_df['target_wins']
    
    split_idx = int(len(X_all) * 0.8)
    X_tr_all, X_val_all = X_all.iloc[:split_idx], X_all.iloc[split_idx:]
    y_tr_all, y_val_all = y_all.iloc[:split_idx], y_all.iloc[split_idx:]
    qid_tr_all, qid_val_all = qid_all.iloc[:split_idx], qid_all.iloc[split_idx:]
    
    w_tr_all_groups = compute_recency_weights(pd.Series(qid_tr_all.unique()), target_season=latest_season, half_life=3.0)
    best_params_all = optimize_xgboost(master_df.iloc[:split_idx], val_yr=latest_season)
    
    master_model = xgb.XGBRanker(**best_params_all, objective='rank:pairwise', eval_metric='ndcg', random_state=42)
    
    eval_set_all = [(X_tr_all, y_tr_all), (X_val_all, y_val_all)]
    master_model.fit(
        X_tr_all, y_tr_all, qid=qid_tr_all,
        sample_weight=w_tr_all_groups,
        eval_set=eval_set_all, eval_qid=[qid_tr_all, qid_val_all],
        verbose=False
    )
    
    evals_result_all = master_model.evals_result()
    df_all_iter_log = pd.DataFrame({
        'iteration': range(1, len(evals_result_all['validation_0']['ndcg']) + 1),
        'train_ndcg': evals_result_all['validation_0']['ndcg'],
        'val_ndcg': evals_result_all['validation_1']['ndcg']
    })
    df_all_iter_log.to_csv(os.path.join(all_out, "master_training_iteration_log.csv"), index=False)
    plot_learning_curves(evals_result_all, all_out, "Master Ranker (All Years)")
    
    all_group_weights = compute_recency_weights(pd.Series(qid_all.unique()), target_season=latest_season, half_life=3.0)
    master_model.fit(X_all, y_all, qid=qid_all, sample_weight=all_group_weights, verbose=False)
    master_json_path = os.path.join(all_out, "xgboost_master_model.json")
    master_model.save_model(master_json_path)
    
    # --- AGGRESSIVE SHAP COMPATIBILITY PATCH FOR XGBOOST 2.1+ ---
    import re
    with open(master_json_path, 'r') as f:
        model_data = json.load(f)
        
    base_score_str = model_data.get('learner', {}).get('learner_model_param', {}).get('base_score', '0.5')
    
    # Robust extraction of the float value, handling brackets and scientific notation
    match = re.search(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', str(base_score_str))
    clean_base_score = float(match.group()) if match else 0.5
    
    patched_booster = xgb.Booster()
    patched_booster.load_model(master_json_path)
    patched_booster.set_param({"base_score": clean_base_score})
    
    print("\n  [Module] Global SHAP Initialization...")
    explainer_all = shap.TreeExplainer(patched_booster)
    # --------------------------------------------------
    
    shap_values_all = explainer_all.shap_values(X_all)
    
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values_all, X_all, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(all_out, "shap_summary_overall.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    vals_all = np.abs(shap_values_all).mean(0)
    fi_all = pd.DataFrame(list(zip(X_all.columns, vals_all)), columns=['feature', 'shap_importance'])
    fi_all.sort_values(by=['shap_importance'], ascending=False, inplace=True)
    fi_all.to_csv(os.path.join(all_out, "shap_feature_importance.csv"), index=False)
    
    print("\n" + "=" * 80)
    print(f"[COMPLETE] Entire Temporal Modeling Pipeline has resolved successfully.")
    print("=" * 80)

if __name__ == "__main__":
    main()