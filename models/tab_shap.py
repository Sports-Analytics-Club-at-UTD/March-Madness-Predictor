import os
import time
import requests
import numpy as np
import pandas as pd
import torch
from pytorch_tabnet.tab_model import TabNetRegressor
import shap
import optuna
import umap.umap_ as umap
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import KFold, train_test_split
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
    print(f"    [Data] Found {len(teams)} team folders for season {season}.")
    
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
    print(f"    [Data] Pre-variance shape: {df_engineered.shape}")
    
    if df_engineered.empty: return df_engineered
        
    df_engineered = df_engineered.loc[:, (df_engineered != 0).any(axis=0)]
    print(f"    [Data] Post-variance shape (zero-variance dropped): {df_engineered.shape}")
    return df_engineered

def compute_recency_weights(seasons_series, target_season, half_life=3.0):
    print(f"  [Weights] Calculating exponential decay weights (Target: {target_season}, Half-life: {half_life} yrs)...")
    delta = target_season - seasons_series.astype(float)
    weights = np.power(0.5, delta / half_life)
    prob_weights = weights / weights.sum()
    return prob_weights.values

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
    print("  [Plot] Generating Training vs Validation Learning Curves...")
    
    try:
        epochs = len(evals_result['train_rmse'])
        x_axis = range(1, epochs + 1)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        train_rmse = evals_result['train_rmse']
        val_rmse = evals_result['val_rmse']
        best_iter_rmse = int(np.argmin(val_rmse))
        
        ax1.plot(x_axis, train_rmse, label='Train RMSE', color='#1f77b4', lw=2)
        ax1.plot(x_axis, val_rmse, label='Validation RMSE', color='#d62728', lw=2)
        ax1.axvline(best_iter_rmse + 1, linestyle='--', color='gray', alpha=0.7, label=f'Min Val RMSE (Epoch {best_iter_rmse + 1})')
        ax1.set_title(f'{run_name} - Learning Curve (RMSE)')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('RMSE')
        ax1.grid(True, linestyle=':', alpha=0.6)
        ax1.legend()
        
        train_mae = evals_result['train_mae']
        val_mae = evals_result['val_mae']
        best_iter_mae = int(np.argmin(val_mae))
        
        ax2.plot(x_axis, train_mae, label='Train MAE', color='#2ca02c', lw=2)
        ax2.plot(x_axis, val_mae, label='Validation MAE', color='#ff7f0e', lw=2)
        ax2.axvline(best_iter_mae + 1, linestyle='--', color='gray', alpha=0.7, label=f'Min Val MAE (Epoch {best_iter_mae + 1})')
        ax2.set_title(f'{run_name} - Learning Curve (MAE)')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('MAE (Tournament Wins)')
        ax2.grid(True, linestyle=':', alpha=0.6)
        ax2.legend()
        
        plt.tight_layout()
        curve_path = os.path.join(out_dir, "learning_curves.png")
        plt.savefig(curve_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  [Plot] Saved: {curve_path}")
    except (KeyError, TypeError):
        print("  [Plot] Warning: Expected metric keys not found in TabNet history. Skipping plot.")

def plot_accuracy_over_time(metrics_df, out_dir):
    print("\n  [Plot] Rendering Model Global Accuracy Over Time...")
    plt.figure(figsize=(10, 5))
    plt.plot(metrics_df['Year'], metrics_df['MARE'], marker='^', color='#d62728', lw=2.5, markersize=8)
    plt.title('Mean Absolute Rank Error (MARE) Over Time', fontsize=14)
    plt.xlabel('Validation Season', fontsize=11)
    plt.ylabel('Placement Difference (Actual vs Predicted Rank)', fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xticks(metrics_df['Year'])
    plt.savefig(os.path.join(out_dir, "rank_error_over_time.png"), dpi=300, bbox_inches='tight')
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
    plt.savefig(os.path.join(out_dir, "accuracy_over_time.png"), dpi=300, bbox_inches='tight')
    plt.close()

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

def optimize_tabnet(X_train, y_train, prob_weights=None):
    print("  [Optuna] Initializing TabNet Hyperparameter Search Space...")
    
    def objective(trial):
        n_da = trial.suggest_int('n_da', 8, 64, step=8)
        n_steps = trial.suggest_int('n_steps', 2, 7)
        gamma = trial.suggest_float('gamma', 1.0, 2.0)
        n_shared = trial.suggest_int('n_shared', 1, 4)
        n_independent = trial.suggest_int('n_independent', 1, 4)
        lambda_sparse = trial.suggest_float('lambda_sparse', 1e-5, 1e-2, log=True)
        lr = trial.suggest_float('lr', 1e-3, 5e-2, log=True)
        
        if prob_weights is not None:
            sample_idx = np.random.choice(len(X_train), size=len(X_train), replace=True, p=prob_weights)
            X_tr_fold = X_train[sample_idx]
            y_tr_fold = y_train[sample_idx]
        else:
            X_tr_fold = X_train
            y_tr_fold = y_train
            
        kf = KFold(n_splits=3, shuffle=True, random_state=42)
        losses = []
        
        for tr_idx, val_idx in kf.split(X_tr_fold):
            X_tr, X_val = X_tr_fold[tr_idx], X_tr_fold[val_idx]
            y_tr, y_val = y_tr_fold[tr_idx], y_tr_fold[val_idx]
            
            model = TabNetRegressor(
                n_d=n_da, n_a=n_da, n_steps=n_steps, gamma=gamma,
                n_shared=n_shared, n_independent=n_independent,
                lambda_sparse=lambda_sparse, 
                optimizer_fn=torch.optim.Adam,
                optimizer_params=dict(lr=lr),
                verbose=0
            )
            
            model.fit(
                X_train=X_tr, y_train=y_tr.reshape(-1, 1),
                eval_set=[(X_val, y_val.reshape(-1, 1))],
                eval_metric=['mse'],
                max_epochs=50, patience=10, batch_size=64, virtual_batch_size=32
            )
            
            preds = model.predict(X_val)
            losses.append(mean_squared_error(y_val, preds))
            
        return np.mean(losses)

    study = optuna.create_study(direction='minimize')
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(objective, n_trials=15)
    print(f"  [Optuna] Search Complete. Best CV MSE: {study.best_value:.4f}")
    return study.best_params

def train_tabnet_with_evals(best_params, X_train, y_train, X_val, y_val, prob_weights=None):
    if prob_weights is not None:
        sample_idx = np.random.choice(len(X_train), size=len(X_train), replace=True, p=prob_weights)
        X_tr = X_train[sample_idx]
        y_tr = y_train[sample_idx]
    else:
        X_tr = X_train
        y_tr = y_train
        
    n_da = best_params['n_da']
    
    model = TabNetRegressor(
        n_d=n_da, n_a=n_da, n_steps=best_params['n_steps'], gamma=best_params['gamma'],
        n_shared=best_params['n_shared'], n_independent=best_params['n_independent'],
        lambda_sparse=best_params['lambda_sparse'], 
        optimizer_fn=torch.optim.Adam,
        optimizer_params=dict(lr=best_params['lr']),
        verbose=0
    )
    
    model.fit(
        X_train=X_tr, y_train=y_tr.reshape(-1, 1),
        eval_set=[(X_tr, y_tr.reshape(-1, 1)), (X_val, y_val.reshape(-1, 1))],
        eval_name=['train', 'val'],
        eval_metric=['rmse', 'mae'],
        max_epochs=100, patience=20,
        batch_size=64, virtual_batch_size=32
    )
    
    return model, model.history

def compute_shap(model, X_train_scaled, X_val_scaled, feature_names, out_dir):
    print("\n  [Module] SHAP Attribution Initialization...")
    print("  [SHAP] Initializing K-Means Background Dataset (k=25) for KernelExplainer...")
    
    def predict_fn(x):
        return model.predict(x).flatten()
            
    background = shap.kmeans(X_train_scaled, 25)
    explainer = shap.KernelExplainer(predict_fn, background)
    
    print(f"  [SHAP] Processing {X_val_scaled.shape[0]} Validation Vectors...")
    shap_values = explainer.shap_values(X_val_scaled, nsamples=100)
    
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_val_scaled, feature_names=feature_names, show=False)
    plt.tight_layout()
    shap_plot_path = os.path.join(out_dir, "shap_summary.png")
    plt.savefig(shap_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [SHAP] Global Summary Plot rendered to {shap_plot_path}")
    
    vals = np.abs(shap_values).mean(0)
    fi = pd.DataFrame(list(zip(feature_names, vals)), columns=['feature', 'shap_importance'])
    fi.sort_values(by=['shap_importance'], ascending=False, inplace=True)
    fi_csv = os.path.join(out_dir, "shap_feature_importance.csv")
    fi.to_csv(fi_csv, index=False)

def main():
    data_dir = "data"
    output_base = "model_outputs"
    tabnet_base = os.path.join(output_base, "tabnet")
    os.makedirs(output_base, exist_ok=True)
    os.makedirs(tabnet_base, exist_ok=True)
    
    print(f"[System] Ensured output directories exist:")
    print(f"         Root: {os.path.abspath(output_base)}")
    print(f"         Models: {os.path.abspath(tabnet_base)}")
    
    existing_seasons = sorted([
        int(f) for f in os.listdir(data_dir) 
        if os.path.isdir(os.path.join(data_dir, f)) and f.isdigit()
    ])
    
    print("=" * 80)
    print("Expanding Window Walk-Forward Model (TabNet) with Deep Logging")
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
        out_dir = os.path.join(tabnet_base, run_name)
        os.makedirs(out_dir, exist_ok=True)
        
        print("\n" + "=" * 75)
        print(f"[EXECUTION] {run_name}")
        print("=" * 75)
        
        df_train = pd.concat([all_data_dict[yr] for yr in train_years], ignore_index=True)
        df_val = all_data_dict[val_yr]
        
        train_probs = compute_recency_weights(df_train['season'], target_season=val_yr, half_life=3.0)
        
        generate_eda(df_train, out_dir, title_prefix=f"{train_years[0]}-{train_years[-1]}")
        
        drop_cols = ['target_wins', 'team', 'season']
        
        X_train_raw = df_train.drop(columns=drop_cols, errors='ignore')
        y_train_raw = df_train['target_wins']
        X_val_raw = df_val.drop(columns=drop_cols, errors='ignore')
        y_val_raw = df_val['target_wins']
        feature_names = X_train_raw.columns.tolist()
        
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train_raw)
        X_val = scaler.transform(X_val_raw)
        y_train = y_train_raw.values
        y_val = y_val_raw.values
        
        print(f"\n  [Module] Optuna Tuning Initialization...")
        best_params = optimize_tabnet(X_train, y_train, prob_weights=train_probs)
        
        print("\n  [Model] Fitting TabNet Regressor (Iterative Epochs)...")
        model, evals_result = train_tabnet_with_evals(
            best_params, X_train, y_train, X_val, y_val, prob_weights=train_probs
        )
        
        try:
            df_iter_log = pd.DataFrame({
                'epoch': range(1, len(evals_result['train_rmse']) + 1),
                'train_rmse': evals_result['train_rmse'],
                'train_mae': evals_result['train_mae'],
                'val_rmse': evals_result['val_rmse'],
                'val_mae': evals_result['val_mae']
            })
            df_iter_log.to_csv(os.path.join(out_dir, "training_iteration_log.csv"), index=False)
            plot_learning_curves(evals_result, out_dir, run_name)
        except (KeyError, TypeError):
            pass
        
        print("  [Evaluation] Running out-of-time inference on Validation set...")
        preds = model.predict(X_val).flatten()
        
        mae = mean_absolute_error(y_val, preds)
        mse = mean_squared_error(y_val, preds)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_val, preds)
        
        ndcg_val = ndcg_score([y_val], [preds]) if len(y_val) > 1 else 0
        spearman_corr, _ = spearmanr(y_val, preds)
        
        actual_rank = pd.Series(y_val).rank(method='average', ascending=False).values
        pred_rank = pd.Series(preds).rank(method='average', ascending=False).values
        mare = np.mean(np.abs(actual_rank - pred_rank))
        
        print(f"  [Metrics] {val_yr} Holdout -> MAE: {mae:.3f} | RMSE: {rmse:.3f} | R2: {r2:.3f}")
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
            f.write(f"MODEL EXECUTION LOG (TABNET): {run_name}\n")
            f.write("=" * 60 + "\n")
            f.write(f"Validation Target Year: {val_yr}\n")
            f.write(f"Training Observations: {len(X_train)}\n")
            f.write(f"Validation Observations: {len(X_val)}\n")
            f.write("--- HYPERPARAMETERS (OPTUNA) ---\n")
            for param, val in best_params.items():
                f.write(f"{param}: {val}\n")
            f.write("\n--- PERFORMANCE METRICS ---\n")
            f.write(f"Validation Mean Absolute Error (MAE): {mae:.4f} wins\n")
            f.write(f"Validation Root Mean Squared Error (RMSE): {rmse:.4f}\n")
            f.write(f"Validation Mean Squared Error (MSE): {mse:.4f}\n")
            f.write(f"Validation R-squared (R2): {r2:.4f}\n")
            f.write(f"Validation Normalized Discounted Cumulative Gain (NDCG): {ndcg_val:.4f}\n")
            f.write(f"Validation Spearman Rank Correlation: {spearman_corr:.4f}\n")
            f.write(f"Validation Mean Absolute Rank Error: {mare:.4f} positions\n")
            
        zip_path = os.path.join(out_dir, "tabnet_model")
        model.save_model(zip_path) # Saves as