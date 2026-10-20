import os
import time
import requests
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.linear_model import LassoCV
import shap
import optuna
import umap.umap_ as umap

# --- MATPLOTLIB BACKEND FIX ---
# Forces non-interactive rendering to prevent Tkinter main loop thread crashes
import matplotlib
matplotlib.use('Agg')
# ------------------------------

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

# Set PyTorch Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def fetch_tournament_wins(season, output_dir):
    """Fetches postseason NCAA tournament wins and caches them in the tournament_wins directory."""
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
    """Parses team subdirectories to build regular-season game volatility and player metrics."""
    season_dir = os.path.join(data_dir, str(season))
    print(f"    [Data] Searching for team directories in {season_dir}...")
    
    if not os.path.exists(season_dir):
        print(f"    [Error] Season directory does not exist: {season_dir}")
        return pd.DataFrame()
        
    tourney_wins = fetch_tournament_wins(season, output_dir)
    teams = sorted([d for d in os.listdir(season_dir) if os.path.isdir(os.path.join(season_dir, d))])
    
    if len(teams) == 0:
        print(f"    [Warning] No teams found for {season}. Returning empty DataFrame.")
        return pd.DataFrame()
    
    engineered_rows = []
    skip_cols = {'gameId', 'season', 'teamId', 'opponentId', 'teamSeed', 'opponentSeed', 
                 'totalMinutes', 'games', 'wins', 'losses', 'starts'}
    
    cols_processed = 0
    p_cols_processed = 0
    
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
        
        features = {
            'team': actual_team_name,
            'season': int(season),
            'target_wins': tourney_wins.get(actual_team_name, 0)
        }
        
        df_reg = df_g[df_g['seasonType'] == 'regular']
        if df_reg.empty: 
            df_reg = df_g 
            
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
            p_num_cols = top5.select_dtypes(include=[np.number]).columns
            for c in p_num_cols:
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
    """Calculates exponential decay weights normalized to mean 1.0."""
    print(f"  [Weights] Calculating exponential decay weights (Target: {target_season}, Half-life: {half_life} yrs)...")
    delta = target_season - seasons_series.astype(float)
    weights = np.power(0.5, delta / half_life)
    normalized_weights = weights / weights.mean()
    return normalized_weights.values

def generate_eda(df, out_dir, title_prefix=""):
    """Generates distribution, custom matplotlib scatter matrices, and heatmaps."""
    os.makedirs(out_dir, exist_ok=True)
    numeric_df = df.select_dtypes(include=[np.number])
    
    plt.figure(figsize=(7, 4))
    sns.countplot(x='target_wins', data=df, palette='magma')
    plt.title(f'{title_prefix} Target Distribution (Tournament Wins)')
    plt.xlabel('Tournament Wins')
    plt.ylabel('Team Count')
    dist_path = os.path.join(out_dir, "target_distribution.png")
    plt.savefig(dist_path, bbox_inches='tight', dpi=200)
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
                if i == num_features - 1:
                    ax.set_xlabel(top_features[j][:15], fontsize=8)
                if j == 0:
                    ax.set_ylabel(top_features[i][:15], fontsize=8)
                    
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        fig.colorbar(scatter, cax=cbar_ax, label='Target Wins')
        fig.suptitle(f"{title_prefix} Top Correlated Features Pairwise Scatter", fontsize=16)
        
        pair_path = os.path.join(out_dir, "top_features_scatter.png")
        plt.savefig(pair_path, bbox_inches='tight', dpi=200)
        plt.close()
        
    top_15 = correlations.index[0:16].tolist()
    plt.figure(figsize=(12, 10))
    sns.heatmap(df[top_15].corr(), annot=True, cmap='vlag', fmt=".2f", cbar=True)
    plt.title(f"{title_prefix} Correlation Matrix (Top 15 Features)")
    corr_path = os.path.join(out_dir, "correlation_matrix.png")
    plt.savefig(corr_path, bbox_inches='tight', dpi=200)
    plt.close()

def plot_learning_curves(evals_result, out_dir, run_name):
    """Plots training vs validation learning curves across Epochs."""
    print("  [Plot] Generating Training vs Validation Learning Curves...")
    epochs = len(evals_result['validation_0']['rmse'])
    
    x_axis = range(1, epochs + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    train_rmse = evals_result['validation_0']['rmse']
    val_rmse = evals_result['validation_1']['rmse']
    best_iter_rmse = int(np.argmin(val_rmse))
    
    ax1.plot(x_axis, train_rmse, label='Train RMSE', color='#1f77b4', lw=2)
    ax1.plot(x_axis, val_rmse, label='Validation RMSE', color='#d62728', lw=2)
    ax1.axvline(best_iter_rmse + 1, linestyle='--', color='gray', alpha=0.7, 
                label=f'Min Val RMSE (Epoch {best_iter_rmse + 1})')
    ax1.set_title(f'{run_name} - Learning Curve (RMSE)')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('RMSE')
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend()
    
    train_mae = evals_result['validation_0']['mae']
    val_mae = evals_result['validation_1']['mae']
    best_iter_mae = int(np.argmin(val_mae))
    
    ax2.plot(x_axis, train_mae, label='Train MAE', color='#2ca02c', lw=2)
    ax2.plot(x_axis, val_mae, label='Validation MAE', color='#ff7f0e', lw=2)
    ax2.axvline(best_iter_mae + 1, linestyle='--', color='gray', alpha=0.7, 
                label=f'Min Val MAE (Epoch {best_iter_mae + 1})')
    ax2.set_title(f'{run_name} - Learning Curve (MAE)')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('MAE (Tournament Wins)')
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.legend()
    
    plt.tight_layout()
    curve_path = os.path.join(out_dir, "learning_curves.png")
    plt.savefig(curve_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_accuracy_over_time(metrics_df, out_dir):
    """Plots tracking metrics measuring prediction vs actual placement accuracy across seasons."""
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

class TeamLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout_rate):
        super(TeamLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, 
                            dropout=dropout_rate if num_layers > 1 else 0)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_dim // 2, 1)
        
    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :] 
        out = self.dropout(self.relu(self.fc1(out)))
        out = self.fc2(out)
        # Squeeze ONLY the feature dimension (-1) to guarantee 
        # the batch dimension remains intact even if batch size == 1.
        return out.squeeze(-1)

def prepare_tensors(X, y, weights):
    X_t = torch.tensor(X, dtype=torch.float32).unsqueeze(1).to(device)
    y_t = torch.tensor(y.values, dtype=torch.float32).to(device)
    w_t = torch.tensor(weights, dtype=torch.float32).to(device) if weights is not None else torch.ones_like(y_t)
    return X_t, y_t, w_t

def optimize_lstm(X_train, y_train, sample_weights=None):
    print("  [Optuna] Initializing LSTM Hyperparameter Search Space...")
    
    def objective(trial):
        hidden_dim = trial.suggest_categorical('hidden_dim', [32, 64, 128])
        num_layers = trial.suggest_int('num_layers', 1, 3)
        dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
        lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
        epochs = trial.suggest_int('epochs', 30, 100)
        
        kf = KFold(n_splits=3, shuffle=True, random_state=42)
        losses = []
        
        for tr_idx, val_idx in kf.split(X_train):
            X_tr, X_val = X_train[tr_idx], X_train[val_idx]
            y_tr, y_val = y_train.iloc[tr_idx], y_train.iloc[val_idx]
            w_tr = sample_weights[tr_idx] if sample_weights is not None else None
            
            X_tr_t, y_tr_t, w_tr_t = prepare_tensors(X_tr, y_tr, w_tr)
            X_val_t, y_val_t, _ = prepare_tensors(X_val, y_val, None)
            
            model = TeamLSTM(X_train.shape[1], hidden_dim, num_layers, dropout_rate).to(device)
            optimizer = optim.Adam(model.parameters(), lr=lr)
            criterion = nn.MSELoss(reduction='none')
            
            model.train()
            for _ in range(epochs):
                optimizer.zero_grad()
                preds = model(X_tr_t)
                loss = (criterion(preds, y_tr_t) * w_tr_t).mean()
                loss.backward()
                optimizer.step()
                
            model.eval()
            with torch.no_grad():
                val_preds = model(X_val_t)
                val_loss = mean_squared_error(y_val_t.cpu().numpy(), val_preds.cpu().numpy())
            losses.append(val_loss)
            
        return np.mean(losses)

    print("  [Optuna] Running 15 tuning trials via 3-Fold Cross Validation...")
    study = optuna.create_study(direction='minimize')
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(objective, n_trials=15)
    print(f"  [Optuna] Search Complete. Best CV MSE: {study.best_value:.4f}")
    return study.best_params

def train_lstm_with_evals(best_params, X_train, y_train, X_val, y_val, sample_weights=None):
    hidden_dim = best_params['hidden_dim']
    num_layers = best_params['num_layers']
    dropout_rate = best_params['dropout_rate']
    lr = best_params['lr']
    epochs = best_params['epochs']
    
    model = TeamLSTM(X_train.shape[1], hidden_dim, num_layers, dropout_rate).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss(reduction='none')
    
    X_tr_t, y_tr_t, w_tr_t = prepare_tensors(X_train, y_train, sample_weights)
    X_val_t, y_val_t, _ = prepare_tensors(X_val, y_val, None)
    
    evals_result = {
        'validation_0': {'rmse': [], 'mae': []},
        'validation_1': {'rmse': [], 'mae': []}
    }
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        tr_preds = model(X_tr_t)
        loss = (criterion(tr_preds, y_tr_t) * w_tr_t).mean()
        loss.backward()
        optimizer.step()
        
        model.eval()
        with torch.no_grad():
            tr_rmse = np.sqrt(mean_squared_error(y_tr_t.cpu().numpy(), tr_preds.cpu().numpy()))
            tr_mae = mean_absolute_error(y_tr_t.cpu().numpy(), tr_preds.cpu().numpy())
            
            val_preds = model(X_val_t)
            val_rmse = np.sqrt(mean_squared_error(y_val_t.cpu().numpy(), val_preds.cpu().numpy()))
            val_mae = mean_absolute_error(y_val_t.cpu().numpy(), val_preds.cpu().numpy())
            
        evals_result['validation_0']['rmse'].append(tr_rmse)
        evals_result['validation_0']['mae'].append(tr_mae)
        evals_result['validation_1']['rmse'].append(val_rmse)
        evals_result['validation_1']['mae'].append(val_mae)
        
    return model, evals_result

def generate_3d_umap(df, out_dir):
    print("  [UMAP] Initiating 3D Projection Engine...")
    X = df.drop(columns=['team', 'season', 'target_wins'], errors='ignore').select_dtypes(include=[np.number])
    X_scaled = StandardScaler().fit_transform(X)
    
    print("  [UMAP] Fitting 2-Component Manifold (n_neighbors=15, min_dist=0.1)...")
    reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
    embedding = reducer.fit_transform(X_scaled)
    
    le = LabelEncoder()
    team_colors = le.fit_transform(df['team'])
    
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(embedding[:, 0], embedding[:, 1], df['target_wins'], c=team_colors, cmap='nipy_spectral', alpha=0.75, s=40, edgecolors='k', linewidth=0.4)
    ax.set_xlabel('UMAP 1 (Play Style/Efficiency)')
    ax.set_ylabel('UMAP 2 (Physicality/Turnover Profile)')
    ax.set_zlabel('Actual Tournament Wins')
    plt.title('3D UMAP Manifold: Team Signatures Elevated by Postseason Success')
    
    umap_path = os.path.join(out_dir, "3d_umap_overall.png")
    plt.savefig(umap_path, dpi=300, bbox_inches='tight')
    plt.close()

def compute_shap(model, X_train_scaled, X_val_scaled, feature_names, out_dir):
    print("\n  [Module] SHAP Attribution Initialization...")
    print("  [SHAP] Initializing K-Means Background Dataset (k=25) for KernelExplainer...")
    model.eval()
    
    def predict_fn(x):
        if x.ndim == 1:
            x = np.expand_dims(x, axis=0)
        x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(1).to(device)
        with torch.no_grad():
            preds = model(x_tensor).cpu().numpy()
            return np.atleast_1d(preds)
            
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
    
    vals = np.abs(shap_values).mean(0)
    fi = pd.DataFrame(list(zip(feature_names, vals)), columns=['feature', 'shap_importance'])
    fi.sort_values(by=['shap_importance'], ascending=False, inplace=True)
    fi_csv = os.path.join(out_dir, "shap_feature_importance.csv")
    fi.to_csv(fi_csv, index=False)

def main():
    # --- DYNAMIC PATH ROUTING ---
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    data_dir = os.path.join(project_root, "data")
    output_base = os.path.join(project_root, "model_outputs")
    meta_base = os.path.join(output_base, "lasso_lstm_meta")
    # ----------------------------
    
    os.makedirs(output_base, exist_ok=True)
    os.makedirs(meta_base, exist_ok=True)
    
    print(f"[System] PyTorch Backend Enabled. Device: {device}")
    print(f"[System] Ensured output directories exist:")
    print(f"         Root: {os.path.abspath(output_base)}")
    print(f"         Models: {os.path.abspath(meta_base)}")
    
    if not os.path.exists(data_dir):
        print(f"\n[Fatal Error] Data directory not found at: {data_dir}")
        return
        
    existing_seasons = sorted([int(f) for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f)) and f.isdigit()])
    print("=" * 80)
    print("Expanding Window Walk-Forward Model (Lasso Selector -> LSTM Meta-Model)")
    print(f"Discovered Seasons: {existing_seasons}")
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
        out_dir = os.path.join(meta_base, run_name)
        os.makedirs(out_dir, exist_ok=True)
        
        print("\n" + "=" * 75)
        print(f"[EXECUTION] {run_name}")
        print("=" * 75)
        
        df_train = pd.concat([all_data_dict[yr] for yr in train_years], ignore_index=True)
        df_val = all_data_dict[val_yr]
        
        train_weights = compute_recency_weights(df_train['season'], target_season=val_yr, half_life=3.0)
        
        generate_eda(df_train, out_dir, title_prefix=f"{train_years[0]}-{train_years[-1]}")
        
        drop_cols = ['target_wins', 'team', 'season']
        y_train = df_train['target_wins']
        X_train_raw = df_train.drop(columns=drop_cols, errors='ignore')
        
        y_val = df_val['target_wins']
        X_val_raw = df_val.drop(columns=drop_cols, errors='ignore')
        
        # =====================================================================
        # META-PIPELINE STEP 1: LASSO FEATURE SELECTION
        # =====================================================================
        print("\n  [Meta-Pipeline] Phase 1: Lasso L1 Regularized Feature Selection...")
        scaler_lasso = StandardScaler()
        X_train_scaled_lasso = scaler_lasso.fit_transform(X_train_raw)
        
        lasso_selector = LassoCV(cv=3, random_state=42, n_jobs=1, max_iter=5000)
        lasso_selector.fit(X_train_scaled_lasso, y_train)
        
        surviving_mask = lasso_selector.coef_ != 0
        surviving_features = X_train_raw.columns[surviving_mask].tolist()
        
        if len(surviving_features) < 3:
            print("  [Warning] Lasso penalization too aggressive. Retaining top 15 correlated features as fallback.")
            corr = X_train_raw.apply(lambda x: x.corr(y_train)).abs().sort_values(ascending=False)
            surviving_features = corr.head(15).index.tolist()
            
        print(f"  [Meta-Pipeline] Features aggressively pruned from {len(X_train_raw.columns)} to {len(surviving_features)}.")
        
        # We must scale the REDUCED feature set because neural networks still require scaled inputs
        X_train_reduced_raw = X_train_raw[surviving_features]
        X_val_reduced_raw = X_val_raw[surviving_features]
        
        scaler_lstm = StandardScaler()
        X_train = scaler_lstm.fit_transform(X_train_reduced_raw)
        X_val = scaler_lstm.transform(X_val_reduced_raw)
        # =====================================================================
        
        print(f"  [Model] Final Train Shape: {X_train.shape} | Final Val Shape: {X_val.shape}")
        
        print(f"\n  [Meta-Pipeline] Phase 2: Optimizing LSTM on Reduced Feature Space...")
        best_params = optimize_lstm(X_train, y_train, sample_weights=train_weights)
        print(f"  [Optuna] Applied Optimal Parameters: {best_params}")
        
        print("\n  [Model] Fitting LSTM Regressor (Iterative Epochs)...")
        model, evals_result = train_lstm_with_evals(
            best_params, X_train, y_train, X_val, y_val, sample_weights=train_weights
        )
        
        df_iter_log = pd.DataFrame({
            'epoch': range(1, len(evals_result['validation_0']['rmse']) + 1),
            'train_rmse': evals_result['validation_0']['rmse'],
            'train_mae': evals_result['validation_0']['mae'],
            'val_rmse': evals_result['validation_1']['rmse'],
            'val_mae': evals_result['validation_1']['mae']
        })
        df_iter_log.to_csv(os.path.join(out_dir, "training_iteration_log.csv"), index=False)
        plot_learning_curves(evals_result, out_dir, run_name)
        
        print("  [Evaluation] Running out-of-time inference on Validation set...")
        model.eval()
        X_val_t = torch.tensor(X_val, dtype=torch.float32).unsqueeze(1).to(device)
        with torch.no_grad():
            preds = model(X_val_t).cpu().numpy()
            
        mae = mean_absolute_error(y_val, preds)
        mse = mean_squared_error(y_val, preds)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_val, preds)
        
        ndcg_val = ndcg_score([y_val.values], [preds]) if len(y_val) > 1 else 0
        spearman_corr, _ = spearmanr(y_val, preds)
        
        if np.isnan(spearman_corr):
            spearman_corr = 0.0
            ndcg_val = 0.0
            
        actual_rank = y_val.rank(method='average', ascending=False).values
        pred_rank = pd.Series(preds).rank(method='average', ascending=False).values
        mare = np.mean(np.abs(actual_rank - pred_rank))
        
        if np.isnan(mare): mare = 34.0
        
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
            f.write(f"MODEL EXECUTION LOG (META LASSO->LSTM): {run_name}\n")
            f.write("=" * 60 + "\n")
            f.write(f"Validation Target Year: {val_yr}\n")
            f.write(f"Training Observations: {len(X_train)}\n")
            f.write(f"Validation Observations: {len(X_val)}\n")
            f.write(f"Original Features: {len(X_train_raw.columns)}\n")
            f.write(f"Lasso-Selected Features: {len(surviving_features)}\n")
            f.write("--- HYPERPARAMETERS (OPTUNA) ---\n")
            for param, val in best_params.items(): f.write(f"{param}: {val}\n")
            f.write("\n--- PERFORMANCE METRICS ---\n")
            f.write(f"Validation Mean Absolute Error (MAE): {mae:.4f} wins\n")
            f.write(f"Validation Root Mean Squared Error (RMSE): {rmse:.4f}\n")
            f.write(f"Validation Mean Squared Error (MSE): {mse:.4f}\n")
            f.write(f"Validation R-squared (R2): {r2:.4f}\n")
            f.write(f"Validation Normalized Discounted Cumulative Gain (NDCG): {ndcg_val:.4f}\n")
            f.write(f"Validation Spearman Rank Correlation: {spearman_corr:.4f}\n")
            f.write(f"Validation Mean Absolute Rank Error: {mare:.4f} positions\n")
            
        pth_path = os.path.join(out_dir, "lstm_meta_model.pth")
        torch.save(model.state_dict(), pth_path)
        print(f"  [I/O] Serialized PyTorch State Dict saved to {pth_path}")
        
        compute_shap(model, X_train, X_val, surviving_features, out_dir)
        
    if yearly_metrics:
        df_metrics = pd.DataFrame(yearly_metrics)
        plot_accuracy_over_time(df_metrics, meta_base)
        
    print("\n" + "=" * 80)
    print("[Phase 3] Executing Master Aggregate Pipeline (All Historical Years Combined)")
    print("=" * 80)
    
    all_out = os.path.join(meta_base, "all_years")
    os.makedirs(all_out, exist_ok=True)
    
    generate_eda(master_df, all_out, title_prefix="All-Time")
    generate_3d_umap(master_df, all_out)
    
    latest_season = master_df['season'].max()
    all_weights = compute_recency_weights(master_df['season'], target_season=latest_season, half_life=3.0)
    
    X_all_raw = master_df.drop(columns=['target_wins', 'team', 'season'], errors='ignore')
    y_all = master_df['target_wins']
    
    split_idx = int(len(X_all_raw) * 0.8)
    X_tr_all_raw, X_val_all_raw = X_all_raw.iloc[:split_idx], X_all_raw.iloc[split_idx:]
    y_tr_all, y_val_all = y_all.iloc[:split_idx], y_all.iloc[split_idx:]
    w_tr_all = all_weights[:split_idx]
    
    # =========================================================================
    # GLOBAL META-PIPELINE STEP 1
    # =========================================================================
    print("\n  [Meta-Pipeline] Global Phase 1: Lasso Feature Selection...")
    scaler_all_lasso = StandardScaler()
    X_tr_all_scaled_lasso = scaler_all_lasso.fit_transform(X_tr_all_raw)
    
    lasso_all = LassoCV(cv=3, random_state=42, n_jobs=1, max_iter=5000)
    lasso_all.fit(X_tr_all_scaled_lasso, y_tr_all)
    
    mask_all = lasso_all.coef_ != 0
    surv_feats_all = X_tr_all_raw.columns[mask_all].tolist()
    
    if len(surv_feats_all) < 3:
        corr_all = X_tr_all_raw.apply(lambda x: x.corr(y_tr_all)).abs().sort_values(ascending=False)
        surv_feats_all = corr_all.head(15).index.tolist()
        
    print(f"  [Meta-Pipeline] Global Features pruned from {len(X_tr_all_raw.columns)} to {len(surv_feats_all)}.")
    
    X_tr_all_reduced_raw = X_tr_all_raw[surv_feats_all]
    X_val_all_reduced_raw = X_val_all_raw[surv_feats_all]
    X_all_reduced_raw = X_all_raw[surv_feats_all]
    
    scaler_lstm_global = StandardScaler()
    X_tr_all_reduced = scaler_lstm_global.fit_transform(X_tr_all_reduced_raw)
    X_val_all_reduced = scaler_lstm_global.transform(X_val_all_reduced_raw)
    X_all_reduced = scaler_lstm_global.fit_transform(X_all_reduced_raw)
    # =========================================================================
    
    print("\n  [Meta-Pipeline] Global Phase 2: Optimizing LSTM...")
    best_params_all = optimize_lstm(X_tr_all_reduced, y_tr_all, sample_weights=w_tr_all)
    
    print("\n  [Model] Fitting Global Diagnostic Regressor (Iterative)...")
    master_model, evals_result_all = train_lstm_with_evals(
        best_params_all, X_tr_all_reduced, y_tr_all, X_val_all_reduced, y_val_all, sample_weights=w_tr_all
    )
    
    df_all_iter_log = pd.DataFrame({
        'epoch': range(1, len(evals_result_all['validation_0']['rmse']) + 1),
        'train_rmse': evals_result_all['validation_0']['rmse'],
        'train_mae': evals_result_all['validation_0']['mae'],
        'val_rmse': evals_result_all['validation_1']['rmse'],
        'val_mae': evals_result_all['validation_1']['mae']
    })
    df_all_iter_log.to_csv(os.path.join(all_out, "master_training_iteration_log.csv"), index=False)
    plot_learning_curves(evals_result_all, all_out, "Master Meta-Model (All Years)")
    
    print("\n  [Model] Final 100% Fit on ALL available data points...")
    final_master_model, _ = train_lstm_with_evals(
        best_params_all, X_all_reduced, y_all, X_all_reduced, y_all, sample_weights=all_weights
    )
    
    master_pth_path = os.path.join(all_out, "lstm_meta_master_model.pth")
    torch.save(final_master_model.state_dict(), master_pth_path)
    print(f"  [I/O] Global Serialized PyTorch State Dict saved to {master_pth_path}")
    
    compute_shap(final_master_model, X_tr_all_reduced, X_all_reduced, surv_feats_all, all_out)
    
    print("\n" + "=" * 80)
    print(f"[COMPLETE] Entire Temporal Meta-Pipeline has resolved successfully.")
    print("=" * 80)

if __name__ == "__main__":
    main()