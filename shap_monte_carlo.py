"""Turn combo_eval.py outputs into Monte Carlo baselines, bracket-tree sims, and EV-optimized brackets.

Enhancements integrated:
  1. Per-model z-score harmonization (exact match to combo_eval.py Phase 2/3 when
     individual model SHAP files exist in model_outputs/, with Rank_1 fallback).
  2. Stratified / heteroskedastic local residual resampling so top contenders draw
     from historical contender error distributions rather than R64 blowouts.
  3. Post-hoc out-of-fold Isotonic Regression calibration on threshold frequencies
     with monotone round enforcement.
  4. Pairwise matchup probability model (Logistic Regression on score + seed
     differentials, evaluated by Log-Loss and Brier Score) feeding a 63-game
     single-elimination Bracket-Tree Monte Carlo simulator.
  5. Exact dynamic-programming Expected Value (EV) & Game-Theory Leverage Bracket
     Optimizer for tournament pools (10-20-40-80-160-320 scoring).
"""

import contextlib
import glob
import io
import os
import re

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.special import expit
from scipy.stats import spearmanr
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.preprocessing import StandardScaler

from monte_carlo import simulate_bracket_tree

THRESHOLDS = list(range(1, 7))
ROUND_LABELS = {1: 'R32', 2: 'Sweet 16', 3: 'Elite 8', 4: 'Final 4', 5: 'Title Game', 6: 'Champion'}
ROUND_POINTS = {1: 10, 2: 20, 3: 40, 4: 80, 5: 160, 6: 320}
PROB_COLS = [f"probability_ge_{k}_win{'s' if k != 1 else ''}" for k in THRESHOLDS]
CAL_PROB_COLS = [f"calibrated_prob_ge_{k}_win{'s' if k != 1 else ''}" for k in THRESHOLDS]
TREE_PROB_COLS = [f"tree_prob_ge_{k}_win{'s' if k != 1 else ''}" for k in THRESHOLDS]


# ---------------------------------------------------------------------------
# Loading combo_eval outputs and historical results
# ---------------------------------------------------------------------------
def load_shap_recipe(ensemble_dir, shap_path=None):
    """Load the best ensemble's blended SHAP weights (Rank 1 by default)."""
    if shap_path is None:
        matches = sorted(glob.glob(os.path.join(str(ensemble_dir), 'top_20_shap_weights', 'Rank_1_*_SHAP.csv')))
        if not matches:
            raise FileNotFoundError(
                f"No Rank_1_*_SHAP.csv under {ensemble_dir}/top_20_shap_weights. Run combo_eval.py first.")
        shap_path = matches[0]
    df = pd.read_csv(shap_path)
    if not {'feature', 'shap_importance'}.issubset(df.columns):
        raise ValueError(f"{shap_path} must contain feature and shap_importance columns.")
    df = df.dropna(subset=['shap_importance']).drop_duplicates('feature')
    if df.empty:
        raise ValueError(f"{shap_path} contains no SHAP weights.")

    info = {'shap_weights_path': str(shap_path)}
    metrics_csv = os.path.join(str(ensemble_dir), 'ALL_ENSEMBLE_COMBINATIONS_METRICS.csv')
    rank = re.search(r'Rank_(\d+)_', os.path.basename(str(shap_path)))
    if rank and os.path.exists(metrics_csv):
        metrics = pd.read_csv(metrics_csv)
        idx = int(rank.group(1)) - 1
        if idx < len(metrics):
            row = metrics.iloc[idx]
            info['ensemble_rank'] = idx + 1
            info['ensemble_name'] = str(row.get('Ensemble_Name'))
            info['ensemble_members'] = [m.strip() for m in str(row.get('Ensemble_Name', '')).split('+') if m.strip()]
            info['ensemble_metrics'] = {k: float(row[k]) for k in
                                        ['Global_Accuracy', 'Global_Blind_Error', 'Global_NDCG',
                                         'Global_Spearman', 'Global_MARE'] if k in row}
    return df, info


def load_member_shap_by_season(training_output_dir, members):
    """Load per-model, per-season SHAP dictionaries from model_outputs/ when available."""
    if not members or not os.path.exists(str(training_output_dir)):
        return {}
    by_model = {}
    for member in members:
        folder = member.strip().lower().replace(' ', '_')
        pattern = os.path.join(str(training_output_dir), folder, 'Train_*_Val_*', 'shap_feature_importance.csv')
        season_map = {}
        for path in sorted(glob.glob(pattern)):
            m = re.search(r'Val_(\d{4})', path)
            if m:
                df = pd.read_csv(path).dropna(subset=['shap_importance'])
                season_map[int(m.group(1))] = dict(zip(df['feature'], df['shap_importance']))
        if season_map:
            by_model[member] = season_map
    return by_model if len(by_model) == len(members) else {}


def discover_seasons(data_dir):
    return sorted(int(d) for d in os.listdir(data_dir)
                  if re.fullmatch(r'\d{4}', d) and os.path.isdir(os.path.join(data_dir, d)))


def load_actual_wins(season, search_dirs):
    """Cached tournament wins (combo_eval: data/, ridge: model_outputs/)."""
    for d in search_dirs:
        path = os.path.join(str(d), 'tournament_wins', f'tournament_wins_{season}.csv')
        if os.path.exists(path):
            df = pd.read_csv(path)
            return {str(t).strip(): float(w) for t, w in zip(df['team'], df['wins'])}
    return None


_BRACKET_GAMES_CACHE = {}


def load_bracket_games_df(season, data_dir):
    """Load data/<season>/tournament_games.csv with standardized column names."""
    key = (int(season), str(data_dir))
    if key in _BRACKET_GAMES_CACHE:
        return _BRACKET_GAMES_CACHE[key]
    path = os.path.join(str(data_dir), str(season), 'tournament_games.csv')
    if not os.path.exists(path):
        _BRACKET_GAMES_CACHE[key] = pd.DataFrame()
        return _BRACKET_GAMES_CACHE[key]
    try:
        df = pd.read_csv(path)
    except Exception:
        _BRACKET_GAMES_CACHE[key] = pd.DataFrame()
        return _BRACKET_GAMES_CACHE[key]
    if df.empty:
        _BRACKET_GAMES_CACHE[key] = df
        return df
    df.columns = df.columns.str.replace(r'[^\x00-\x7F]+', '', regex=True).str.strip()
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
        elif clean_c in ['homeseed', 'seedhome']:
            col_map[c] = 'homeSeed'
        elif clean_c in ['awayseed', 'seedaway']:
            col_map[c] = 'awaySeed'
    df = df.rename(columns=col_map)
    if {'homeTeam', 'awayTeam'}.issubset(df.columns):
        df['homeTeam'] = df['homeTeam'].astype(str).str.strip()
        df['awayTeam'] = df['awayTeam'].astype(str).str.strip()
    if 'homeWinner' in df.columns and {'homeTeam', 'awayTeam'}.issubset(df.columns):
        df['Actual_Winner'] = np.where(
            df['homeWinner'].astype(str).str.lower() == 'true', df['homeTeam'], df['awayTeam']
        )
    _BRACKET_GAMES_CACHE[key] = df
    return df


def extract_team_seeds(df_games):
    """Extract {team: seed} from bracket games."""
    seeds = {}
    if df_games.empty:
        return seeds
    for team_col, seed_col in [('homeTeam', 'homeSeed'), ('awayTeam', 'awaySeed')]:
        if team_col in df_games.columns and seed_col in df_games.columns:
            for t, s in zip(df_games[team_col], pd.to_numeric(df_games[seed_col], errors='coerce')):
                if pd.notna(s) and 1 <= int(s) <= 16:
                    seeds[str(t).strip()] = int(s)
    return seeds


def tournament_field(season, data_dir, actual_wins=None):
    """Tournament teams from cached results, else from data/<season>/tournament_games.csv."""
    if actual_wins:
        return set(actual_wins)
    df = load_bracket_games_df(season, data_dir)
    if not df.empty and {'homeTeam', 'awayTeam'}.issubset(df.columns):
        return set(df['homeTeam']) | set(df['awayTeam'])
    return None


def _dot_zscore(X, X_scaled, shap_weights):
    used = [f for f in shap_weights if f in X.columns]
    if not used:
        return np.zeros(len(X), dtype=float), list(shap_weights)
    missing = [f for f in shap_weights if f not in X.columns]
    weights = np.array([shap_weights[f] for f in used], dtype=float)
    raw = X_scaled[:, [X.columns.get_loc(f) for f in used]] @ weights
    std = raw.std() if raw.std() > 0 else 1.0
    return (raw - raw.mean()) / std, missing


def score_season(data_dir, season, shap_weights, member_shap_by_season=None):
    """Score every team in a season.

    When member_shap_by_season is available, computes each model's z-scored
    SHAP dot product and averages them (matching combo_eval Phase 2 & 3).
    """
    from combo_eval import build_team_features_for_scoring
    with contextlib.redirect_stdout(io.StringIO()):
        feats = build_team_features_for_scoring(str(data_dir), season)
    if feats.empty:
        return pd.DataFrame(columns=['team', 'season', 'ensemble_score']), list(shap_weights)
    feats['team'] = feats['team'].astype(str).str.strip()
    feats = feats.drop_duplicates('team').reset_index(drop=True)
    X = feats.drop(columns=['team'])
    X_scaled = StandardScaler().fit_transform(X)

    member_cols = {}
    if member_shap_by_season:
        missing_all = set()
        for model_name, season_dict in member_shap_by_season.items():
            w = season_dict.get(season) or season_dict[max(season_dict)]
            z, missing = _dot_zscore(X, X_scaled, w)
            member_cols[f"score__{model_name}"] = z
            missing_all.update(missing)
        ensemble_z = np.mean(np.column_stack(list(member_cols.values())), axis=1)
        missing_list = sorted(missing_all)
    else:
        ensemble_z, missing_list = _dot_zscore(X, X_scaled, shap_weights)

    scores = pd.DataFrame({
        'team': feats['team'],
        'season': int(season),
        'ensemble_score': ensemble_z,
        **member_cols,
    })
    return scores, missing_list


def build_history(seasons, data_dir, wins_dirs, shap_weights, member_shap_by_season=None):
    """Scored tournament teams with known results and seeds for every historical season."""
    frames = []
    for season in seasons:
        wins = load_actual_wins(season, wins_dirs)
        if not wins:
            continue
        scores, _ = score_season(data_dir, season, shap_weights, member_shap_by_season)
        scores = scores[scores['team'].isin(wins)].copy()
        scores['actual_wins'] = scores['team'].map(wins)
        df_games = load_bracket_games_df(season, data_dir)
        seeds = extract_team_seeds(df_games)
        scores['seed'] = scores['team'].map(seeds)
        print(f"    [History] {season}: scored {len(scores)}/{len(wins)} tournament teams")
        frames.append(scores)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# ---------------------------------------------------------------------------
# Calibration, Stratified Residuals & Isotonic Post-Processing
# ---------------------------------------------------------------------------
def _fit_line(df):
    if len(df) < 2 or df['ensemble_score'].nunique() < 2:
        return 0.0, float(df['actual_wins'].mean()) if len(df) else 0.0
    slope, intercept = np.polyfit(df['ensemble_score'], df['actual_wins'], 1)
    return float(slope), float(intercept)


def calibrate(history, target):
    """Fit on seasons != target; residuals from nested leave-one-season-out fits."""
    others = history[history['season'] != target].copy()
    if others['season'].nunique() < 2:
        pool = others if not others.empty else history.copy()
        slope, intercept = _fit_line(pool)
        res = pool['actual_wins'] - (slope * pool['ensemble_score'] + intercept)
        return slope, intercept, res.to_numpy(float), True
    slope, intercept = _fit_line(others)
    residuals = []
    for s in sorted(others['season'].unique()):
        a, b = _fit_line(others[others['season'] != s])
        hold = others[others['season'] == s]
        residuals.extend(hold['actual_wins'] - (a * hold['ensemble_score'] + b))
    return slope, intercept, np.asarray(residuals, dtype=float), False


def build_stratified_residuals(history, target, target_teams_df, window_fraction=0.28, min_pool=80):
    """Build per-team local/stratified residual pools conditioned on ensemble_score.

    For each team in target_teams_df, finds the nearest historical out-of-fold
    tournament teams in ensemble_score space and returns their held-out residuals.
    This captures heteroskedasticity (contenders have wider upside up to 6 wins;
    low seeds cluster near 0-1 wins) without leaking target-season data.
    """
    others = history[history['season'] != target].copy()
    if others['season'].nunique() < 2:
        _, _, flat, _ = calibrate(history, target)
        return {t: flat for t in target_teams_df['team']}

    oof_residuals = np.empty(len(others), dtype=float)
    oof_scores = others['ensemble_score'].to_numpy(float)
    for s in sorted(others['season'].unique()):
        mask = (others['season'] == s).to_numpy()
        a, b = _fit_line(others[~mask])
        oof_residuals[mask] = others.loc[mask, 'actual_wins'] - (a * oof_scores[mask] + b)

    k = max(min_pool, int(round(len(others) * window_fraction)))
    k = min(k, len(others))
    residual_map = {}
    for _, row in target_teams_df.iterrows():
        z = float(row['ensemble_score'])
        nearest_idx = np.argsort(np.abs(oof_scores - z))[:k]
        pool = oof_residuals[nearest_idx]
        residual_map[row['team']] = pool
    return residual_map


def fit_isotonic_calibrators(oof_tables):
    """Fit one IsotonicRegression per win threshold k=1..6 on out-of-fold tables."""
    calibrators = {}
    for col, k in zip(PROB_COLS, THRESHOLDS):
        x = oof_tables[col].to_numpy(float)
        y = (oof_tables['actual_wins'].to_numpy(float) >= k).astype(float)
        iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds='clip')
        iso.fit(x, y)
        calibrators[k] = iso
    return calibrators


def apply_isotonic_calibration(table, calibrators):
    """Add calibrated_prob_ge_1..6 columns with monotone nonincreasing enforcement."""
    table = table.copy()
    cal_matrix = np.column_stack([
        calibrators[k].transform(table[col].to_numpy(float))
        for col, k in zip(PROB_COLS, THRESHOLDS)
    ])
    cal_matrix = np.clip(np.minimum.accumulate(cal_matrix, axis=1), 0.0, 1.0)
    for idx, col in enumerate(CAL_PROB_COLS):
        table[col] = cal_matrix[:, idx]
    return table


# ---------------------------------------------------------------------------
# Pairwise Matchup Model & 63-Game Bracket-Tree Simulator
# ---------------------------------------------------------------------------
def build_matchup_training_data(seasons, data_dir, history, exclude_season):
    """Build symmetric pairwise game dataset (delta_score, delta_seed) -> win {0, 1}."""
    score_lookup = {(int(r['season']), r['team']): float(r['ensemble_score'])
                    for _, r in history.iterrows()}
    X_rows, y_rows, season_labels = [], [], []
    for s in seasons:
        if s == exclude_season:
            continue
        df_games = load_bracket_games_df(s, data_dir)
        if df_games.empty or 'Actual_Winner' not in df_games.columns:
            continue
        seeds = extract_team_seeds(df_games)
        for _, g in df_games.iterrows():
            h, a, w = g['homeTeam'], g['awayTeam'], g['Actual_Winner']
            if (s, h) not in score_lookup or (s, a) not in score_lookup:
                continue
            dz = score_lookup[(s, h)] - score_lookup[(s, a)]
            sh, sa = seeds.get(h, 8.5), seeds.get(a, 8.5)
            dseed = float(sa - sh)  # positive when home has better (lower) seed
            h_won = 1 if w == h else 0
            # Add both perspectives for exact antisymmetry (zero intercept bias)
            X_rows.append([dz, dseed])
            y_rows.append(h_won)
            season_labels.append(s)
            X_rows.append([-dz, -dseed])
            y_rows.append(1 - h_won)
            season_labels.append(s)
    return np.asarray(X_rows, dtype=float), np.asarray(y_rows, dtype=int), np.asarray(season_labels, dtype=int)


def fit_matchup_model(seasons, data_dir, history, target_season):
    """Fit logistic matchup model on seasons != target and evaluate out-of-fold Log-Loss & Brier."""
    X, y, s_labels = build_matchup_training_data(seasons, data_dir, history, exclude_season=target_season)
    if len(X) < 20:
        return None, {}
    model = LogisticRegression(fit_intercept=False, C=1.0)
    model.fit(X, y)

    # Evaluate out-of-fold log-loss, Brier score, and accuracy across historical seasons
    unique_s = sorted(np.unique(s_labels))
    oof_probs, oof_true = [], []
    for s in unique_s:
        mask = s_labels == s
        if mask.sum() == 0 or (~mask).sum() < 20:
            continue
        m_fold = LogisticRegression(fit_intercept=False, C=1.0)
        m_fold.fit(X[~mask], y[~mask])
        p = m_fold.predict_proba(X[mask])[:, 1]
        oof_probs.extend(p)
        oof_true.extend(y[mask])
    oof_probs = np.asarray(oof_probs, dtype=float)
    oof_true = np.asarray(oof_true, dtype=int)
    # Deduplicate symmetric pairs (take every 2nd row = original games)
    orig_probs, orig_true = oof_probs[::2], oof_true[::2]
    diagnostics = {
        'games_evaluated': int(len(orig_true)),
        'oof_log_loss': float(log_loss(orig_true, orig_probs)),
        'oof_brier_score': float(brier_score_loss(orig_true, orig_probs)),
        'oof_accuracy': float(np.mean((orig_probs >= 0.5) == orig_true)),
        'coef_ensemble_score_diff': float(model.coef_[0][0]),
        'coef_seed_diff': float(model.coef_[0][1]),
    }
    return model, diagnostics


def parse_bracket_tree(df_games):
    """Parse 63-game single-elimination bracket topology from tournament_games.csv."""
    if df_games.empty or len(df_games) < 63:
        return None
    if 'startDate' in df_games.columns:
        df = df_games.sort_values('startDate').tail(63).reset_index(drop=True)
    else:
        df = df_games.tail(63).reset_index(drop=True)
    if len(df) != 63 or 'Actual_Winner' not in df.columns:
        return None

    r1 = df.iloc[:32]
    teams64 = sorted(set(r1['homeTeam']).union(set(r1['awayTeam'])))
    if len(teams64) != 64:
        return None
    team_to_idx = {t: i for i, t in enumerate(teams64)}
    round1_pairs = [(team_to_idx[r['homeTeam']], team_to_idx[r['awayTeam']]) for _, r in r1.iterrows()]

    tree_edges = []
    for i in range(32, 63):
        h_team = df.iloc[i]['homeTeam']
        a_team = df.iloc[i]['awayTeam']
        left_src, right_src = None, None
        for j in range(i):
            if df.iloc[j]['Actual_Winner'] == h_team:
                left_src = j
            elif df.iloc[j]['Actual_Winner'] == a_team:
                right_src = j
        if left_src is None or right_src is None:
            return None
        tree_edges.append((left_src, right_src))

    seeds = extract_team_seeds(df_games)
    actual_slot_winners = [team_to_idx.get(w, -1) for w in df['Actual_Winner']]
    slot_rounds = [1] * 32 + [2] * 16 + [3] * 8 + [4] * 4 + [5] * 2 + [6] * 1
    return {
        'df_63': df,
        'teams64': teams64,
        'team_to_idx': team_to_idx,
        'round1_pairs': round1_pairs,
        'tree_edges': tree_edges,
        'seeds': seeds,
        'actual_slot_winners': actual_slot_winners,
        'slot_rounds': slot_rounds,
    }


def build_pairwise_prob_matrix(teams64, score_map, seeds, matchup_model):
    """Compute (64, 64) matrix P[i, j] = P(team i beats team j)."""
    n = len(teams64)
    z = np.array([score_map.get(t, 0.0) for t in teams64], dtype=float)
    s = np.array([seeds.get(t, 8.5) for t in teams64], dtype=float)
    dz = z[:, None] - z[None, :]
    dseed = s[None, :] - s[:, None]
    if matchup_model is not None:
        w_z, w_s = matchup_model.coef_[0]
        logits = w_z * dz + w_s * dseed
    else:
        logits = 0.85 * dz + 0.12 * dseed
    P = expit(logits)
    np.fill_diagonal(P, 0.5)
    return P


def build_public_prob_matrix(teams64, score_map, seeds):
    """Model public pick behavior: heavily weighted toward seed number + chalk favorites."""
    z = np.array([score_map.get(t, 0.0) for t in teams64], dtype=float)
    s = np.array([seeds.get(t, 8.5) for t in teams64], dtype=float)
    dz = z[:, None] - z[None, :]
    dseed = s[None, :] - s[:, None]
    # Public pickers strongly anchor on seed differential and over-pick favorites
    logits = 0.65 * dz + 0.28 * dseed
    P_pub = expit(logits)
    np.fill_diagonal(P_pub, 0.5)
    return P_pub


def run_bracket_tree_simulation(season, data_dir, scores_df, matchup_model, n_simulations=10000, seed=42):
    """Run 63-game bracket tree Monte Carlo and return tree probabilities + slot state."""
    df_games = load_bracket_games_df(season, data_dir)
    parsed = parse_bracket_tree(df_games)
    if parsed is None:
        return None

    teams64 = parsed['teams64']
    score_map = dict(zip(scores_df['team'], scores_df['ensemble_score']))
    P_model = build_pairwise_prob_matrix(teams64, score_map, parsed['seeds'], matchup_model)
    win_counts, slot_winners = simulate_bracket_tree(
        parsed['round1_pairs'], parsed['tree_edges'], teams64, P_model,
        n_simulations=n_simulations, seed=seed,
    )

    rows = []
    for idx, team in enumerate(teams64):
        w = win_counts[idx]
        row = {
            'team': team,
            'seed': parsed['seeds'].get(team, np.nan),
            'tree_expected_wins': float(w.mean()),
            'tree_std_wins': float(w.std()),
        }
        for col, k in zip(TREE_PROB_COLS, THRESHOLDS):
            row[col] = float(np.mean(w >= k))
        rows.append(row)

    tree_df = pd.DataFrame(rows)
    parsed['P_model'] = P_model
    parsed['win_counts'] = win_counts
    parsed['slot_winners'] = slot_winners
    return tree_df, parsed


# ---------------------------------------------------------------------------
# Expected Value (EV) & Game-Theory Bracket Pool Optimizer
# ---------------------------------------------------------------------------
def _exact_dp_optimal_bracket(round1_pairs, tree_edges, n_teams, slot_rewards):
    """Find the globally optimal valid 63-game bracket via bottom-up dynamic programming."""
    subtree_teams = [None] * 63
    dp = np.full((63, n_teams), -np.inf, dtype=float)
    back_left = {}
    back_right = {}

    for g, (h, a) in enumerate(round1_pairs):
        subtree_teams[g] = [h, a]
        dp[g, h] = slot_rewards[g, h]
        dp[g, a] = slot_rewards[g, a]

    for offset, (left_g, right_g) in enumerate(tree_edges):
        g = 32 + offset
        left_candidates = subtree_teams[left_g]
        right_candidates = subtree_teams[right_g]
        subtree_teams[g] = left_candidates + right_candidates

        best_right_team = max(right_candidates, key=lambda u: dp[right_g, u])
        best_left_team = max(left_candidates, key=lambda u: dp[left_g, u])

        for t in left_candidates:
            dp[g, t] = slot_rewards[g, t] + dp[left_g, t] + dp[right_g, best_right_team]
            back_left[(g, t)] = t
            back_right[(g, t)] = best_right_team
        for t in right_candidates:
            dp[g, t] = slot_rewards[g, t] + dp[right_g, t] + dp[left_g, best_left_team]
            back_left[(g, t)] = best_left_team
            back_right[(g, t)] = t

    champ = int(np.argmax(dp[62]))
    picks = [None] * 63

    def _reconstruct(g, winner_idx):
        picks[g] = int(winner_idx)
        if g >= 32:
            left_g, right_g = tree_edges[g - 32]
            _reconstruct(left_g, back_left[(g, winner_idx)])
            _reconstruct(right_g, back_right[(g, winner_idx)])

    _reconstruct(62, champ)
    return picks


def optimize_bracket_pool(parsed_tree, scores_df, pool_size=100, n_simulations=10000, seed=42):
    """Compute Max-EV (Chalk) and Pool-Leverage Optimal 63-game brackets and simulate pool finish."""
    teams64 = parsed_tree['teams64']
    n_teams = len(teams64)
    round1_pairs = parsed_tree['round1_pairs']
    tree_edges = parsed_tree['tree_edges']
    slot_winners = parsed_tree['slot_winners']
    slot_rounds = np.asarray(parsed_tree['slot_rounds'], dtype=int)
    round_pts = np.array([ROUND_POINTS[r] for r in slot_rounds], dtype=float)

    # Model slot win probability matrix: slot_prob_model[g, t]
    slot_prob_model = np.zeros((63, n_teams), dtype=float)
    for g in range(63):
        counts = np.bincount(slot_winners[g], minlength=n_teams)
        slot_prob_model[g] = counts / slot_winners.shape[1]

    # Public slot win probability matrix via public bracket tree simulation
    score_map = dict(zip(scores_df['team'], scores_df['ensemble_score']))
    P_pub = build_public_prob_matrix(teams64, score_map, parsed_tree['seeds'])
    _, pub_slot_winners = simulate_bracket_tree(
        round1_pairs, tree_edges, teams64, P_pub, n_simulations=max(2000, n_simulations // 2), seed=seed + 99
    )
    slot_prob_public = np.zeros((63, n_teams), dtype=float)
    for g in range(63):
        counts = np.bincount(pub_slot_winners[g], minlength=n_teams)
        slot_prob_public[g] = counts / pub_slot_winners.shape[1]

    # 1. Chalk / Max Expected Points Bracket
    ev_rewards = round_pts[:, None] * slot_prob_model
    max_ev_picks = _exact_dp_optimal_bracket(round1_pairs, tree_edges, n_teams, ev_rewards)

    # 2. Game-Theory Pool Leverage Bracket
    alpha = min(0.40, 0.08 * np.log(max(pool_size, 1)))
    leverage_ratio = slot_prob_model / np.maximum(slot_prob_public, 0.01)
    # Apply stronger leverage weighting in later rounds where pool differentiation matters most
    round_scale = np.array([0.25 if r <= 2 else (0.65 if r <= 4 else 1.0) for r in slot_rounds])[:, None]
    leverage_multiplier = np.power(leverage_ratio, alpha * round_scale)
    lev_rewards = ev_rewards * leverage_multiplier
    leverage_picks = _exact_dp_optimal_bracket(round1_pairs, tree_edges, n_teams, lev_rewards)

    # Evaluate both brackets across the N simulated true tournaments vs (pool_size - 1) public brackets
    n_eval_sims = min(2000, slot_winners.shape[1])
    n_opponents = max(1, min(pool_size - 1, 250))
    opp_brackets = pub_slot_winners[:, :n_opponents]  # (63, n_opponents)
    true_outcomes = slot_winners[:, :n_eval_sims]     # (63, n_eval_sims)

    max_ev_arr = np.asarray(max_ev_picks, dtype=np.int32)[:, None]
    lev_arr = np.asarray(leverage_picks, dtype=np.int32)[:, None]

    ev_scores = ((max_ev_arr == true_outcomes) * round_pts[:, None]).sum(axis=0)
    lev_scores = ((lev_arr == true_outcomes) * round_pts[:, None]).sum(axis=0)

    # Vectorized opponent scores: (n_eval_sims, n_opponents)
    opp_scores = np.zeros((n_eval_sims, n_opponents), dtype=float)
    for g in range(63):
        opp_scores += (true_outcomes[g, :, None] == opp_brackets[g, None, :]) * round_pts[g]
    max_opp = opp_scores.max(axis=1)
    top10_cutoff = np.percentile(opp_scores, 90, axis=1)

    actual_winners = np.asarray(parsed_tree['actual_slot_winners'], dtype=np.int32)
    actual_ev_pts = float((( np.asarray(max_ev_picks) == actual_winners) * round_pts).sum())
    actual_lev_pts = float(((np.asarray(leverage_picks) == actual_winners) * round_pts).sum())

    rows = []
    for g in range(63):
        r_num = int(slot_rounds[g])
        if g < 32:
            h_idx, a_idx = round1_pairs[g]
            matchup_label = f"R64: ({parsed_tree['seeds'].get(teams64[h_idx], '?')}) {teams64[h_idx]} vs ({parsed_tree['seeds'].get(teams64[a_idx], '?')}) {teams64[a_idx]}"
        else:
            lg, rg = tree_edges[g - 32]
            matchup_label = f"{ROUND_LABELS[r_num]}: Winner(G{lg+1}) vs Winner(G{rg+1})"
        ev_t = max_ev_picks[g]
        lev_t = leverage_picks[g]
        act_t = actual_winners[g]
        rows.append({
            'game_slot': g + 1,
            'round_number': r_num,
            'round_name': ROUND_LABELS[r_num],
            'round_points': int(round_pts[g]),
            'matchup': matchup_label,
            'max_ev_pick': teams64[ev_t],
            'max_ev_win_prob': float(slot_prob_model[g, ev_t]),
            'max_ev_public_pct': float(slot_prob_public[g, ev_t]),
            'max_ev_expected_pts': float(ev_rewards[g, ev_t]),
            'leverage_pick': teams64[lev_t],
            'leverage_win_prob': float(slot_prob_model[g, lev_t]),
            'leverage_public_pct': float(slot_prob_public[g, lev_t]),
            'leverage_ratio': float(leverage_ratio[g, lev_t]),
            'leverage_expected_pts': float(ev_rewards[g, lev_t]),
            'actual_winner': teams64[act_t] if 0 <= act_t < n_teams else '',
            'max_ev_correct': bool(ev_t == act_t),
            'leverage_correct': bool(lev_t == act_t),
        })

    bracket_df = pd.DataFrame(rows)
    summary = {
        'pool_size': int(pool_size),
        'leverage_alpha': float(alpha),
        'max_ev_champion': teams64[max_ev_picks[62]],
        'leverage_champion': teams64[leverage_picks[62]],
        'actual_champion': teams64[actual_winners[62]] if 0 <= actual_winners[62] < n_teams else None,
        'max_ev_expected_points': float(ev_scores.mean()),
        'leverage_expected_points': float(lev_scores.mean()),
        'max_ev_pool_win_prob': float(np.mean(ev_scores >= max_opp)),
        'leverage_pool_win_prob': float(np.mean(lev_scores >= max_opp)),
        'max_ev_top10pct_prob': float(np.mean(ev_scores >= top10_cutoff)),
        'leverage_top10pct_prob': float(np.mean(lev_scores >= top10_cutoff)),
        'actual_max_ev_points': actual_ev_pts,
        'actual_leverage_points': actual_lev_pts,
    }
    return bracket_df, summary, slot_prob_model, slot_prob_public


def add_ranks(table):
    table = table.copy()
    table.insert(2, 'simulated_rank', np.arange(1, len(table) + 1))
    if 'actual_wins' in table and table['actual_wins'].notna().all():
        table.insert(3, 'actual_rank', table['actual_wins'].rank(ascending=False, method='min').astype(int))
    return table


def season_metrics(table):
    actual = table['actual_wins'].to_numpy(float)
    pred = table['simulated_mean'].to_numpy(float)
    rho = spearmanr(actual, pred)[0] if len(table) > 1 else np.nan
    champ = table.loc[table['actual_wins'].idxmax()]
    prob_cols_to_use = TREE_PROB_COLS if all(c in table.columns for c in TREE_PROB_COLS) and table[TREE_PROB_COLS[-1]].notna().any() else (
        CAL_PROB_COLS if all(c in table.columns for c in CAL_PROB_COLS) else PROB_COLS
    )
    out = {
        'Season': int(table['season'].iloc[0]),
        'Teams': len(table),
        'MAE': float(np.mean(np.abs(actual - pred))),
        'RMSE': float(np.sqrt(np.mean((actual - pred) ** 2))),
        'Spearman': float(0.0 if np.isnan(rho) else rho),
        'Coverage_90': float(np.mean((actual >= table['percentile_5']) & (actual <= table['percentile_95']))),
        'Champion': champ['team'],
        'Champion_Sim_Rank': int(champ['simulated_rank']),
        'Champion_P_ge_6': float(champ[prob_cols_to_use[-1]]),
        'Champion_Raw_P_ge_6': float(champ[PROB_COLS[-1]]),
    }
    for col, k in zip(prob_cols_to_use, THRESHOLDS):
        valid = table[col].notna()
        out[f'Brier_ge_{k}'] = float(np.mean((table.loc[valid, col] - (actual[valid] >= k)) ** 2))
    return out


# ---------------------------------------------------------------------------
# Visualizations
# ---------------------------------------------------------------------------
def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches='tight', format='jpg')
    plt.close(fig)
    print(f"      [Plot] {path}")


def _has_actuals(table):
    return 'actual_wins' in table and table['actual_wins'].notna().all()


def _best_prob_cols(table):
    if all(c in table.columns for c in TREE_PROB_COLS) and table[TREE_PROB_COLS[-1]].notna().any():
        return TREE_PROB_COLS, 'Bracket-Tree Simulated Probability'
    if all(c in table.columns for c in CAL_PROB_COLS):
        return CAL_PROB_COLS, 'Isotonic-Calibrated Frequency'
    return PROB_COLS, 'Simulated Frequency'


def plot_forest(table, season, path, top_n=25):
    top = table.head(top_n).iloc[::-1]
    y = np.arange(len(top))
    fig, ax = plt.subplots(figsize=(11, max(6, 0.38 * len(top))))
    ax.hlines(y, top['percentile_5'], top['percentile_95'], color='#9ecae1', lw=7, label='90% simulated interval')
    ax.scatter(top['simulated_mean'], y, color='#08519c', s=50, zorder=3, label='Bootstrap mean')
    if 'tree_expected_wins' in top.columns and top['tree_expected_wins'].notna().any():
        ax.scatter(top['tree_expected_wins'], y, marker='s', color='#2ca02c', s=45, zorder=4, label='Bracket-tree expected wins')
    if _has_actuals(table):
        ax.scatter(top['actual_wins'], y, marker='D', color='#d62728', s=45, zorder=5, label='Actual wins')
    ax.set_yticks(y)
    ax.set_yticklabels(top['team'])
    ax.set_xlabel('Tournament Wins')
    ax.set_title(f'{season}: Top {len(top)} Teams - Simulated Win Range & Bracket Expectation', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right')
    _save(fig, path)


def plot_probability_heatmap(table, season, path, top_n=25):
    cols, label = _best_prob_cols(table)
    top = table.head(top_n).set_index('team')[cols].fillna(0.0)
    top.columns = [ROUND_LABELS[k] for k in THRESHOLDS]
    fig, ax = plt.subplots(figsize=(10, max(6, 0.36 * len(top))))
    sns.heatmap(top, annot=True, fmt='.1%', cmap='YlGnBu', vmin=0, vmax=1,
                cbar_kws={'label': label}, ax=ax)
    ax.set_title(f'{season}: Round-Advancement Probabilities ({label})', fontsize=13, fontweight='bold')
    ax.set_xlabel('Round reached (>= k wins)')
    ax.set_ylabel('')
    _save(fig, path)


def plot_violins(table, samples, season, path, top_n=12):
    teams = table.head(top_n)['team'].tolist()
    fig, ax = plt.subplots(figsize=(14, 7))
    parts = ax.violinplot([samples[t] for t in teams], showmedians=True, widths=0.85)
    for body, color in zip(parts['bodies'], sns.color_palette('viridis', len(teams))):
        body.set_facecolor(color)
        body.set_alpha(0.65)
    if _has_actuals(table):
        actual = table.set_index('team').loc[teams, 'actual_wins']
        ax.scatter(np.arange(1, len(teams) + 1), actual, marker='D', color='#d62728', zorder=5, label='Actual wins')
        ax.legend()
    ax.set_xticks(np.arange(1, len(teams) + 1))
    ax.set_xticklabels(teams, rotation=40, ha='right')
    ax.set_ylabel('Simulated Tournament Wins')
    ax.set_title(f'{season}: Stratified Outcome Distributions (Top {len(teams)})', fontsize=14, fontweight='bold')
    _save(fig, path)


def plot_championship_odds(table, season, path, top_n=15):
    cols, label = _best_prob_cols(table)
    champ_col = cols[-1]
    top = table.dropna(subset=[champ_col]).sort_values(champ_col, ascending=False).head(top_n)
    colors = ['#4c72b0'] * len(top)
    title = f'{season}: Championship Probability ({label})'
    if _has_actuals(table):
        champ_wins = table['actual_wins'].max()
        colors = ['#d62728' if w == champ_wins else '#4c72b0' for w in top['actual_wins']]
        title += ' [red = actual champion]'
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(top['team'], top[champ_col], color=colors, edgecolor='black')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:.1%}'))
    ax.set_ylabel('P(Champion)')
    ax.set_title(title, fontsize=13, fontweight='bold')
    plt.setp(ax.get_xticklabels(), rotation=40, ha='right')
    _save(fig, path)


def plot_score_vs_baseline(table, season, slope, intercept, path):
    cols, _ = _best_prob_cols(table)
    fig, ax = plt.subplots(figsize=(10, 6))
    sc = ax.scatter(table['ensemble_score'], table['simulated_mean'], c=table[cols[3]].fillna(0), cmap='plasma',
                    s=55, edgecolor='black')
    ax.errorbar(table['ensemble_score'], table['simulated_mean'],
                yerr=[table['simulated_mean'] - table['percentile_5'], table['percentile_95'] - table['simulated_mean']],
                fmt='none', ecolor='gray', alpha=0.3)
    for _, r in table.head(8).iterrows():
        ax.annotate(r['team'], (r['ensemble_score'], r['simulated_mean']), fontsize=8,
                    xytext=(4, 4), textcoords='offset points')
    fig.colorbar(sc, label='P(Final 4)')
    ax.set_xlabel('SHAP Ensemble Score (z-units)')
    ax.set_ylabel('Simulated Mean Wins')
    ax.set_title(f'{season}: Ensemble Score -> Simulated Wins (calib {slope:.2f}x + {intercept:.2f})',
                 fontsize=13, fontweight='bold')
    _save(fig, path)


def plot_pred_vs_actual(table, season, path):
    cols, _ = _best_prob_cols(table)
    jitter = np.random.default_rng(0).uniform(-0.15, 0.15, len(table))
    fig, ax = plt.subplots(figsize=(10, 8))
    sc = ax.scatter(table['simulated_mean'], table['actual_wins'] + jitter, c=table[cols[3]].fillna(0),
                    cmap='plasma', s=60, edgecolor='black', alpha=0.8)
    lim = [min(table['simulated_mean'].min(), 0) - 0.3, max(table['simulated_mean'].max(), 6) + 0.3]
    ax.plot(lim, lim, '--', color='gray', label='Perfect prediction')
    for idx, r in table.iterrows():
        if r['simulated_rank'] <= 8 or r['actual_wins'] >= 4:
            ax.annotate(r['team'], (r['simulated_mean'], r['actual_wins'] + jitter[idx]),
                        fontsize=8, xytext=(4, 4), textcoords='offset points')
    fig.colorbar(sc, label='P(Final 4)')
    ax.set_xlabel('Simulated Mean Wins')
    ax.set_ylabel('Actual Wins (jittered)')
    ax.set_title(f'{season}: Simulated vs Actual Tournament Wins', fontsize=14, fontweight='bold')
    ax.legend()
    _save(fig, path)


def plot_rank_comparison(table, season, path, top_n=20):
    top = table.head(top_n)
    fig, ax = plt.subplots(figsize=(9, 10))
    for _, r in top.iterrows():
        color = '#2ca02c' if r['actual_rank'] <= r['simulated_rank'] else '#d62728'
        ax.plot([0, 1], [r['simulated_rank'], r['actual_rank']], marker='o', color=color, alpha=0.7)
        ax.text(-0.03, r['simulated_rank'], r['team'], ha='right', va='center', fontsize=8)
        ax.text(1.03, r['actual_rank'], r['team'], ha='left', va='center', fontsize=8)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Simulated Rank', 'Actual Rank'])
    ax.set_xlim(-0.45, 1.45)
    ax.invert_yaxis()
    ax.set_title(f'{season}: Simulated vs Actual Rank (Top {len(top)})', fontsize=13, fontweight='bold')
    sns.despine(left=True, bottom=True)
    _save(fig, path)


def plot_bracket_leverage(bracket_df, pool_summary, season, path):
    """Plot Final Four / Elite 8 / Title game leverage vs model win probability."""
    late = bracket_df[bracket_df['round_number'] >= 3].copy()
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))

    sc = axes[0].scatter(
        late['leverage_public_pct'], late['leverage_win_prob'],
        c=late['round_number'], cmap='viridis', s=90, edgecolor='black', alpha=0.85
    )
    axes[0].plot([0, 1], [0, 1], '--', color='gray', label='Public = Model (Fair odds)')
    for _, r in late.iterrows():
        if r['round_number'] >= 4 or r['leverage_ratio'] >= 1.25:
            axes[0].annotate(f"{r['leverage_pick']} ({r['round_name']})",
                             (r['leverage_public_pct'], r['leverage_win_prob']),
                             fontsize=8, xytext=(4, 4), textcoords='offset points')
    axes[0].set_xlabel('Estimated Public Pick %')
    axes[0].set_ylabel('Model Advancement Probability')
    axes[0].set_title(f'{season}: Model vs Public Odds (Above line = +EV Leverage)', fontsize=12, fontweight='bold')
    axes[0].legend()
    fig.colorbar(sc, ax=axes[0], label='Round (3=E8, 4=F4, 5=Title, 6=Champ)')

    metrics_names = ['Expected Pool Pts', '1st-Place Win % (x1000)', 'Top-10% Rate (x1000)', 'Actual Historical Pts']
    ev_vals = [
        pool_summary['max_ev_expected_points'],
        pool_summary['max_ev_pool_win_prob'] * 1000,
        pool_summary['max_ev_top10pct_prob'] * 1000,
        pool_summary['actual_max_ev_points'],
    ]
    lev_vals = [
        pool_summary['leverage_expected_points'],
        pool_summary['leverage_pool_win_prob'] * 1000,
        pool_summary['leverage_top10pct_prob'] * 1000,
        pool_summary['actual_leverage_points'],
    ]
    x = np.arange(len(metrics_names))
    w = 0.36
    axes[1].bar(x - w / 2, ev_vals, w, label=f"Max-EV Chalk ({pool_summary['max_ev_champion']})", color='#4c72b0', edgecolor='black')
    axes[1].bar(x + w / 2, lev_vals, w, label=f"Leverage Optimal ({pool_summary['leverage_champion']})", color='#2ca02c', edgecolor='black')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(metrics_names, rotation=20, ha='right')
    axes[1].set_title(f"{season}: Bracket Optimizer ({pool_summary['pool_size']}-Entry Pool Simulation)",
                      fontsize=12, fontweight='bold')
    axes[1].legend()
    _save(fig, path)


def plot_calibration_scatter(history, target, slope, intercept, path):
    fit_df = history[history['season'] != target]
    fig, ax = plt.subplots(figsize=(11, 7))
    sns.scatterplot(data=fit_df, x='ensemble_score', y='actual_wins', hue='season', palette='viridis',
                    alpha=0.7, ax=ax)
    xs = np.linspace(history['ensemble_score'].min(), history['ensemble_score'].max(), 100)
    ax.plot(xs, slope * xs + intercept, color='#d62728', lw=2.5, label=f'Calibration: {slope:.2f}x + {intercept:.2f}')
    ax.set_xlabel('SHAP Ensemble Score (z-units)')
    ax.set_ylabel('Actual Tournament Wins')
    ax.set_title(f'Score-to-Wins Calibration (seasons excluding {target})', fontsize=14, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
    _save(fig, path)


def plot_residual_pool(residuals, path):
    fig, ax = plt.subplots(figsize=(11, 6))
    sns.histplot(residuals, bins=40, kde=True, color='#4c72b0', ax=ax)
    for q, ls in [(5, ':'), (50, '-'), (95, ':')]:
        v = np.percentile(residuals, q)
        ax.axvline(v, color='#d62728', ls=ls, label=f'P{q} = {v:.2f}')
    ax.set_xlabel('Held-out Residual (actual - calibrated prediction)')
    ax.set_title(f'Empirical Residual Pool (n={len(residuals)})', fontsize=14, fontweight='bold')
    ax.legend()
    _save(fig, path)


def plot_shap_recipe(df_shap, label, path, top_n=20):
    top = df_shap.reindex(df_shap['shap_importance'].abs().sort_values(ascending=False).index).head(top_n)
    fig, ax = plt.subplots(figsize=(12, 8))
    colors = ['#2ca02c' if v >= 0 else '#d62728' for v in top['shap_importance']]
    ax.barh(top['feature'][::-1], top['shap_importance'][::-1], color=colors[::-1], edgecolor='black')
    ax.set_xlabel('Blended SHAP weight')
    ax.set_title(f'Best Ensemble Feature Recipe (Top {len(top)})\n{label}', fontsize=12, fontweight='bold')
    _save(fig, path)


def plot_reliability(all_tables, path):
    """Compare Raw Bootstrap, Isotonic Calibrated, and Bracket-Tree advancement reliability."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10), sharex=True, sharey=True)
    has_cal = all(c in all_tables.columns for c in CAL_PROB_COLS)
    has_tree = all(c in all_tables.columns for c in TREE_PROB_COLS) and all_tables[TREE_PROB_COLS[0]].notna().any()

    for ax, raw_col, cal_col, tree_col, k in zip(axes.flat, PROB_COLS, CAL_PROB_COLS, TREE_PROB_COLS, THRESHOLDS):
        ax.plot([0, 1], [0, 1], '--', color='gray', label='Perfect calibration')
        series_list = [('Raw Stratified Bootstrap', raw_col, '#4c72b0', 'o')]
        if has_cal:
            series_list.append(('Isotonic Calibrated', cal_col, '#ff7f0e', 's'))
        if has_tree:
            series_list.append(('Bracket-Tree Simulator', tree_col, '#2ca02c', 'D'))

        for name, col, color, marker in series_list:
            sub = all_tables[[col, 'actual_wins']].dropna()
            df = pd.DataFrame({'p': sub[col], 'o': (sub['actual_wins'] >= k).astype(float)})
            df['bin'] = pd.cut(df['p'], np.linspace(0, 1, 9), include_lowest=True)
            g = df.groupby('bin', observed=True).agg(p=('p', 'mean'), o=('o', 'mean'), n=('o', 'size')).dropna()
            ax.plot(g['p'], g['o'], marker=marker, color=color, lw=1.8, alpha=0.85, label=name)
        ax.set_title(f'>= {k} wins ({ROUND_LABELS[k]})')
        ax.set_xlabel('Predicted probability')
        ax.set_ylabel('Observed frequency')
    axes.flat[0].legend(loc='upper left', fontsize=8)
    fig.suptitle('Reliability Diagram: Raw Bootstrap vs Isotonic vs Bracket-Tree (Backtest)',
                 fontsize=15, fontweight='bold')
    _save(fig, path)


def plot_brier(metrics, path):
    fig, ax = plt.subplots(figsize=(13, 6))
    for k in THRESHOLDS:
        ax.plot(metrics['Season'], metrics[f'Brier_ge_{k}'], marker='o', lw=2, label=ROUND_LABELS[k])
    ax.set_xticks(metrics['Season'])
    ax.set_xlabel('Season')
    ax.set_ylabel('Brier Score (lower is better)')
    ax.set_title('Brier Score by Round Threshold Over Time', fontsize=14, fontweight='bold')
    ax.legend(title='Threshold')
    _save(fig, path)


def plot_season_diagnostics(metrics, path):
    fig, axes = plt.subplots(3, 1, figsize=(13, 11), sharex=True)
    axes[0].plot(metrics['Season'], metrics['MAE'], marker='o', color='#d62728', label='MAE')
    axes[0].plot(metrics['Season'], metrics['RMSE'], marker='s', color='#ff7f0e', label='RMSE')
    axes[0].set_ylabel('Wins error')
    axes[0].legend()
    axes[1].bar(metrics['Season'], metrics['Spearman'], color='#4c72b0', edgecolor='black')
    axes[1].set_ylabel('Spearman rho')
    axes[2].plot(metrics['Season'], metrics['Coverage_90'], marker='o', color='#2ca02c', label='Observed')
    axes[2].axhline(0.9, ls='--', color='gray', label='Nominal 90%')
    axes[2].set_ylim(0, 1.05)
    axes[2].set_ylabel('90% interval coverage')
    axes[2].legend()
    axes[2].set_xticks(metrics['Season'])
    axes[2].set_xlabel('Season')
    fig.suptitle('Monte Carlo Accuracy Diagnostics by Season (backtest)', fontsize=15, fontweight='bold')
    _save(fig, path)


def plot_champion_tracking(metrics, path):
    fig, ax1 = plt.subplots(figsize=(13, 6))
    ax1.bar(metrics['Season'], metrics['Champion_P_ge_6'], color='#2ca02c', alpha=0.75,
            label='Bracket-Tree / Calibrated P(Champion)')
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:.1%}'))
    ax1.set_ylabel('Simulated title probability of actual champion')
    ax2 = ax1.twinx()
    ax2.plot(metrics['Season'], metrics['Champion_Sim_Rank'], color='#d62728', marker='D', lw=2, label='Simulated rank')
    ax2.set_ylabel('Simulated rank of actual champion')
    ax2.invert_yaxis()
    for _, r in metrics.iterrows():
        ax1.text(r['Season'], r['Champion_P_ge_6'], r['Champion'], rotation=90, ha='center', va='bottom', fontsize=8)
    ax1.set_xticks(metrics['Season'])
    ax1.set_title('How Did the Upgraded Pipeline Rate Each Actual Champion?', fontsize=14, fontweight='bold')
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc='upper left')
    _save(fig, path)


def plot_top_programs(all_tables, path, top_n=10):
    counts = all_tables[all_tables['simulated_rank'] <= top_n]['team'].value_counts()
    pivot = all_tables[all_tables['team'].isin(counts.head(25).index)].pivot_table(
        index='team', columns='season', values='simulated_mean')
    pivot = pivot.loc[pivot.mean(axis=1).sort_values(ascending=False).index]
    fig, ax = plt.subplots(figsize=(1.0 * pivot.shape[1] + 5, 0.4 * len(pivot) + 3))
    sns.heatmap(pivot, annot=True, fmt='.1f', cmap='rocket_r', ax=ax, cbar_kws={'label': 'Simulated mean wins'})
    ax.set_title('Simulated Mean Wins of Recurring Top Programs (backtest)', fontsize=13, fontweight='bold')
    ax.set_ylabel('')
    _save(fig, path)


def render_season_plots(table, samples, season, slope, intercept, out_dir, bracket_df=None, pool_summary=None):
    os.makedirs(out_dir, exist_ok=True)
    plot_forest(table, season, os.path.join(out_dir, 'forest_top25.jpg'))
    plot_probability_heatmap(table, season, os.path.join(out_dir, 'advancement_heatmap.jpg'))
    plot_violins(table, samples, season, os.path.join(out_dir, 'distribution_violins.jpg'))
    plot_championship_odds(table, season, os.path.join(out_dir, 'championship_odds.jpg'))
    plot_score_vs_baseline(table, season, slope, intercept, os.path.join(out_dir, 'score_to_wins.jpg'))
    if _has_actuals(table):
        plot_pred_vs_actual(table, season, os.path.join(out_dir, 'simulated_vs_actual.jpg'))
        plot_rank_comparison(table, season, os.path.join(out_dir, 'rank_comparison.jpg'))
    if bracket_df is not None and pool_summary is not None:
        plot_bracket_leverage(bracket_df, pool_summary, season, os.path.join(out_dir, 'bracket_pool_optimizer.jpg'))


def render_backtest_plots(all_tables, metrics, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    plot_reliability(all_tables, os.path.join(out_dir, 'reliability_diagram.jpg'))
    plot_brier(metrics, os.path.join(out_dir, 'brier_by_threshold.jpg'))
    plot_season_diagnostics(metrics, os.path.join(out_dir, 'season_diagnostics.jpg'))
    plot_champion_tracking(metrics, os.path.join(out_dir, 'champion_tracking.jpg'))
    plot_top_programs(all_tables, os.path.join(out_dir, 'top_programs_across_seasons.jpg'))
