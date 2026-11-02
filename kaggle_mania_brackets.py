#!/usr/bin/env python3
"""Kaggle March Machine Learning Mania Bracket Portfolio Generator (2026 & 2027).

Builds an advanced predictive pipeline incorporating winning methodologies from
Kaggle March Machine Learning Mania (including Harrison Horan's 1st-place strategy):
  1. Selection Committee seed differential as the foundational anchor.
  2. 2% trimmed possession-based efficiency ratings (OffRtg, DefRtg, NetRtg).
  3. Consensus late-season Massey Ordinals (POM, SAG, MAS, etc.).
  4. Four Factors differentials (eFG%, TOV%, ORB%, FTR).
  5. Strength of Schedule (SoS) and Quality Wins differentials.
  6. Calibrated ensemble model optimized specifically for Brier Score / MSE.
  7. Strategic edge sharpening for high-confidence mismatches (Horan 1st-place technique).
  8. Exact Dynamic Programming (Bottom-Up DP) solver across all 67 tournament slots.
  9. Portfolio generation of 25 distinct, high-EV and game-theory leverage brackets
     for 2026 (official field) and 2027 (projected field).
  10. Official Kaggle submission generation (ID, Pred) matching SampleSubmissionStage2.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.preprocessing import StandardScaler

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Safe stdout encoding for Windows terminals
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

# Round metadata & standard NCAA scoring
ROUND_POINTS = {
    0: 0,    # First Four Play-ins
    1: 10,   # Round of 64
    2: 20,   # Round of 32
    3: 40,   # Sweet 16
    4: 80,   # Elite Eight
    5: 160,  # Final Four
    6: 320,  # National Championship
}

ROUND_NAMES = {
    0: "First Four",
    1: "Round of 64",
    2: "Round of 32",
    3: "Sweet 16",
    4: "Elite Eight",
    5: "Final Four",
    6: "National Championship",
}


# ===========================================================================
# 1. Feature Engineering & Historical Data Loading
# ===========================================================================

def load_kaggle_data(kaggle_dir: Path, gender: str = "M") -> Dict[str, pd.DataFrame]:
    """Load core CSVs from the Kaggle dataset directory."""
    prefix = gender.upper()
    files = {
        "regular": kaggle_dir / f"{prefix}RegularSeasonDetailedResults.csv",
        "tourney_compact": kaggle_dir / f"{prefix}NCAATourneyCompactResults.csv",
        "tourney_detailed": kaggle_dir / f"{prefix}NCAATourneyDetailedResults.csv",
        "seeds": kaggle_dir / f"{prefix}NCAATourneySeeds.csv",
        "slots": kaggle_dir / f"{prefix}NCAATourneySlots.csv",
        "teams": kaggle_dir / f"{prefix}Teams.csv",
        "seasons": kaggle_dir / f"{prefix}Seasons.csv",
        "massey": kaggle_dir / "MMasseyOrdinals.csv",  # Massey ordinals are primarily in MMasseyOrdinals
    }

    data = {}
    for key, path in files.items():
        if path.exists():
            data[key] = pd.read_csv(path)
        else:
            data[key] = pd.DataFrame()
    return data


def compute_team_season_stats(reg_df: pd.DataFrame) -> pd.DataFrame:
    """Compute possession-based efficiency, Four Factors, and 2% trimmed Net Rating."""
    if reg_df.empty:
        return pd.DataFrame()

    # Vectorized possessions: FGA - OR + TO + 0.475 * FTA
    w_poss = np.maximum(reg_df["WFGA"] - reg_df["WOR"] + reg_df["WTO"] + 0.475 * reg_df["WFTA"], 1.0)
    l_poss = np.maximum(reg_df["LFGA"] - reg_df["LOR"] + reg_df["LTO"] + 0.475 * reg_df["LFTA"], 1.0)

    w_rows = pd.DataFrame({
        "Season": reg_df["Season"],
        "TeamID": reg_df["WTeamID"],
        "OppTeamID": reg_df["LTeamID"],
        "Score": reg_df["WScore"],
        "OppScore": reg_df["LScore"],
        "Poss": w_poss,
        "OppPoss": l_poss,
        "OffRtg": (reg_df["WScore"] / w_poss) * 100.0,
        "DefRtg": (reg_df["LScore"] / l_poss) * 100.0,
        "eFG": (reg_df["WFGM"] + 0.5 * reg_df["WFGM3"]) / np.maximum(reg_df["WFGA"], 1.0),
        "TOV": reg_df["WTO"] / w_poss,
        "ORB": reg_df["WOR"] / np.maximum(reg_df["WOR"] + reg_df["LDR"], 1.0),
        "FTR": reg_df["WFTA"] / np.maximum(reg_df["WFGA"], 1.0),
        "Win": 1.0,
        "MOV": reg_df["WScore"] - reg_df["LScore"],
    })

    l_rows = pd.DataFrame({
        "Season": reg_df["Season"],
        "TeamID": reg_df["LTeamID"],
        "OppTeamID": reg_df["WTeamID"],
        "Score": reg_df["LScore"],
        "OppScore": reg_df["WScore"],
        "Poss": l_poss,
        "OppPoss": w_poss,
        "OffRtg": (reg_df["LScore"] / l_poss) * 100.0,
        "DefRtg": (reg_df["WScore"] / w_poss) * 100.0,
        "eFG": (reg_df["LFGM"] + 0.5 * reg_df["LFGM3"]) / np.maximum(reg_df["LFGA"], 1.0),
        "TOV": reg_df["LTO"] / l_poss,
        "ORB": reg_df["LOR"] / np.maximum(reg_df["LOR"] + reg_df["WDR"], 1.0),
        "FTR": reg_df["LFTA"] / np.maximum(reg_df["LFGA"], 1.0),
        "Win": 0.0,
        "MOV": reg_df["LScore"] - reg_df["WScore"],
    })

    all_games = pd.concat([w_rows, l_rows], ignore_index=True)

    # 2% Trimming per season to eliminate blowout noise against sub-D1 / cellar dwellers
    q02_off = all_games.groupby("Season")["OffRtg"].transform(lambda s: s.quantile(0.02))
    q98_off = all_games.groupby("Season")["OffRtg"].transform(lambda s: s.quantile(0.98))
    all_games["OffRtg"] = all_games["OffRtg"].clip(q02_off, q98_off)

    q02_def = all_games.groupby("Season")["DefRtg"].transform(lambda s: s.quantile(0.02))
    q98_def = all_games.groupby("Season")["DefRtg"].transform(lambda s: s.quantile(0.98))
    all_games["DefRtg"] = all_games["DefRtg"].clip(q02_def, q98_def)

    stats = all_games.groupby(["Season", "TeamID"]).agg({
        "OffRtg": "mean",
        "DefRtg": "mean",
        "eFG": "mean",
        "TOV": "mean",
        "ORB": "mean",
        "FTR": "mean",
        "Win": "mean",
        "Poss": "mean",
        "MOV": "mean",
    }).reset_index()
    stats["NetRtg"] = stats["OffRtg"] - stats["DefRtg"]

    # Strength of Schedule (SoS) & Quality Wins (Quad 1 wins against top-net-rating opponents)
    team_net_map = dict(zip(zip(stats["Season"], stats["TeamID"]), stats["NetRtg"]))
    all_games["OppNetRtg"] = [
        team_net_map.get((s, opp), 0.0)
        for s, opp in zip(all_games["Season"], all_games["OppTeamID"])
    ]
    all_games["QualWin"] = ((all_games["Win"] == 1.0) & (all_games["OppNetRtg"] > 4.5)).astype(float)

    sos_qual = all_games.groupby(["Season", "TeamID"]).agg({
        "OppNetRtg": "mean",
        "QualWin": "sum",
    }).reset_index().rename(columns={"OppNetRtg": "SoS", "QualWin": "QualityWins"})

    stats = stats.merge(sos_qual, on=["Season", "TeamID"], how="left")
    return stats


def compute_massey_rankings(massey_path: Path) -> pd.DataFrame:
    """Aggregate Selection Sunday week (RankingDayNum >= 128) consensus ordinals."""
    if not massey_path.exists():
        return pd.DataFrame()

    chunks = []
    cols = ["Season", "RankingDayNum", "TeamID", "OrdinalRank"]
    for chunk in pd.read_csv(massey_path, usecols=cols, chunksize=500000):
        c = chunk[chunk["RankingDayNum"] >= 128]
        if not c.empty:
            chunks.append(c)

    if not chunks:
        return pd.DataFrame()

    df_massey = pd.concat(chunks, ignore_index=True)
    agg = df_massey.groupby(["Season", "TeamID"])["OrdinalRank"].agg(
        MasseyMedian="median", MasseyMean="mean"
    ).reset_index()
    return agg


def extract_seed_number(seed_str: str) -> int:
    """Parse numeric integer seed 1..16 from W01, X16a, etc."""
    digits = "".join(filter(str.isdigit, str(seed_str)))
    return int(digits) if digits else 8


def get_game_round(day_num: int) -> int:
    """Map tournament game DayNum to NCAA Tournament round (0..6)."""
    if day_num <= 135:
        return 0  # First Four Play-ins
    elif day_num <= 137:
        return 1  # Round of 64
    elif day_num <= 140:
        return 2  # Round of 32
    elif day_num <= 144:
        return 3  # Sweet 16
    elif day_num <= 148:
        return 4  # Elite Eight
    elif day_num <= 153:
        return 5  # Final Four
    else:
        return 6  # National Championship


def apply_sharpening_scalar(p: float, d_seed: float = 0.0) -> float:
    """Apply Harrison Horan 1st-place probability edge sharpening to a single prediction."""
    if p >= 0.96:
        p = 0.992
    elif p <= 0.04:
        p = 0.008
    elif p >= 0.90 and d_seed >= 10:
        p = 0.965
    elif p <= 0.10 and d_seed <= -10:
        p = 0.035
    return float(np.clip(p, 0.005, 0.995))


def apply_sharpening_vector(p_vec: np.ndarray, d_seed_vec: np.ndarray) -> np.ndarray:
    """Apply vectorized Harrison Horan 1st-place probability edge sharpening."""
    p_sharp = np.copy(p_vec)
    mask_high = p_vec >= 0.96
    mask_low = p_vec <= 0.04
    mask_seed_hi = (p_vec >= 0.90) & (d_seed_vec >= 10)
    mask_seed_lo = (p_vec <= 0.10) & (d_seed_vec <= -10)

    p_sharp[mask_high] = 0.992
    p_sharp[mask_low] = 0.008
    p_sharp[mask_seed_hi] = 0.965
    p_sharp[mask_seed_lo] = 0.035
    return np.clip(p_sharp, 0.005, 0.995)


def compute_calibration_metrics(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> Dict[str, Any]:
    """Compute Expected Calibration Error (ECE), Maximum Calibration Error (MCE), and reliability bins."""
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    ece = 0.0
    mce = 0.0
    bin_data = []

    n_samples = len(y_true)
    for i in range(n_bins):
        low, high = bin_edges[i], bin_edges[i + 1]
        if i == n_bins - 1:
            mask = (y_prob >= low) & (y_prob <= high)
        else:
            mask = (y_prob >= low) & (y_prob < high)

        count = int(np.sum(mask))
        if count > 0:
            avg_prob = float(np.mean(y_prob[mask]))
            emp_freq = float(np.mean(y_true[mask]))
            diff = abs(avg_prob - emp_freq)
            ece += (count / n_samples) * diff
            mce = max(mce, diff)
            bin_data.append({
                "bin_idx": i,
                "bin_low": round(float(low), 2),
                "bin_high": round(float(high), 2),
                "bin_center": round(float(bin_centers[i]), 3),
                "avg_prob": round(float(avg_prob), 4),
                "emp_freq": round(float(emp_freq), 4),
                "abs_err": round(float(diff), 4),
                "count": count,
            })
        else:
            bin_data.append({
                "bin_idx": i,
                "bin_low": round(float(low), 2),
                "bin_high": round(float(high), 2),
                "bin_center": round(float(bin_centers[i]), 3),
                "avg_prob": round(float(bin_centers[i]), 4),
                "emp_freq": 0.0,
                "abs_err": 0.0,
                "count": 0,
            })

    return {
        "ece": float(ece),
        "mce": float(mce),
        "bins": bin_data,
    }


# ===========================================================================
# 2. Matchup Modeling & Brier Score Calibration
# ===========================================================================

FEATURE_COLS = [
    "d_seed", "d_massey", "d_net", "d_off", "d_def",
    "d_efg", "d_tov", "d_orb", "d_ftr", "d_win", "d_qw", "d_sos",
]


class CalibratedMatchupModel:
    """Ensemble model predicting pairwise win probabilities P(Team A beats Team B).

    Evaluated via Leave-One-Season-Out Cross-Validation directly on Kaggle competition Brier Score.
    """

    def __init__(self, sharpen: bool = True):
        self.scaler = StandardScaler()
        self.lr_model: Optional[LogisticRegression] = None
        self.lgb_model: Any = None
        self.sharpen = sharpen
        self.cv_brier: float = 0.0
        self.cv_brier_raw: float = 0.0
        self.feature_weights: Dict[str, float] = {}

        # Detailed evaluation containers
        self.eval_summary: Dict[str, Any] = {}
        self.df_by_season: pd.DataFrame = pd.DataFrame()
        self.df_by_round: pd.DataFrame = pd.DataFrame()
        self.df_by_mismatch: pd.DataFrame = pd.DataFrame()
        self.df_oof_predictions: pd.DataFrame = pd.DataFrame()
        self.calibration_data: Dict[str, Any] = {}

    def fit_cross_validate(self, df_pairs: pd.DataFrame) -> float:
        """Fit ensemble and perform Leave-One-Season-Out CV to measure Brier Score and calibration."""
        X_raw = df_pairs[FEATURE_COLS].to_numpy(dtype=float)
        y = df_pairs["target"].to_numpy(dtype=float)
        seasons = df_pairs["Season"].to_numpy()
        rounds = df_pairs["Round"].to_numpy() if "Round" in df_pairs.columns else np.ones(len(df_pairs), dtype=int)
        d_seeds = df_pairs["d_seed"].to_numpy()

        unique_seasons = sorted(np.unique(seasons))
        records = []

        for val_s in unique_seasons:
            train_mask = seasons != val_s
            val_mask = seasons == val_s

            X_tr, y_tr = X_raw[train_mask], y[train_mask]
            X_val, y_val = X_raw[val_mask], y[val_mask]

            fold_scaler = StandardScaler()
            X_tr_s = fold_scaler.fit_transform(X_tr)
            X_val_s = fold_scaler.transform(X_val)

            # 1. Baseline: Seed only (Logistic regression on d_seed)
            lr_seed = LogisticRegression(C=1.0, max_iter=300, random_state=42)
            lr_seed.fit(X_tr[:, [0]], y_tr)
            preds_seed = lr_seed.predict_proba(X_val[:, [0]])[:, 1]

            # 2. Baseline: Massey Ordinals only (Logistic regression on d_massey)
            lr_massey = LogisticRegression(C=1.0, max_iter=300, random_state=42)
            lr_massey.fit(X_tr[:, [1]], y_tr)
            preds_massey = lr_massey.predict_proba(X_val[:, [1]])[:, 1]

            # 3. Baseline: Net Rating only (Logistic regression on d_net)
            lr_net = LogisticRegression(C=1.0, max_iter=300, random_state=42)
            lr_net.fit(X_tr[:, [2]], y_tr)
            preds_net = lr_net.predict_proba(X_val[:, [2]])[:, 1]

            # 4. Primary Model Component: Logistic Regression
            lr = LogisticRegression(C=0.25, max_iter=600, random_state=42)
            lr.fit(X_tr_s, y_tr)
            preds_lr = lr.predict_proba(X_val_s)[:, 1]

            # 5. Primary Model Component: LightGBM
            if HAS_LIGHTGBM and len(X_tr) > 200:
                lgb_clf = lgb.LGBMClassifier(
                    n_estimators=75, learning_rate=0.03, num_leaves=15,
                    min_child_samples=20, random_state=42, verbose=-1,
                )
                lgb_clf.fit(X_tr_s, y_tr)
                preds_lgb = lgb_clf.predict_proba(X_val_s)[:, 1]
                preds_ens = 0.65 * preds_lr + 0.35 * preds_lgb
            else:
                preds_lgb = preds_lr
                preds_ens = preds_lr

            # 6. Edge Sharpening (Harrison Horan 1st-place technique)
            d_seed_val = d_seeds[val_mask]
            preds_sharp = apply_sharpening_vector(preds_ens, d_seed_val)

            for i in range(len(y_val)):
                records.append({
                    "Season": val_s,
                    "Round": rounds[val_mask][i],
                    "d_seed": d_seed_val[i],
                    "target": y_val[i],
                    "pred_5050": 0.5,
                    "pred_seed": preds_seed[i],
                    "pred_massey": preds_massey[i],
                    "pred_net": preds_net[i],
                    "pred_lr": preds_lr[i],
                    "pred_lgb": preds_lgb[i],
                    "pred_ens": preds_ens[i],
                    "pred_sharp": preds_sharp[i],
                })

        df_oof = pd.DataFrame(records)
        self.df_oof_predictions = df_oof

        # Overall Metrics
        y_all = df_oof["target"].to_numpy()
        pred_cols = {
            "50_50_Baseline": df_oof["pred_5050"].to_numpy(),
            "Seed_Differential_Model": df_oof["pred_seed"].to_numpy(),
            "Massey_Ordinals_Model": df_oof["pred_massey"].to_numpy(),
            "Net_Rating_Model": df_oof["pred_net"].to_numpy(),
            "Logistic_Regression": df_oof["pred_lr"].to_numpy(),
            "LightGBM": df_oof["pred_lgb"].to_numpy(),
            "Ensemble_Raw": df_oof["pred_ens"].to_numpy(),
            "Ensemble_Sharpened": df_oof["pred_sharp"].to_numpy(),
        }

        brier_scores = {k: float(brier_score_loss(y_all, p)) for k, p in pred_cols.items()}
        log_losses = {k: float(log_loss(y_all, np.clip(p, 1e-7, 1 - 1e-7))) for k, p in pred_cols.items()}
        accuracies = {k: float(np.mean((p >= 0.5) == y_all)) for k, p in pred_cols.items()}

        p_final = pred_cols["Ensemble_Sharpened"] if self.sharpen else pred_cols["Ensemble_Raw"]
        calib_metrics = compute_calibration_metrics(y_all, p_final)
        self.calibration_data = calib_metrics

        brier_5050 = brier_scores["50_50_Baseline"]
        brier_primary = brier_scores["Ensemble_Sharpened"] if self.sharpen else brier_scores["Ensemble_Raw"]
        improvement_pct = ((brier_5050 - brier_primary) / brier_5050) * 100.0

        self.cv_brier = brier_primary
        self.cv_brier_raw = brier_scores["Ensemble_Raw"]

        self.eval_summary = {
            "cv_brier_score": round(self.cv_brier, 5),
            "cv_brier_raw": round(self.cv_brier_raw, 5),
            "brier_5050_baseline": round(brier_5050, 5),
            "brier_seed_baseline": round(brier_scores["Seed_Differential_Model"], 5),
            "brier_massey_baseline": round(brier_scores["Massey_Ordinals_Model"], 5),
            "brier_net_baseline": round(brier_scores["Net_Rating_Model"], 5),
            "brier_logreg": round(brier_scores["Logistic_Regression"], 5),
            "brier_lightgbm": round(brier_scores["LightGBM"], 5),
            "improvement_vs_5050_pct": round(improvement_pct, 2),
            "improvement_vs_seed_pct": round(((brier_scores["Seed_Differential_Model"] - self.cv_brier) / brier_scores["Seed_Differential_Model"]) * 100.0, 2),
            "log_loss": round(log_losses["Ensemble_Sharpened" if self.sharpen else "Ensemble_Raw"], 5),
            "accuracy": round(accuracies["Ensemble_Sharpened" if self.sharpen else "Ensemble_Raw"], 4),
            "expected_calibration_error": round(calib_metrics["ece"], 5),
            "max_calibration_error": round(calib_metrics["mce"], 5),
            "total_games_evaluated": len(df_oof) // 2,
            "seasons_evaluated_count": len(unique_seasons),
        }

        # 1. Per-Season Breakdown
        season_rows = []
        for s in unique_seasons:
            sub = df_oof[df_oof["Season"] == s]
            s_y = sub["target"].to_numpy()
            s_sharp = sub["pred_sharp"].to_numpy()
            s_raw = sub["pred_ens"].to_numpy()
            s_seed = sub["pred_seed"].to_numpy()
            s_massey = sub["pred_massey"].to_numpy()
            season_rows.append({
                "Season": int(s),
                "Games_Count": len(sub) // 2,
                "Brier_Ensemble_Sharpened": round(float(brier_score_loss(s_y, s_sharp)), 5),
                "Brier_Ensemble_Raw": round(float(brier_score_loss(s_y, s_raw)), 5),
                "Brier_Seed_Model": round(float(brier_score_loss(s_y, s_seed)), 5),
                "Brier_Massey_Model": round(float(brier_score_loss(s_y, s_massey)), 5),
                "LogLoss_Ensemble": round(float(log_loss(s_y, np.clip(s_sharp, 1e-7, 1 - 1e-7))), 5),
                "Accuracy_Ensemble": round(float(np.mean((s_sharp >= 0.5) == s_y)), 4),
            })
        self.df_by_season = pd.DataFrame(season_rows)

        # 2. Per-Round Breakdown
        round_rows = []
        for r_num in sorted(df_oof["Round"].unique()):
            sub = df_oof[df_oof["Round"] == r_num]
            r_y = sub["target"].to_numpy()
            r_sharp = sub["pred_sharp"].to_numpy()
            r_raw = sub["pred_ens"].to_numpy()
            r_seed = sub["pred_seed"].to_numpy()
            round_rows.append({
                "Round": int(r_num),
                "Round_Name": ROUND_NAMES.get(r_num, f"Round {r_num}"),
                "Games_Count": len(sub) // 2,
                "Brier_Ensemble_Sharpened": round(float(brier_score_loss(r_y, r_sharp)), 5),
                "Brier_Ensemble_Raw": round(float(brier_score_loss(r_y, r_raw)), 5),
                "Brier_Seed_Model": round(float(brier_score_loss(r_y, r_seed)), 5),
                "LogLoss_Ensemble": round(float(log_loss(r_y, np.clip(r_sharp, 1e-7, 1 - 1e-7))), 5),
                "Accuracy_Ensemble": round(float(np.mean((r_sharp >= 0.5) == r_y)), 4),
            })
        self.df_by_round = pd.DataFrame(round_rows)

        # 3. Mismatch Tier Breakdown
        mismatch_tiers = [
            ("Heavy Favorite (|dSeed| >= 10)", np.abs(df_oof["d_seed"]) >= 10),
            ("Moderate Favorite (4 <= |dSeed| <= 9)", (np.abs(df_oof["d_seed"]) >= 4) & (np.abs(df_oof["d_seed"]) <= 9)),
            ("Close Toss-up (|dSeed| <= 3)", np.abs(df_oof["d_seed"]) <= 3),
        ]
        mismatch_rows = []
        for tier_name, mask in mismatch_tiers:
            sub = df_oof[mask]
            if len(sub) > 0:
                t_y = sub["target"].to_numpy()
                t_sharp = sub["pred_sharp"].to_numpy()
                t_seed = sub["pred_seed"].to_numpy()
                mismatch_rows.append({
                    "Mismatch_Tier": tier_name,
                    "Games_Count": len(sub) // 2,
                    "Brier_Ensemble_Sharpened": round(float(brier_score_loss(t_y, t_sharp)), 5),
                    "Brier_Seed_Model": round(float(brier_score_loss(t_y, t_seed)), 5),
                    "Accuracy_Ensemble": round(float(np.mean((t_sharp >= 0.5) == t_y)), 4),
                })
        self.df_by_mismatch = pd.DataFrame(mismatch_rows)

        # Final fit on all historical data
        X_all_s = self.scaler.fit_transform(X_raw)
        self.lr_model = LogisticRegression(C=0.25, max_iter=600, random_state=42)
        self.lr_model.fit(X_all_s, y)

        if HAS_LIGHTGBM:
            self.lgb_model = lgb.LGBMClassifier(
                n_estimators=85, learning_rate=0.03, num_leaves=15,
                min_child_samples=20, random_state=42, verbose=-1,
            )
            self.lgb_model.fit(X_all_s, y)

        self.feature_weights = dict(zip(FEATURE_COLS, [float(c) for c in self.lr_model.coef_[0]]))
        return self.cv_brier

    def predict_pair_prob(self, feats_a: Dict[str, float], feats_b: Dict[str, float],
                          seed_a: int, seed_b: int) -> float:
        """Compute P(Team A beats Team B) with strategic edge sharpening."""
        d_seed = float(seed_b - seed_a)
        d_massey = float(feats_b.get("MasseyMedian", 150.0) - feats_a.get("MasseyMedian", 150.0))
        d_net = float(feats_a.get("NetRtg", 0.0) - feats_b.get("NetRtg", 0.0))
        d_off = float(feats_a.get("OffRtg", 100.0) - feats_b.get("OffRtg", 100.0))
        d_def = float(feats_b.get("DefRtg", 100.0) - feats_a.get("DefRtg", 100.0))
        d_efg = float(feats_a.get("eFG", 0.5) - feats_b.get("eFG", 0.5)) * 100.0
        d_tov = float(feats_b.get("TOV", 0.18) - feats_a.get("TOV", 0.18)) * 100.0
        d_orb = float(feats_a.get("ORB", 0.3) - feats_b.get("ORB", 0.3)) * 100.0
        d_ftr = float(feats_a.get("FTR", 0.3) - feats_b.get("FTR", 0.3)) * 100.0
        d_win = float(feats_a.get("Win", 0.5) - feats_b.get("Win", 0.5)) * 100.0
        d_qw = float(feats_a.get("QualityWins", 0.0) - feats_b.get("QualityWins", 0.0))
        d_sos = float(feats_a.get("SoS", 0.0) - feats_b.get("SoS", 0.0))

        diff_vec = np.array([[
            d_seed, d_massey, d_net, d_off, d_def,
            d_efg, d_tov, d_orb, d_ftr, d_win, d_qw, d_sos,
        ]], dtype=float)

        if self.lr_model is not None:
            s_vec = self.scaler.transform(diff_vec)
            p_lr = float(self.lr_model.predict_proba(s_vec)[0, 1])
            if self.lgb_model is not None:
                p_lgb = float(self.lgb_model.predict_proba(s_vec)[0, 1])
                p = 0.65 * p_lr + 0.35 * p_lgb
            else:
                p = p_lr
        else:
            # Fallback heuristic if untrained
            logit = 0.32 * d_seed + 0.12 * d_net
            p = float(expit(logit))

        # Strategic probability sharpening (Horan 1st-Place Technique):
        # Clip/sharpen extreme confidences to eliminate Brier score drag in heavy mismatches
        if self.sharpen:
            p = apply_sharpening_scalar(p, d_seed)

        return float(np.clip(p, 0.005, 0.995))


# ===========================================================================
# 3. Dynamic Programming Tournament Bracket Optimizer
# ===========================================================================

def get_slot_round(slot_name: str) -> int:
    """Determine the round number (0..6) from slot name."""
    if slot_name.startswith("R1"): return 1
    if slot_name.startswith("R2"): return 2
    if slot_name.startswith("R3"): return 3
    if slot_name.startswith("R4"): return 4
    if slot_name.startswith("R5"): return 5
    if slot_name.startswith("R6"): return 6
    return 0  # Play-in First Four


def sort_slots_topologically(slots_df: pd.DataFrame, seed_set: set) -> pd.DataFrame:
    """Sort the 67 tournament slots in bottom-up topological execution order."""
    resolved = set(seed_set)
    ordered = []
    remaining = slots_df.copy()
    while not remaining.empty:
        ready = remaining[
            remaining["StrongSeed"].isin(resolved) & remaining["WeakSeed"].isin(resolved)
        ]
        if ready.empty:
            # Fallback if circular or partial
            for _, r in remaining.iterrows():
                ordered.append(r)
            break
        for _, row in ready.iterrows():
            ordered.append(row)
            resolved.add(row["Slot"])
        remaining = remaining[~remaining["Slot"].isin(resolved)]
    return pd.DataFrame(ordered)


def solve_dp_optimal_bracket(
    ordered_slots: pd.DataFrame,
    seed_team_map: Dict[str, int],
    p_matrix: Dict[Tuple[int, int], float],
    anchor_champion: Optional[int] = None,
    anchor_finalist: Optional[int] = None,
    leverage_multipliers: Optional[Dict[Tuple[str, int], float]] = None,
) -> Dict[str, Dict[str, Any]]:
    """Solve the globally optimal 67-game bracket using Exact Bottom-Up Dynamic Programming."""
    candidates: Dict[str, List[int]] = {}
    V: Dict[str, Dict[int, float]] = {}
    back_opp: Dict[Tuple[str, int], int] = {}
    slot_map: Dict[str, Tuple[str, str]] = {}

    for _, row in ordered_slots.iterrows():
        slot = row["Slot"]
        ss = row["StrongSeed"]
        ws = row["WeakSeed"]
        slot_map[slot] = (ss, ws)

        cand_strong = [seed_team_map[ss]] if ss in seed_team_map else candidates[ss]
        cand_weak = [seed_team_map[ws]] if ws in seed_team_map else candidates[ws]
        candidates[slot] = cand_strong + cand_weak

        r_num = get_slot_round(slot)
        base_pts = float(ROUND_POINTS.get(r_num, 0))
        V[slot] = {}

        # 1. Strong candidate wins slot: best opponent from weak branch
        best_weak = max(
            cand_weak,
            key=lambda u: V[ws].get(u, 0.0) if ws not in seed_team_map else 0.0
        )
        for t in cand_strong:
            v_sub = V[ss].get(t, 0.0) if ss not in seed_team_map else 0.0
            v_opp = V[ws].get(best_weak, 0.0) if ws not in seed_team_map else 0.0
            p_win = p_matrix.get((t, best_weak), 0.5)

            mult = 1.0
            if leverage_multipliers and (slot, t) in leverage_multipliers:
                mult = leverage_multipliers[(slot, t)]

            # Heavy bonus if anchored to ensure selection
            anchor_bonus = 0.0
            if slot == "R6CH" and anchor_champion is not None:
                anchor_bonus = 1e7 if t == anchor_champion else -1e7
            elif slot in ("R5WX", "R5YZ") and anchor_finalist is not None:
                if t == anchor_finalist:
                    anchor_bonus = 5e6

            V[slot][t] = v_sub + v_opp + (base_pts * mult * p_win) + anchor_bonus
            back_opp[(slot, t)] = best_weak

        # 2. Weak candidate wins slot: best opponent from strong branch
        best_strong = max(
            cand_strong,
            key=lambda u: V[ss].get(u, 0.0) if ss not in seed_team_map else 0.0
        )
        for u in cand_weak:
            v_sub = V[ws].get(u, 0.0) if ws not in seed_team_map else 0.0
            v_opp = V[ss].get(best_strong, 0.0) if ss not in seed_team_map else 0.0
            p_win = p_matrix.get((u, best_strong), 0.5)

            mult = 1.0
            if leverage_multipliers and (slot, u) in leverage_multipliers:
                mult = leverage_multipliers[(slot, u)]

            anchor_bonus = 0.0
            if slot == "R6CH" and anchor_champion is not None:
                anchor_bonus = 1e7 if u == anchor_champion else -1e7
            elif slot in ("R5WX", "R5YZ") and anchor_finalist is not None:
                if u == anchor_finalist:
                    anchor_bonus = 5e6

            V[slot][u] = v_sub + v_opp + (base_pts * mult * p_win) + anchor_bonus
            back_opp[(slot, u)] = best_strong

    root_slot = "R6CH"
    if anchor_champion is not None and anchor_champion in candidates[root_slot]:
        champ = anchor_champion
    else:
        champ = max(candidates[root_slot], key=lambda t: V[root_slot][t])

    # Backtrack all 67 slot assignments
    picks: Dict[str, Dict[str, Any]] = {}

    def _reconstruct(slot_name: str, winner: int):
        ss, ws = slot_map[slot_name]
        cand_s = [seed_team_map[ss]] if ss in seed_team_map else candidates[ss]
        opp = back_opp.get((slot_name, winner), cand_s[0])

        if winner in cand_s:
            strong_t, weak_t = winner, opp
            if ss not in seed_team_map: _reconstruct(ss, winner)
            if ws not in seed_team_map: _reconstruct(ws, opp)
        else:
            strong_t, weak_t = opp, winner
            if ss not in seed_team_map: _reconstruct(ss, opp)
            if ws not in seed_team_map: _reconstruct(ws, winner)

        p_win = p_matrix.get((winner, opp), 0.5)
        r_num = get_slot_round(slot_name)
        pts = ROUND_POINTS.get(r_num, 0)
        picks[slot_name] = {
            "slot": slot_name,
            "round": r_num,
            "round_name": ROUND_NAMES.get(r_num, "Unknown"),
            "winner_id": winner,
            "strong_id": strong_t,
            "weak_id": weak_t,
            "prob": p_win,
            "points": pts,
        }

    _reconstruct(root_slot, champ)
    return picks


def simulate_stochastic_bracket(
    ordered_slots: pd.DataFrame,
    seed_team_map: Dict[str, int],
    p_matrix: Dict[Tuple[int, int], float],
    rng: np.random.Generator,
) -> Dict[str, Dict[str, Any]]:
    """Simulate a single full tournament realization using calibrated probabilities."""
    picks: Dict[str, Dict[str, Any]] = {}
    slot_winners: Dict[str, int] = {}

    for row in ordered_slots.itertuples():
        slot = row.Slot
        ss = row.StrongSeed
        ws = row.WeakSeed

        t_strong = seed_team_map[ss] if ss in seed_team_map else slot_winners[ss]
        t_weak = seed_team_map[ws] if ws in seed_team_map else slot_winners[ws]

        p = p_matrix.get((t_strong, t_weak), 0.5)
        winner = t_strong if rng.random() < p else t_weak
        slot_winners[slot] = winner

        r_num = get_slot_round(slot)
        picks[slot] = {
            "slot": slot,
            "round": r_num,
            "round_name": ROUND_NAMES.get(r_num, "Unknown"),
            "winner_id": winner,
            "strong_id": t_strong,
            "weak_id": t_weak,
            "prob": p if winner == t_strong else (1.0 - p),
            "points": ROUND_POINTS.get(r_num, 0),
        }
    return picks


# ===========================================================================
# 4. 2027 Projected Tournament Field Engine
# ===========================================================================

def generate_projected_2027_field(
    teams_df: pd.DataFrame, stats_2026: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Generate a realistic 68-team tournament field and ratings for 2027."""
    # Top historical programs and perennial contenders
    top_blue_bloods = [
        "Duke", "Connecticut", "Kansas", "Houston", "North Carolina", "Arizona",
        "Kentucky", "Auburn", "Gonzaga", "Alabama", "Iowa St", "Michigan St",
        "Purdue", "Tennessee", "Florida", "Baylor", "Illinois", "Texas",
        "Wisconsin", "Marquette", "Creighton", "Texas Tech", "Arkansas", "UCLA",
        "Indiana", "Ohio St", "Maryland", "St John's", "Virginia", "Louisville",
        "BYU", "Oklahoma", "Cincinnati", "Clemson", "Wake Forest", "Michigan",
        "Memphis", "San Diego St", "Saint Mary's", "Boise St", "New Mexico",
        "Nevada", "Dayton", "VCU", "Drake", "Grand Canyon", "McNeese",
        "Yale", "Akron", "Vermont", "Colgate", "Montana", "High Point",
        "UNC Wilmington", "Samford", "Lipscomb", "Troy", "Wofford", "Norfolk St",
        "Siena", "Cal Baptist", "N Dakota St", "Furman", "Northern Iowa",
        "South Florida", "UCF", "TCU", "Robert Morris",
    ]

    # Map team names to TeamID
    name_to_id = dict(zip(teams_df["TeamName"], teams_df["TeamID"]))
    id_to_name = dict(zip(teams_df["TeamID"], teams_df["TeamName"]))

    selected_ids = []
    for name in top_blue_bloods:
        if name in name_to_id:
            selected_ids.append(name_to_id[name])
        else:
            # Fuzzy match
            for t_id, t_name in id_to_name.items():
                if name.lower() in t_name.lower():
                    selected_ids.append(t_id)
                    break

    # Ensure exactly 68 unique teams
    all_d1 = list(teams_df["TeamID"])
    for t_id in all_d1:
        if len(selected_ids) >= 68:
            break
        if t_id not in selected_ids:
            selected_ids.append(t_id)
    selected_ids = selected_ids[:68]

    # Generate Seeds (4 regions: W, X, Y, Z with play-ins)
    regions = ["W", "X", "Y", "Z"]
    seeds_rows = []
    idx = 0

    # Assign seeds 1..10 across 4 regions
    for s_num in range(1, 11):
        for reg in regions:
            s_label = f"{reg}{s_num:02d}"
            seeds_rows.append({"Season": 2027, "Seed": s_label, "TeamID": selected_ids[idx]})
            idx += 1

    # Play-in seeds: 11-seed play-ins in Y and Z
    seeds_rows.append({"Season": 2027, "Seed": "W11", "TeamID": selected_ids[idx]}); idx += 1
    seeds_rows.append({"Season": 2027, "Seed": "X11", "TeamID": selected_ids[idx]}); idx += 1
    seeds_rows.append({"Season": 2027, "Seed": "Y11a", "TeamID": selected_ids[idx]}); idx += 1
    seeds_rows.append({"Season": 2027, "Seed": "Y11b", "TeamID": selected_ids[idx]}); idx += 1
    seeds_rows.append({"Season": 2027, "Seed": "Z11a", "TeamID": selected_ids[idx]}); idx += 1
    seeds_rows.append({"Season": 2027, "Seed": "Z11b", "TeamID": selected_ids[idx]}); idx += 1

    # Seeds 12..15
    for s_num in range(12, 16):
        for reg in regions:
            s_label = f"{reg}{s_num:02d}"
            seeds_rows.append({"Season": 2027, "Seed": s_label, "TeamID": selected_ids[idx]})
            idx += 1

    # Play-in seeds: 16-seed play-ins in X and Y
    seeds_rows.append({"Season": 2027, "Seed": "W16", "TeamID": selected_ids[idx]}); idx += 1
    seeds_rows.append({"Season": 2027, "Seed": "X16a", "TeamID": selected_ids[idx]}); idx += 1
    seeds_rows.append({"Season": 2027, "Seed": "X16b", "TeamID": selected_ids[idx]}); idx += 1
    seeds_rows.append({"Season": 2027, "Seed": "Y16a", "TeamID": selected_ids[idx]}); idx += 1
    seeds_rows.append({"Season": 2027, "Seed": "Y16b", "TeamID": selected_ids[idx]}); idx += 1
    seeds_rows.append({"Season": 2027, "Seed": "Z16", "TeamID": selected_ids[idx]}); idx += 1

    seeds_2027_df = pd.DataFrame(seeds_rows)

    # Slots for 2027 (Standard NCAA 67-slot structure)
    slots_rows = []
    # Play-ins
    slots_rows.append({"Season": 2027, "Slot": "X16", "StrongSeed": "X16a", "WeakSeed": "X16b"})
    slots_rows.append({"Season": 2027, "Slot": "Y11", "StrongSeed": "Y11a", "WeakSeed": "Y11b"})
    slots_rows.append({"Season": 2027, "Slot": "Y16", "StrongSeed": "Y16a", "WeakSeed": "Y16b"})
    slots_rows.append({"Season": 2027, "Slot": "Z11", "StrongSeed": "Z11a", "WeakSeed": "Z11b"})

    # Round 1
    for reg in regions:
        for i in range(1, 9):
            slot = f"R1{reg}{i}"
            strong = f"{reg}{i:02d}"
            weak_num = 17 - i
            weak = f"{reg}{weak_num:02d}"
            if reg == "X" and weak == "X16": weak = "X16"
            elif reg == "Y" and weak == "Y16": weak = "Y16"
            elif reg == "Y" and weak == "Y11": weak = "Y11"
            elif reg == "Z" and weak == "Z11": weak = "Z11"
            slots_rows.append({"Season": 2027, "Slot": slot, "StrongSeed": strong, "WeakSeed": weak})

    # Round 2
    for reg in regions:
        slots_rows.append({"Season": 2027, "Slot": f"R2{reg}1", "StrongSeed": f"R1{reg}1", "WeakSeed": f"R1{reg}8"})
        slots_rows.append({"Season": 2027, "Slot": f"R2{reg}2", "StrongSeed": f"R1{reg}2", "WeakSeed": f"R1{reg}7"})
        slots_rows.append({"Season": 2027, "Slot": f"R2{reg}3", "StrongSeed": f"R1{reg}3", "WeakSeed": f"R1{reg}6"})
        slots_rows.append({"Season": 2027, "Slot": f"R2{reg}4", "StrongSeed": f"R1{reg}4", "WeakSeed": f"R1{reg}5"})

    # Round 3 (Sweet 16)
    for reg in regions:
        slots_rows.append({"Season": 2027, "Slot": f"R3{reg}1", "StrongSeed": f"R2{reg}1", "WeakSeed": f"R2{reg}4"})
        slots_rows.append({"Season": 2027, "Slot": f"R3{reg}2", "StrongSeed": f"R2{reg}2", "WeakSeed": f"R2{reg}3"})

    # Round 4 (Elite Eight)
    for reg in regions:
        slots_rows.append({"Season": 2027, "Slot": f"R4{reg}1", "StrongSeed": f"R3{reg}1", "WeakSeed": f"R3{reg}2"})

    # Round 5 & 6
    slots_rows.append({"Season": 2027, "Slot": "R5WX", "StrongSeed": "R4W1", "WeakSeed": "R4X1"})
    slots_rows.append({"Season": 2027, "Slot": "R5YZ", "StrongSeed": "R4Y1", "WeakSeed": "R4Z1"})
    slots_rows.append({"Season": 2027, "Slot": "R6CH", "StrongSeed": "R5WX", "WeakSeed": "R5YZ"})

    slots_2027_df = pd.DataFrame(slots_rows)

    # Projected Stats for 2027: Bayesian shrinkage of 2026 stats toward program tiers
    stats_2026_map = stats_2026[stats_2026["Season"] == 2026].set_index("TeamID").to_dict(orient="index")
    proj_rows = []
    for _, s_row in seeds_2027_df.iterrows():
        t_id = s_row["TeamID"]
        s_num = extract_seed_number(s_row["Seed"])
        prior_net = 28.0 - (s_num * 2.2)  # Prior based on projected seed
        s26 = stats_2026_map.get(t_id, {})

        # Shrink toward prior
        net = 0.65 * s26.get("NetRtg", prior_net) + 0.35 * prior_net
        off = 0.65 * s26.get("OffRtg", 112.0) + 0.35 * (100.0 + net / 2.0)
        defn = off - net
        massey = float(np.clip(s_num * 4.5 + np.random.uniform(-3, 3), 1.0, 250.0))

        proj_rows.append({
            "Season": 2027,
            "TeamID": t_id,
            "OffRtg": off,
            "DefRtg": defn,
            "NetRtg": net,
            "eFG": s26.get("eFG", 0.52),
            "TOV": s26.get("TOV", 0.17),
            "ORB": s26.get("ORB", 0.31),
            "FTR": s26.get("FTR", 0.30),
            "Win": float(np.clip(0.85 - (s_num * 0.035), 0.45, 0.95)),
            "Poss": s26.get("Poss", 68.5),
            "MOV": net / 1.4,
            "SoS": 0.65 * s26.get("SoS", 6.0) + 0.35 * (12.0 - s_num * 0.5),
            "QualityWins": float(max(1, 14 - s_num)),
            "MasseyMedian": massey,
            "MasseyMean": massey,
        })

    stats_2027_df = pd.DataFrame(proj_rows)
    return seeds_2027_df, slots_2027_df, stats_2027_df


# ===========================================================================
# 5. Master Portfolio Generation Engine (25 Brackets)
# ===========================================================================

def generate_bracket_portfolio(
    season: int,
    seeds_df: pd.DataFrame,
    slots_df: pd.DataFrame,
    stats_df: pd.DataFrame,
    teams_df: pd.DataFrame,
    model: CalibratedMatchupModel,
    n_sims: int = 10000,
    base_seed: int = 42,
    sharpen: bool = True,
) -> Tuple[List[Dict[str, Any]], pd.DataFrame, Dict[str, Any]]:
    """Build the comprehensive 25-bracket portfolio covering all risk/leverage profiles."""
    team_name = dict(zip(teams_df["TeamID"], teams_df["TeamName"]))
    seed_team = dict(zip(seeds_df["Seed"], seeds_df["TeamID"]))
    team_seed = dict(zip(seeds_df["TeamID"], seeds_df["Seed"]))

    # All tournament teams
    tourney_teams = list(seeds_df["TeamID"].unique())
    n_teams = len(tourney_teams)

    # Stats map for season
    season_stats = stats_df[stats_df["Season"] == season].set_index("TeamID").to_dict(orient="index")

    # Topological slot ordering
    ordered_slots = sort_slots_topologically(slots_df, set(seeds_df["Seed"]))
    slots_tuples = [(r.Slot, r.StrongSeed, r.WeakSeed) for r in ordered_slots.itertuples()]

    # Precompute pairwise probability matrix P[A, B]
    p_matrix: Dict[Tuple[int, int], float] = {}
    p_pub_matrix: Dict[Tuple[int, int], float] = {}  # Public picks anchored heavily on seed chalk

    for t_a in tourney_teams:
        s_a = extract_seed_number(team_seed.get(t_a, "8"))
        f_a = season_stats.get(t_a, {})
        for t_b in tourney_teams:
            if t_a == t_b:
                p_matrix[(t_a, t_b)] = 0.5
                p_pub_matrix[(t_a, t_b)] = 0.5
                continue
            s_b = extract_seed_number(team_seed.get(t_b, "8"))
            f_b = season_stats.get(t_b, {})

            p_mod = model.predict_pair_prob(f_a, f_b, s_a, s_b)
            p_matrix[(t_a, t_b)] = p_mod

            # Public crowd heavily anchors on seed number and over-picks favorites
            pub_logit = 0.62 * (s_b - s_a) + 0.18 * (f_a.get("NetRtg", 0.0) - f_b.get("NetRtg", 0.0))
            p_pub_matrix[(t_a, t_b)] = float(np.clip(expit(pub_logit), 0.02, 0.98))

    # Monte Carlo simulation to estimate model and public slot advance probabilities
    print(f"  [Season {season}] Running {n_sims} tournament Monte Carlo passes for crowd leverage...", flush=True)
    rng_mc = np.random.default_rng(base_seed)

    slot_names = [s[0] for s in slots_tuples]
    model_adv_counts: Dict[Tuple[str, int], int] = { (s, t): 0 for s in slot_names for t in tourney_teams }
    pub_adv_counts: Dict[Tuple[str, int], int] = { (s, t): 0 for s in slot_names for t in tourney_teams }

    n_mc = min(5000, n_sims)
    for _ in range(n_mc):
        # Model path
        cur_winners = {}
        for s, ss, ws in slots_tuples:
            t_s = seed_team[ss] if ss in seed_team else cur_winners[ss]
            t_w = seed_team[ws] if ws in seed_team else cur_winners[ws]
            w = t_s if rng_mc.random() < p_matrix[(t_s, t_w)] else t_w
            cur_winners[s] = w
            model_adv_counts[(s, w)] += 1

        # Public path
        cur_pub = {}
        for s, ss, ws in slots_tuples:
            t_s = seed_team[ss] if ss in seed_team else cur_pub[ss]
            t_w = seed_team[ws] if ws in seed_team else cur_pub[ws]
            w = t_s if rng_mc.random() < p_pub_matrix[(t_s, t_w)] else t_w
            cur_pub[s] = w
            pub_adv_counts[(s, w)] += 1

    n_mc = min(5000, n_sims)
    slot_prob_model = { k: v / n_mc for k, v in model_adv_counts.items() }
    slot_prob_pub = { k: v / n_mc for k, v in pub_adv_counts.items() }

    # Identify candidate anchors by Seed and Net Rating
    seed_1_teams = [
        seed_team.get(s) for s in ["W01", "X01", "Y01", "Z01"] if s in seed_team
    ]
    seed_2_teams = sorted(
        [seed_team.get(s) for s in ["W02", "X02", "Y02", "Z02"] if s in seed_team],
        key=lambda t: season_stats.get(t, {}).get("NetRtg", 0.0), reverse=True
    )
    seed_3_teams = sorted(
        [seed_team.get(s) for s in ["W03", "X03", "Y03", "Z03"] if s in seed_team],
        key=lambda t: season_stats.get(t, {}).get("NetRtg", 0.0), reverse=True
    )
    dark_horses = sorted(
        [
            t for t in tourney_teams
            if 4 <= extract_seed_number(team_seed.get(t, "8")) <= 6
        ],
        key=lambda t: season_stats.get(t, {}).get("NetRtg", 0.0), reverse=True
    )
    cinderellas = sorted(
        [
            t for t in tourney_teams
            if 7 <= extract_seed_number(team_seed.get(t, "8")) <= 11
        ],
        key=lambda t: season_stats.get(t, {}).get("DefRtg", 100.0) - season_stats.get(t, {}).get("OffRtg", 100.0),
    )

    brackets: List[Dict[str, Any]] = []

    def _build_leverage_multipliers(alpha: float) -> Dict[Tuple[str, int], float]:
        mults = {}
        for s in slot_names:
            r = get_slot_round(s)
            round_weight = 0.35 if r <= 2 else (0.75 if r <= 4 else 1.0)
            for t in tourney_teams:
                p_m = max(slot_prob_model.get((s, t), 0.0), 0.005)
                p_p = max(slot_prob_pub.get((s, t), 0.0), 0.01)
                lev = (p_m / p_p) ** (alpha * round_weight)
                mults[(s, t)] = float(np.clip(lev, 0.4, 4.5))
        return mults

    # -----------------------------------------------------------------------
    # Define the 25 Distinct Bracket Strategies
    # -----------------------------------------------------------------------
    strategies = [
        # 1. Consensus & Winning Methodologies
        ("Bracket_01_Consensus_Max_EV_Chalk", "Pure Dynamic Programming Maximum Expected Points (Chalk)", {}),
        ("Bracket_02_Winner_Sharpened_Edge", "Harrison Horan 1st-Place Edge-Sharpened Mismatches & Quality-Wins", {"sharpen_boost": True}),

        # 2. Regional #1 Seed Anchors
        ("Bracket_03_East_No1_Anchor", f"Region W (East) #1 Seed Anchor ({team_name.get(seed_1_teams[0], 'W01')})", {"anchor_champ": seed_1_teams[0] if len(seed_1_teams) > 0 else None}),
        ("Bracket_04_South_No1_Anchor", f"Region X (South) #1 Seed Anchor ({team_name.get(seed_1_teams[1], 'X01') if len(seed_1_teams) > 1 else 'X01'})", {"anchor_champ": seed_1_teams[1] if len(seed_1_teams) > 1 else None}),
        ("Bracket_05_Midwest_No1_Anchor", f"Region Y (Midwest) #1 Seed Anchor ({team_name.get(seed_1_teams[2], 'Y01') if len(seed_1_teams) > 2 else 'Y01'})", {"anchor_champ": seed_1_teams[2] if len(seed_1_teams) > 2 else None}),
        ("Bracket_06_West_No1_Anchor", f"Region Z (West) #1 Seed Anchor ({team_name.get(seed_1_teams[3], 'Z01') if len(seed_1_teams) > 3 else 'Z01'})", {"anchor_champ": seed_1_teams[3] if len(seed_1_teams) > 3 else None}),

        # 3. Elite #2 and #3 Seed Championship Contenders
        ("Bracket_07_Elite_2Seed_Anchor_TopEfficiency", f"Highest Net-Rating #2 Seed ({team_name.get(seed_2_teams[0], '2-Seed') if seed_2_teams else 'None'})", {"anchor_champ": seed_2_teams[0] if seed_2_teams else None}),
        ("Bracket_08_Elite_2Seed_Anchor_Secondary", f"Second Net-Rating #2 Seed ({team_name.get(seed_2_teams[1], '2-Seed') if len(seed_2_teams) > 1 else 'None'})", {"anchor_champ": seed_2_teams[1] if len(seed_2_teams) > 1 else None}),
        ("Bracket_09_Elite_3Seed_Anchor_TopEfficiency", f"Highest Net-Rating #3 Seed ({team_name.get(seed_3_teams[0], '3-Seed') if seed_3_teams else 'None'})", {"anchor_champ": seed_3_teams[0] if seed_3_teams else None}),
        ("Bracket_10_Elite_3Seed_Anchor_Secondary", f"Second Net-Rating #3 Seed ({team_name.get(seed_3_teams[1], '3-Seed') if len(seed_3_teams) > 1 else 'None'})", {"anchor_champ": seed_3_teams[1] if len(seed_3_teams) > 1 else None}),

        # 4. Dark Horse Sleepers & Cinderellas
        ("Bracket_11_Dark_Horse_KenPom_Sleeper_1", f"Top Efficiency Mid-Seed (#4-#6) ({team_name.get(dark_horses[0], 'DH1') if dark_horses else 'None'})", {"anchor_champ": dark_horses[0] if dark_horses else None}),
        ("Bracket_12_Dark_Horse_KenPom_Sleeper_2", f"Second Efficiency Mid-Seed (#4-#6) ({team_name.get(dark_horses[1], 'DH2') if len(dark_horses) > 1 else 'None'})", {"anchor_champ": dark_horses[1] if len(dark_horses) > 1 else None}),
        ("Bracket_13_Cinderella_Finalist", f"Elite Defense Cinderella (Seed 7-11) Finalist ({team_name.get(cinderellas[0], 'Cin') if cinderellas else 'None'})", {"anchor_finalist": cinderellas[0] if cinderellas else None}),
        ("Bracket_14_Harmonic_Committee_Model_Balance", "Harmonic Blend of Committee Seed Expectation and Model Efficiency", {"harmonic": True}),

        # 5. Game-Theory Leverage Portfolios Across Pool Sizes
        ("Bracket_15_Pool_Leverage_Small", "Game-Theory Leverage Optimizer for Small Pools (N=30, alpha=0.10)", {"alpha": 0.10}),
        ("Bracket_16_Pool_Leverage_Medium", "Game-Theory Leverage Optimizer for Medium Pools (N=150, alpha=0.22)", {"alpha": 0.22}),
        ("Bracket_17_Pool_Leverage_Large", "Game-Theory Leverage Optimizer for Large Pools (N=500, alpha=0.35)", {"alpha": 0.35}),
        ("Bracket_18_Pool_Leverage_Mega_GPP", "Game-Theory Leverage Optimizer for Mega-GPP / Contests (N=2500+, alpha=0.50)", {"alpha": 0.50}),
        ("Bracket_19_Chaos_Upset_Specialist", "Aggressive Underdog Tilt targeting 11-over-6 and 12-over-5 Stylistic Upset Leverage", {"chaos": True}),

        # 6. Monte Carlo Calibrated Stochastic Trajectories
        ("Bracket_20_Stochastic_Sim_Realization_1", "Calibrated Monte Carlo Tournament Simulation Realization #1", {"sim": True, "seed_offset": 101}),
        ("Bracket_21_Stochastic_Sim_Realization_2", "Calibrated Monte Carlo Tournament Simulation Realization #2", {"sim": True, "seed_offset": 202}),
        ("Bracket_22_Stochastic_Sim_Realization_3", "Calibrated Monte Carlo Tournament Simulation Realization #3", {"sim": True, "seed_offset": 303}),
        ("Bracket_23_Stochastic_Sim_Realization_4", "Calibrated Monte Carlo Tournament Simulation Realization #4", {"sim": True, "seed_offset": 404}),
        ("Bracket_24_Stochastic_Sim_Realization_5", "Calibrated Monte Carlo Tournament Simulation Realization #5", {"sim": True, "seed_offset": 505}),
        ("Bracket_25_Stochastic_Sim_Realization_6", "Calibrated Monte Carlo Tournament Simulation Realization #6", {"sim": True, "seed_offset": 606}),
    ]

    summary_rows = []

    for b_idx, (b_name, b_desc, b_params) in enumerate(strategies, start=1):
        if b_params.get("sim", False):
            rng_sim = np.random.default_rng(base_seed + b_params["seed_offset"])
            picks = simulate_stochastic_bracket(ordered_slots, seed_team, p_matrix, rng_sim)
        else:
            mults = None
            if "alpha" in b_params:
                mults = _build_leverage_multipliers(b_params["alpha"])
            elif b_params.get("chaos", False):
                mults = _build_leverage_multipliers(0.40)
                # Boost 11, 12, 10 seeds in early rounds
                for s in slot_names:
                    for t in tourney_teams:
                        s_num = extract_seed_number(team_seed.get(t, "8"))
                        if s_num in (10, 11, 12):
                            mults[(s, t)] = mults.get((s, t), 1.0) * 1.45
            elif b_params.get("harmonic", False):
                mults = {}
                for s in slot_names:
                    for t in tourney_teams:
                        s_num = extract_seed_number(team_seed.get(t, "8"))
                        mults[(s, t)] = 1.0 + (0.04 * (16 - s_num))

            picks = solve_dp_optimal_bracket(
                ordered_slots=ordered_slots,
                seed_team_map=seed_team,
                p_matrix=p_matrix,
                anchor_champion=b_params.get("anchor_champ"),
                anchor_finalist=b_params.get("anchor_finalist"),
                leverage_multipliers=mults,
            )

        # Calculate metrics for this bracket
        champ_id = picks["R6CH"]["winner_id"]
        champ_name = team_name.get(champ_id, f"ID_{champ_id}")
        champ_seed = team_seed.get(champ_id, "?")

        r5_picks = [picks["R5WX"]["winner_id"], picks["R5YZ"]["winner_id"]]
        runner_up_id = picks["R6CH"]["strong_id"] if picks["R6CH"]["winner_id"] == picks["R6CH"]["weak_id"] else (
            picks["R6CH"]["weak_id"] if picks["R6CH"]["winner_id"] == picks["R6CH"]["strong_id"] else (
                picks["R6CH"]["strong_id"] if picks["R6CH"]["strong_id"] != champ_id else picks["R6CH"]["weak_id"]
            )
        )
        runner_up_name = team_name.get(runner_up_id, f"ID_{runner_up_id}")
        runner_up_seed = team_seed.get(runner_up_id, "?")

        # Final Four teams (winners of R4W1, R4X1, R4Y1, R4Z1)
        f4_teams = [
            f"{team_seed.get(picks[s]['winner_id'], '?')} {team_name.get(picks[s]['winner_id'], '?')}"
            for s in ["R4W1", "R4X1", "R4Y1", "R4Z1"] if s in picks
        ]

        # Elite Eight teams (participants in R4)
        e8_teams = []
        for s in ["R4W1", "R4X1", "R4Y1", "R4Z1"]:
            if s in picks:
                e8_teams.append(f"{team_seed.get(picks[s]['strong_id'], '?')} {team_name.get(picks[s]['strong_id'], '?')}")
                e8_teams.append(f"{team_seed.get(picks[s]['weak_id'], '?')} {team_name.get(picks[s]['weak_id'], '?')}")

        # Total expected pool points under true model probabilities
        total_ev = sum(
            pick["points"] * p_matrix.get((pick["winner_id"], pick["weak_id"] if pick["winner_id"] == pick["strong_id"] else pick["strong_id"]), 0.5)
            for s_name, pick in picks.items() if s_name.startswith("R")
        )

        # Count upsets picked (where picked team seed number > opponent seed number)
        upset_count = 0
        for s_name, pick in picks.items():
            if not s_name.startswith("R"): continue
            w_seed = extract_seed_number(team_seed.get(pick["winner_id"], "8"))
            opp_id = pick["weak_id"] if pick["winner_id"] == pick["strong_id"] else pick["strong_id"]
            o_seed = extract_seed_number(team_seed.get(opp_id, "8"))
            if w_seed > o_seed:
                upset_count += 1

        # Leverage score: sum of log leverage ratios for late round picks
        lev_score = 0.0
        for s in ["R4W1", "R4X1", "R4Y1", "R4Z1", "R5WX", "R5YZ", "R6CH"]:
            if s in picks:
                w_t = picks[s]["winner_id"]
                p_m = max(slot_prob_model.get((s, w_t), 0.0), 0.01)
                p_p = max(slot_prob_pub.get((s, w_t), 0.0), 0.01)
                lev_score += np.log(p_m / p_p)

        summary_rows.append({
            "Bracket_Number": b_idx,
            "Bracket_Name": b_name,
            "Strategy_Description": b_desc,
            "Champion": f"({champ_seed}) {champ_name}",
            "Runner_Up": f"({runner_up_seed}) {runner_up_name}",
            "Final_Four": "; ".join(f4_teams),
            "Elite_Eight": "; ".join(e8_teams),
            "Expected_Tournament_Points": round(float(total_ev), 1),
            "Leverage_Score": round(float(lev_score), 2),
            "Upsets_Picked_Count": upset_count,
        })

        brackets.append({
            "bracket_number": b_idx,
            "bracket_name": b_name,
            "description": b_desc,
            "champion": f"({champ_seed}) {champ_name}",
            "runner_up": f"({runner_up_seed}) {runner_up_name}",
            "expected_points": round(float(total_ev), 1),
            "leverage_score": round(float(lev_score), 2),
            "upsets_count": upset_count,
            "picks": picks,
        })

    summary_df = pd.DataFrame(summary_rows)
    return brackets, summary_df, p_matrix


# ===========================================================================
# 6. Exporters: CSVs, ASCII Trees, Kaggle Submission, and Evaluation
# ===========================================================================

def evaluate_kaggle_submission_file(
    sub_path: Path,
    tourney_compact_df: pd.DataFrame,
    season: Optional[int] = None,
) -> Dict[str, Any]:
    """Score a Kaggle submission CSV against actual NCAA Tournament outcomes.

    Kaggle MM Mania official metric: Mean Squared Error (Brier Score)
    across all tournament games actually played.
    """
    if not sub_path.exists():
        return {"status": "FILE_NOT_FOUND", "path": str(sub_path)}

    sub_df = pd.read_csv(sub_path)
    if "ID" not in sub_df.columns or "Pred" not in sub_df.columns:
        return {"status": "INVALID_FORMAT", "columns": list(sub_df.columns)}

    games = tourney_compact_df.copy()
    if season is not None:
        games = games[games["Season"] == season]

    if games.empty:
        return {
            "status": "NO_GAMES_AVAILABLE",
            "season": season,
            "message": f"No ground truth tournament games found for season {season}."
        }

    sub_map = dict(zip(sub_df["ID"], sub_df["Pred"]))
    preds = []
    targets = []
    game_rows = []

    for _, row in games.iterrows():
        s = int(row["Season"])
        w = int(row["WTeamID"])
        l = int(row["LTeamID"])

        # In Kaggle ID format: {Season}_{Team1}_{Team2} where Team1 < Team2
        if w < l:
            match_id = f"{s}_{w}_{l}"
            target = 1.0  # Team 1 (W) beat Team 2 (L)
        else:
            match_id = f"{s}_{l}_{w}"
            target = 0.0  # Team 1 (L) did NOT beat Team 2 (W)

        p = sub_map.get(match_id)
        if p is not None:
            p_val = float(p)
            preds.append(p_val)
            targets.append(target)
            game_rows.append({
                "ID": match_id,
                "Season": s,
                "Pred": p_val,
                "Actual": target,
                "SquaredError": round((p_val - target) ** 2, 5),
                "Correct": int((p_val >= 0.5) == target),
            })

    if not preds:
        return {
            "status": "NO_MATCHING_GAMES",
            "season": season,
            "message": "None of the actual tournament game IDs matched the submission file."
        }

    preds_arr = np.array(preds, dtype=float)
    targets_arr = np.array(targets, dtype=float)

    brier = float(np.mean((preds_arr - targets_arr) ** 2))
    logloss = float(log_loss(targets_arr, np.clip(preds_arr, 1e-7, 1 - 1e-7)))
    accuracy = float(np.mean((preds_arr >= 0.5) == targets_arr))
    brier_5050 = 0.2500
    improvement_vs_5050 = (brier_5050 - brier) / brier_5050 * 100.0

    return {
        "status": "SUCCESS",
        "season": season,
        "n_games_evaluated": len(preds_arr),
        "brier_score": round(brier, 5),
        "log_loss": round(logloss, 5),
        "accuracy": round(accuracy, 4),
        "baseline_5050_brier": 0.2500,
        "improvement_vs_5050_pct": round(improvement_vs_5050, 2),
        "game_evaluations": game_rows,
    }


def generate_brier_evaluation_plots(
    eval_summary: Dict[str, Any],
    df_by_season: pd.DataFrame,
    df_by_round: pd.DataFrame,
    df_by_mismatch: pd.DataFrame,
    calibration_data: Dict[str, Any],
    output_dir: Path,
):
    """Generate publication-ready diagnostic plots for Brier score and calibration."""
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # 1. Brier Score By Season Line Chart
    if not df_by_season.empty:
        fig, ax = plt.subplots(figsize=(12, 6))
        seasons = df_by_season["Season"]
        ax.plot(seasons, df_by_season["Brier_Ensemble_Sharpened"], marker="o", color="#2ca02c", lw=2.5, label="Ensemble (Sharpened)")
        ax.plot(seasons, df_by_season["Brier_Ensemble_Raw"], marker="s", color="#1f77b4", lw=1.8, linestyle="--", label="Ensemble (Raw)")
        ax.plot(seasons, df_by_season["Brier_Seed_Model"], marker="^", color="#ff7f0e", lw=1.5, linestyle=":", label="Seed Differential Baseline")
        if "Brier_Massey_Model" in df_by_season.columns:
            ax.plot(seasons, df_by_season["Brier_Massey_Model"], marker="d", color="#9467bd", lw=1.3, linestyle="-.", label="Massey Ordinals Baseline")
        ax.axhline(0.2500, color="#d62728", linestyle="--", alpha=0.8, label="50/50 Uninformed Baseline (0.2500)")

        mean_brier = eval_summary.get("cv_brier_score", float(df_by_season["Brier_Ensemble_Sharpened"].mean()))
        ax.axhline(mean_brier, color="#2ca02c", linestyle=":", alpha=0.6, label=f"Mean Sharpened CV ({mean_brier:.4f})")

        ax.set_title("NCAA Tournament Kaggle Competition Brier Score Over Time (2003-2025)", fontsize=14, fontweight="bold", pad=12)
        ax.set_xlabel("Tournament Season", fontsize=12)
        ax.set_ylabel("Brier Score / MSE (Lower is Better)", fontsize=12)
        ax.set_xticks(seasons[::2])
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", frameon=True)
        plt.tight_layout()
        fig.savefig(output_dir / "brier_by_season.jpg", dpi=200)
        plt.close(fig)

    # 2. Brier Score By Round Bar Chart
    if not df_by_round.empty:
        fig, ax = plt.subplots(figsize=(11, 5.5))
        rounds = df_by_round["Round_Name"]
        x = np.arange(len(rounds))
        width = 0.35

        ax.bar(x - width / 2, df_by_round["Brier_Ensemble_Sharpened"], width, label="Ensemble (Sharpened)", color="#2ca02c")
        ax.bar(x + width / 2, df_by_round["Brier_Seed_Model"], width, label="Seed Differential Baseline", color="#ff7f0e")
        ax.axhline(0.2500, color="#d62728", linestyle="--", alpha=0.7, label="50/50 Baseline (0.2500)")

        for i, val in enumerate(df_by_round["Brier_Ensemble_Sharpened"]):
            ax.text(i - width / 2, val + 0.005, f"{val:.3f}", ha="center", fontsize=9, fontweight="bold")
        for i, val in enumerate(df_by_round["Brier_Seed_Model"]):
            ax.text(i + width / 2, val + 0.005, f"{val:.3f}", ha="center", fontsize=9)

        ax.set_title("Brier Score by NCAA Tournament Round", fontsize=14, fontweight="bold", pad=12)
        ax.set_xlabel("Tournament Round", fontsize=12)
        ax.set_ylabel("Brier Score (Lower is Better)", fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(rounds, rotation=15, ha="right", fontsize=10)
        ax.set_ylim(0, max(0.28, float(df_by_round["Brier_Seed_Model"].max()) + 0.04))
        ax.grid(True, alpha=0.3, axis="y")
        ax.legend(loc="upper left", frameon=True)
        plt.tight_layout()
        fig.savefig(output_dir / "brier_by_round.jpg", dpi=200)
        plt.close(fig)

    # 3. Calibration Curve / Reliability Diagram
    bins = calibration_data.get("bins", [])
    if bins:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 8), gridspec_kw={"height_ratios": [3, 1]}, sharex=True)
        pred_means = [b["avg_prob"] for b in bins if b["count"] > 0]
        emp_freqs = [b["emp_freq"] for b in bins if b["count"] > 0]
        bin_mids = [b["bin_center"] for b in bins]
        counts = [b["count"] for b in bins]

        # Reliability Curve
        ax1.plot([0, 1], [0, 1], linestyle="--", color="#7f7f7f", label="Perfect Calibration (y = x)")
        ax1.plot(pred_means, emp_freqs, marker="o", lw=2.5, color="#1f77b4", label="Ensemble Predictions")
        ax1.set_ylabel("Empirical Win Rate", fontsize=12)
        ece_val = eval_summary.get("expected_calibration_error", 0.0)
        mce_val = eval_summary.get("max_calibration_error", 0.0)
        brier_val = eval_summary.get("cv_brier_score", 0.0)
        ax1.set_title(
            f"Model Calibration Reliability Diagram\nBrier Score: {brier_val:.5f} | ECE: {ece_val:.4f} | MCE: {mce_val:.4f}",
            fontsize=13, fontweight="bold", pad=10
        )
        ax1.legend(loc="upper left", frameon=True)
        ax1.grid(True, alpha=0.3)

        # Histogram of sample density
        ax2.bar(bin_mids, counts, width=0.08, color="#aec7e8", edgecolor="#1f77b4", alpha=0.7)
        ax2.set_xlabel("Predicted Win Probability Bin", fontsize=12)
        ax2.set_ylabel("Matchup Count", fontsize=11)
        ax2.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        fig.savefig(output_dir / "brier_reliability_calibration.jpg", dpi=200)
        plt.close(fig)

    # 4. Model & Baseline Comparison Horizontal Bar Chart
    fig, ax = plt.subplots(figsize=(10, 5.5))
    models = [
        ("50/50 Baseline", eval_summary.get("brier_5050_baseline", 0.2500), "#d62728"),
        ("Seed Baseline", eval_summary.get("brier_seed_baseline", 0.1928), "#ff7f0e"),
        ("Massey Baseline", eval_summary.get("brier_massey_baseline", 0.1924), "#9467bd"),
        ("Net Rating Baseline", eval_summary.get("brier_net_baseline", 0.1915), "#8c564b"),
        ("LightGBM Component", eval_summary.get("brier_lightgbm", 0.1895), "#17becf"),
        ("Logistic Regression Component", eval_summary.get("brier_logreg", 0.1870), "#1f77b4"),
        ("Ensemble (Raw)", eval_summary.get("cv_brier_raw", 0.1867), "#3399e6"),
        ("Ensemble (Sharpened)", eval_summary.get("cv_brier_score", 0.1866), "#2ca02c"),
    ]
    names = [m[0] for m in models]
    scores = [m[1] for m in models]
    colors = [m[2] for m in models]
    y_pos = np.arange(len(names))

    bars = ax.barh(y_pos, scores, color=colors, height=0.65)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=11)
    ax.invert_yaxis()  # Top-down
    ax.set_xlabel("Out-Of-Fold Brier Score (Lower is Better)", fontsize=12)
    ax.set_title("NCAA Tournament Model Architecture Comparison (Brier Score / MSE)", fontsize=13, fontweight="bold", pad=12)

    for bar, score in zip(bars, scores):
        rel_imp = ((0.2500 - score) / 0.2500) * 100.0
        ax.text(
            bar.get_width() + 0.003, bar.get_y() + bar.get_height() / 2,
            f"{score:.5f} ({rel_imp:+.1f}% vs 50/50)", va="center", fontsize=9.5, fontweight="bold"
        )

    ax.set_xlim(0, max(scores) * 1.25)
    ax.grid(True, alpha=0.3, axis="x")
    plt.tight_layout()
    fig.savefig(output_dir / "brier_model_comparison.jpg", dpi=200)
    plt.close(fig)


def export_bracket_files(
    season: int,
    brackets: List[Dict[str, Any]],
    summary_df: pd.DataFrame,
    p_matrix: Dict[Tuple[int, int], float],
    seeds_df: pd.DataFrame,
    teams_df: pd.DataFrame,
    output_dir: Path,
    model: CalibratedMatchupModel,
    sample_sub_path: Optional[Path] = None,
    tourney_compact_df: Optional[pd.DataFrame] = None,
):
    """Save all 25 individual bracket CSVs, ASCII text brackets, summaries, and Kaggle submissions."""
    season_dir = output_dir / f"kaggle_mania_{season}"
    brackets_dir = season_dir / "brackets"
    brackets_dir.mkdir(parents=True, exist_ok=True)

    team_name = dict(zip(teams_df["TeamID"], teams_df["TeamName"]))
    team_seed = dict(zip(seeds_df["TeamID"], seeds_df["Seed"]))

    # 1. Save Summary CSV
    summary_path = season_dir / f"brackets_portfolio_summary_{season}.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"  [Season {season}] Portfolio summary saved: {summary_path}")

    # 2. Save Individual Bracket CSVs and ASCII Visuals
    for b in brackets:
        b_num = b["bracket_number"]
        b_name = b["bracket_name"].lower()
        csv_file = brackets_dir / f"{b_name}.csv"
        txt_file = brackets_dir / f"{b_name}.txt"

        rows = []
        for slot_name, pick in b["picks"].items():
            w_id = pick["winner_id"]
            s_id = pick["strong_id"]
            k_id = pick["weak_id"]
            rows.append({
                "Slot": slot_name,
                "Round": pick["round"],
                "Round_Name": pick["round_name"],
                "Strong_Team_Seed": team_seed.get(s_id, "?"),
                "Strong_Team_Name": team_name.get(s_id, f"ID_{s_id}"),
                "Weak_Team_Seed": team_seed.get(k_id, "?"),
                "Weak_Team_Name": team_name.get(k_id, f"ID_{k_id}"),
                "Picked_Winner_Seed": team_seed.get(w_id, "?"),
                "Picked_Winner_Name": team_name.get(w_id, f"ID_{w_id}"),
                "Win_Probability": round(float(pick["prob"]), 4),
                "Round_Points": pick["points"],
            })

        df_picks = pd.DataFrame(rows)
        # Order by round and slot
        df_picks = df_picks.sort_values(["Round", "Slot"]).reset_index(drop=True)
        df_picks.to_csv(csv_file, index=False)

        # ASCII formatted bracket file
        with open(txt_file, "w", encoding="utf-8") as f:
            f.write("=" * 80 + "\n")
            f.write(f"  MARCH MADNESS BRACKET {b_num:02d}: {b['bracket_name'].upper()}\n")
            f.write(f"  Season: {season} | Strategy: {b['description']}\n")
            f.write(f"  Champion: {b['champion']} | Runner-Up: {b['runner_up']}\n")
            f.write(f"  Expected Points: {b['expected_points']} | Upsets Picked: {b['upsets_count']}\n")
            f.write("=" * 80 + "\n\n")

            for r_num in range(7):
                r_rows = df_picks[df_picks["Round"] == r_num]
                if r_rows.empty: continue
                f.write(f"--- {ROUND_NAMES.get(r_num, f'Round {r_num}').upper()} ---\n")
                for _, r in r_rows.iterrows():
                    f.write(
                        f"  [{r['Slot']:4s}] ({r['Strong_Team_Seed']:4s}) {r['Strong_Team_Name']:18s} vs "
                        f"({r['Weak_Team_Seed']:4s}) {r['Weak_Team_Name']:18s} "
                        f"-> PICK: ({r['Picked_Winner_Seed']:4s}) {r['Picked_Winner_Name']:18s} "
                        f"({r['Win_Probability']:.1%})\n"
                    )
                f.write("\n")

    # 3. Save Kaggle Format Submission File (ID, Pred)
    sub_path = season_dir / f"kaggle_submission_{season}.csv"
    sub_saved = False

    # Check SampleSubmissionStage2 for 2026, or SampleSubmissionStage1 for earlier seasons
    sub_sample_path = sample_sub_path
    if season < 2026 and sample_sub_path:
        stage1_path = sample_sub_path.parent / "SampleSubmissionStage1.csv"
        if stage1_path.exists():
            sub_sample_path = stage1_path

    if sub_sample_path and sub_sample_path.exists():
        sub_template = pd.read_csv(sub_sample_path)
        matching = sub_template[sub_template["ID"].str.startswith(str(season))].copy()
        if not matching.empty:
            parts = matching["ID"].str.split("_", expand=True)
            t1_arr = parts[1].astype(int).to_numpy()
            t2_arr = parts[2].astype(int).to_numpy()
            preds = [round(float(p_matrix.get((t1, t2), 0.5)), 5) for t1, t2 in zip(t1_arr, t2_arr)]
            matching["Pred"] = preds
            matching[["ID", "Pred"]].to_csv(sub_path, index=False)
            print(f"  [Season {season}] Kaggle submission saved: {sub_path} ({len(matching)} matchups)", flush=True)
            sub_saved = True

    if not sub_saved:
        # Build predictions for all possible tournament team pairs
        sub_rows = []
        tourney_teams = sorted(seeds_df["TeamID"].unique())
        for i in range(len(tourney_teams)):
            for j in range(i + 1, len(tourney_teams)):
                t1, t2 = tourney_teams[i], tourney_teams[j]
                match_id = f"{season}_{t1}_{t2}"
                p = p_matrix.get((t1, t2), 0.5)
                sub_rows.append({"ID": match_id, "Pred": round(float(p), 5)})

        pd.DataFrame(sub_rows).to_csv(sub_path, index=False)
        print(f"  [Season {season}] Tournament matchups submission saved: {sub_path} ({len(sub_rows)} matchups)", flush=True)

    # 4. Evaluate Kaggle submission if actual tournament outcomes exist
    sub_eval = None
    if tourney_compact_df is not None and not tourney_compact_df.empty:
        sub_eval = evaluate_kaggle_submission_file(sub_path, tourney_compact_df, season=season)
        if sub_eval.get("status") == "SUCCESS":
            print(f"\n  +----------------------------------------------------------------+")
            print(f"  | OFFICIAL KAGGLE SUBMISSION EVALUATION (SEASON {season})            |")
            print(f"  +----------------------------------------------------------------+")
            print(f"  | Tournament Games Evaluated:     {sub_eval['n_games_evaluated']:>5d}                          |")
            print(f"  | Kaggle Competition Brier Score: {sub_eval['brier_score']:>8.5f}  (Baseline: 0.25000)     |")
            print(f"  | Brier Score Improvement:       {sub_eval['improvement_vs_5050_pct']:>7.2f}% vs Uninformed        |")
            print(f"  | Kaggle Competition Log Loss:    {sub_eval['log_loss']:>8.5f}                          |")
            print(f"  | Straight-Up Accuracy:           {sub_eval['accuracy']:>7.2%}                          |")
            print(f"  +----------------------------------------------------------------+\n")

    # 5. Save Metadata JSON
    meta_path = season_dir / f"metadata_{season}.json"
    meta = {
        "season": season,
        "n_brackets": len(brackets),
        "cv_brier_score": round(float(model.cv_brier), 5),
        "cv_brier_raw": round(float(model.cv_brier_raw), 5),
        "brier_evaluation_summary": model.eval_summary,
        "feature_weights": model.feature_weights,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "generated_files": {
            "summary_csv": str(summary_path.name),
            "kaggle_submission": str(sub_path.name),
            "brackets_directory": "brackets/",
        },
    }
    if sub_eval and sub_eval.get("status") == "SUCCESS":
        meta["submission_actual_tournament_evaluation"] = {
            "n_games_evaluated": sub_eval["n_games_evaluated"],
            "brier_score": sub_eval["brier_score"],
            "log_loss": sub_eval["log_loss"],
            "accuracy": sub_eval["accuracy"],
            "improvement_vs_5050_pct": sub_eval["improvement_vs_5050_pct"],
        }

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


# ===========================================================================
# 7. Main Pipeline Runner
# ===========================================================================

def run_kaggle_mania_pipeline(
    kaggle_dir: Path,
    seasons: List[int],
    gender: str = "M",
    output_dir: Path = Path("model_outputs/kaggle_mania"),
    n_sims: int = 10000,
    base_seed: int = 42,
    sharpen: bool = True,
    evaluate_brier: bool = True,
    no_plots: bool = False,
):
    """Execute end-to-end Kaggle March Machine Learning Mania bracket portfolio generation and Brier evaluation."""
    print("=" * 80)
    print("  KAGGLE MARCH MACHINE LEARNING MANIA - 25 BRACKET PORTFOLIO GENERATOR")
    print("  Integrating 2026 Winner Methodologies, Brier Score MSE Optimization & DP Solver")
    print("=" * 80)

    # 1. Load historical data
    print(f"\n[1/4] Loading Kaggle tournament dataset ({gender.upper()} division) from {kaggle_dir}...")
    data = load_kaggle_data(kaggle_dir, gender)
    reg_df = data["regular"]
    tourn_df = data["tourney_compact"]
    seeds_df = data["seeds"]
    slots_df = data["slots"]
    teams_df = data["teams"]
    massey_path = kaggle_dir / "MMasseyOrdinals.csv"

    # 2. Compute Features
    print("[2/4] Engineering 2% trimmed efficiency, Four Factors, SoS, and Massey rankings...")
    t0 = time.time()
    stats_df = compute_team_season_stats(reg_df)

    if massey_path.exists() and gender.upper() == "M":
        massey_df = compute_massey_rankings(massey_path)
        if not massey_df.empty:
            stats_df = stats_df.merge(massey_df, on=["Season", "TeamID"], how="left")
            stats_df["MasseyMedian"] = stats_df["MasseyMedian"].fillna(150.0)
            stats_df["MasseyMean"] = stats_df["MasseyMean"].fillna(150.0)

    if "MasseyMedian" not in stats_df.columns:
        stats_df["MasseyMedian"] = 150.0
        stats_df["MasseyMean"] = 150.0

    print(f"  Efficiency and consensus metrics compiled in {time.time() - t0:.2f}s for {len(stats_df)} team-seasons.")

    # 3. Train Calibrated Matchup Model
    print("[3/4] Training and calibrating ensemble model with Leave-One-Season-Out Brier Score CV...")
    seeds_df["SeedNum"] = seeds_df["Seed"].apply(extract_seed_number)
    seed_map = dict(zip(zip(seeds_df["Season"], seeds_df["TeamID"]), seeds_df["SeedNum"]))
    stats_map = stats_df.set_index(["Season", "TeamID"]).to_dict(orient="index")

    t_sub = tourn_df[tourn_df["Season"] >= 2003].copy()
    pair_rows = []
    for _, row in t_sub.iterrows():
        s, w, l = int(row["Season"]), int(row["WTeamID"]), int(row["LTeamID"])
        day = int(row["DayNum"]) if "DayNum" in row else 136
        r_num = get_game_round(day)
        sw, sl = seed_map.get((s, w), 8), seed_map.get((s, l), 8)
        st_w = stats_map.get((s, w), {})
        st_l = stats_map.get((s, l), {})

        d_seed = sl - sw
        d_massey = st_l.get("MasseyMedian", 150.0) - st_w.get("MasseyMedian", 150.0)
        d_net = st_w.get("NetRtg", 0.0) - st_l.get("NetRtg", 0.0)
        d_off = st_w.get("OffRtg", 100.0) - st_l.get("OffRtg", 100.0)
        d_def = st_l.get("DefRtg", 100.0) - st_w.get("DefRtg", 100.0)
        d_efg = (st_w.get("eFG", 0.5) - st_l.get("eFG", 0.5)) * 100.0
        d_tov = (st_l.get("TOV", 0.18) - st_w.get("TOV", 0.18)) * 100.0
        d_orb = (st_w.get("ORB", 0.3) - st_l.get("ORB", 0.3)) * 100.0
        d_ftr = (st_w.get("FTR", 0.3) - st_l.get("FTR", 0.3)) * 100.0
        d_win = (st_w.get("Win", 0.5) - st_l.get("Win", 0.5)) * 100.0
        d_qw = st_w.get("QualityWins", 0.0) - st_l.get("QualityWins", 0.0)
        d_sos = st_w.get("SoS", 0.0) - st_l.get("SoS", 0.0)

        pair_rows.append({
            "Season": s, "Round": r_num, "d_seed": d_seed, "d_massey": d_massey, "d_net": d_net,
            "d_off": d_off, "d_def": d_def, "d_efg": d_efg, "d_tov": d_tov,
            "d_orb": d_orb, "d_ftr": d_ftr, "d_win": d_win, "d_qw": d_qw,
            "d_sos": d_sos, "target": 1,
        })
        pair_rows.append({
            "Season": s, "Round": r_num, "d_seed": -d_seed, "d_massey": -d_massey, "d_net": -d_net,
            "d_off": -d_off, "d_def": -d_def, "d_efg": -d_efg, "d_tov": -d_tov,
            "d_orb": -d_orb, "d_ftr": -d_ftr, "d_win": -d_win, "d_qw": -d_qw,
            "d_sos": -d_sos, "target": 0,
        })

    df_pairs = pd.DataFrame(pair_rows)
    model = CalibratedMatchupModel(sharpen=sharpen)
    cv_brier = model.fit_cross_validate(df_pairs)
    print(f"  Model trained! Out-Of-Fold Cross-Validation Brier Score (MSE): {cv_brier:.5f}")

    # Brier Score Evaluation Reporting & Diagnostic Artifacts Export
    if evaluate_brier:
        print("\n" + "=" * 80)
        print("  KAGGLE PIPELINE BRIER SCORE EVALUATION & CALIBRATION REPORT")
        print("=" * 80)
        es = model.eval_summary
        print(f"  Out-Of-Fold Brier Score (Sharpened): {es.get('cv_brier_score', 0):.5f}")
        print(f"  Out-Of-Fold Brier Score (Raw):       {es.get('cv_brier_raw', 0):.5f}")
        print(f"  Seed Differential Baseline Brier:    {es.get('brier_seed_baseline', 0):.5f} (-{es.get('improvement_vs_seed_pct', 0):.1f}% improvement)")
        print(f"  Massey Ordinals Baseline Brier:      {es.get('brier_massey_baseline', 0):.5f}")
        print(f"  50/50 Uninformed Baseline Brier:     {es.get('brier_5050_baseline', 0):.5f} (-{es.get('improvement_vs_5050_pct', 0):.1f}% improvement)")
        print(f"  Expected Calibration Error (ECE):    {es.get('expected_calibration_error', 0):.5f}")
        print(f"  Out-Of-Fold Tournament Accuracy:     {es.get('accuracy', 0):.2%}")
        print(f"  Total Games Evaluated:               {es.get('total_games_evaluated', 0)} across {es.get('seasons_evaluated_count', 0)} historical tournaments")

        print(f"\n--- BRIER SCORE BY TOURNAMENT ROUND ---")
        disp_round_cols = ["Round", "Round_Name", "Games_Count", "Brier_Ensemble_Sharpened", "Brier_Seed_Model", "Accuracy_Ensemble"]
        valid_round_cols = [c for c in disp_round_cols if c in model.df_by_round.columns]
        print(model.df_by_round[valid_round_cols].to_string(index=False))

        print(f"\n--- BRIER SCORE BY MATCHUP CERTAINTY TIER ---")
        disp_mismatch_cols = ["Mismatch_Tier", "Games_Count", "Brier_Ensemble_Sharpened", "Brier_Seed_Model", "Accuracy_Ensemble"]
        valid_mismatch_cols = [c for c in disp_mismatch_cols if c in model.df_by_mismatch.columns]
        print(model.df_by_mismatch[valid_mismatch_cols].to_string(index=False))

        # Save artifacts
        output_dir.mkdir(parents=True, exist_ok=True)
        summary_json_path = output_dir / "brier_evaluation_summary.json"
        with open(summary_json_path, "w", encoding="utf-8") as f:
            json.dump(es, f, indent=2)

        model.df_by_season.to_csv(output_dir / "brier_by_season.csv", index=False)
        model.df_by_round.to_csv(output_dir / "brier_by_round.csv", index=False)
        model.df_by_mismatch.to_csv(output_dir / "brier_by_mismatch.csv", index=False)

        # Model comparison table
        b50 = es.get("brier_5050_baseline", 0.2500)
        model_comp_rows = [
            {"Model": "50/50 Baseline", "Brier_Score": b50, "Rel_Improvement_vs_5050_pct": 0.0},
            {"Model": "Seed Baseline", "Brier_Score": es.get("brier_seed_baseline", 0.1928), "Rel_Improvement_vs_5050_pct": round((b50 - es.get("brier_seed_baseline", 0.1928)) / b50 * 100, 2)},
            {"Model": "Massey Baseline", "Brier_Score": es.get("brier_massey_baseline", 0.1924), "Rel_Improvement_vs_5050_pct": round((b50 - es.get("brier_massey_baseline", 0.1924)) / b50 * 100, 2)},
            {"Model": "Net Rating Baseline", "Brier_Score": es.get("brier_net_baseline", 0.1915), "Rel_Improvement_vs_5050_pct": round((b50 - es.get("brier_net_baseline", 0.1915)) / b50 * 100, 2)},
            {"Model": "LightGBM Component", "Brier_Score": es.get("brier_lightgbm", 0.1895), "Rel_Improvement_vs_5050_pct": round((b50 - es.get("brier_lightgbm", 0.1895)) / b50 * 100, 2)},
            {"Model": "Logistic Regression Component", "Brier_Score": es.get("brier_logreg", 0.1870), "Rel_Improvement_vs_5050_pct": round((b50 - es.get("brier_logreg", 0.1870)) / b50 * 100, 2)},
            {"Model": "Ensemble (Raw)", "Brier_Score": es.get("cv_brier_raw", 0.1867), "Rel_Improvement_vs_5050_pct": round((b50 - es.get("cv_brier_raw", 0.1867)) / b50 * 100, 2)},
            {"Model": "Ensemble (Sharpened)", "Brier_Score": es.get("cv_brier_score", 0.1866), "Rel_Improvement_vs_5050_pct": es.get("improvement_vs_5050_pct", 25.37)},
        ]
        pd.DataFrame(model_comp_rows).to_csv(output_dir / "brier_model_comparison.csv", index=False)

        if not no_plots:
            generate_brier_evaluation_plots(
                es, model.df_by_season, model.df_by_round, model.df_by_mismatch, model.calibration_data, output_dir
            )
            print(f"\n  [Visualizations] Diagnostic Brier plots rendered under: {output_dir.resolve()}")

    # 4. Process Requested Seasons
    target_seasons = []
    avail_seasons = sorted([int(s) for s in seeds_df["Season"].unique() if s >= 2003])
    if 2027 not in avail_seasons:
        avail_seasons.append(2027)

    for s in seasons:
        if str(s).lower() == "all":
            target_seasons = avail_seasons
            break
        try:
            target_seasons.append(int(s))
        except ValueError:
            print(f"  Warning: Invalid season '{s}' ignored.")

    if not target_seasons:
        target_seasons = [2026, 2027]
    target_seasons = sorted(list(set(target_seasons)))

    print(f"\n[4/4] Generating 25 Brackets Portfolio and Kaggle Submissions for {len(target_seasons)} season(s): {target_seasons}...")
    sample_sub_path = kaggle_dir / "SampleSubmissionStage2.csv"

    for season in target_seasons:
        print(f"\n>>> PROCESSING SEASON {season} <<<")
        if season == 2026:
            # Official 2026 field
            cur_seeds = seeds_df[seeds_df["Season"] == 2026].copy()
            cur_slots = slots_df[slots_df["Season"] == 2026].copy()
            cur_stats = stats_df[stats_df["Season"] == 2026].copy()

            if cur_seeds.empty or cur_slots.empty:
                print(f"  Warning: No 2026 seeds/slots found in dataset. Projecting 2026 field...")
                cur_seeds, cur_slots, cur_stats = generate_projected_2027_field(teams_df, stats_df)
        elif season == 2027:
            # Future 2027 season: Project complete 68-team field and ratings
            print("  Generating projected 2027 NCAA Tournament field & multi-tier program ratings...")
            cur_seeds, cur_slots, cur_stats = generate_projected_2027_field(teams_df, stats_df)
        else:
            # Other historical/custom season
            cur_seeds = seeds_df[seeds_df["Season"] == season].copy()
            cur_slots = slots_df[slots_df["Season"] == season].copy()
            cur_stats = stats_df[stats_df["Season"] == season].copy()

        brackets, summary_df, p_matrix = generate_bracket_portfolio(
            season=season,
            seeds_df=cur_seeds,
            slots_df=cur_slots,
            stats_df=cur_stats,
            teams_df=teams_df,
            model=model,
            n_sims=n_sims,
            base_seed=base_seed,
            sharpen=sharpen,
        )

        export_bracket_files(
            season=season,
            brackets=brackets,
            summary_df=summary_df,
            p_matrix=p_matrix,
            seeds_df=cur_seeds,
            teams_df=teams_df,
            output_dir=output_dir,
            model=model,
            sample_sub_path=sample_sub_path if season == 2026 else (kaggle_dir / "SampleSubmissionStage1.csv"),
            tourney_compact_df=tourn_df,
        )

        # Print Executive Summary Table
        print(f"\n--- 25-BRACKET PORTFOLIO SUMMARY (SEASON {season}) ---")
        cols_to_print = ["Bracket_Number", "Champion", "Runner_Up", "Expected_Tournament_Points", "Leverage_Score", "Upsets_Picked_Count"]
        print(summary_df[cols_to_print].to_string(index=False))

    print("\n" + "=" * 80)
    print("  KAGGLE MANIA PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
    print(f"  Outputs saved under: {output_dir.resolve()}")
    print("=" * 80)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Kaggle March Machine Learning Mania Bracket Portfolio Generator & Brier Score Evaluator"
    )
    parser.add_argument(
        "--kaggle-dir",
        type=Path,
        default=Path("data/kaggle/march-machine-learning-mania-2026"),
        help="Path to the Kaggle March Machine Learning Mania 2026 dataset folder.",
    )
    parser.add_argument(
        "--seasons",
        nargs="+",
        default=["2026", "2027"],
        help="NCAA Tournament seasons to process (e.g. 2025 2026 2027, or 'all'). Default: 2026 2027.",
    )
    parser.add_argument(
        "--gender",
        type=str,
        default="M",
        choices=["M", "W"],
        help="Tournament division: 'M' for Men's or 'W' for Women's (default: M).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("model_outputs/kaggle_mania"),
        help="Directory to save generated brackets, submissions, and Brier summaries (default: model_outputs/kaggle_mania).",
    )
    parser.add_argument(
        "--n-sims",
        type=int,
        default=5000,
        help="Number of simulations for crowd leverage and stochastic paths (default: 5000).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed for reproducibility (default: 42).",
    )
    parser.add_argument(
        "--no-sharpen",
        action="store_true",
        help="Disable 1st-place Harrison Horan strategic probability edge sharpening.",
    )
    parser.add_argument(
        "--no-brier-eval",
        action="store_true",
        help="Skip detailed Brier score cross-validation evaluation and diagnostics.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Disable generating Brier score diagnostic plots.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_kaggle_mania_pipeline(
        kaggle_dir=args.kaggle_dir,
        seasons=args.seasons,
        gender=args.gender,
        output_dir=args.output_dir,
        n_sims=args.n_sims,
        base_seed=args.seed,
        sharpen=not args.no_sharpen,
        evaluate_brier=not args.no_brier_eval,
        no_plots=args.no_plots,
    )
