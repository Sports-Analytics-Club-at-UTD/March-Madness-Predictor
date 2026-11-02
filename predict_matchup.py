#!/usr/bin/env python3
"""Direct Inference Matchup Predictor using Primary Pipeline Methodology and Live CBBD API.

Pulls live/recent team statistics and adjusted ratings from the College Basketball Data (CBBD)
API, applies the primary pipeline's pairwise matchup methodology (Four Factors stylistic
clash differentials, tempo-free net rating, and calibrated logistic win probability), and
outputs win probabilities and projected point spreads with ZERO runtime model training.

Usage Examples:
  # Positional arguments (defaults to latest season 2026, neutral court)
  uv run python predict_matchup.py "Duke" "North Carolina"

  # Named arguments with specific season and home court advantage
  uv run python predict_matchup.py --team1 "UConn" --team2 "Houston" --season 2026 --home 1

  # Interactive mode (prompts for teams if run without arguments)
  uv run python predict_matchup.py
"""

from __future__ import annotations

import argparse
import difflib
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import requests
from dotenv import load_dotenv
from scipy.special import expit

load_dotenv()

CBBD_API_KEY = os.getenv("CBBD_API_KEY")
BASE_URL = "https://api.collegebasketballdata.com"

# Common team colloquialisms and abbreviations mapped to official CBBD school names
KNOWN_ALIASES: Dict[str, str] = {
    "uconn": "UConn",
    "connecticut": "UConn",
    "unc": "North Carolina",
    "tar heels": "North Carolina",
    "blue devils": "Duke",
    "pitt": "Pittsburgh",
    "nova": "Villanova",
    "ole miss": "Ole Miss",
    "mississippi": "Ole Miss",
    "smu": "SMU",
    "byu": "BYU",
    "lsu": "LSU",
    "ucf": "UCF",
    "vcu": "VCU",
    "tcu": "TCU",
    "ucla": "UCLA",
    "usc": "USC",
    "unlv": "UNLV",
    "utep": "UTEP",
    "utsa": "UTSA",
    "fau": "FAU",
    "fiu": "FIU",
    "umass": "UMass",
    "st marys": "Saint Mary's",
    "saint marys": "Saint Mary's",
    "st. mary's": "Saint Mary's",
    "st johns": "St. John's",
    "saint johns": "St. John's",
    "st. john's": "St. John's",
    "st bonaventure": "St. Bonaventure",
    "st. bonaventure": "St. Bonaventure",
    "st joseph's": "Saint Joseph's",
    "st. joseph's": "Saint Joseph's",
    "saint joes": "Saint Joseph's",
    "texas a&m": "Texas A&M",
    "a&m": "Texas A&M",
    "nc state": "NC State",
    "north carolina state": "NC State",
    "miami": "Miami",
    "miami fl": "Miami",
    "miami oh": "Miami (OH)",
    "penn state": "Penn State",
    "michigan state": "Michigan State",
    "michigan st": "Michigan State",
    "iowa state": "Iowa State",
    "iowa st": "Iowa State",
    "ohio state": "Ohio State",
    "ohio st": "Ohio State",
    "kansas state": "Kansas State",
    "kansas st": "Kansas State",
    "florida state": "Florida State",
    "florida st": "Florida State",
    "arizona state": "Arizona State",
    "arizona st": "Arizona State",
    "colorado state": "Colorado State",
    "colorado st": "Colorado State",
    "san diego state": "San Diego State",
    "san diego st": "San Diego State",
    "sdsu": "San Diego State",
    "new mexico": "New Mexico",
    "unm": "New Mexico",
    "wash state": "Washington State",
    "washington state": "Washington State",
    "wazzu": "Washington State",
}

# In-memory session cache for API lookups
_CACHE: Dict[str, Any] = {}


def get_headers() -> Dict[str, str]:
    if not CBBD_API_KEY:
        print("[WARNING] CBBD_API_KEY not found in environment or .env file.", file=sys.stderr)
        return {}
    return {"Authorization": f"Bearer {CBBD_API_KEY}"}


def fetch_all_teams() -> List[Dict[str, Any]]:
    """Fetch master list of all teams from CBBD to enable fuzzy school matching."""
    cache_key = "cbbd_master_teams"
    if cache_key in _CACHE:
        return _CACHE[cache_key]

    headers = get_headers()
    try:
        r = requests.get(f"{BASE_URL}/teams", headers=headers, timeout=12)
        if r.status_code == 200:
            teams = r.json()
            _CACHE[cache_key] = teams
            return teams
    except Exception as e:
        print(f"  [API Error] Failed to fetch /teams: {e}", file=sys.stderr)
    return []


def resolve_team_name(query: str) -> str:
    """Normalize user input string to canonical CBBD school name."""
    clean = query.strip()
    lower = clean.lower()

    if lower in KNOWN_ALIASES:
        return KNOWN_ALIASES[lower]

    all_teams = fetch_all_teams()
    if not all_teams:
        return clean

    # 1. Exact match on school, mascot, or abbreviation
    for t in all_teams:
        school = t.get("school", "")
        abbrev = t.get("abbreviation", "")
        mascot = t.get("mascot", "")
        display = t.get("displayName", "")
        short_display = t.get("shortDisplayName", "")

        if lower in (school.lower(), abbrev.lower(), mascot.lower(), display.lower(), short_display.lower()):
            return school

    # 2. Substring match
    schools = [t.get("school", "") for t in all_teams if t.get("school")]
    for s in schools:
        if lower == s.lower():
            return s
        if lower in s.lower() and len(lower) >= 4:
            return s

    # 3. Fuzzy ratio match
    matches = difflib.get_close_matches(clean, schools, n=1, cutoff=0.7)
    if matches:
        return matches[0]

    return clean


def get_national_ranks(season: int) -> Dict[str, int]:
    """Fetch all team adjusted ratings for the season to determine real national ranks."""
    cache_key = f"cbbd_ranks_{season}"
    if cache_key in _CACHE:
        return _CACHE[cache_key]

    headers = get_headers()
    try:
        r = requests.get(f"{BASE_URL}/ratings/adjusted", params={"season": season}, headers=headers, timeout=10)
        if r.status_code == 200:
            all_ratings = sorted(r.json(), key=lambda x: float(x.get("netRating", -999)), reverse=True)
            ranks = {t.get("team"): idx for idx, t in enumerate(all_ratings, start=1)}
            _CACHE[cache_key] = ranks
            return ranks
    except Exception as e:
        print(f"  [Notice] National ranks lookup fallback: {e}", file=sys.stderr)
    return {}


def fetch_team_data(team_name: str, season: int = 2026) -> Dict[str, Any]:
    """Fetch adjusted efficiency ratings and season stats from CBBD API."""
    canonical_name = resolve_team_name(team_name)
    cache_key = f"{canonical_name}_{season}"
    if cache_key in _CACHE:
        return _CACHE[cache_key]

    headers = get_headers()
    adj_data: Dict[str, Any] = {}
    stats_data: Dict[str, Any] = {}

    # 1. Adjusted ratings (Offensive Rating, Defensive Rating, Net Rating, Rankings)
    try:
        r_adj = requests.get(
            f"{BASE_URL}/ratings/adjusted",
            params={"season": season, "team": canonical_name},
            headers=headers,
            timeout=10,
        )
        if r_adj.status_code == 200 and r_adj.json():
            adj_data = r_adj.json()[0]
    except Exception as e:
        print(f"  [API Error] /ratings/adjusted for {canonical_name}: {e}", file=sys.stderr)

    # 2. Team season stats & Four Factors
    try:
        r_stats = requests.get(
            f"{BASE_URL}/stats/team/season",
            params={"season": season, "team": canonical_name},
            headers=headers,
            timeout=10,
        )
        if r_stats.status_code == 200 and r_stats.json():
            stats_data = r_stats.json()[0]
    except Exception as e:
        print(f"  [API Error] /stats/team/season for {canonical_name}: {e}", file=sys.stderr)

    # If recent season returned nothing, fallback to previous season (e.g. 2025)
    if not adj_data and not stats_data and season > 2020:
        fallback_s = season - 1
        print(f"  [Notice] No {season} records for '{canonical_name}'; attempting {fallback_s} fallback...", file=sys.stderr)
        return fetch_team_data(canonical_name, season=fallback_s)

    # Extract parsed fields
    tf = stats_data.get("teamStats", {}).get("fourFactors", {})
    of = stats_data.get("opponentStats", {}).get("fourFactors", {})
    ts = stats_data.get("teamStats", {})
    os_dict = stats_data.get("opponentStats", {})

    off_rtg = float(adj_data.get("offensiveRating", 108.0))
    def_rtg = float(adj_data.get("defensiveRating", 100.0))
    net_rtg = float(adj_data.get("netRating", off_rtg - def_rtg))

    pace = float(stats_data.get("pace", 68.5))
    games = int(stats_data.get("games", 0))
    wins = int(stats_data.get("wins", 0))
    losses = int(stats_data.get("losses", 0))

    # Four factors
    off_efg = float(tf.get("effectiveFieldGoalPct", 50.0))
    def_efg = float(of.get("effectiveFieldGoalPct", 50.0))
    off_tov = float(tf.get("turnoverRatio", 0.17))
    def_tov = float(of.get("turnoverRatio", 0.17))
    off_orb = float(tf.get("offensiveReboundPct", 30.0))
    opp_orb = float(of.get("offensiveReboundPct", 30.0))
    def_drb = 100.0 - opp_orb
    off_ftr = float(tf.get("freeThrowRate", 30.0))
    def_ftr = float(of.get("freeThrowRate", 30.0))

    pts_raw = ts.get("points", 0)
    pts_val = float(pts_raw.get("total", 0) if isinstance(pts_raw, dict) else (pts_raw or 0))
    pts_pg = pts_val / max(games, 1) if games > 0 else 72.0

    opp_pts_raw = os_dict.get("points", 0)
    opp_pts_val = float(opp_pts_raw.get("total", 0) if isinstance(opp_pts_raw, dict) else (opp_pts_raw or 0))
    opp_pts_pg = opp_pts_val / max(games, 1) if games > 0 else 68.0

    ranks_map = get_national_ranks(season)
    national_rank = ranks_map.get(canonical_name)

    team_dict = {
        "team": canonical_name,
        "season": season,
        "conference": adj_data.get("conference", stats_data.get("conference", "N/A")),
        "record": f"{wins}-{losses}",
        "games": games,
        "wins": wins,
        "losses": losses,
        "national_rank": national_rank,
        "off_rtg": off_rtg,
        "def_rtg": def_rtg,
        "net_rtg": net_rtg,
        "pace": pace,
        "pts_pg": pts_pg,
        "opp_pts_pg": opp_pts_pg,
        "off_efg": off_efg,
        "def_efg": def_efg,
        "off_tov": off_tov,
        "def_tov": def_tov,
        "off_orb": off_orb,
        "def_drb": def_drb,
        "off_ftr": off_ftr,
        "def_ftr": def_ftr,
    }

    _CACHE[cache_key] = team_dict
    return team_dict


# ===========================================================================
# Pairwise Direct Inference Methodology
# ===========================================================================

def compute_pairwise_win_probability(
    team1: Dict[str, Any],
    team2: Dict[str, Any],
    home_court: int = 0,  # 0 = neutral, 1 = team1 home, 2 = team2 home
) -> Dict[str, Any]:
    """Compute direct model inference win probability using primary pipeline methodology."""
    # 1. Primary Efficiency & Continuous Score Differentials
    d_net = (team1["net_rtg"] - team2["net_rtg"]) / 10.0
    dz = (team1["net_rtg"] - team2["net_rtg"]) / 7.5

    # 2. Four Factors Stylistic Clash Differentials
    d_efg = ((team1["off_efg"] - team2["def_efg"]) - (team2["off_efg"] - team1["def_efg"])) / 10.0
    d_tov = ((team1["def_tov"] - team2["off_tov"]) - (team2["def_tov"] - team1["off_tov"])) * 10.0
    d_reb = ((team1["off_orb"] - (100.0 - team2["def_drb"])) - (team2["off_orb"] - (100.0 - team1["def_drb"]))) / 10.0
    d_ftr = ((team1["off_ftr"] - team2["def_ftr"]) - (team2["off_ftr"] - team1["def_ftr"])) / 10.0

    # 3. Home Court Advantage Adjustment
    hca_logit = 0.0
    hca_pts = 0.0
    if home_court == 1:
        hca_logit = 0.35
        hca_pts = 3.2
    elif home_court == 2:
        hca_logit = -0.35
        hca_pts = -3.2

    # Calibrated primary pipeline logistic weights
    logit = (
        0.58 * dz +
        0.38 * d_net +
        0.35 * d_efg +
        0.28 * d_tov +
        0.22 * d_reb +
        0.14 * d_ftr +
        hca_logit
    )

    p1 = float(expit(logit))
    p1 = float(np.clip(p1, 0.01, 0.99))
    p2 = 1.0 - p1

    # 4. Projected Pace, Total, and Final Scores
    ncaa_avg_pace = 68.5
    exp_pace = (team1["pace"] * team2["pace"]) / ncaa_avg_pace

    exp_margin = ((team1["net_rtg"] - team2["net_rtg"]) * (exp_pace / 100.0)) + hca_pts

    t1_score = exp_pace * ((team1["off_rtg"] + team2["def_rtg"]) / 200.0) + (hca_pts / 2.0)
    t2_score = exp_pace * ((team2["off_rtg"] + team1["def_rtg"]) / 200.0) - (hca_pts / 2.0)

    # Format point spread
    if exp_margin >= 0:
        spread_str = f"{team1['team']} -{abs(exp_margin):.1f}"
    else:
        spread_str = f"{team2['team']} -{abs(exp_margin):.1f}"

    return {
        "prob_team1": p1,
        "prob_team2": p2,
        "exp_margin": exp_margin,
        "spread_str": spread_str,
        "exp_pace": exp_pace,
        "score_team1": round(t1_score, 1),
        "score_team2": round(t2_score, 1),
        "d_net": d_net * 10.0,
        "d_efg": d_efg * 10.0,
        "d_tov": d_tov / 10.0,
        "d_reb": d_reb * 10.0,
        "d_ftr": d_ftr * 10.0,
    }


# ===========================================================================
# Formatted Output Display
# ===========================================================================

def print_matchup_analysis(
    t1: Dict[str, Any],
    t2: Dict[str, Any],
    res: Dict[str, Any],
    home_court: int = 0,
):
    """Print an executive-level head-to-head comparison and probability breakdown."""
    loc_str = "Neutral Court" if home_court == 0 else (
        f"Home Court: {t1['team']}" if home_court == 1 else f"Home Court: {t2['team']}"
    )

    t1_rank_str = f"#{t1['national_rank']}" if t1['national_rank'] else "Unranked"
    t2_rank_str = f"#{t2['national_rank']}" if t2['national_rank'] else "Unranked"

    width = 82
    print("\n" + "=" * width)
    print(f"  MATCHUP PREDICTION: {t1['team'].upper()} vs {t2['team'].upper()} ({loc_str})")
    print(f"  Season: {t1['season']} | Primary Pipeline Direct Model Inference")
    print("=" * width)

    # Main Win Probabilities
    winner_name = t1['team'] if res['prob_team1'] >= 0.50 else t2['team']
    winner_prob = max(res['prob_team1'], res['prob_team2'])
    loser_name = t2['team'] if winner_name == t1['team'] else t1['team']
    loser_prob = min(res['prob_team1'], res['prob_team2'])

    print(f"\n  >>> PROJECTED WINNER: {winner_name.upper()} ({winner_prob:.1%}) <<<")
    print(f"  {t1['team']}: {res['prob_team1']:.1%}  |  {t2['team']}: {res['prob_team2']:.1%}")
    print(f"  Projected Score : {t1['team']} {res['score_team1']:.0f}, {t2['team']} {res['score_team2']:.0f}")
    print(f"  Projected Spread: {res['spread_str']}  |  Estimated Total: {res['score_team1'] + res['score_team2']:.1f} pts (Pace: {res['exp_pace']:.1f} poss)")

    # Head-to-Head Comparison Table
    print("\n" + "-" * width)
    print(f"  {'METRIC':<28s} | {t1['team']:<22s} | {t2['team']:<22s}")
    print("-" * width)
    print(f"  {'Conference':<28s} | {t1['conference']:<22s} | {t2['conference']:<22s}")
    print(f"  {'Record':<28s} | {t1['record']:<22s} | {t2['record']:<22s}")
    print(f"  {'National Net Rank':<28s} | {t1_rank_str:<22s} | {t2_rank_str:<22s}")
    print(f"  {'Adjusted Net Rating':<28s} | {t1['net_rtg']:>+6.1f}                 | {t2['net_rtg']:>+6.1f}")
    print(f"  {'Adjusted Offense (pts/100)':<28s} | {t1['off_rtg']:>6.1f}                 | {t2['off_rtg']:>6.1f}")
    print(f"  {'Adjusted Defense (pts/100)':<28s} | {t1['def_rtg']:>6.1f}                 | {t2['def_rtg']:>6.1f}")
    print(f"  {'Tempo / Pace (poss/40m)':<28s} | {t1['pace']:>6.1f}                 | {t2['pace']:>6.1f}")
    print("-" * width)
    print(f"  {'Effective FG% (Off / Def)':<28s} | {t1['off_efg']:.1f}% / {t1['def_efg']:.1f}%          | {t2['off_efg']:.1f}% / {t2['def_efg']:.1f}%")
    print(f"  {'Turnover Ratio (Off / Def)':<28s} | {t1['off_tov']*100:.1f}% / {t1['def_tov']*100:.1f}%          | {t2['off_tov']*100:.1f}% / {t2['def_tov']*100:.1f}%")
    print(f"  {'Rebound Rate (ORB / DRB)':<28s} | {t1['off_orb']:.1f}% / {t1['def_drb']:.1f}%          | {t2['off_orb']:.1f}% / {t2['def_drb']:.1f}%")
    print(f"  {'Free Throw Rate (Off / Def)':<28s} | {t1['off_ftr']:.1f}% / {t1['def_ftr']:.1f}%          | {t2['off_ftr']:.1f}% / {t2['def_ftr']:.1f}%")
    print("-" * width)

    # Strategic Insights
    print("\n  MATCHUP TACTICAL BREAKDOWN:")
    if abs(t1['net_rtg'] - t2['net_rtg']) >= 8.0:
        superior = t1['team'] if t1['net_rtg'] > t2['net_rtg'] else t2['team']
        diff = abs(t1['net_rtg'] - t2['net_rtg'])
        print(f"  * Efficiency Edge: {superior} holds a significant +{diff:.1f} Net Rating advantage.")
    else:
        print("  * Efficiency Edge: Teams are closely matched in overall efficiency (< 8.0 Net diff).")

    if abs(res['d_efg']) >= 2.0:
        edge_team = t1['team'] if res['d_efg'] > 0 else t2['team']
        print(f"  * Shooting Clash: {edge_team} has a distinct advantage in the eFG% battle.")

    if abs(res['d_tov']) >= 0.02:
        edge_team = t1['team'] if res['d_tov'] > 0 else t2['team']
        print(f"  * Ball Security: {edge_team} projects to win the turnover margin.")

    if abs(res['d_reb']) >= 3.0:
        edge_team = t1['team'] if res['d_reb'] > 0 else t2['team']
        print(f"  * Glass Dominance: {edge_team} holds an edge in offensive second-chance points.")

    print("=" * width + "\n")


# ===========================================================================
# Main CLI Driver
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Predict head-to-head NCAA win probability using live CBBD data & primary pipeline methodology."
    )
    parser.add_argument("teams", nargs="*", help="Optional positional team names (e.g., 'Duke' 'North Carolina').")
    parser.add_argument("--team1", "-t1", type=str, help="Name of Team 1 (e.g., 'Duke', 'UConn').")
    parser.add_argument("--team2", "-t2", type=str, help="Name of Team 2 (e.g., 'North Carolina', 'Houston').")
    parser.add_argument("--season", "-s", type=int, default=2026, help="College basketball season (default: 2026).")
    parser.add_argument(
        "--home",
        type=int,
        default=0,
        choices=[0, 1, 2],
        help="Court location: 0 = Neutral court (default), 1 = Team 1 Home, 2 = Team 2 Home.",
    )

    args = parser.parse_args()

    # Determine teams from positional or named args
    team1_name = args.team1
    team2_name = args.team2

    if args.teams and len(args.teams) >= 2:
        team1_name = args.teams[0]
        team2_name = args.teams[1]
    elif args.teams and len(args.teams) == 1 and not team1_name:
        team1_name = args.teams[0]

    # Interactive prompt fallback if team names not provided
    if not team1_name:
        try:
            team1_name = input("Enter Team 1 (e.g. Duke, UConn, Houston): ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            sys.exit(0)

    if not team2_name:
        try:
            team2_name = input("Enter Team 2 (e.g. North Carolina, Kansas, Purdue): ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            sys.exit(0)

    if not team1_name or not team2_name:
        print("Error: Both Team 1 and Team 2 must be specified.", file=sys.stderr)
        sys.exit(1)

    print(f"\n[CBBD API] Fetching live data for '{team1_name}' and '{team2_name}' (Season {args.season})...")
    t1_data = fetch_team_data(team1_name, season=args.season)
    t2_data = fetch_team_data(team2_name, season=args.season)

    print(f"[Inference] Computing pairwise win probability via primary pipeline methodology...")
    result = compute_pairwise_win_probability(t1_data, t2_data, home_court=args.home)

    print_matchup_analysis(t1_data, t2_data, result, home_court=args.home)


if __name__ == "__main__":
    main()
