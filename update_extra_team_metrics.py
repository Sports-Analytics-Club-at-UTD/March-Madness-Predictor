import os
import time
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("CBBD_API_KEY")
BASE_URL = "https://api.collegebasketballdata.com"
HEADERS = {"Authorization": f"Bearer {API_KEY}"}
SEASONS = list(range(2005, 2026))


def safe_get(endpoint, params=None):
    url = f"{BASE_URL}{endpoint}"
    try:
        response = requests.get(url, params=params, headers=HEADERS, timeout=60)
        if response.status_code == 404:
            print(f"[404] {endpoint} {params}")
            return None
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] {endpoint}: {e}")
        return None


def team_columns(df):
    if df is None or df.empty:
        return {}
    if "team" in df.columns:
        values = df["team"].dropna()
        if not values.empty:
            return {"team": str(values.iloc[0])}
    return {}


def compute_efficiency_frame(df):
    if df is None or df.empty:
        return df

    base = df.copy()
    points_col = "teamStats.points.total"
    opp_points_col = "opponentStats.points.total"
    possessions_col = "teamStats.possessions"

    if points_col not in base.columns:
        points_col = "points"
    if opp_points_col not in base.columns:
        opp_points_col = "opponent_points"
    if possessions_col not in base.columns:
        possessions_col = "possessions"

    points = pd.to_numeric(base[points_col], errors="coerce")
    opp_points = pd.to_numeric(base[opp_points_col], errors="coerce")
    possessions = pd.to_numeric(base[possessions_col], errors="coerce")

    fga = pd.to_numeric(base.get("teamStats.fieldGoals.attempted", pd.Series([None] * len(base), index=base.index)), errors="coerce")
    fta = pd.to_numeric(base.get("teamStats.freeThrows.attempted", pd.Series([None] * len(base), index=base.index)), errors="coerce")
    tov = pd.to_numeric(base.get("teamStats.turnovers.total", pd.Series([None] * len(base), index=base.index)), errors="coerce")
    orb = pd.to_numeric(base.get("teamStats.rebounds.offensive", pd.Series([None] * len(base), index=base.index)), errors="coerce")

    est_possessions = fga + 0.44 * fta + tov - orb
    possessions = possessions.where(possessions.notna() & (possessions > 0)).combine_first(est_possessions)

    base["offensiveEfficiency"] = float("nan")
    base["defensiveEfficiency"] = float("nan")
    valid = possessions.notna() & (possessions > 0)
    base.loc[valid, "offensiveEfficiency"] = (points[valid] / possessions[valid]) * 100
    base.loc[valid, "defensiveEfficiency"] = (opp_points[valid] / possessions[valid]) * 100
    return base


def build_elo_lookup(season):
    games = safe_get("/games", {"season": season})
    if not games:
        return {}

    lookup = {}
    for record in games:
        if not isinstance(record, dict):
            continue
        home = record.get("homeTeam")
        away = record.get("awayTeam")
        start_date = record.get("startDate")
        if not home or not away or not start_date:
            continue

        payload = {
            "homeTeamEloStart": record.get("homeTeamEloStart"),
            "homeTeamEloEnd": record.get("homeTeamEloEnd"),
            "awayTeamEloStart": record.get("awayTeamEloStart"),
            "awayTeamEloEnd": record.get("awayTeamEloEnd"),
        }
        lookup[(start_date, home, away)] = payload
        lookup[(start_date, away, home)] = payload
    return lookup


def merge_elo_columns_for_team(df, season):
    if df is None or df.empty:
        return df

    elo_lookup = build_elo_lookup(season)
    if not elo_lookup:
        return df

    for col in ["homeTeamEloStart", "homeTeamEloEnd", "awayTeamEloStart", "awayTeamEloEnd"]:
        if col not in df.columns:
            df[col] = None

    for idx, row in df.iterrows():
        start_date = row.get("startDate")
        team_name = row.get("team")
        opponent = row.get("opponent")
        if not start_date or not team_name or not opponent:
            continue
        payload = elo_lookup.get((start_date, team_name, opponent))
        if payload is None:
            payload = elo_lookup.get((start_date, opponent, team_name))
        if payload is None:
            continue
        for col, value in payload.items():
            df.at[idx, col] = value

    return df


def update_season_file(year):
    season_path = Path("data") / str(year) / f"{year}_teams_season_stats.csv"
    if not season_path.exists():
        return

    df = pd.read_csv(season_path)
    df = compute_efficiency_frame(df)
    df.to_csv(season_path, index=False)
    print(f"[OK] season file: {season_path}")


def update_game_files(year):
    year_dir = Path("data") / str(year)
    if not year_dir.exists():
        return

    for team_dir in sorted(year_dir.iterdir()):
        if not team_dir.is_dir():
            continue

        for csv_path in sorted(team_dir.glob("*_game_stats.csv")):
            try:
                df = pd.read_csv(csv_path)
            except Exception:
                continue

            if df.empty:
                continue

            team_name = team_columns(df).get("team")
            if not team_name:
                continue

            df = compute_efficiency_frame(df)
            df = merge_elo_columns_for_team(df, year)
            df.to_csv(csv_path, index=False)
            print(f"[OK] game file: {csv_path}")


def main():
    for year in SEASONS:
        print(f"\n== Updating {year} ==")
        update_season_file(year)
        update_game_files(year)
        time.sleep(0.2)


if __name__ == "__main__":
    main()
