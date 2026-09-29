import os
import time
import requests
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("CBBD_API_KEY")
BASE_URL = "https://api.collegebasketballdata.com"
HEADERS = {"Authorization": f"Bearer {API_KEY}"}
SEASONS = range(2000, 2009)  # 2000 through 2008, inclusive.

def safe_get(endpoint, params=None):
    url = f"{BASE_URL}{endpoint}"
    try:
        response = requests.get(url, params=params, headers=HEADERS, timeout=60)
        if response.status_code == 404:
            print(f"      [404 Not Found] {endpoint} with params {params}")
            return None
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"      [Request Error] {endpoint}: {e}")
        return None

def sanitize(name):
    return "".join(c for c in str(name) if c.isalnum() or c in (" ", "_", "-")).strip().replace(" ", "_").lower()


def collect_teams_from_games(games):
    teams = set()
    if not games:
        return teams

    for game in games:
        if not isinstance(game, dict):
            continue

        for key in ("team", "opponent", "homeTeam", "awayTeam", "home_team", "away_team"):
            value = game.get(key)
            if value:
                teams.add(str(value))

        nested_teams = game.get("teams")
        if isinstance(nested_teams, list):
            for value in nested_teams:
                if value:
                    teams.add(str(value))

    return sorted(teams)


def choose_team_list(tournament_teams, raw_games):
    """Merge teams from both sources; coverage may still be incomplete."""
    tournament_list = sorted(set(tournament_teams or []))
    raw_team_list = collect_teams_from_games(raw_games)

    if not tournament_list:
        return raw_team_list, "games"
    if set(raw_team_list) - set(tournament_list):
        return sorted(set(raw_team_list) | set(tournament_list)), "games"
    return tournament_list, "tournament"


def calculate_missing_four_factors(df, prefix=""):
    """Fill missing factors in calc_* columns; preserve API values.

    Turnovers: season fraction, game/player percent. Other factors: percent.
    Use API_column.combine_first(calc_column) to combine values.
    """
    if df is None or df.empty:
        return df

    def numeric(column):
        if column not in df.columns:
            return pd.Series(float("nan"), index=df.index)
        return pd.to_numeric(df[column], errors="coerce").replace(
            [float("inf"), -float("inf")], float("nan")
        )

    def divide(numerator, denominator):
        return (numerator / denominator.where(denominator > 0)).replace(
            [float("inf"), -float("inf")], float("nan")
        )

    # Team fields are nested; player fields are unprefixed.
    is_team = any(c.startswith("teamStats.") for c in df.columns)
    root = "teamStats." if is_team else ""
    api_root = "teamStats.fourFactors." if is_team else ""

    def add_fallback(api_name, fallback_name, calculate):
        missing = numeric(api_root + api_name).isna()
        if missing.any():
            values = calculate().where(missing)
            # Missing inputs remain empty.
            df[f"{prefix}calc_{fallback_name}"] = values

    fgm = numeric(root + "fieldGoals.made")
    fga = numeric(root + "fieldGoals.attempted")
    threes = numeric(root + "threePointFieldGoals.made")
    fta = numeric(root + "freeThrows.attempted")
    tov = numeric("teamStats.turnovers.total" if is_team else "turnovers")
    orb = numeric(root + "rebounds.offensive")

    add_fallback("effectiveFieldGoalPct", "efg_pct",
                 lambda: 100 * divide(fgm + 0.5 * threes, fga))
    add_fallback("freeThrowRate", "ft_rate",
                 lambda: 100 * divide(fta, fga))

    def turnover_ratio():
        # Estimate possessions only when unavailable.
        estimate = fga + 0.44 * fta + tov
        if is_team:
            estimate = estimate - orb
        possessions = numeric(root + "possessions")
        possessions = possessions.where(possessions > 0).combine_first(estimate)
        scale = 100 if "gameId" in df.columns or not is_team else 1
        return scale * divide(tov, possessions)

    add_fallback("turnoverRatio", "turnover_ratio", turnover_ratio)
    def offensive_rebound_pct():
        if not is_team:
            # Player rebound context is unavailable.
            return pd.Series(float("nan"), index=df.index)
        if "gameId" in df.columns:
            # Observed CBBD game definition.
            return 100 * divide(orb, numeric("teamStats.rebounds.total"))
        return 100 * divide(orb, orb + numeric("opponentStats.rebounds.defensive"))

    add_fallback("offensiveReboundPct", "orb_pct", offensive_rebound_pct)
    return df


def main():
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)
    
    print("=" * 70)
    print(f"Starting Extraction for seasons: {list(SEASONS)}")
    print(f"Destination: {os.path.abspath(data_dir)}")
    print("=" * 70)

    for season in SEASONS:
        print(f"\n{'#' * 60}")
        print(f"--> Processing Season: {season}")
        print(f"{'#' * 60}")
        
        season_dir = os.path.join(data_dir, str(season))
        os.makedirs(season_dir, exist_ok=True)
        
        all_teams_season_data = []
        
        print(f"[API] Fetching tournament teams for {season}...")
        plays = safe_get("/plays/tournament", {"season": season, "tournament": "NCAA"})
        tournament_teams = set()
        if plays:
            for play in plays:
                if play.get("team"):
                    tournament_teams.add(play.get("team"))
                if play.get("opponent"):
                    tournament_teams.add(play.get("opponent"))

        raw_games = safe_get("/games", {"season": season, "tournament": "NCAA"})
        if not tournament_teams:
            print(f"[Fallback] /plays/tournament empty for {season}; using /games fallback.")
            team_list, source = choose_team_list(tournament_teams, raw_games)
            total_teams = len(team_list)
            print(f"[Fallback] Found {total_teams} teams from raw game data for {season}.")
        else:
            team_list, source = choose_team_list(tournament_teams, raw_games)
            total_teams = len(team_list)
            if source == "games":
                print(f"[Fallback] /plays/tournament looks partial for {season}; using full raw games list ({total_teams} teams).")
            else:
                print(f"[Success] Found {total_teams} tournament teams.")

        if not team_list:
            print(f"[Warning] No teams found from raw games fallback for {season}. Skipping season.")
            continue

        for idx, team in enumerate(team_list, 1):
            team_safe = sanitize(team)
            print(f"\n  [{idx}/{total_teams}] Team: {team} ({team_safe})")
            
            team_dir = os.path.join(season_dir, team_safe)
            os.makedirs(team_dir, exist_ok=True)

            # Team season stats and ratings.
            print(f"    Fetching season stats & ratings...")
            team_row = {"season": season, "team": team}
            t_stats = safe_get("/stats/team/season", {"season": season, "team": team})
            
            if t_stats and len(t_stats) > 0:
                df_team_season = pd.json_normalize(t_stats[0])
                
                # Adjusted efficiency.
                adj = safe_get("/ratings/adjusted", {"season": season, "team": team})
                if adj:
                    df_adj = pd.json_normalize(adj[0]).add_prefix("adj_")
                    df_team_season = pd.concat([df_team_season, df_adj], axis=1)

                # Fetch SRS rating
                srs = safe_get("/ratings/srs", {"season": season, "team": team})
                if srs:
                    df_team_season["srs_rating"] = srs[0].get("rating")

                # Fetch Elo rating
                elo = safe_get("/ratings/elo", {"season": season, "team": team})
                if elo:
                    df_team_season["elo_rating"] = elo[0].get("elo")

                df_team_season = calculate_missing_four_factors(df_team_season)
                team_row.update(df_team_season.iloc[0].to_dict())
                team_row.update(season=season, team=team)
                print(f"      [OK] Added {team} to season aggregate ledger.")
            else:
                print(f"      [--] No season stats found for {team}; preserving roster row with blank stats.")

            all_teams_season_data.append(team_row)

            # 2. Team Game Stats
            print(f"    Fetching regular season games...")
            t_games = safe_get("/games/teams", {"season": season, "team": team, "seasonType": "regular"})
            if t_games:
                df_games = pd.json_normalize(t_games)
                df_games = calculate_missing_four_factors(df_games)
                game_csv_path = os.path.join(team_dir, f"{team_safe}_game_stats.csv")
                df_games.to_csv(game_csv_path, index=False)
                print(f"      [Saved] {game_csv_path} ({len(df_games)} games)")
            else:
                print(f"      [--] No regular season game records found.")

            # Player season stats per team.
            print(f"    Fetching player rosters and season stats...")
            p_stats = safe_get("/stats/player/season", {"season": season, "team": team})
            if p_stats:
                df_players = pd.json_normalize(p_stats)
                df_players = calculate_missing_four_factors(df_players)
                player_csv_path = os.path.join(team_dir, f"{team_safe}_players_season_stats.csv")
                df_players.to_csv(player_csv_path, index=False)
                print(f"      [Saved] {player_csv_path} ({len(df_players)} players in single CSV)")
            else:
                print(f"      [--] No player records found.")

            time.sleep(0.3)

        # Save the season aggregate.
        print(f"\n[Aggregate] Saving all teams season stats for {season}...")
        if all_teams_season_data:
            df_master = pd.DataFrame(all_teams_season_data)
            master_path = os.path.join(season_dir, f"{season}_teams_season_stats.csv")
            df_master.to_csv(master_path, index=False)
            print(f"[Done] Master file written: {master_path} with {len(df_master)} teams.")
        else:
            print(f"[Warning] No data aggregated for season {season}.")

    print("\n" + "=" * 70)
    print("Database build complete.")
    print("=" * 70)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n[Halted] Script terminated by user. Processed files remain saved.")
