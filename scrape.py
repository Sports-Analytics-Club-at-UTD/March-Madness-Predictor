import os
import time
import requests
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("CBBD_API_KEY")
BASE_URL = "https://api.collegebasketballdata.com"
HEADERS = {"Authorization": f"Bearer {API_KEY}"}

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

def calculate_missing_four_factors(df, prefix=""):
    if df is None or df.empty:
        return df

    def get_col(suffix_options):
        for col in df.columns:
            clean = col.lower().split(".")[-1]
            if clean in suffix_options:
                return pd.to_numeric(df[col], errors='coerce')
        return None

    fga = get_col(["attempted", "fga"])
    fgm = get_col(["made", "fgm"])
    fg3m = get_col(["made", "3pm"])
    fta = get_col(["attempted", "fta"])
    tov = get_col(["total", "turnovers", "tov"])

    calc_count = 0
    if fga is not None and fgm is not None and fg3m is not None:
        if f"{prefix}calc_efg_pct" not in df.columns:
            df[f"{prefix}calc_efg_pct"] = (fgm + 0.5 * fg3m) / fga
            calc_count += 1

    if tov is not None and fga is not None and fta is not None:
        if f"{prefix}calc_tov_pct" not in df.columns:
            df[f"{prefix}calc_tov_pct"] = tov / (fga + 0.44 * fta + tov)
            calc_count += 1

    if fta is not None and fga is not None:
        if f"{prefix}calc_ft_rate" not in df.columns:
            df[f"{prefix}calc_ft_rate"] = fta / fga
            calc_count += 1

    if calc_count > 0:
        print(f"      [Calculation] Derived {calc_count} missing Four Factors metrics.")

    return df

def main():
    seasons = range(2009, 2018)
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)
    
    print("=" * 70)
    print(f"Starting Extraction for seasons: {list(seasons)}")
    print(f"Destination: {os.path.abspath(data_dir)}")
    print("=" * 70)

    for season in seasons:
        print(f"\n{'#' * 60}")
        print(f"--> Processing Season: {season}")
        print(f"{'#' * 60}")
        
        season_dir = os.path.join(data_dir, str(season))
        os.makedirs(season_dir, exist_ok=True)
        
        all_teams_season_data = []
        
        print(f"[API] Fetching tournament teams for {season}...")
        plays = safe_get("/plays/tournament", {"season": season, "tournament": "NCAA"})
        
        if not plays:
            print(f"[Warning] No tournament plays found for {season}. Skipping season.")
            continue
            
        teams = set()
        for play in plays:
            if play.get("team"):
                teams.add(play.get("team"))
            if play.get("opponent"):
                teams.add(play.get("opponent"))
            
        team_list = sorted(list(teams))
        total_teams = len(team_list)
        print(f"[Success] Found {total_teams} tournament teams.")

        for idx, team in enumerate(team_list, 1):
            team_safe = sanitize(team)
            print(f"\n  [{idx}/{total_teams}] Team: {team} ({team_safe})")
            
            team_dir = os.path.join(season_dir, team_safe)
            os.makedirs(team_dir, exist_ok=True)

            # 1. Collect Team Season Stats & Ratings into master aggregation list
            print(f"    Fetching season stats & ratings...")
            t_stats = safe_get("/stats/team/season", {"season": season, "team": team})
            
            if t_stats and len(t_stats) > 0:
                df_team_season = pd.json_normalize(t_stats[0])
                
                # Fetch Adjusted Efficiency ratings
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
                all_teams_season_data.append(df_team_season.iloc[0].to_dict())
                print(f"      [OK] Added {team} to season aggregate ledger.")
            else:
                print(f"      [--] No season stats found for {team}.")

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

            # 3. All Players into One Combined CSV per Team
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

        # 4. Save Master Season CSV containing all teams for the year
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