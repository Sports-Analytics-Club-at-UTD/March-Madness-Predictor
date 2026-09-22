import os
import requests
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("CBBD_API_KEY")
BASE_URL = "https://api.collegebasketballdata.com"

HEADERS = {
    "Authorization": f"Bearer {API_KEY}"
}


def fetch_ncaa_tournament_plays(season):
    url = f"{BASE_URL}/plays/tournament"

    params = {
        "season": season,
        "tournament": "NCAA",
    }

    response = requests.get(
        url,
        params=params,
        headers=HEADERS,
        timeout=60,
    )

    response.raise_for_status()

    return response.json()


def get_tournament_teams(season):
    plays = fetch_ncaa_tournament_plays(season)

    teams = set()

    for play in plays:
        team = play.get("team")
        opponent = play.get("opponent")

        if team:
            teams.add(team)

        if opponent:
            teams.add(opponent)

    return teams


def main():
    # 2000-2026, excluding canceled 2020 tournament
    seasons = [
        year
        for year in range(2000, 2027)
        if year != 2020
    ]

    appearances = {}

    for season in seasons:
        print(f"Fetching {season} NCAA tournament...")

        try:
            teams = get_tournament_teams(season)
        except requests.HTTPError as e:
            print(f"Failed {season}: {e}")
            continue

        print(f"  Found {len(teams)} teams")

        for team in teams:
            appearances.setdefault(team, [])
            appearances[team].append(season)

    rows = []

    for team, years in appearances.items():
        rows.append({
            "team": team,
            "appearances": len(years),
            "first_appearance": min(years),
            "last_appearance": max(years),
            "seasons": ",".join(map(str, years)),
        })

    df = pd.DataFrame(rows)

    df.sort_values(
        ["appearances", "team"],
        ascending=[False, True],
        inplace=True,
    )

    df.to_csv(
        "march_madness_teams_2000_2026.csv",
        index=False,
    )

    print()
    print(f"Unique tournament teams: {len(df)}")
    print()
    print(df.head(30))


if __name__ == "__main__":
    main()