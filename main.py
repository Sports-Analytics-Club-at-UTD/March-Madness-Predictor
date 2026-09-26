import os
import requests

from dotenv import load_dotenv



load_dotenv()

API_KEY = os.getenv("CBBD_API_KEY")

url = "https://api.collegebasketballdata.com/games"

headers = {
    "Authorization": f"Bearer {API_KEY}"
}

for season in range(2026, 1979, -1):
    response = requests.get(
        url,
        params={
            "season": season,
            "team": "Duke"
        },
        headers=headers,
        timeout=30
    )

    response.raise_for_status()
    games = response.json()

    if games:
        print(f"{season}: {len(games)} games")
    else:
        print(f"{season}: no data")
 