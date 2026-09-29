import os
import time
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("CBBD_API_KEY")
BASE_URL = "https://api.collegebasketballdata.com"
OUT_DIR = Path("data/2000_2008")
SEASONS = range(2000, 2009)  # 2000 through 2008
MAX_ATTEMPTS = 4

if not API_KEY:
    raise SystemExit("CBBD_API_KEY is missing. Set it in .env.")

SEEN_COLUMNS = {}


def get_json(path, params):
    url = f"{BASE_URL}{path}"
    headers = {"Authorization": f"Bearer {API_KEY}"}
    season = params.get("season")

    for attempt in range(MAX_ATTEMPTS):
        try:
            response = requests.get(
                url,
                params=params,
                headers=headers,
                timeout=60,
            )
        except requests.Timeout:
            if attempt == MAX_ATTEMPTS - 1:
                print(f"Failed {path} season {season}: timeout")
                return []
            time.sleep(2**attempt)
            continue
        except requests.RequestException as error:
            print(f"Failed {path} season {season}: {error}")
            return []

        status = response.status_code

        if status == 404 or not response.content or not response.text.strip():
            return []

        if status == 429 or status >= 500:
            if attempt == MAX_ATTEMPTS - 1:
                print(f"Failed {path} season {season}: {status}")
                return []
            retry_after = response.headers.get("Retry-After")
            try:
                wait = float(retry_after) if retry_after else 2**attempt
            except ValueError:
                wait = 2**attempt
            time.sleep(wait)
            continue

        if status != 200:
            print(f"Failed {path} season {season}: {status}")
            return []

        try:
            payload = response.json()
        except ValueError:
            print(f"Failed {path} season {season}: invalid JSON")
            return []

        if isinstance(payload, list):
            return payload

        print(f"Failed {path} season {season}: expected a list")
        return []

    return []


def save_table(slug, season, rows):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"{slug}_{season}.csv"

    if rows:
        frame = pd.DataFrame(rows)
        if "id" in frame.columns:
            frame = frame.drop_duplicates(subset=["id"])
        SEEN_COLUMNS[slug] = list(frame.columns)
    else:
        columns = SEEN_COLUMNS.get(slug, ["season"])
        frame = pd.DataFrame(columns=columns)
        print(f"Empty response: {slug} {season}")

    frame.to_csv(path, index=False)
    print(f"{slug} {season}: {len(frame)} rows -> {path}")
    return path


def validate():
    expected = set(SEASONS)
    files = sorted(OUT_DIR.glob("*.csv"))
    if not files:
        print(f"No CSV files in {OUT_DIR}")
        return

    by_slug = {}
    for path in files:
        slug, _, season_text = path.stem.rpartition("_")
        try:
            season = int(season_text)
        except ValueError:
            print(f"Unexpected filename: {path.name}")
            continue
        by_slug.setdefault(slug, {})[season] = path

    for slug, season_paths in by_slug.items():
        found = set(season_paths)
        missing = sorted(expected - found)
        extra = sorted(found - expected)
        if missing:
            print(f"Missing {slug}: {missing}")
        if extra:
            print(f"Unexpected seasons in {slug}: {extra}")

        for season in sorted(found):
            path = season_paths[season]
            try:
                frame = pd.read_csv(path)
            except pd.errors.EmptyDataError as error:
                print(f"Failed to load {path.name}: {error}")
                continue

            print(f"{path.name}: {len(frame)} rows")
            if len(frame) == 0:
                print(f"Empty file: {path.name}")

            if "season" in frame.columns and len(frame):
                present = set(frame["season"].dropna().astype(int).unique())
                outside = sorted(present - expected)
                if outside:
                    print(f"Seasons outside 2000-2008 in {path.name}: {outside}")

            if "id" not in frame.columns:
                print(f"{path.name}: no id column")
                continue

            duplicates = int(frame["id"].duplicated().sum())
            print(f"{path.name}: {duplicates} duplicate ids")


def main():
    for season in SEASONS:
        rows = get_json("/games", {"season": season, "tournament": "NCAA"})
        save_table("games", season, rows)
        time.sleep(1)

    validate()


if __name__ == "__main__":
    main()
