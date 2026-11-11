"""Acquire whole-season CBBD schedules into a separate, validated cache root."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import os
from pathlib import Path
import tempfile

import requests
from dotenv import load_dotenv

from bayesian_evaluation import digest, load_config, write_csv, write_json
from bayesian_ratings import load_games


def acquire_season(spec, output_dir):
    """One league-wide request, no entrant filter; publish only a validated season."""
    output_dir = Path(output_dir)
    destination = output_dir / str(spec.season)
    if destination.exists():
        raise FileExistsError(f"Refusing to replace {destination}")
    load_dotenv()
    key = os.getenv("CBBD_API_KEY")
    if not key:
        raise ValueError("Schedule acquisition requires CBBD_API_KEY in the environment or .env")
    params = {"season": spec.season, "seasonType": "regular"}
    response = requests.get("https://api.collegebasketballdata.com/games", params=params,
                            headers={"Authorization": f"Bearer {key}"}, timeout=45)
    response.raise_for_status()
    raw = response.json()
    if not isinstance(raw, list) or not raw or any(not isinstance(r, dict) for r in raw):
        raise ValueError("Expected a nonempty list of season games")
    rows = []
    for r in raw:
        if r["season"] != spec.season or str(r["seasonType"]).lower() != "regular":
            raise ValueError("API returned games outside the requested regular season")
        rows.append(dict(gameId=r.get("id", r.get("gameId")), season=r["season"],
                         seasonType=r["seasonType"], startDate=r["startDate"],
                         teamId=r.get("homeTeamId", r.get("homeId")), team=r["homeTeam"],
                         opponentId=r.get("awayTeamId", r.get("awayId")), opponent=r["awayTeam"],
                         neutralSite=r["neutralSite"], isHome=True, status=r.get("status", ""),
                         tournament=r.get("tournament") or "",
                         **{"teamStats.points.total": r["homePoints"],
                            "opponentStats.points.total": r["awayPoints"]}))
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".coverage-", dir=output_dir) as temporary:
        root = Path(temporary)
        season_dir = root / str(spec.season)
        cache = season_dir / "league" / "league_game_stats.csv"
        cache.parent.mkdir(parents=True)
        write_csv(cache, rows)
        dataset = load_games(root, spec.season, spec.cutoff)
        counts = Counter(t for g in dataset.games for t in (g.team_a, g.team_b))
        write_json(season_dir / "coverage.source.json", {
            "endpoint": "https://api.collegebasketballdata.com/games", "params": params,
            "retrieved_at": datetime.now(timezone.utc).isoformat(),
            "cutoff": spec.cutoff.isoformat(), "raw_rows": len(raw),
            "cache_sha256": digest(cache), "loader": dataset.report,
            "team_game_counts": dict(sorted(counts.items())),
            "limitations": "League-wide request, not verified complete schedules or Division I membership. Historical corrections and completion times are not independently archived.",
        })
        # Save the original response for diagnosing truncation or schema changes.
        write_json(season_dir / "games.response.json", raw)
        season_dir.rename(destination)
    return {"season": spec.season, "games": len(dataset.games), "teams": len(counts),
            "below_10_games": sum(n < 10 for n in counts.values())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Separate schedule root; existing seasons are never overwritten")
    args = parser.parse_args()
    specs, targets = load_config(args.config)
    for spec in specs:
        if spec.season in targets:
            print(acquire_season(spec, args.output_dir))


if __name__ == "__main__":
    main()
