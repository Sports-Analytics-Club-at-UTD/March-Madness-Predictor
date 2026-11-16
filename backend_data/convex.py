"""Convert saved forecasts and import them through the site's authenticated Convex CLI.

Run `python -m backend_data.convex --help`. No training or API server is involved.
"""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from pydantic import Field

from .models import ContractModel, ForecastDocument, Identifier, Text


class SiteTeam(ContractModel):
    id: Identifier
    name: Text
    seed: int = Field(ge=1, le=16)
    logoUrl: Text


class SiteRoster(ContractModel):
    season: int
    teams: list[SiteTeam]
    logoStorageId: Text


def convert_forecast(document, roster, aliases=None):
    """Produce existing site records, resolving participants separately per strategy.

    Aliases map forecast team IDs (or exact names) to canonical site IDs. Otherwise
    only an unambiguous, case-insensitive exact name match is accepted.
    """
    # Revalidate even when callers pass previously constructed model instances.
    document = ForecastDocument.model_validate(document.model_dump(mode="json"))
    roster = SiteRoster.model_validate(roster)
    if document.tournament.competition != "Men's college basketball":
        raise ValueError("The demo currently supports men's college basketball only")
    if document.tournament.season != roster.season:
        raise ValueError("Forecast season does not match the site roster")
    if not document.games or not document.brackets:
        raise ValueError("Forecast needs topology and complete bracket picks; Ridge team scores alone cannot populate the demo")
    if document.tournament.includes_first_four:
        raise ValueError("Import the main bracket without First Four games for this importer")
    aliases = aliases or {}
    if not isinstance(aliases, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in aliases.items()):
        raise ValueError("Aliases must map forecast team IDs or names to site team IDs")
    canonical = {team.id: team for team in roster.teams}
    if len(canonical) != len(roster.teams):
        raise ValueError("Site roster contains duplicate team IDs")
    entrants = {p.team_id for game in document.games for p in game.participants if p.team_id}
    mapped, used = {}, set()
    for team in document.teams:
        if team.id not in entrants:
            continue  # e.g. First Four losers that have team statistics only
        target = aliases.get(team.id, aliases.get(team.name))
        if target is None:
            matches = [t.id for t in roster.teams if t.name.strip().casefold() == team.name.strip().casefold()]
            if len(matches) != 1:
                raise ValueError(f"Provide an alias for {team.name!r} ({team.id}); no unique exact site match")
            target = matches[0]
        if target not in canonical:
            raise ValueError(f"Unknown site team ID {target!r} for {team.name!r}")
        if target in used:
            raise ValueError(f"Multiple forecast teams map to site team {target!r}")
        mapped[team.id], used = target, used | {target}

    games = {game.id: game for game in document.games}
    sources = {p.source_game_id for game in document.games for p in game.participants if p.source_game_id}
    final = next(game for game in document.games if game.id not in sources)
    if not 2 <= final.round_number <= 6 or len(entrants) != 2 ** final.round_number:
        raise ValueError("Demo imports require a balanced main bracket of 4–64 teams")
    order, regions, counts = {}, {}, {}

    def visit(game, region=None):
        # Depth-first traversal preserves bracket order even if input slots are shuffled.
        order[game.id] = counts.get(game.round_number, 0)
        counts[game.round_number] = order[game.id] + 1
        regions[game.id] = region
        for participant in game.participants:
            if game.round_number == 1:
                if participant.team_id is None:
                    raise ValueError("Main-bracket round 1 must have direct team entrants")
            else:
                source = games.get(participant.source_game_id)
                if source is None or source.round_number != game.round_number - 1:
                    raise ValueError("Demo layout requires consecutive rounds and two feeder games")
                next_region = region
                if source.round_number == final.round_number - 2:
                    next_region = f"Region {counts.get(source.round_number, 0) + 1}"
                visit(source, next_region)

    visit(final)
    runs = {run.id: run for run in document.prediction_runs}
    items = []
    for bracket in document.brackets:
        identity = json.dumps([document.tournament.id, bracket.run_id, bracket.id], separators=(",", ":"))
        slug = f"forecast-{roster.season}-{hashlib.sha256(identity.encode()).hexdigest()}"
        picks = {pick.game_id: pick for pick in bracket.picks}
        converted = []
        for game in sorted(document.games, key=lambda g: (g.round_number, order[g.id])):
            participants = [mapped[p.team_id or picks[p.source_game_id].winner_team_id] for p in game.participants]
            label = {1: "First round", 2: "Second round", 3: "Sweet 16", 4: "Elite Eight"}.get(game.round_number, f"Round {game.round_number}")
            if game.round_number == final.round_number:
                label = "National championship"
            elif game.round_number == final.round_number - 1:
                label = "Final Four"
            if regions[game.id]:
                label = f"{regions[game.id]} · {label}"
            record = {
                "id": game.id, "round": game.round_number, "order": order[game.id], "label": label,
                "topTeamId": participants[0], "bottomTeamId": participants[1],
                "winnerTeamId": mapped[picks[game.id].winner_team_id],
            }
            for key, participant in zip(("topSourceGameId", "bottomSourceGameId"), game.participants):
                if participant.source_game_id:
                    record[key] = participant.source_game_id
            converted.append(record)
        run = runs[bracket.run_id]
        strategy = {"max_ev": "Max expected points", "leverage": "Leverage"}.get(bracket.strategy, bracket.strategy)
        title = f"{roster.season} {strategy} · {run.model_name}"
        items.append({
            "tournament": {
                "slug": slug, "name": title, "season": roster.season,
                "winnerTeamId": mapped[bracket.summary.champion_team_id], "sourceUrls": [],
                "teams": [canonical[team_id].model_dump() for team_id in sorted(used)], "games": converted,
            },
            "title": title,
            "description": f"{run.method} · {bracket.strategy} · run {run.id} · generated {run.generated_at.isoformat()}",
            "logoStorageId": roster.logoStorageId,
        })
    return {"items": items}


def run_convex(site_dir, function, arguments, production=False):
    """Use the site's installed CLI and its existing deployment authentication."""
    cli = site_dir / "node_modules" / "convex" / "bin" / "main.js"
    if not cli.is_file():
        raise ValueError("Install the site dependencies first; node_modules/convex/bin/main.js is missing")
    if not (site_dir / "convex" / "forecastImports.ts").is_file():
        raise ValueError("Copy integrations/convex/forecastImports.ts into the site's convex/ directory and sync it first")
    encoded = json.dumps(arguments, separators=(",", ":"), allow_nan=False)
    if len(encoded.encode()) > 100_000:
        raise ValueError("Import exceeds the CLI argument limit; split the forecast into smaller run bundles")
    command = ["node", str(cli), "run", function, encoded]
    if production:
        command.append("--prod")
    result = subprocess.run(command, cwd=site_dir, text=True, capture_output=True, check=False)
    if result.returncode:
        raise ValueError(f"Convex {function} failed: {result.stderr.strip() or result.stdout.strip()}")
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise ValueError(f"Convex {function} returned unexpected output; verify the site CLI setup") from error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("forecast", type=Path)
    parser.add_argument("--site-dir", type=Path, help="Checkout of the site branch, with Convex configured")
    parser.add_argument("--roster", type=Path, help="Saved site roster for a fully offline dry run")
    parser.add_argument("--aliases", type=Path, help="JSON object mapping forecast team IDs/names to site IDs")
    parser.add_argument("--output", type=Path, help="Save converted mutation arguments for review")
    parser.add_argument("--dry-run", action="store_true", help="Validate and convert without database writes")
    parser.add_argument("--prod", action="store_true", help="Explicitly use the site's production deployment")
    args = parser.parse_args()
    if not args.site_dir and not (args.dry_run and args.roster):
        parser.error("--site-dir is required unless using --dry-run with --roster")
    if args.roster and not args.dry_run:
        parser.error("--roster is for offline dry runs; live imports always read the target deployment's roster")
    try:
        document = ForecastDocument.model_validate_json(args.forecast.read_text(encoding="utf-8"))
        # Fail before making even a read request when the forecast cannot display a bracket.
        if not document.games or not document.brackets:
            raise ValueError("Forecast needs topology and complete bracket picks; Ridge team scores alone cannot populate the demo")
        roster = json.loads(args.roster.read_text(encoding="utf-8")) if args.roster else run_convex(
            args.site_dir.resolve(), "forecastImports:getRoster", {"season": document.tournament.season}, args.prod)
        aliases = json.loads(args.aliases.read_text(encoding="utf-8")) if args.aliases else None
        payload = convert_forecast(document, roster, aliases)
        if args.output:
            args.output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        if args.dry_run:
            print(f"Dry run: {len(payload['items'])} brackets validated; no database writes.")
        else:
            print(json.dumps(run_convex(args.site_dir.resolve(), "forecastImports:importForecast", payload, args.prod), indent=2))
    except (ValueError, TypeError, KeyError, OSError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
