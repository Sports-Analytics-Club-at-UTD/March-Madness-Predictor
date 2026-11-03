"""Import the supplied historical CSV; no network calls or inferred game joins."""

import csv
import hashlib
from datetime import datetime, timezone
from pathlib import Path

from .models import MarketDataset

COMPETITION = "Men's college basketball"


def source_artifact(path, media_type):
    path = Path(path)
    return {"uri": str(path), "media_type": media_type,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def read_kalshi(path, season, team_ids, mapping=None, competition=COMPETITION):
    """Select dated records in [July 1 previous year, July 1 season year).

    Mapping format: {"team_aliases": {"vendor name": "canonical name"},
                     "event_games": {"event ticker": "internal game ID"}}.
    Unknown teams remain explicitly unmatched. Row numbers include the header.
    """
    mapping = mapping or {}
    if set(mapping) - {"team_aliases", "event_games"}:
        raise ValueError("Unknown Kalshi mapping sections")
    aliases, event_games = mapping.get("team_aliases", {}), mapping.get("event_games", {})
    if any(not isinstance(section, dict) or any(not isinstance(k, str) or not isinstance(v, str)
                                               for k, v in section.items())
           for section in (aliases, event_games)):
        raise ValueError("Kalshi mappings must be string-to-string objects")
    start = datetime(season - 1, 7, 1, tzinfo=timezone.utc)
    end = datetime(season, 7, 1, tzinfo=timezone.utc)
    snapshots, excluded, rows_read = [], [], 0
    base_columns = {"game", "competition", "scheduled_start_utc", "event_ticker", "game_status",
                    "game_id", "markets_for_same_game", "requested_snapshot_utc", "timing_basis",
                    "timing_source_url", "price_basis", "source_url", "exception_source_url"}
    side_columns = {"win_probability_pct", "probability_basis", "midpoint_probability_pct",
                    "bid_probability_pct", "ask_probability_pct", "last_trade_probability_pct",
                    "quote_asof_utc", "seconds_before_start", "data_status", "market_ticker"}
    required = base_columns | {f"team_{i}" for i in (1, 2)} | {
        f"team_{i}_{field}" for i in (1, 2) for field in side_columns}
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Kalshi CSV missing columns: {sorted(missing)}")
        for row_number, row in enumerate(reader, 2):
            rows_read += 1
            if None in row or any(v is None for v in row.values()):
                raise ValueError(f"Malformed Kalshi CSV at row {row_number}")
            reason = None
            if row["competition"] != competition:
                reason = "other_competition"
            elif not row["scheduled_start_utc"]:
                reason = "missing_start_time"
            else:
                try:
                    scheduled = datetime.fromisoformat(row["scheduled_start_utc"].replace("Z", "+00:00"))
                    if scheduled.utcoffset() is None:
                        raise ValueError("timestamp must include a timezone")
                except ValueError as error:
                    raise ValueError(f"Invalid scheduled start at CSV row {row_number}: {error}") from error
                if not start <= scheduled < end:
                    reason = "other_season"
            if reason:
                excluded.append({"row_number": row_number, "event_ticker": row["event_ticker"] or None,
                                 "reason": reason})
                continue
            snapshot = {k: row[k] or None for k in base_columns - {"game", "game_id", "markets_for_same_game"}}
            snapshot.update({
                "id": "kalshi-" + hashlib.sha256(
                    (row["event_ticker"] + "|" + row["requested_snapshot_utc"]).encode()
                ).hexdigest()[:24],
                "title": row["game"] or None,
                "provider_game_id": row["game_id"],
                "game_id": event_games.get(row["event_ticker"]),
                "markets_for_same_game": int(row["markets_for_same_game"]) if row["markets_for_same_game"] else None,
                "outcomes": [],
                "source_fields": {k: v or None for k, v in row.items() if k not in required},
            })
            for i in (1, 2):
                prefix = f"team_{i}"
                name = row[prefix]
                canonical = aliases.get(name, name)
                internal_id = team_ids.get(canonical)
                outcome = {
                    "team_name": name, "team_id": internal_id,
                    "team_mapping": ("alias" if name in aliases else "exact_name") if internal_id else "unmatched",
                    "market_ticker": row[f"{prefix}_market_ticker"],
                    "probability_basis": row[f"{prefix}_probability_basis"],
                    "quote_asof_utc": row[f"{prefix}_quote_asof_utc"] or None,
                    "seconds_before_start": float(row[f"{prefix}_seconds_before_start"]) if row[f"{prefix}_seconds_before_start"] else None,
                    "data_status": row[f"{prefix}_data_status"],
                }
                for kind in ("win", "midpoint", "bid", "ask", "last_trade"):
                    raw = row[f"{prefix}_{kind}_probability_pct"]
                    outcome[f"{kind}_probability"] = float(raw) / 100 if raw else None
                snapshot["outcomes"].append(outcome)
            snapshots.append(snapshot)
    return MarketDataset.model_validate({
        "source": source_artifact(path, "text/csv"), "imported_at": datetime.now(timezone.utc),
        "season_start_utc": start, "season_end_exclusive_utc": end,
        "import_summary": {"rows_read": rows_read, "included": len(snapshots), "excluded": excluded,
                           "unmatched_outcomes": sum(o["team_id"] is None for s in snapshots for o in s["outcomes"])},
        "snapshots": snapshots,
    })
