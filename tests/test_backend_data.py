import contextlib
import csv
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from jsonschema import Draft202012Validator, FormatChecker
from pydantic import ValidationError

from backend_data.__main__ import json_schema, main, merge_documents
from backend_data.export import build_forecast, normalize, save_forecast, stable_team_id
from backend_data.kalshi import read_kalshi
from backend_data.models import ForecastDocument


def bracket_fixture():
    """Synthetic 64-team topology and two internally consistent strategies."""
    teams = [f"Example Team {i:02}" for i in range(64)]
    pairs = [(i, i + 1) for i in range(0, 64, 2)]
    rounds, edges, previous = [1] * 32, [], list(range(32))
    for round_number in range(2, 7):
        current = []
        for i in range(0, len(previous), 2):
            current.append(32 + len(edges))
            edges.append((previous[i], previous[i + 1]))
            rounds.append(round_number)
        previous = current
    ev, leverage, rows = [], [], []
    for i in range(63):
        ev.append(teams[pairs[i][0]] if i < 32 else ev[edges[i - 32][0]])
        leverage.append(teams[pairs[i][1]] if i < 32 else leverage[edges[i - 32][1]])
        points = 10 * 2 ** (rounds[i] - 1)
        row = {"game_slot": i + 1, "round_number": rounds[i], "round_points": points,
               "leverage_ratio": 1.2}
        for strategy, winner in (("max_ev", ev[-1]), ("leverage", leverage[-1])):
            row.update({f"{strategy}_pick": winner, f"{strategy}_win_prob": .4,
                        f"{strategy}_public_pct": .3, f"{strategy}_expected_pts": points * .4})
        rows.append(row)
    table = pd.DataFrame({"team": teams, "season": 2026, "ensemble_score": 1.,
                          "simulated_mean": 1.1, "percentile_5": -.2, "percentile_95": 2.4,
                          "probability_ge_1_win": .6, "probability_ge_2_wins": .3})
    tree = {"teams64": teams, "round1_pairs": pairs, "tree_edges": edges,
            "slot_rounds": rounds, "actual_slot_winners": [-1] * 63}
    return table, pd.DataFrame(rows), tree


def market_row(**updates):
    row = {
        "game": "Example matchup", "competition": "Men's college basketball",
        "scheduled_start_utc": "2025-11-03T16:00:00Z", "event_ticker": "EVENT-1",
        "game_id": "PROVIDER-1", "game_status": "settled", "markets_for_same_game": "1",
        "requested_snapshot_utc": "2025-11-03T15:59:00Z", "timing_basis": "scheduled start",
        "timing_source_url": "https://example.test/timing", "price_basis": "midpoint or last trade",
        "source_url": "https://example.test/market", "exception_source_url": "",
    }
    for i, name in ((1, "Example Team 00"), (2, "Vendor Team 01")):
        row[f"team_{i}"] = name
        row.update({f"team_{i}_{k}": v for k, v in {
            "win_probability_pct": "50" if i == 1 else "51", "probability_basis": "bid_ask_midpoint",
            "midpoint_probability_pct": "50" if i == 1 else "51", "bid_probability_pct": "0",
            "ask_probability_pct": "", "last_trade_probability_pct": "",
            "quote_asof_utc": "2025-11-03T15:59:00Z" if i == 1 else "2025-11-03T15:55:00Z",
            "seconds_before_start": "60" if i == 1 else "300",
            "data_status": "ok" if i == 1 else "older_than_requested_minute",
            "market_ticker": f"MARKET-{i}",
        }.items()})
    row.update(updates)
    return row


class BackendDataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.metadata = {"season": 2026, "seed": 42, "n_simulations_per_team": 100,
                         "example_only": True}
        self.table, self.brackets, self.tree = bracket_fixture()

    def forecast(self, **kwargs):
        return build_forecast(self.table, self.metadata, "shap", self.brackets, self.tree, **kwargs)

    def write_markets(self, rows):
        path = self.root / "markets.csv"
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        return path

    def test_strategies_topology_and_confidence(self):
        self.tree["scheduled_starts_utc"] = ["2026-03-19T16:00:00Z"] * 63
        document = self.forecast()
        self.assertEqual(len(document.games), 63)
        self.assertEqual(document.tournament.bracket_team_count, 64)
        self.assertFalse(document.tournament.includes_first_four)
        self.assertEqual([b.strategy for b in document.brackets], ["max_ev", "leverage"])
        self.assertNotEqual(document.brackets[0].summary.champion_team_id,
                            document.brackets[1].summary.champion_team_id)
        self.assertEqual(document.games[32].participants[0].source_game_id, "ncaa-men-2026-g1")
        self.assertEqual(document.brackets[0].picks[0].confidence.meaning, "probability_team_wins_slot")
        self.assertIsNone(document.brackets[0].picks[0].actual_correct)
        self.assertEqual(document.team_predictions[0].intervals[0].kind, "empirical_prediction")
        self.assertIsNone(document.team_predictions[0].posterior)
        self.assertEqual(document.games[0].scheduled_start_utc.year, 2026)

    def test_invalid_probability_and_monotonicity(self):
        for changes in ({"probability_ge_1_win": 1.1},
                        {"probability_ge_2_wins": .8}, {"simulated_mean": float("inf")}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                build_forecast(self.table.assign(**changes), self.metadata, "shap")
        self.brackets.loc[0, "max_ev_win_prob"] = -.1
        with self.assertRaises(ValidationError):
            self.forecast()

    def test_missing_values_round_trip_and_unique_storage(self):
        table = self.table.iloc[:1].assign(actual_wins=np.nan)
        document = build_forecast(table, self.metadata, "ridge")
        self.assertEqual(document.brackets, [])
        self.assertIsNone(document.market_data)
        self.assertIsNone(document.team_predictions[0].statistics["actual_wins"])
        first, second = save_forecast(document, self.root), save_forecast(document, self.root)
        self.assertNotEqual(first, second)
        self.assertEqual(ForecastDocument.model_validate_json(first.read_text()), document)
        self.assertEqual(len(list(self.root.iterdir())), 2)
        with self.assertRaises(TypeError):
            normalize({"unsupported": object()})

    def test_topology_and_pick_consistency_failures(self):
        payload = self.forecast().model_dump(mode="json")
        bad = json.loads(json.dumps(payload))
        bad["brackets"][0]["picks"][-1]["winner_team_id"] = stable_team_id("Example Team 63")
        with self.assertRaisesRegex(ValidationError, "Inconsistent pick"):
            ForecastDocument.model_validate(bad)
        bad = json.loads(json.dumps(payload))
        bad["games"][32]["participants"][0]["source_game_id"] = "ncaa-men-2026-g63"
        with self.assertRaisesRegex(ValidationError, "future, cyclic"):
            ForecastDocument.model_validate(bad)
        bad = json.loads(json.dumps(payload))
        bad["brackets"][0]["picks"].pop()
        with self.assertRaisesRegex(ValidationError, "each game"):
            ForecastDocument.model_validate(bad)
        bad = json.loads(json.dumps(payload))
        bad["team_predictions"][0]["run_id"] = "missing"
        with self.assertRaisesRegex(ValidationError, "unknown run"):
            ForecastDocument.model_validate(bad)

    def test_invalid_input_rosters_and_seasons(self):
        for table in (self.table.assign(season=2025), self.table.assign(team="same"),
                      self.table.assign(team=None), self.table.iloc[:0]):
            with self.subTest(), self.assertRaises(ValueError):
                build_forecast(table, self.metadata, "ridge")
        with self.assertRaisesRegex(ValueError, "topology"):
            build_forecast(self.table, self.metadata, "shap", self.brackets)

    def test_market_selection_probabilities_and_aliases(self):
        path = self.write_markets([
            market_row(), market_row(competition="Women's college basketball"),
            market_row(scheduled_start_utc=""),
            market_row(scheduled_start_utc="2024-11-03T16:00:00Z"),
        ])
        document = self.forecast(kalshi_csv=path, kalshi_mapping={
            "team_aliases": {"Vendor Team 01": "Example Team 01"},
            "event_games": {"EVENT-1": "ncaa-men-2026-g1"},
        })
        markets = document.market_data
        self.assertEqual(markets.import_summary.rows_read, 4)
        self.assertEqual(markets.import_summary.included, 1)
        self.assertEqual([r.reason for r in markets.import_summary.excluded],
                         ["other_competition", "missing_start_time", "other_season"])
        first, second = markets.snapshots[0].outcomes
        self.assertEqual((first.win_probability, second.win_probability), (.5, .51))
        self.assertEqual(first.bid_probability, 0)
        self.assertIsNone(first.ask_probability)
        self.assertEqual(second.team_mapping, "alias")
        self.assertEqual(second.data_status, "older_than_requested_minute")
        self.assertNotEqual(first.quote_asof_utc, second.quote_asof_utc)
        self.assertEqual(markets.snapshots[0].game_id, "ncaa-men-2026-g1")
        self.assertEqual(len(markets.source.sha256), 64)

    def test_unmatched_missing_and_bad_market_records(self):
        ids = {"Example Team 00": stable_team_id("Example Team 00")}
        path = self.write_markets([market_row(team_1_win_probability_pct="")])
        data = read_kalshi(path, 2026, ids)
        self.assertIsNone(data.snapshots[0].outcomes[0].win_probability)
        self.assertIsNone(data.snapshots[0].outcomes[1].team_id)
        self.assertEqual(data.import_summary.unmatched_outcomes, 1)
        self.assertIsNone(data.snapshots[0].game_id)
        for updates in ({"team_1_win_probability_pct": "101"},
                        {"team_1_win_probability_pct": "NaN"},
                        {"team_1_quote_asof_utc": "2025-11-03T16:01:00Z"},
                        {"team_1_seconds_before_start": "999"},
                        {"scheduled_start_utc": "2025-11-03T16:00:00"}):
            with self.subTest(updates=updates), self.assertRaises(ValueError):
                read_kalshi(self.write_markets([market_row(**updates)]), 2026, ids)

    def test_market_game_mapping_rejects_same_branch_and_wrong_season(self):
        path = self.write_markets([market_row()])
        mapping = {"team_aliases": {"Vendor Team 01": "Example Team 01"},
                   "event_games": {"EVENT-1": "ncaa-men-2026-g63"}}
        with self.assertRaisesRegex(ValidationError, "cannot meet"):
            self.forecast(kalshi_csv=path, kalshi_mapping=mapping)
        document = self.forecast(kalshi_csv=path).model_dump(mode="json")
        document["market_data"]["season_start_utc"] = "2024-07-01T00:00:00Z"
        with self.assertRaisesRegex(ValidationError, "season window differs"):
            ForecastDocument.model_validate(document)

    def test_schema_is_current_and_accepts_exports(self):
        schema = json_schema()
        Draft202012Validator.check_schema(schema)
        saved = json.loads((Path(__file__).resolve().parents[1] / "schemas/forecast.schema.json").read_text())
        self.assertEqual(saved, schema)
        validator = Draft202012Validator(schema, format_checker=FormatChecker())
        validator.validate(self.forecast().model_dump(mode="json"))
        validator.validate(self.forecast(kalshi_csv=self.write_markets([market_row()])).model_dump(mode="json"))
        example_path = Path(__file__).resolve().parents[1] / "examples/forecast.example.json"
        example = ForecastDocument.model_validate_json(example_path.read_text())
        validator.validate(example.model_dump(mode="json"))

    def test_multiple_runs_merge_and_future_bayesian_summary(self):
        first, second = self.forecast(), build_forecast(self.table, self.metadata, "ridge")
        merged = merge_documents([first, second])
        self.assertEqual(len(merged.prediction_runs), 2)
        self.assertEqual(len(merged.team_predictions), 128)
        with self.assertRaisesRegex(ValidationError, "Duplicate run"):
            merge_documents([first, first])
        payload = second.model_dump(mode="json")
        payload["prediction_runs"][0]["method"] = "bayesian_example_only"
        payload["team_predictions"][0]["posterior"] = {
            "target": "tournament_success_score", "mean": 1.2,
            "intervals": [{"kind": "credible", "target": "tournament_success_score",
                           "level": .9, "lower": .2, "upper": 2.4, "method": "example_posterior"}],
        }
        self.assertEqual(ForecastDocument.model_validate(payload).team_predictions[0].posterior.mean, 1.2)
        payload["team_predictions"][0]["posterior"]["intervals"][0]["kind"] = "empirical_prediction"
        with self.assertRaisesRegex(ValidationError, "credible"):
            ForecastDocument.model_validate(payload)

    def test_export_and_validation_commands(self):
        self.table.to_csv(self.root / "teams.csv", index=False)
        (self.root / "metadata.json").write_text(json.dumps(self.metadata))
        args = ["backend_data", "export", "--team-scores", str(self.root / "teams.csv"),
                "--metadata", str(self.root / "metadata.json"), "--mode", "ridge",
                "--output-dir", str(self.root)]
        with patch("sys.argv", args), contextlib.redirect_stdout(io.StringIO()):
            main()
        path = next(self.root.glob("forecast_*.json"))
        exported = ForecastDocument.model_validate_json(path.read_text())
        self.assertEqual(len(exported.prediction_runs[0].artifacts), 2)
        self.assertEqual(len(exported.prediction_runs[0].artifacts[0].sha256), 64)
        with patch("sys.argv", ["backend_data", "validate", str(path)]), contextlib.redirect_stdout(io.StringIO()) as output:
            main()
        self.assertIn("Valid schema", output.getvalue())


if __name__ == "__main__":
    unittest.main()
