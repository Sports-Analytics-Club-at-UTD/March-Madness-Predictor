import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from backend_data.convex import convert_forecast, main, run_convex
from backend_data.export import build_forecast
from backend_data.models import ForecastDocument
from examples.generate_forecast_example import example_document
from test_backend_data import bracket_fixture


ROOT = Path(__file__).resolve().parents[1]


class ConvexImportTests(unittest.TestCase):
    def setUp(self):
        self.document = example_document()
        self.roster = json.loads((ROOT / "examples/convex-roster.example.json").read_text())

    def test_distinct_strategies_advance_their_own_winners(self):
        payload = convert_forecast(self.document, self.roster)
        first, second = payload["items"]
        self.assertEqual(first["tournament"]["games"][-1]["topTeamId"], "site-example-0")
        self.assertEqual(second["tournament"]["games"][-1]["topTeamId"], "site-example-1")
        self.assertEqual(second["tournament"]["winnerTeamId"], "site-example-3")
        self.assertNotEqual(first["tournament"]["slug"], second["tournament"]["slug"])
        self.assertEqual(payload, convert_forecast(self.document, self.roster))
        self.assertTrue(all("topScore" not in g for g in first["tournament"]["games"]))
        self.assertNotIn("market_data", payload)

    def test_aliases_are_explicit_and_injective(self):
        self.roster["teams"][0]["name"] = "Other spelling"
        with self.assertRaisesRegex(ValueError, "Provide an alias"):
            convert_forecast(self.document, self.roster)
        aliases = {"Example Team 0": "site-example-0"}
        self.assertEqual(convert_forecast(self.document, self.roster, aliases)["items"][0]["tournament"]["teams"][0]["name"], "Other spelling")
        with self.assertRaisesRegex(ValueError, "Multiple forecast teams"):
            convert_forecast(self.document, self.roster, {"example-team-0": "site-example-1"})
        with self.assertRaisesRegex(ValueError, "Unknown site team"):
            convert_forecast(self.document, self.roster, {"example-team-0": "missing"})
        self.roster["teams"].append({**self.roster["teams"][1], "id": "duplicate-name"})
        with self.assertRaisesRegex(ValueError, "Provide an alias"):
            convert_forecast(self.document, self.roster, aliases)

    def test_missing_bracket_wrong_season_and_competition(self):
        self.roster["season"] = 2025
        with self.assertRaisesRegex(ValueError, "season"):
            convert_forecast(self.document, self.roster)
        self.roster["season"] = 2026
        payload = self.document.model_dump(mode="json")
        payload["brackets"] = []
        with self.assertRaisesRegex(ValueError, "complete bracket"):
            convert_forecast(ForecastDocument.model_validate(payload), self.roster)
        payload = self.document.model_dump(mode="json")
        payload["market_data"] = None
        payload["tournament"]["competition"] = "Women's college basketball"
        with self.assertRaisesRegex(ValueError, "men's"):
            convert_forecast(ForecastDocument.model_validate(payload), self.roster)

    def test_real_export_shape_layout_and_extra_prediction_team(self):
        table, brackets, tree = bracket_fixture()
        document = build_forecast(table, {"season": 2026}, "shap", brackets, tree)
        data = document.model_dump(mode="json")
        data["teams"].append({"id": "extra-team", "name": "First Four loser"})
        # Input game list order must not control canvas ordering.
        data["games"].reverse()
        document = ForecastDocument.model_validate(data)
        roster = {"season": 2026, "logoStorageId": "synthetic-logo", "teams": [
            {"id": f"canonical-{i}", "name": team.name, "seed": i % 16 + 1, "logoUrl": "/favicon.ico"}
            for i, team in enumerate(document.teams) if team.id != "extra-team"
        ]}
        items = convert_forecast(document, roster)["items"]
        self.assertEqual(len(items), 2)
        for item in items:
            tournament = item["tournament"]
            self.assertEqual(len(tournament["teams"]), 64)
            self.assertEqual(len(tournament["games"]), 63)
            games = {game["id"]: game for game in tournament["games"]}
            for game in games.values():
                for source, participant in (("topSourceGameId", "topTeamId"), ("bottomSourceGameId", "bottomTeamId")):
                    if source in game:
                        self.assertEqual(game[participant], games[game[source]]["winnerTeamId"])
            labels = {game["label"].split(" · ")[0] for game in games.values() if game["round"] == 4}
            self.assertEqual(labels, {"Region 1", "Region 2", "Region 3", "Region 4"})

    def test_invalid_progression_is_rejected(self):
        data = self.document.model_dump(mode="json")
        data["brackets"][0]["picks"][-1]["winner_team_id"] = "example-team-1"
        with self.assertRaises(ValueError):
            ForecastDocument.model_validate(data)

    def test_first_four_and_gapped_rounds_are_rejected(self):
        data = self.document.model_dump(mode="json")
        data["tournament"]["includes_first_four"] = True
        with self.assertRaisesRegex(ValueError, "First Four"):
            convert_forecast(ForecastDocument.model_validate(data), self.roster)
        data["tournament"]["includes_first_four"] = False
        data["games"][0]["round_number"] = 2
        data["games"][1]["round_number"] = 2
        data["games"][2]["round_number"] = 3
        with self.assertRaisesRegex(ValueError, "balanced"):
            convert_forecast(ForecastDocument.model_validate(data), self.roster)

    def test_team_only_cli_fails_before_network_access(self):
        data = self.document.model_dump(mode="json")
        data["games"], data["brackets"], data["market_data"] = [], [], None
        data["tournament"].update(topology_status="unavailable", bracket_team_count=None)
        with tempfile.TemporaryDirectory() as temp:
            forecast = Path(temp) / "team-only.json"
            forecast.write_text(json.dumps(data))
            with patch("sys.argv", ["convex", str(forecast), "--site-dir", "/tmp/site"]), patch("backend_data.convex.run_convex") as remote, contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as failure:
                    main()
            self.assertEqual(failure.exception.code, 1)
            remote.assert_not_called()

    def test_offline_cli_never_calls_convex(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "preview.json"
            argv = ["convex", str(ROOT / "examples/forecast.example.json"), "--roster", str(ROOT / "examples/convex-roster.example.json"), "--dry-run", "--output", str(output)]
            with patch("sys.argv", argv), patch("backend_data.convex.run_convex") as remote, contextlib.redirect_stdout(io.StringIO()):
                main()
            remote.assert_not_called()
            self.assertEqual(len(json.loads(output.read_text())["items"]), 2)

    def test_cli_uses_same_deployment_for_reads_and_writes(self):
        argv = ["convex", str(ROOT / "examples/forecast.example.json"), "--site-dir", "/tmp/site", "--prod"]
        with patch("sys.argv", argv), patch("backend_data.convex.run_convex", side_effect=[self.roster, []]) as remote, contextlib.redirect_stdout(io.StringIO()):
            main()
        self.assertEqual([call.args[1] for call in remote.call_args_list], ["forecastImports:getRoster", "forecastImports:importForecast"])
        self.assertTrue(all(call.args[-1] is True for call in remote.call_args_list))

    def test_convex_cli_is_invoked_without_a_shell(self):
        with tempfile.TemporaryDirectory() as temp:
            site = Path(temp)
            cli = site / "node_modules/convex/bin/main.js"
            cli.parent.mkdir(parents=True)
            cli.touch()
            (site / "convex").mkdir()
            (site / "convex/forecastImports.ts").touch()
            with patch("backend_data.convex.subprocess.run") as run:
                run.return_value.returncode = 0
                run.return_value.stdout = "[]"
                run_convex(site, "forecastImports:importForecast", {"title": "$(do-not-execute)"}, True)
                self.assertNotIn("shell", run.call_args.kwargs)
                self.assertEqual(run.call_args.args[0][-1], "--prod")
                self.assertEqual(json.loads(run.call_args.args[0][-2]), {"title": "$(do-not-execute)"})


if __name__ == "__main__":
    unittest.main()
