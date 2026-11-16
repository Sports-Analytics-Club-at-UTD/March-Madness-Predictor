"""Convert a 64-team forecast against the site's real 2026 roster and topology.

`tests/fixtures/site-2026-roster.json` is extracted from the `site` branch's
`convex/data/marchMadness2026.ts` (68 teams including First Four, 32 round-one
pairs). `convex-payload-2026.json` is the committed converter output, which
`forecastImports.test.cjs` feeds to the site's real validator and mutation.
Regenerate it with `PYTHONPATH=. uv run python tests/test_convex_site_roster.py --write`.
"""

import json
from pathlib import Path
import sys
import unittest

import pandas as pd

from backend_data.convex import convert_forecast
from backend_data.export import build_forecast
from backend_data.models import ForecastDocument

FIXTURES = Path(__file__).parent / "fixtures"
ALIASES = {"Miami OH": "miami-ohio"}  # the forecast spells one team differently


def site_forecast():
    site = json.loads((FIXTURES / "site-2026-roster.json").read_text())
    names = {team["id"]: team["name"] for team in site["roster"]["teams"]}
    seeds = {team["name"]: team["seed"] for team in site["roster"]["teams"]}
    forecast_name = lambda name: "Miami OH" if name == "Miami (Ohio)" else name
    teams = [forecast_name(names[i]) for pair in site["round1"] for i in pair]
    pairs = [(i, i + 1) for i in range(0, 64, 2)]
    rounds, edges, previous = [1] * 32, [], list(range(32))
    for round_number in range(2, 7):
        current = []
        for i in range(0, len(previous), 2):
            current.append(32 + len(edges))
            edges.append((previous[i], previous[i + 1]))
            rounds.append(round_number)
        previous = current
    seed = lambda name: seeds["Miami (Ohio)" if name == "Miami OH" else name]
    ev, leverage, rows = [], [], []
    for slot in range(63):
        if slot < 32:
            ev_options = leverage_options = [teams[i] for i in pairs[slot]]
        else:
            a, b = edges[slot - 32]
            ev_options, leverage_options = [ev[a], ev[b]], [leverage[a], leverage[b]]
        ev.append(min(ev_options, key=seed))  # chalk: better seed advances
        # The leverage strategy takes an upset in every fifth game.
        leverage.append(max(leverage_options, key=seed) if slot % 5 == 0 else min(leverage_options, key=seed))
        points = 10 * 2 ** (rounds[slot] - 1)
        row = {"game_slot": slot + 1, "round_number": rounds[slot], "round_points": points, "leverage_ratio": 1.2}
        for strategy, winner in (("max_ev", ev[-1]), ("leverage", leverage[-1])):
            row.update({f"{strategy}_pick": winner, f"{strategy}_win_prob": .4,
                        f"{strategy}_public_pct": .3, f"{strategy}_expected_pts": points * .4})
        rows.append(row)
    table = pd.DataFrame({"team": teams, "season": 2026, "ensemble_score": 1., "simulated_mean": 1.1,
                          "percentile_5": -.2, "percentile_95": 2.4,
                          "probability_ge_1_win": .6, "probability_ge_2_wins": .3})
    tree = {"teams64": teams, "round1_pairs": pairs, "tree_edges": edges,
            "slot_rounds": rounds, "actual_slot_winners": [-1] * 63}
    document = build_forecast(table, {"season": 2026}, "shap", pd.DataFrame(rows), tree, run_id="site-2026-fixture")
    data = document.model_dump(mode="json")
    for run in data["prediction_runs"]:
        run["generated_at"] = "2026-03-15T12:00:00Z"  # exporter stamps the clock; pin it for a stable fixture
    return ForecastDocument.model_validate(data), site["roster"], site["round1"]


class SiteRosterTests(unittest.TestCase):
    def test_converts_against_real_site_roster_and_topology(self):
        document, roster, round1 = site_forecast()
        self.assertEqual(len(roster["teams"]), 68)  # includes First Four losers the forecast omits
        with self.assertRaisesRegex(ValueError, "Provide an alias.*Miami OH"):
            convert_forecast(document, roster)
        payload = convert_forecast(document, roster, ALIASES)
        self.assertEqual(len(payload["items"]), 2)
        site_ids = {team["id"] for team in roster["teams"]}
        for item in payload["items"]:
            tournament = item["tournament"]
            self.assertEqual((len(tournament["teams"]), len(tournament["games"])), (64, 63))
            self.assertLessEqual({team["id"] for team in tournament["teams"]}, site_ids)
            # Round one must reproduce the official matchups in official order.
            first = sorted((g for g in tournament["games"] if g["round"] == 1), key=lambda g: g["order"])
            self.assertEqual([[g["topTeamId"], g["bottomTeamId"]] for g in first], round1)
        self.assertNotEqual(*(item["tournament"]["winnerTeamId"] for item in payload["items"]))
        # 100 KB is the CLI argument cap enforced by run_convex.
        self.assertLess(len(json.dumps(payload, separators=(",", ":"))), 100_000)

    def test_committed_payload_matches_converter(self):
        document, roster, _ = site_forecast()
        expected = json.loads((FIXTURES / "convex-payload-2026.json").read_text())
        self.assertEqual(convert_forecast(document, roster, ALIASES), expected)


if __name__ == "__main__":
    if sys.argv[1:] == ["--write"]:
        document, roster, _ = site_forecast()
        (FIXTURES / "convex-payload-2026.json").write_text(
            json.dumps(convert_forecast(document, roster, ALIASES), indent=1) + "\n")
    else:
        unittest.main()
