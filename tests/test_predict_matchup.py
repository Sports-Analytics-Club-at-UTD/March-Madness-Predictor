"""Unit tests for predict_matchup.py."""

import unittest
from predict_matchup import (
    resolve_team_name,
    compute_pairwise_win_probability,
    KNOWN_ALIASES,
)


class TestPredictMatchup(unittest.TestCase):

    def test_alias_resolution(self):
        self.assertEqual(resolve_team_name("uconn"), "UConn")
        self.assertEqual(resolve_team_name("connecticut"), "UConn")
        self.assertEqual(resolve_team_name("unc"), "North Carolina")
        self.assertEqual(resolve_team_name("pitt"), "Pittsburgh")
        self.assertEqual(resolve_team_name("nova"), "Villanova")
        self.assertEqual(resolve_team_name("ole miss"), "Ole Miss")
        self.assertEqual(resolve_team_name("st marys"), "Saint Mary's")
        self.assertEqual(resolve_team_name("st johns"), "St. John's")

    def test_symmetry_neutral_court(self):
        t1 = {
            "team": "Team A",
            "net_rtg": 25.0,
            "off_rtg": 118.0,
            "def_rtg": 93.0,
            "pace": 68.0,
            "off_efg": 54.0,
            "def_efg": 46.0,
            "off_tov": 0.16,
            "def_tov": 0.18,
            "off_orb": 32.0,
            "def_drb": 72.0,
            "off_ftr": 32.0,
            "def_ftr": 26.0,
        }
        t2 = {
            "team": "Team B",
            "net_rtg": 15.0,
            "off_rtg": 110.0,
            "def_rtg": 95.0,
            "pace": 66.0,
            "off_efg": 50.0,
            "def_efg": 48.0,
            "off_tov": 0.18,
            "def_tov": 0.16,
            "off_orb": 28.0,
            "def_drb": 68.0,
            "off_ftr": 28.0,
            "def_ftr": 30.0,
        }

        # A vs B
        res1 = compute_pairwise_win_probability(t1, t2, home_court=0)
        # B vs A
        res2 = compute_pairwise_win_probability(t2, t1, home_court=0)

        # Symmetry: P(A beats B) + P(B beats A) == 1.0
        self.assertAlmostEqual(res1["prob_team1"] + res1["prob_team2"], 1.0, places=5)
        self.assertAlmostEqual(res1["prob_team1"], res2["prob_team2"], places=4)
        self.assertGreater(res1["prob_team1"], 0.50)

    def test_home_court_advantage_impact(self):
        t1 = {
            "team": "Team A",
            "net_rtg": 20.0,
            "off_rtg": 115.0,
            "def_rtg": 95.0,
            "pace": 67.0,
            "off_efg": 52.0,
            "def_efg": 47.0,
            "off_tov": 0.16,
            "def_tov": 0.17,
            "off_orb": 30.0,
            "def_drb": 70.0,
            "off_ftr": 30.0,
            "def_ftr": 28.0,
        }
        t2 = dict(t1)
        t2["team"] = "Team B"

        # Equal teams on neutral court should be 50/50
        res_neutral = compute_pairwise_win_probability(t1, t2, home_court=0)
        self.assertAlmostEqual(res_neutral["prob_team1"], 0.50, places=4)

        # Team 1 at home should have > 50%
        res_home1 = compute_pairwise_win_probability(t1, t2, home_court=1)
        self.assertGreater(res_home1["prob_team1"], 0.55)

        # Team 2 at home should have Team 1 < 50%
        res_home2 = compute_pairwise_win_probability(t1, t2, home_court=2)
        self.assertLess(res_home2["prob_team1"], 0.45)


if __name__ == "__main__":
    unittest.main()
