import unittest
import pandas as pd
from pandas.testing import assert_frame_equal
from scrape import calculate_missing_four_factors, choose_team_list


class FourFactorsTests(unittest.TestCase):
    def frame(self):
        return pd.DataFrame({
            'teamStats.fieldGoals.made': [20, 20],
            'teamStats.fieldGoals.attempted': [50, 50],
            'teamStats.threePointFieldGoals.made': [6, 6],
            'teamStats.freeThrows.attempted': [10, 10],
            'teamStats.turnovers.total': [8, 8],
            'teamStats.rebounds.offensive': [10, 10],
            'opponentStats.rebounds.defensive': [30, 30],
            'teamStats.possessions': [60, 60],
            'teamStats.fourFactors.effectiveFieldGoalPct': [99, None],
            'teamStats.fourFactors.freeThrowRate': [0, None],
            'teamStats.fourFactors.turnoverRatio': [.9, None],
            'teamStats.fourFactors.offensiveReboundPct': [88, None],
        }, index=[4, 9])

    def test_preserves_api_and_fills_only_missing_rows(self):
        original = self.frame()
        result = calculate_missing_four_factors(original.copy())
        assert_frame_equal(result[original.columns], original)
        expected = {'calc_efg_pct': 46, 'calc_ft_rate': 20,
                    'calc_turnover_ratio': 8/60, 'calc_orb_pct': 25}
        for col, value in expected.items():
            self.assertTrue(pd.isna(result.loc[4, col]))
            self.assertAlmostEqual(result.loc[9, col], value)

    def test_complete_api_adds_nothing(self):
        original = self.frame().iloc[:1].copy()
        assert_frame_equal(calculate_missing_four_factors(original.copy()), original)

    def test_zero_missing_and_nonfinite_inputs(self):
        frame = self.frame().iloc[1:].copy()
        frame['teamStats.fieldGoals.attempted'] = 0
        frame['teamStats.possessions'] = 0
        frame['teamStats.freeThrows.attempted'] = float('inf')
        frame = frame.drop(columns='opponentStats.rebounds.defensive')
        result = calculate_missing_four_factors(frame)
        self.assertTrue(result.filter(like='calc_').isna().all().all())

    def test_estimated_team_possessions(self):
        frame = self.frame().iloc[1:].drop(columns='teamStats.possessions')
        result = calculate_missing_four_factors(frame)
        self.assertAlmostEqual(result.iloc[0]['calc_turnover_ratio'], 8/(50+4.4-10+8))

    def test_player_schema_and_missing_rebound_context(self):
        frame = pd.DataFrame({'fieldGoals.made':[20], 'fieldGoals.attempted':[50],
                              'threePointFieldGoals.made':[6], 'freeThrows.attempted':[10],
                              'turnovers':[8], 'rebounds.offensive':[10]})
        result = calculate_missing_four_factors(frame)
        self.assertAlmostEqual(result.iloc[0]['calc_efg_pct'],46)
        self.assertAlmostEqual(result.iloc[0]['calc_turnover_ratio'],100*8/62.4)
        self.assertTrue(pd.isna(result.iloc[0]['calc_orb_pct']))

    def test_game_turnover_scale(self):
        frame = self.frame().iloc[1:].copy()
        frame["gameId"] = 123
        result = calculate_missing_four_factors(frame)
        self.assertAlmostEqual(result.iloc[0]["calc_turnover_ratio"], 100*8/60)

    def test_choose_team_list_uses_raw_games_when_tournament_is_partial(self):
        tournament = {"A", "B", "C"}
        raw_games = [
            {"team": "A", "opponent": "B"},
            {"team": "C", "opponent": "D"},
            {"team": "E", "opponent": "F"},
        ]
        result, source = choose_team_list(tournament, raw_games)
        self.assertEqual(source, "games")
        self.assertEqual(result, ["A", "B", "C", "D", "E", "F"])

    def test_choose_team_list_keeps_tournament_when_it_is_full_field(self):
        tournament = {"A", "B", "C", "D", "E", "F"}
        raw_games = [{"team": "A", "opponent": "B"}, {"team": "C", "opponent": "D"}]
        result, source = choose_team_list(tournament, raw_games)
        self.assertEqual(source, "tournament")
        self.assertEqual(result, ["A", "B", "C", "D", "E", "F"])

    def test_empty(self):
        self.assertIsNone(calculate_missing_four_factors(None))
        self.assertTrue(calculate_missing_four_factors(pd.DataFrame()).empty)

    def test_game_orb_uses_team_total_only_for_missing_api(self):
        original = self.frame()
        original["gameId"] = [123, 124]
        original["teamStats.rebounds.total"] = [50, 50]
        result = calculate_missing_four_factors(original.copy())
        assert_frame_equal(result[original.columns], original)
        self.assertTrue(pd.isna(result.loc[4, "calc_orb_pct"]))
        self.assertAlmostEqual(result.loc[9, "calc_orb_pct"], 20)

    def test_game_orb_missing_or_zero_total_stays_missing(self):
        for total in (None, 0, float("inf")):
            with self.subTest(total=total):
                frame = self.frame().iloc[1:].copy()
                frame["gameId"] = 123
                if total is not None:
                    frame["teamStats.rebounds.total"] = total
                result = calculate_missing_four_factors(frame)
                self.assertTrue(pd.isna(result.iloc[0]["calc_orb_pct"]))

if __name__ == '__main__':
    unittest.main()
