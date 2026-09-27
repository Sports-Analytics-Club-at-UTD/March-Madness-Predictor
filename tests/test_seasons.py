import unittest
from scrape import SEASONS, choose_team_list

class SeasonTests(unittest.TestCase):
    def test_assigned_seasons_include_2008(self):
        self.assertEqual(list(SEASONS), [2000, 2001, 2002, 2003, 2004,
                                         2005, 2006, 2007, 2008])

    def test_large_partial_field_and_disjoint_sources(self):
        teams={f'Team {i}' for i in range(63)}
        result, source=choose_team_list(teams,[{'homeTeam':'Team 0','awayTeam':'Missing'}])
        self.assertEqual(set(result),teams|{'Missing'})
        self.assertEqual(source,'games')
